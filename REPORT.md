# MiniMax-M3.1-Flash 双方案实测报告

**A 方案**：4× DGX Spark GB10，vLLM TP4（社区 leo-dcfa 路线）
**B 方案**：1× DGX Spark，EXL3 2.15bpw 单机（社区 vcruz305 路线）

- 测试日期：2026-10-24
- 硬件：4× DGX Spark GB10（每台 121.6 GiB 统一内存，200GbE RoCE 环网，无交换机）+ RTX 6000D 备机
- 统一口径：流式请求、`temperature=0`、剥离 TTFT、三次取中位数；预填用唯一前缀（nonce）绕开 prefix cache，并核对 `cached_tokens=0`
- 所有数字均为本机实测，未引用社区数据

---

## 一、结论先说

| 项目 | A：四机 vLLM TP4 | B：单机 EXL3 2.15bpw |
|---|---|---|
| 是否跑通 | ❌ **未跑通**（能启动、能出 token，但输出不可用） | ✅ **完整跑通并实测** |
| 权重加载 | ✅ 252.4 GB，178.86 s | ✅ 143 GB |
| 服务启动 | ✅ `/health` 200 | ✅ `/v1/models` 200 |
| 推理输出 | ❌ `17*23` 返回 NUL 字节流 | ✅ `17×23=391` 正确 |
| 单流解码 | 未能测得 | **19.7 tok/s**（开草稿）/ 13.8 tok/s（无草稿） |
| 冷预填 | 未能测得 | **357–369 tok/s**（开草稿）/ 374–389（无草稿） |
| 上下文 | 32K（另测 262K 启动） | 8K（开草稿）/ 16K（无草稿） |
| 并发 | 设计 8 序列 | 不支持（`max_num_seqs=1`） |
| 内存余量 | — | 0.5–3.4 GiB，非常紧张 |

**一句话**：单机 EXL3 这条路在我们机器上走通了；四机 vLLM TP4 卡在 GB10（sm_121）的 MXFP8 内核支持上。

---

## 二、B 方案：单机 EXL3 2.15bpw ✅

### 2.1 环境搭建

| 项目 | 内容 |
|---|---|
| 权重 | `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`，38 文件 / 143 GB（16 主分片 + dspark 草稿） |
| 基础镜像 | `vllm/vllm-openai:minimax-m31-flash`（digest `86f582d…`），与 A 用的是同一个镜像 |
| 插件 | `vcruz305/vllm-exl3` @ `41fdb7d`（mixed-K 修复、indexer MXFP8、q4 KV 池、UMMA 加载整形） |
| 运行时 | `vcruz305/exllamav3` @ `fdd7736`（act-3 修复，缺了每个 MoE 层都出乱码） |
| 编译 | 容器内编译 CUDA 扩展（`TORCH_CUDA_ARCH_LIST=12.1a`, `MAX_JOBS=4`），**13 分钟**出镜像 |
| 机器 | `10.0.0.X`（aidgx4），121.6 GiB 统一内存 |

### 2.2 实测数据（配置一：DSpark 开，8K 上下文，fp8 KV 0.85 GB）

| 指标 | 实测值 |
|---|---|
| **单流解码**（代码提示，512 tok） | 19.5 / 19.7 / 21.4 tok/s → **中位 19.7** |
| 单流解码（per-stream 中位） | 20.3 / 20.5 / 22.5 tok/s |
| TTFT（短提示） | 0.98 / 1.13 / 1.13 s |
| **冷预填 1,672 tok** | **357 tok/s**（TTFT 4.68 s） |
| **冷预填 3,146 tok** | **358 tok/s**（TTFT 8.78 s） |
| **冷预填 4,587 tok** | **369 tok/s**（TTFT 12.42 s） |
| 缓存命中校验 | `cached_tokens=0` ✓（非缓存假象） |
| DSpark 投机 | 2.83 token/步，接受率 61.1% |
| 正确性 | `17×23=391` ✓，中文与推理链正常 |

### 2.3 实测数据（配置二：无草稿，16K 上下文，fp8 KV 1.24 GB）

| 指标 | 实测值 |
|---|---|
| 单流解码 | **13.8 tok/s**（13.8 / 13.8 / 13.8，per-stream 14.1） |
| TTFT | 0.69 / 0.70 / 0.73 s |
| 冷预填 1,670 / 3,145 / 6,058 / 11,879 tok | **386 / 389 / 382 / 374 tok/s** |
| 并发 C4 | 聚合仍 13.8 tok/s（请求串行，TTFT 56 s）——`max_num_seqs=1` |

### 2.4 与作者公布值对照

| 项目 | 作者公布 | 我们实测 | 差异原因 |
|---|---|---|---|
| 解码（代码提示） | 28.97 tok/s | 19.7 tok/s | 作者用 48K 上下文 + q4 KV；我们只能压到 8K（内存少约 6 GiB） |
| 预填 | 379.4 / 369.1 / 327.6 tok/s（6.6K/30K/45K） | 357 / 358 / 369 tok/s（1.7K/3.1K/4.6K） | 同量级，趋势一致 |
| 上下文 | 48K（推荐档） | 8K（可用最大稳定档） | 见 2.5 |
| 并发 | 单请求 | 单请求 | 一致 |

### 2.5 踩坑（B 路线）

1. **`serve.sh` 变量名笔误**：第 233 行 `$PERIODIC_TR_S` 应为 `$PERIODIC_TRIM_S`，`set -u` 下脚本 30 秒即 abort
2. **默认 `q4-48k` 必 OOM**：加载完 111.38 GiB 已分配，只剩 0.69 GiB，guard 在 `avail=185 MiB` 时杀容器
3. **`q4-32k` 同样被杀**：`avail=383 MiB` < 204 MiB 阈值
4. **`fp8-16k` 无草稿可跑**，开 DSpark 则会瞬时打到 383 MiB
5. **要开 DSpark 必须三压**：guard 阈值 384→128 MiB、KV 池 1.24 GB→0.85 GB、上下文 16K→8K
6. **`KVB=600000000` 直接报错**：8K 上下文需 0.57 GiB KV，实际只有 0.55 GiB，KV 池必须 ≥ 0.85 GB
7. 连 `1Panel-ollama`（17 MB）都要停掉，否则启动期内存更紧

### 2.6 B 的定位

- 适合：**离线批处理、单会话私有助手、长文档摘要**（预填 370 tok/s 很稳，8K 上下文够单篇文档）
- 不适合：多用户并发（`max_num_seqs=1`）、32K 以上上下文、低延迟交互
- 额外好处：**只占一台机器**，另外三台可以继续跑 DeepSeek 生产

---

## 三、A 方案：四机 vLLM TP4 ❌ 未跑通

### 3.1 做成的事

| 环节 | 结果 |
|---|---|
| 权重下载 | 252.4 GB / 49 文件，逐文件字节校验与 HF 清单**完全一致** |
| 四机权重分发 | RoCE 网内 rsync，单文件 10.2 GB 约 40 秒，四台一致 |
| 镜像 | `vllm/vllm-openai:minimax-m31-flash`（39.6 GB）四台到位 |
| NCCL | 打了 sparkring 补丁的 NCCL 2.30.7 分发到四台，ring 通信正常建立 |
| **权重加载** | ✅ **178.86 秒**；Marlin FP4 回退正常触发（日志：`Your GPU does not have native support for FP4 computation… Weight-only FP4 compression will be used leveraging the Marlin kernel`） |
| **KV 池** | ✅ 262K 配置下分配 **838,784 tokens**（3.2× 并发） |
| **服务启动** | ✅ `/health` 返回 **200** |

**结论：模型加载、显存规划、四机 TP 通信全部没问题，卡死在 kernel 层。**

### 3.2 卡住的三个点（按出现顺序）

**① FlashInfer CuTe-DSL MXFP8 量化内核（sm_121 不支持）**

```
flashinfer/quantization/fp8_quantization.py:248 → mxfp8_quantize
  → kernels/mxfp8_quantize.py:1479 → mxfp8_quantize_cute_dsl
  → cutlass_dsl/tvm_ffi_provider.py:1084
cutlass.cutlass_dsl.tvm_ffi_provider.CUDADialectError:
  error: cudaErrorIllegalAddress (700)
```

原因：vLLM 的 `mxfp8_utils.py` 判断 `has_device_capability(100)`，GB10 是 **sm_121**，满足条件后走 CuTe-DSL 后端，该后端只对 SM100 验证过。
处理：patch 成 `not has_device_capability(120)` 强制走 torch 回退 —— **有效，越过了这一步**。

**② FlashInfer cutlass MXFP8 稠密 GEMM（第二道墙）**

```
vllm/utils/flashinfer.py:1072 mm_mxfp8
  → flashinfer/gemm/gemm_base.py:7165
  → flashinfer/autotuner/autotuner.py:2367 choose_one
torch.AcceleratorError: CUDA error: an illegal memory access
```

- 把 `backend="cute-dsl"` 改成 `"cutlass"` → **无效**，cutlass 的 MXFP8 GEMM 在 sm_121 上同样非法访存
- 关 DSpark 草稿 → **无效**（与投机解码无关）
- `--enforce-eager` 关图捕获 → **无效**（不是 cudagraph 问题）

**③ 最小参数集能起、不能推理**

按上次跑通时的最小参数（`--trust-remote-code`、32K、fp8 KV、`--enforce-eager`、无 attention/moe 覆盖）：

- `/health` = 200 ✅
- `POST /v1/chat/completions` → 返回 `\u0000\u0000\u0000…`（NUL 字节流）❌

稠密层/MoE 的量化路径落到了错误的 kernel 上，权重被按错误布局解读。

### 3.3 为什么社区能跑、我们跑不了

| 对比项 | leo-dcfa | 我们 |
|---|---|---|
| 镜像 digest | `86f582d…` | **同一份** |
| 权重 | 官方 NVFP4 experts + MXFP8 dense | **同一份**（逐字节校验一致） |
| 显存 | 121 GiB（GB10） | 121.6 GiB |
| 结果 | 31 / 42–68 tok/s | 启动 OK，推理崩溃或输出 NUL |

镜像和权重都一样，差异只能在**驱动/固件版本**或**他没写出来的环境变量**上。他日志同样出现 Marlin 回退，但没提 MXFP8 GEMM 崩溃——说明他那台 cutlass MXFP8 GEMM 能跑，我们的不能，最可能是 CUDA 驱动版本差异（我们 DGX OS 580 / CUDA 13.0）。

### 3.4 下一步三条路

1. **等 vLLM 修 sm_121**：issue #50925 已讨论 NVFP4 在 sm121 回退 Marlin，MXFP8 GEMM 的 sm121 支持还没人提
2. **换 SGLang 路线**：`knapcio` 用 `lmsysorg/sglang:dev-minimax-m31-flash` + DFlash2 在 4×GB10 上跑到 45.5 tok/s（speedbench 几何均值）；arm64 镜像需再确认
3. **稠密层降 bf16**（`mxfp8/emulation.py` 的 `VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD`）：每 rank 权重 63 GB→约 80 GB，20 GB KV 要砍，成本高、收益不确定

---

## 四、两方案横向对比（只比能测到的）

| 维度 | B（单机 EXL3） | A（四机 vLLM） |
|---|---|---|
| 机器占用 | 1 台 | 4 台 |
| 权重体积 | 143 GB | 252.4 GB |
| 量化 | EXL3 SAGE 2.15 bpw | MXFP8 稠密 + NVFP4 专家 |
| 单流解码 | **19.7 tok/s** | 未测得（社区 31 / 42–68） |
| 预填 | **357–369 tok/s** | 未测得（社区 ~2.1k） |
| 上下文 | 8K 稳定 | 262K 启动成功、推理崩 |
| 并发 | 1 | 设计 8 |
| 显存余量 | 极紧（0.5–3.4 GiB） | 社区报 26–29 GiB |
| 可用性 | ✅ 立即可用 | ❌ 待内核支持 |

---

## 五、排障记录（对后续复用有价值）

### 5.1 下载侧

| 问题 | 现象 | 解法 |
|---|---|---|
| hf-mirror 重定向到 Xet | 下载卡在 0 B/s，进程活着但不增长 | `HF_HUB_DISABLE_XET=1`（前期有效）；后期大分片仍 302 到 `cas-bridge.xethub.hf.co` |
| Xet CDN 直连被墙 | `curl` 连不上，`http=000` | 走 VPS 代理 `74.82.201.147:28888`（单连接 2.5 MB/s） |
| 镜像站限速 | 多连接只有 0.4–1.5 MB/s | 关掉其他下载独占，或代理 + 12–16 段并发（4–7 MB/s） |
| 中断续传踩坑 | `curl -C -` 追加导致文件超出预期（10.24 GB vs 10.21 GB） | 必须下 `.part` 再按字节数校验改名，不能直接续写到目标文件 |

### 5.2 分发侧

| 问题 | 现象 | 解法 |
|---|---|---|
| `docker save \| pv \| ssh docker load` 产空壳 | 只有 9.47 kB | 去掉 `pv`，直接 `docker save \| ssh docker load`，39.6 GB 实测成功 |
| `ctr images export` | 立即失败 | 放弃 containerd 导出 |
| rsync 复制到半成品 | 目标端出现 26 MB 的"完整"分片 | 必须校验字节数；源端落盘后再同步 |
| 节点间无 SSH 互信 | `Permission denied (publickey)` | 把 `.134` 的 ed25519 公钥写入各节点对应用户（`aidgx2/3/4`）的 `authorized_keys` |

### 5.3 启动侧

| 问题 | 解法 |
|---|---|
| `--gpu-memory-utilization 0.90` 起不来 | 降 0.85（0.90 需 109.5 GB > 可用 108.93 GB） |
| `--kv-cache-dtype nvfp4` 报错 | 降 `fp8`（NVFP4 KV 只支持 SM100 MSA 后端） |
| DSpark `FLASH_ATTN` 不支持 fp8 KV | 改 `TRITON_ATTN` |
| 多机 `mp` 后端 | 需 `--nnodes/--node-rank/--master-addr`，rank 0 出 API、其余 `--headless`（已固化脚本） |

---

## 六、结论与建议

1. **短期要用 MiniMax-M3.1-Flash**：走 B 方案（单机 EXL3）——19.7 tok/s 解码、370 tok/s 预填、8K 上下文、单请求，够私有化单用户场景，且不占生产集群。
2. **四机 vLLM TP4 目前不可用**：不是配置问题，是 FlashInfer 的 MXFP8 内核在 GB10（sm_121）上非法访存；社区仓库能跑说明存在能工作的驱动组合，值得后续对照他的驱动版本再试。
3. **生产已恢复**：四机 DeepSeek-V4.1-Flash-1017 MXFP4 TP4（C1 106.7 / C8 317.0 / C16 439.9 tok/s，预填 ≈5,000 tok/s），8888 与 5701 均 200。
4. **磁盘**：A 权重 252.4 GB × 4 台 + B 权重 143 GB × 1 台，合计约 1.15 TB 仍占用；回收可删 A 的权重（B 建议保留备用）。

---

*报告生成：2026-10-24　全部数据来自本机实测，社区数据均已标注出处*
