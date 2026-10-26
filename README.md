# MiniMax-M3.1-Flash 在 DGX Spark (GB10) 上的双方案实测与部署

MiniMax-M3.1-Flash（428B 总参 / 23B 激活 / 1M 上下文）在 GB10 上的两条路线全记录：单机 EXL3 2.15bpw 跑通，四机 vLLM TP4 卡在 SM121 内核。含 A/B 两套完整部署步骤。

**交流群**：QQ 群 **1128010665**（加群备注：GitHub）

---

- 测试时间：2026-10-24
- 硬件：4 × DGX Spark（GB10，121.6 GiB 可用统一内存）+ 1 台备机
- 两条路线：**A** 四机 vLLM TP4（社区 leo-dcfa 路线） / **B** 单机 EXL3 2.15bpw（社区 vcruz305 路线）
- 结论：**B 跑通、A 未跑通**（加载 OK、服务 OK、推理崩在 SM121 的内核上）

---

## 0. 模型规格

| 项 | 数值 |
|---|---|
| 总参数 / 激活 | ~428B / ~23B（128 专家 top-4 + 1 共享） |
| 层数 | 60 层，每层含 MiniMax Sparse Attention (MSA) |
| KV heads | 4 |
| 上下文 | 1,048,576 |
| 量化（官方） | 路由专家 **NVFP4 group-16**；稠密/注意力 **MXFP8** |
| 权重体积 | 252.4 GB / 49 个 safetensors / ~89,632 张量 |
| 内置草稿头 | DSpark（5 层小模型） |
| 多模态 | 支持图像/视频输入 |
| 许可 | MiniMax Community License（非商业免费；商用需署名 + 报备 api@minimax.io；年收入 >$20M 需书面授权） |

---

## 1. 方案 A：四机 vLLM TP4（未跑通）

### 1.1 环境

| 项 | 配置 |
|---|---|
| 镜像 | `vllm/vllm-openai:minimax-m31-flash`（39.6 GB，digest `sha256:86f582d…`） |
| 引擎版本 | vLLM `0.1.dev23341+g8744dd2fb` / torch 2.14.0+cu130 / triton 3.8.0 / flashinfer 0.7.2 / transformers 5.19.0 / NCCL 2.30.7 |
| 后端 | `--distributed-executor-backend mp`，`--nnodes 4`，每机一 rank |
| 量化后端 | `--moe-backend marlin`，KV `fp8` |
| 通信 | 200GbE RoCE 环网 + sparkring 补丁 NCCL |

### 1.2 做成的部分

| 环节 | 结果 |
|---|---|
| 权重下载 | 252.4 GB / 49 文件，逐文件字节校验与 HF 清单**完全一致** |
| 四机分发 | RoCE 网内 rsync，单文件 10.2 GB 约 40 秒 |
| 权重加载 | ✅ **178.86 秒** |
| Marlin FP4 回退 | ✅ 日志确认：`Your GPU does not have native support for FP4 computation… Weight-only FP4 compression will be used leveraging the Marlin kernel` |
| KV 池 | ✅ 262K 配置下 **838,784 tokens**（3.2× 并发） |
| 服务启动 | ✅ `/health` 返回 **200** |
| 推理 | ❌ 见 1.3 |

### 1.3 三道墙（按出现顺序）

**① FlashInfer CuTe-DSL MXFP8 量化内核**

```
flashinfer/quantization/fp8_quantization.py:248 → mxfp8_quantize
  → kernels/mxfp8_quantize.py:1479 → mxfp8_quantize_cute_dsl
  → nvidia_cutlass_dsl/.../tvm_ffi_provider.py:1084
cutlass.cutlass_dsl.tvm_ffi_provider.CUDADialectError:
  error: cudaErrorIllegalAddress (error code: 700)
```

原因：vLLM `mxfp8_utils.py` 判断 `has_device_capability(100)`，GB10 是 **sm_121**，满足条件 → 走只对 SM100 验证过的 CuTe-DSL 后端。
处理：patch 成 `not has_device_capability(120)` 强制走 torch 回退 —— **有效，越过**。

**② FlashInfer cutlass MXFP8 稠密 GEMM**

```
vllm/utils/flashinfer.py:1072 mm_mxfp8
  → flashinfer/gemm/gemm_base.py:7165
  → flashinfer/autotuner/autotuner.py:2367 choose_one
torch.AcceleratorError: CUDA error: an illegal memory access
```

| 尝试 | 结果 |
|---|---|
| `backend="cute-dsl"` → `"cutlass"` | ❌ 同样非法访存 |
| 关闭 DSpark 草稿 | ❌ 无关，仍崩 |
| `--enforce-eager` 关图捕获 | ❌ 无关 |

**③ 最小参数集能起不能推理**

`--trust-remote-code` + 32K 上下文 + fp8 KV + `--enforce-eager`：

- `/health` = 200 ✅
- `POST /v1/chat/completions` → 返回 `\u0000\u0000\u0000…`（NUL 字节流）❌

### 1.4 与社区对照

| 对比项 | 社区 leo-dcfa | 我们 |
|---|---|---|
| 镜像 digest | `86f582d…` | 同一份 |
| 权重 | 官方 NVFP4 + MXFP8 | 同一份（逐字节校验） |
| 显存 | 121 GiB | 121.6 GiB |
| 结果 | 31 / 42–68 tok/s | 启动 OK，推理崩或 NUL |

镜像与权重完全相同 → 差异只能在**驱动/固件版本**或未公开的环境变量。

---

## 2. 方案 B：单机 EXL3 2.15bpw（跑通）

### 2.1 环境

| 项 | 配置 |
|---|---|
| 权重 | `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`，38 文件 / **143 GB**（16 主分片 + dspark） |
| 基础镜像 | 同 A：`vllm/vllm-openai:minimax-m31-flash` |
| exllamav3 | `vcruz305/exllamav3` @ `fdd7736`（act-3 修复，缺了就每层出乱码） |
| vllm-exl3 插件 | `vcruz305/vllm-exl3` @ `41fdb7d`（mixed-K、indexer MXFP8、q4 KV 池） |
| 编译 | `TORCH_CUDA_ARCH_LIST=12.1a`、`MAX_JOBS=4`，约 **13 分钟** |
| 机器 | 1 × DGX Spark（121.6 GiB） |

### 2.2 两套可用配置

**配置一：DSpark 开，8K 上下文，fp8 KV 0.85 GB**

| 指标 | 实测 |
|---|---|
| 单流解码（代码，512 tok） | 19.5 / 19.7 / 21.4 tok/s → 中位 **19.7** |
| 单流 per-stream 中位 | 20.3 / 20.5 / 22.5 tok/s |
| TTFT | 0.98 / 1.13 / 1.13 s |
| 冷预填 1,672 tok | **357 tok/s**（TTFT 4.68 s） |
| 冷预填 3,146 tok | **358 tok/s**（TTFT 8.78 s） |
| 冷预填 4,587 tok | **369 tok/s**（TTFT 12.42 s） |
| 缓存校验 | `cached_tokens=0` ✓ |
| DSpark 投机 | 2.83 token/步，接受率 61.1% |
| 正确性 | `17×23=391` ✓ |

**配置二：无草稿，16K 上下文，fp8 KV 1.24 GB**

| 指标 | 实测 |
|---|---|
| 单流解码 | **13.8 tok/s**（三次均 13.8，per-stream 14.1） |
| TTFT | 0.69 / 0.70 / 0.73 s |
| 冷预填 1,670 / 3,145 / 6,058 / 11,879 tok | **386 / 389 / 382 / 374 tok/s** |
| 并发 C4 | 聚合仍 13.8 tok/s（请求串行，TTFT 56 s）—— `max_num_seqs=1` |

### 2.3 内存剖面（日志实测）

| 阶段 | cuda_alloc | avail |
|---|---:|---:|
| PRE_KV | 109.17 GiB | 2.82 GiB |
| POST_KV | 111.38 GiB | **0.69 GiB** |
| ready（48K 档） | — | 0.32 GiB → 被 guard 杀 |
| ready（32K 档） | — | 0.38 GiB → 被 guard 杀 |
| ready（8K + 无草稿） | — | **3.45 GiB** ✅ |

### 2.4 与作者公布值对照

| 项目 | 作者公布 | 我们实测 | 差异原因 |
|---|---|---|---|
| 解码（代码） | 28.97 tok/s | 19.7 tok/s | 作者 48K 上下文 + q4 KV；我们只能压到 8K（内存少 ~6 GiB） |
| 预填 | 379.4 / 369.1 / 327.6（6.6K/30K/45K） | 357 / 358 / 369（1.7K/3.1K/4.6K） | 同量级 |
| 上下文 | 48K | 8K 稳定 | 见 2.3 |
| 并发 | 单请求 | 单请求 | 一致 |

---

## 3. 踩坑记录

### B 路线

1. `serve.sh` 第 233 行笔误 `$PERIODIC_TR_S`（应为 `$PERIODIC_TRIM_S`），`set -u` 下 30 秒 abort
2. 默认 `q4-48k` **必 OOM**：加载后 111.38 GiB 已占，剩 0.69 GiB，guard 在 avail=185 MiB 杀容器
3. `q4-32k` 同样被杀（avail=383 MiB < 204 MiB）
4. 要开 DSpark 必须三压：guard 阈值 384→128 MiB、KV 1.24→0.85 GB、上下文 16K→8K
5. `KVB=600000000` 报错：8K 需 0.57 GiB KV，实际只有 0.55 GiB → 池必须 ≥0.85 GB
6. 连 `1Panel-ollama`（17 MB）都要停掉

### 下载 / 分发

| 问题 | 现象 | 解法 |
|---|---|---|
| hf-mirror 重定向到 Xet | 下载卡 0 B/s，进程活着不增长 | `HF_HUB_DISABLE_XET=1`（部分有效），大分片仍需代理 |
| Xet CDN 直连被墙 | `http=000` | 走 VPS 代理，单连接 2.5 MB/s |
| 镜像站限速 | 0.4–1.5 MB/s | 代理 + 12–16 段并发（4–7 MB/s） |
| `curl -C -` 续写 | 文件超预期（10.24 GB vs 10.21 GB） | 必须 `.part` + 字节校验后改名 |
| `docker save \| pv \| ssh load` | 只产出 9.47 kB 空壳 | 去掉 `pv`，直接 `docker save \| ssh docker load` |
| rsync 复制半成品 | 目标端 26 MB 的"完整"分片 | 校验字节数；源端落盘后再同步 |

---

## 4. 结论

| | A 四机 vLLM | B 单机 EXL3 |
|---|---|---|
| 跑通 | ❌ | ✅ |
| 机器占用 | 4 台 | 1 台 |
| 权重 | 252.4 GB | 143 GB |
| 解码 | 未测得（社区 31 / 42–68） | **19.7 tok/s** |
| 预填 | 未测得（社区 ~2.1k） | **357–389 tok/s** |
| 上下文 | 262K 启动成功 | 8K |
| 并发 | 设计 8 | 1 |
| 可用性 | 待内核修复 | ✅ 立即可用 |

**短期要用**：走 B（单机 EXL3），适合离线批处理、单会话助手、长文档摘要；不适合并发与 >8K 上下文。
**四机路线**：等 vLLM 修 sm_121 的 MXFP8 内核，或转 SGLang（社区 knapcio 报 45.5 tok/s，需确认 arm64 镜像）。

---

## 5. 数据来源

| 内容 | 来源 |
|---|---|
| 模型规格 | 官方权重仓库 `config.json` + 文件清单 |
| 社区对照数据 | `leo-dcfa/minimax-m31-flash-4x-spark`、`vcruz305/MiniMax-M3.1-Flash-EXL3-Spark-recipe`、`knapcio/MiniMax-M3.1-Flash-4x-DGX-Spark-TP4` |
| 其余全部数字 | 本机实测 |

*报告生成：2026-10-26　交流群：QQ 1128010665*

---

## 附录 A · 部署方法

### A.1 方案 B（单机 EXL3）部署步骤

```bash
# 1. 环境准备（DGX Spark 上）
sudo apt-get install -y git build-essential python3-venv zstd
pip install -U "huggingface_hub[cli]"

# 2. 下载权重（143 GB / 38 文件）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw \
  --local-dir ~/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw \
  --max-workers 8

# 3. 克隆两个固定 revision 的 fork
mkdir -p ~/m31B && cd ~/m31B
git clone -b feat/moe-act3-batched https://github.com/vcruz305/exllamav3
git -C exllamav3 checkout fdd77361971144e8d175e09e7b50b7100739fd95
git clone -b feat/minimax-q4-prefill https://github.com/vcruz305/vllm-exl3
git -C vllm-exl3 checkout 41fdb7d99a441d37938c2a1bd600ac963c18a3cb

# 4. 构建镜像（基础镜像 + exllamav3 编译 + 插件）
git clone https://github.com/vcruz305/MiniMax-M3.1-Flash-EXL3-Spark-recipe
cd MiniMax-M3.1-Flash-EXL3-Spark-recipe
SRC_DIR=$HOME/m31B MODEL_DIR=$HOME/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw \
  MAX_JOBS=4 bash scripts/build_image.sh        # 约 13 分钟

# 5. 启动（按内存余量选档）
#    48K 档（作者推荐，我们的机器 OOM）
bash scripts/serve.sh
#    8K 档（我们实测可用）
SRC_DIR=$HOME/m31B PROFILE=fp8-16k MAXLEN=8192 KVB=850000000 \
  WARMUP=0 DSPARK=1 GUARD_MEM_FLOOR_KIB=131072 bash scripts/serve.sh
```

**启动前必须做**：

```bash
sudo docker rm -f m31-flash-tp1 2>/dev/null
sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'   # 清页缓存
free -g                                            # 要求 ≤117 GiB available
```

### A.2 内存档位对照（我们机器实测）

| 档位 | 上下文 | KV 池 | DSpark | 结果 | 就绪时 avail |
|---|---:|---:|---|---|---:|
| q4-48k（作者推荐） | 48K | 2.37 GB | 开 | ❌ guard 杀 | 0.32 GiB |
| q4-32k | 32K | 1.58 GB | 开 | ❌ guard 杀 | 0.38 GiB |
| fp8-16k | 16K | 1.24 GB | 开 | ❌ guard 杀 | 0.38 GiB |
| fp8-16k | 16K | 1.24 GB | **关** | ✅ 稳定 | **3.45 GiB** |
| fp8-16k | **8K** | **0.85 GB** | 开 | ✅ 稳定 | 0.5–0.8 GiB |

### A.3 验证

```bash
# 健康
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
# → MiniMax-M3.1-Flash-2.15bpw

# 正确性
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"MiniMax-M3.1-Flash-2.15bpw",
  "messages":[{"role":"user","content":"17*23=?"}],"max_tokens":32
}' | jq -r '.choices[0].message.content'
# → 391

# 投机接受率（DSpark 开时）
sudo docker logs m31-flash-tp1 2>&1 | grep 'SpecDecoding' | tail -1
# → Mean acceptance length: 2.83, Avg Draft acceptance rate: 61.1%
```

### A.4 方案 A（四机 vLLM）部署步骤

```bash
# 1. 四台拉同一镜像（39.6 GB/台，走代理）
sudo docker pull vllm/vllm-openai:minimax-m31-flash

# 2. 四台同路径放权重 252.4 GB
# 3. 挂补丁 NCCL（switchless ring）
# 4. 每台起一个 rank：
sudo docker run -d --name m31f-tp4 --gpus all --network host --shm-size 16g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v /models/MiniMax-M3.1-Flash:/model:ro \
  vllm/vllm-openai:minimax-m31-flash /model \
  --served-model-name minimax-m3.1-flash \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --nnodes 4 --node-rank 0 --master-addr 10.10.10.1 --master-port 29511 \
  --block-size 128 --disable-hybrid-kv-cache-manager \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.80 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --moe-backend marlin --trust-remote-code \
  --host 0.0.0.0 --port 8000      # rank 1-3 用 --headless
```

> 按此配置能加载完、能起 `/health` 200，但**推理会在 FlashInfer MXFP8 内核崩**（详见正文第 1.3 节）。这不是参数问题，是 sm_121 的内核支持缺失。

### A.5 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `PERIODIC_TR_S: unbound variable` | serve.sh 第 233 行笔误 | 改 `$PERIODIC_TRIM_S` |
| `guard: KILL(avail)` | 可用内存跌破阈值 | 降档：上下文/KV/DSpark 三选一 |
| `0.57 GiB KV needed > 0.55 available` | KV 池太小装不下设定上下文 | `KVB` ≥ 0.85 GB |
| 输出乱码 `reatreat...` | exllamav3 缺 act-3 修复 | 必须 checkout `fdd7736` |
| 输出 NUL 字节 | 量化路径错（vLLM 路线） | 见正文 1.3 |

---

*文档生成：2026-10-26　数据口径与来源见各节*
