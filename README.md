# MiniMax-M3.1-Flash 在 DGX Spark (GB10) 上的双方案实测

MiniMax-M3.1-Flash（428B 总参 / 23B 激活 / 1M ctx）在 4 台 GB10 上的两条路线：**单机 EXL3 量化**跑通，**四机 vLLM TP4** 卡在 SM121 的内核支持。

**交流群**：QQ 群 **1128010665**（加群备注：GitHub）

## 结论一句话

单机 EXL3 能跑（19.7 tok/s / 8K 上下文 / 单请求），四机 vLLM 跑到 `/health` 200 但推理崩在 FlashInfer 的 MXFP8 内核上（`cudaErrorIllegalAddress`）。

## 实测数据

**B 方案 · 1 台 GB10 · EXL3 2.15bpw**

| 指标 | 数值 |
|---|---|
| 单流解码（DSpark 开） | 19.5 / 19.7 / 21.4 → 中位 **19.7 tok/s** |
| 单流解码（无草稿） | 13.8 tok/s |
| 冷预填 | 357 / 358 / 369 tok/s（1.7K / 3.1K / 4.6K 输入） |
| 上下文上限 | 8K（48K/32K 档在本机必 OOM） |
| 并发 | 不支持（`max_num_seqs=1`） |

**A 方案 · 4 台 GB10 · vLLM TP4**

| 环节 | 结果 |
|---|---|
| 权重加载 | ✅ 252.4GB / 178.86 s |
| KV 池 | ✅ 838,784 tokens |
| `/health` | ✅ 200 |
| 推理 | ❌ NUL 字节流 / cutlass MXFP8 GEMM 崩溃 |

## 文档

- [REPORT.md](REPORT.md)：双方案完整报告（环境、逐项数据、失败堆栈、排障过程、下载与分发踩坑）

## 为什么值得看

全网关于 MiniMax-M3.1-Flash 在 GB10 上的资料几乎都只报成功。这份记录把**失败的那一半**写全了：三道墙、每道的报错原文、以及哪些绕过尝试无效。
