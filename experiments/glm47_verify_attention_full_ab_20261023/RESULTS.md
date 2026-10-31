# GLM-4.7 A100 Verify Attention 修复验证

## 修复方法

原配置开启 deterministic inference。Verify 实际从 `forward_extend` 提前进入 `_forward_extend_unified`，使用 M128 的 query tile 并串行扫描长 KV；普通 Decode 则使用 grouped Split-KV。这是同一个 Triton backend 内部的路径差异。此前将本轮路径描述为普通 `extend_attention_fwd` fallback 不准确。

修复限定 topk1、Q3 的因果链：把三个 query 展开为可见长度分别为 P+1、P+2、P+3 的 Decode 行，复用原 Decode 内核和固定 256-token KV 分块。每层都在 GPU 上重建指针、物理 KV slot 索引及 split 数；复用 Graph 地址稳定的 scratch，支持非连续索引，不缓存会随 Graph replay 改变的元数据。计时包含这些开销。

默认关闭的开关：`SGLANG_SM80_CAUSAL_VERIFY_DECODE=1`。仅资格测试通过的 SM80、BF16、page64、24Q/2KV heads、D128、deterministic、topk1/Q3、无 SWA/sinks/logit cap 等修饰时启用；其他配置保持原路径。

## 真实模型配对测量

065／167，8×A100，GLM-4.7-AWQ / Humming INT4，BF16 KV；Attention TP4×DP2，Expert EP8，MTP 均开启。B8，每条逻辑 prefix P4096；计时前原生 HiCache 已有 CPU4032/GPU0，因此首轮重算尾部64加新 token1。生成64 token。计时含 HiCache→GPU，排除权重加载和源 Bank→HiCache 导入。两组复用同一语义 target+draft KV Bank，不使用合成 KV。

| 配置 / cohort | 吞吐 output tok/s | TTFT mean / median / P99 (s) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (s) |
|---|---:|---|---|---|
| ep8_verify_original / timed | 95.17 | 0.554 / 0.554 / 0.554 | 70.691 / 71.399 / 76.603 | 5.007 / 5.052 / 5.380 |
| ep8_verify_original / repeat | 94.94 | 0.542 / 0.542 / 0.542 | 71.079 / 71.791 / 76.993 | 5.020 / 5.065 / 5.393 |
| ep8_verify_decode / timed | 131.52 | 0.528 / 0.528 / 0.528 | 47.291 / 47.770 / 53.415 | 3.507 / 3.537 / 3.893 |
| ep8_verify_decode / repeat | 131.10 | 0.588 / 0.588 / 0.588 | 46.537 / 47.015 / 52.663 | 3.519 / 3.549 / 3.905 |

P99 是 N8 请求样本的 nearest-rank 最大值，不代表稳定尾延迟估计。两次 cohort 是同一进程独立恢复、重新执行，输出逐 token 一致；不是跨重启统计置信区间。

## 分段归因

每卡、每次真实 target Verify Graph，八卡均值；每卡采12次，另有24次 draft。Attention 包含本轮标注内的投影和 KV 操作，不仅是纯 attention kernel。

| target Verify 分项 (ms) | 原版 | 修复版 |
|---|---:|---:|
| Attention | 78.290 | 17.546 |
| Expert | 24.750 | 25.058 |
| 通信 | 33.776 | 33.875 |
| 其他未覆盖 Graph 时间 | 26.100 | 26.064 |
| Graph 总计 | 162.916 | 102.543 |

Attention 降低 77.6%，target Graph 降低 37.1%；首个 production cohort TPOT 降低 33.1%，吞吐提高 38.2%。Expert 与通信基本不变，支持性能收益主要来自 Attention 路径修正。

外部 CUDA event 节点在两组 Graph 中相同；额外的逐 Graph 完成同步和读回只用于后续诊断 cohort。因此分段不能直接相加解释 HTTP TPOT，也不能把八张卡时间相加。

## 正确性与启用边界

单卡资格测试覆盖4K/16K、Q3及Q1边界、空 prefix、ragged长度、非连续物理索引和修改元数据后的 Graph replay；与 FP32 参考在 atol0.002/rtol0.02 内一致，并与逐 query 普通 Decode 结果 bitwise 一致。详情见 [单卡结果](../glm47_verify_attention_sm80_20261023_v3/RESULTS.md)。

本次两组各自的两次输出均完全可重复。但与同 Bank、同输入、同提交版本的历史 MTP-off 结果相比，原版只有 2/8 条完全一致，修复版为 7/8；修复版 request1 在第60个生成 token 开始分歧。不同 reduction 顺序可能影响 BF16 多层模型的 greedy 决策，现有证据不能证明严格无损 MTP。该选项保持默认关闭；还需定位剩余端到端数值差异。

本轮仅重测 EP8；没有用 EP8 数据代替 PP2×EP4 的新性能结果。

## 实现及审计

- 引擎实现：`python/sglang/kernels/ops/attention/verify_as_decode.py`；入口：`TritonAttnBackend._forward_extend_unified`；配置：`srt/environ.py`。
- 本次实际运行使用固定9e48f876引擎加冻结的等效 runtime adapter；未改写旧实验冻结源码或历史数据。根工作树中的核心与 GPU 资格测试版 AST 一致。
- 扩展既有 `test_verify_splitkv.py`；CPU support gate 测试通过，新增 registered GPU 测试文件本身尚未单独执行；GPU资格测试与实际模型运行已有独立证据。
- 历史报告路径说明已更正，16项报告生成测试通过，旧测量值不变。
- 原版无修复 receipt；修复版八个 rank 的启用 receipt 均存在。两组 performance/profile 有效，Guard postcheck、模型退出和 GPU 清理通过，共享租约已释放。
- 原始运行：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_verify_attention_full_ab_20261023/runs/original_vs_causal_decode_r1`
- 机器可读汇总：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_verify_attention_full_ab_20261023/summary.json`
- Guard postcheck：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_verify_attention_full_ab_20261023/runs/original_vs_causal_decode_r1/dashboard_guard/checks/1792741602014785935-post.json`
