# Q06：35B BF16 的 A100 缺失点已补齐

固定两张 A100069 GPU0/1（NV12），完整 Qwen3.6-35B-A3B，EP2 对两槽 PP2。8 请求，每点正式 3 轮；Prefill 每条无旧缓存、4096 新输入、输出 1，Decode 每条 GPU 已缓存 4096、新输入 1、输出 128。

| 本轮吞吐，token/s | EP2 | PP2 | PP 变化 |
|---|---:|---:|---:|
| Prefill 新输入 | 27644.14 | 31841.16 | +15.18% |
| Decode 输出 | 90.80 | 121.76 | +34.09% |

可以展示的 insight：同为 35B Decode，A100 的 PP +34.09%，已有 H100 的 PP -25.08%；H100 35B 的 Prefill 均值则为 PP +14.15%。本轮 35B Prefill 的 Routed Expert 为 EP 2.245／PP 2.222 ms／每卡每层调用，没有明显增加，同时 PP 避免 EP 约 0.957 ms 的逐层 Dispatch＋Combine。此前 122B A100 两槽 PP 的 Expert 则增加约 13.6%。因此应联合考虑通信节省与本地 Expert 执行效率。

这些是具体部署组合的证据：A100／H100 的内核、Graph、overlap 不同；35B BF16 与 122B FP8 也不是大小单变量对照。H100 Prefill 有波动；本次仅 NVLink 对内，不能外推跨节点阈值。

[完整短报告、四点统计和三维对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/REPORT.md) · [均值／中位数／P99](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/full_metrics.csv) · [分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2/analysis/components.csv) · [归档清单](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_a100_ep2_pp2_20261025_r2/manifest.json)。正式 96 请求计时、8 份完整 rank trace、Guard／cache／数值资格与清理核验均通过；GPU 租约已释放。
