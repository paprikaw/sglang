# 35B 五种四卡部署：简单报告

五种布局均在同一台 4×H100 80GB NVLink 节点测量；Qwen3.6-35B-A3B BF16，全部开启 HiCache。每批64请求，正式点各3轮。

P 为每请求缓存的前缀长度。Prefill 每请求新增512 tokens、输出1 token；Decode 每请求新增1 token、输出128 tokens。下表单位均为 tokens/s，Prefill 仅计算新增输入，吞吐越高越好。

| 部署 | P=8,256 Prefill | P=8,256 Decode | P=130,304 Prefill | P=130,304 Decode | 长前缀 Decode 的 CPU 命中比例 |
|---|---:|---:|---:|---:|---:|
| EP4 | 85,767 | 5,474 | 33,360 | 2,292 | 0.0% |
| 2×EP2 | 99,446 | 4,961 | 17,406 | 1,553 | 57.8% |
| 4×EP1 | 123,670 | 5,373 | 14,336 | 636 | 99.5% |
| 2×PP2 | 105,066 | 4,148 | 33,966 | 1,537 | 49.0% |
| PP4 | 89,664 | 3,276 | 29,117 | 1,6.,58 | 0.0% |

长前缀合计缓存8,339,456 tokens。开启 HiCache 后，EP4 对 4×EP1 的本轮长前缀 Decode 优势为 3.61×；4×EP1 主要从 CPU 命中，原先未开 HiCache 的24.1×不适用于本表。

GPU KV token 池：EP4 8.693M；2×EP2 6.400M；4×EP1 0.641M；2×PP2 7.006M；PP4 8.419M。长前缀占 EP4 GPU 池95.9%、PP4 GPU 池99.1%。

CPU 命中比例是实际原生计数的 CPU 前缀 tokens / 全部前缀 tokens，属于请求层面的命中，不是迁移字节比例。各部署每卡 CPU 池固定64 decimal GB KV + 8 decimal GB GDN。

| 受控 CPU 重放，P=130,304 | Prefill tokens/s | Decode tokens/s | CPU 前缀 token 命中比例（Decode） |
|---|---:|---:|---:|
| EP4 | 12,432 | 1,230 | 100.0% |
| 2×EP2 | 14,126 | 1,327 | 100.0% |
| 4×EP1 | 14,379 | 636 | 100.0% |
| 2×PP2 | 24,827 | 1,405 | 100.0% |
| PP4 | 22,607 | 1,669 | 100.0% |

CPU→GPU 恢复包含在请求耗时中；批前写入 fence 和主动移出 GPU 缓存位于计时外。这两种重放有不同缓存历史和生成分支，不能用两表差值单独归因 CPU 传输成本。这是合成的重复前缀负载，尚未测生产 trace 的命中率，也未搜索 HiCache 总容量上限。

PP 按阶段、前缀长度、缓存模式选择本轮最快的有效候选。PP2/PP4 最终使用原生或 early-send 与 microbatch 调优；PP2 lookahead 实验路径未通过输出资格，tail-local 反馈在长上下文出现 slot 复用错误，均排除在正式比较外。这是最新 PP benchmark 仍需补齐的正确性资格。

所有正式数据已从逐请求样本独立重算；冻结输入、缓存来源、Guard 后检、自有模型清理和租约释放已核验。

[完整结果与配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/REPORT.md) · [逐轮数据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/full_metrics.csv) · [核验记录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/validation.json)
