# 35B 单节点 Decode：是否值得动态调整 PP／EP 比例

更新：2026-10-30。[返回统一索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。

**建议暂缓在这个场景实现 PP／EP 动态配比控制器。** 找到了更大的性能差距，但优化 PP2 没有形成独占的生成速度优势：低并发有更快的单卡副本，高并发 EP4 更快。当前更适合把结果整理为部署边界 benchmark。

## 1. 直接看绝对数字

以下均在 **同一个四卡 H100 节点 011** 测量。P 是每请求已有的 GPU 缓存长度，N 是整个节点同时生成的请求数；Q1，输出 128 token。每格是三轮的逐请求算术平均 TPOT，单位 **ms/token**，没有挑最快样本。

| P | 全组 N | EP4 | 2×EP2 | 4×单卡 | 2×优化 PP2 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 6.122 | 6.534 | **4.752** | 5.495 |
| 4096 | 64 | **9.220** | 10.683 | 10.310 | 11.181 |
| 256 | 128 | **10.630** | 12.919 | 未通过运行资格 | 15.402 |
| 32768 | 32 | 9.386 | 10.410 | 9.392 | 10.115 |
| 131072 | 8 | 7.970 | 8.354 | **6.550** | 7.306 |
| 131072 | 32 | **14.936** | 15.598 | 容量不足 | 15.293 |

32K、N32 的 EP4 与单卡差约 0.006 ms，视为近似持平。

- **确实找到了更强反差。** P256、N128 时，EP4 的 TPOT 比优化 PP2 低约 **31.0%**。
- **PP 的低并发优势被更强候选覆盖。** P4096、N8 时，PP2 比 EP4 快，但四个单卡副本又比 PP2 低约 **13.5%**。
- 主扫描覆盖 **16 个 workload**，包括 N8–128、P256–131072 和不同主题输入。PP2 在 **0/16** 个工况取得最低平均 TPOT。原生 PP4 的补充测试也未提供更快的候选。

单卡本身也能避免专家交换；35B 能放进单张 H100 时，PP 的少通信优势没有独占性。这是与结果相符的结构解释，本轮没有测量各组件的因果贡献。

## 2. 一个真实但较小的 PP 优势

P131072、N32 时，PP2 的**整请求时间**略好，生成间隔仍略慢：

| 指标，均为请求／轮次算术平均 | EP4 | 2×PP2 |
|---|---:|---:|
| TPOT，ms/token | **14.936** | 15.293 |
| TTFT，ms | 387.45 | **239.30** |
| 请求 E2E，ms | 2284.38 | **2181.57** |
| 整批完成时间，ms | 2290.39 | **2216.47** |

PP2 的请求 E2E 低 **4.5%**，整批时间低 **3.2%**。按指标恒等式，TTFT 省 148.15 ms，随后 127 个生成间隔多花 45.34 ms，净省 102.81 ms。这里的 TTFT 是 GPU 热缓存上的 Q1 首 token，不能当成冷 Prefill 性能。

另一个 H100 节点按相反次序做五轮复测，E2E 为 EP4 **2311.8 ms**、PP2 **2208.9 ms**，仍约 4.5%；TPOT 为 **15.197 / 15.501 ms**。该小优势可重复，但没有得到我们想找的强双向 TPOT 区间。

## 3. 理想动态选型还有多少空间

在同节点、同内存档、所有候选都成功测量的两工况混合格点中，假设每批都能免费选择布局：

| 指标 | 相对最佳固定候选的最高代理改进 | 加入 PP2 的额外改进 |
|---|---:|---:|
| workload 加权平均 TPOT | 7.46% | 0 |
| workload 加权平均请求 E2E | 4.83% | 0 |
| 串行 cohort 的整批时间 | 6.74% | 0 |

这些收益来自 **EP4 与单卡副本**之间的选择。例：60% 的 cohort 为 P4096/N64，40% 为 P4096/N8 的多主题输入，理想平均 TPOT 从最佳固定 EP4 的 8.052 ms 降至 7.451 ms；删除 PP 候选不改变结果。

这是封闭同步批次的选型代理，**不是在线 autoscaling 收益，也不是异质混合实例池收益的普遍上界**。128K/N32 的容量条件不同，未混入这张共同可行性表；其 PP 整请求小收益已单列。排队、分流后的 batch、缓存迁移及扩缩容成本均未实测。

## 4. 范围、失败点与可信度

- 完整 Qwen3.6-35B-A3B：40 层、256 experts、top-8，BF16 权重／KV、FP32 GDN。实际 EP4 每层每卡持有 64 个专家，PP 的负责层持有 256 个专家；此次直接运行完整布局，未使用旧组件图的理想查表值替代实测。
- 两个已预约 H100 节点并行，每个候选始终只用单节点四卡。未测试单节点八卡、跨节点布局或更大模型。
- 三轮主扫描，关键点另换节点、反向布局顺序、五轮复测。所有完整正式样本保留；逐请求检查 GPU 缓存命中与输出长度。PP2 每个正式输出与同组原生 PP2 逐 token 相同，并核对四个 stage 的实际 lookahead、退休与 Graph 资格计数。
- **N128 四单卡没有合格性能值。** 0.95 内存档启动失败；提高到 0.98 并限制 KV 后，在未计时预热中仍因 GDN CoW 临时申请 1.82 GiB 而 OOM。它是本引擎的未解决基线问题，不能写成单卡的普遍性能／容量边界，也没有被当作 PP 的收益。
- 128K/N32 的四单卡容量限制有独立依据：每卡仅 BF16 routed Expert 权重约 60 GiB，加八条 128K 前缀的 BF16 KV 约 20 GiB，已占 80 GiB，尚未包含其他权重、GDN 和 workspace。[计算依据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/capacity_derivation.json)。
- 主扫描末尾的单卡 dense 启动失败使整次 run 的 postcheck 失败；**44 个已完成性能点另按原始 Guard 校验器复核权重来源、完整模型资格及清理，原失败标记保留**。其余五次纳入统计的运行完整通过 postcheck。[逐次审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/RUN_AUDIT.md)。

共归档 **122 条布局×负载×运行性能记录、388 个正式 cohort、16376 个请求**，包括独立复测；不同节点、协议、内存档没有混合求平均。PP4 使用原生调度，PP2 使用当前已验证的优化；结论不代表所有可能 PP 实现的上限。

## 5. 下一步建议

**先收束这个 35B 场景，保留 benchmark，不投入完整 PP／EP 配比控制器。** 若继续单节点机制线，先研究单卡副本与 EP4 的选择，确认真实到达流下是否仍有净收益。若继续 PP／EP 主线，下一次应选权重必须跨卡放置的模型，重新验证 PP 能否进入最佳配置区间；本报告没有对这些更大模型作结论。

## 数据入口

- [主扫描全部绝对均值](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/primary/TABLES.md) · [独立五轮复测](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/confirmation/TABLES.md) · [EP2／PP4 补充对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/controls/TABLES.md)
- [完整均值 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/full_metrics.csv) · [全部正式轮次 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/formal_rounds.csv) · [实际 rank／stage batch 计数](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/forward_histograms.csv)
- [逐工况 PP 增量](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/pp_marginal_by_case.csv) · [混合权重代理](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/primary/oracle_pairs.csv) · [决策统计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/decision.json) · [已完成 arm 的独立审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/accepted_arm_audit.json)
- [实验协议](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/PROTOCOL.md) · [原始来源与复现清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/manifest.json)
