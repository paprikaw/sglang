# H100 Prefill: EP2 and PP2 after existing optimizations

Q4096长Prefill中，PP early相对EP的B32重复/多主题吞吐高17.6%/19.0%，首token跨布局一致；B8独立10轮复测高20.3%。提前发送相对原生PP仅约0.15%，未见稳定Prefill收益。**Q128/Q512的原生EP/PP存在显著数值差异，其吞吐只能作为实现观测，暂不作为等质量部署结论。**

完整 Qwen3.6-35B-A3B BF16，固定 SG8702697022，H100178 NVLink GPU0/1，顺序运行。EP2 的 Attention 为 DP2×TP1、MoE 为 EP2，每卡40层；PP2 为两级、TP1/EP1，每级20层、两个流水槽。

每请求缓存上下文 P=0，新输入 Q=128、512、4096，输出1个token。B表示同时提交的请求数，所测 B8/B32；总新输入为 B×Q。重复文档以及B32/Q4096的32主题输入分别报告。两种布局按相同偶/奇文档分组等待请求到齐，然后进入原生调度；等待计入客户端延迟。EP每个DP副本拿B/2个请求；PP微批次请求上限B/2。每个Attention实例的Prefill token预算4096，实际每次forward的token数量从独立trace核对。

双方保留此前相同的Expert内核选择器：只有本地输入T≤16会选小tile内核，本轮Prefill大形状使用原生DeepGEMM；因此不能把Decode的小tile加速写成本轮Prefill加速。EP保留原生shared-expert overlap开关，DeepEP auto在大Prefill上采用normal路径，TBO关闭。PP对照policy0与既有policy1：提前提交原始activation发送，保留原生序列化、producer event和消息内容。Decode专用固定小形状传输与下一槽位提前forward未使用。

每配置先3轮冷Prefill预热，再5轮profiler关闭的正式计时；PP两种policy交替、正反排序。每轮先清缓存，逐请求验证cached_tokens=0、输入Q、输出1。输入吞吐=整批B×Q/首请求开始至最后请求首token的墙钟时间；TTFT和E2E为逐请求均值。本实验只有1个输出token，TTFT=E2E，没有TPOT。

## 吞吐与延迟

| Workload | EP2 input tok/s ↑ | PP2 native input tok/s ↑ | PP2 early input tok/s ↑ | Early vs native PP | Early PP vs EP | EP TTFT/E2E ms ↓ | Early PP TTFT/E2E ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| b8_q128_repeated | 5,906.0 | 9,069.4 | 9,085.7 | +0.18% | +53.84% | 171.57 | 94.81 |
| b8_q512_repeated | 20,757.5 | 33,600.7 | 33,483.7 | -0.35% | +61.31% | 195.22 | 102.86 |
| b8_q4096_repeated | 59,734.5 | 74,451.5 | 66,941.5 | -10.09% | +12.06% | 422.80 | 341.99 |
| b32_q128_repeated | 18,565.0 | 28,350.4 | 27,959.4 | -1.38% | +50.60% | 215.97 | 124.16 |
| b32_q512_repeated | 52,294.1 | 64,236.4 | 64,251.2 | +0.02% | +22.87% | 285.80 | 184.00 |
| b32_q4096_repeated | 71,351.9 | 83,822.8 | 83,884.9 | +0.07% | +17.57% | 1100.89 | 840.91 |
| b32_q4096_diverse | 70,424.6 | 83,531.1 | 83,777.8 | +0.30% | +18.96% | 1116.13 | 841.01 |

![Prefill throughput](prefill_throughput.png)

图中误差线为5个整批吞吐样本的标准差；逐轮吞吐和makespan见cohorts.csv。

## Expert GEMM 与通信

以下来自另一次冷Prefill GPU profile，只统计host launch归属于真实Prefill层的GPU kernel，排除Decode调用。Expert GEMM只计routed Expert中的实际grouped GEMM kernel，包含本轮Prefill的grouped-contiguous GemmType1；不把完整MoE模块时间当成GEMM。Dispatch/combine计对应操作内GPU kernel，PP的Expert在本地，列为0，流水边界通信另属PP调度。每层时间先按各rank实际层调用数归一化，再对两个rank平均。GPU分项可能重叠，不能直接相加替代E2E。

| Workload | Layout | Local source tokens/forward mean | Actual layer calls/card | Expert GEMM µs/card/layer | Dispatch+combine µs/card/layer | Expert GEMM ms/card/cohort | Attention module ms/card/cohort |
|---|---|---:|---:|---:|---:|---:|---:|
| b8_q128_repeated | EP2 | 512.0 | 40 | 246.39 | 134.83 | 9.86 | 6.22 |
| b8_q128_repeated | PP2 native | 512.0 | 40 | 611.32 | 0.00 | 24.45 | 6.17 |
| b8_q128_repeated | PP2 early activation | 512.0 | 40 | 612.44 | 0.00 | 24.50 | 6.19 |
| b8_q512_repeated | EP2 | 2048.0 | 40 | 385.10 | 368.96 | 15.40 | 15.24 |
| b8_q512_repeated | PP2 native | 2048.0 | 40 | 775.15 | 0.00 | 31.01 | 15.09 |
| b8_q512_repeated | PP2 early activation | 2048.0 | 40 | 774.80 | 0.00 | 30.99 | 15.15 |
| b8_q4096_repeated | EP2 | 4096.0 | 160 | 460.02 | 585.57 | 73.60 | 125.30 |
| b8_q4096_repeated | PP2 native | 4096.0 | 160 | 878.30 | 0.00 | 140.53 | 124.95 |
| b8_q4096_repeated | PP2 early activation | 4096.0 | 160 | 879.18 | 0.00 | 140.67 | 124.52 |
| b32_q128_repeated | EP2 | 2048.0 | 40 | 308.04 | 288.13 | 12.32 | 18.90 |
| b32_q128_repeated | PP2 native | 2048.0 | 40 | 759.61 | 0.00 | 30.38 | 18.79 |
| b32_q128_repeated | PP2 early activation | 2048.0 | 40 | 760.01 | 0.00 | 30.40 | 18.76 |
| b32_q512_repeated | EP2 | 4096.0 | 80 | 440.17 | 565.73 | 35.21 | 57.35 |
| b32_q512_repeated | PP2 native | 4096.0 | 80 | 913.89 | 0.00 | 73.11 | 57.07 |
| b32_q512_repeated | PP2 early activation | 4096.0 | 80 | 915.69 | 0.00 | 73.26 | 56.68 |
| b32_q4096_repeated | EP2 | 4096.0 | 640 | 484.73 | 584.04 | 310.22 | 501.13 |
| b32_q4096_repeated | PP2 native | 4096.0 | 640 | 881.18 | 0.00 | 563.95 | 500.68 |
| b32_q4096_repeated | PP2 early activation | 4096.0 | 640 | 880.88 | 0.00 | 563.76 | 500.19 |
| b32_q4096_diverse | EP2 | 4096.0 | 640 | 494.32 | 562.03 | 316.37 | 501.17 |
| b32_q4096_diverse | PP2 native | 4096.0 | 640 | 881.58 | 0.00 | 564.21 | 500.87 |
| b32_q4096_diverse | PP2 early activation | 4096.0 | 640 | 881.48 | 0.00 | 564.14 | 500.22 |

![Prefill components](prefill_components.png)

Attention module包含projection及其他Attention模块内kernel，口径与此前Decode的Attention core不同；不能直接跨阶段相减。每rank的实际chunk范围、调用数量和kernel清单在分析文件中。

这些分项支持Attention相近、Expert执行与通信存在取舍，但不能用GEMM与通信kernel的和解释全部吞吐差距。例如B32/Q128的EP GEMM+dispatch/combine之和小于PP GEMM，PP的端到端吞吐仍更高；还需考虑完整执行顺序、调度、CPU开销与重叠。已检查的trace没有证据支持将差距直接归因于显式CUDA同步。

## 生效与正确性

- 每个正式PP policy1批次的stage0提前提交计数增加，stage1无提前提交；所有pending均清空。下一槽位lookahead、固定形状fast transport计数保持0，符合Prefill适用边界。
- PP两种policy的首token输出逐请求一致，EP原生重复轮输出一致性单独保留；双方输入SHA256一致。
- 原生GPU Prefill、客户端性能计时、独立profile各自保存；18项CPU检查通过。冻结输入、必需CPU权重缓存使用记录、Guard postcheck、owned cleanup和lease release通过。

初始r1在EP原生跨轮输出一致性检查时中止，PP尚未运行，清理通过。已保留诊断，r2补上EP完整本地请求分组和额外预热。本报告所有性能数据来自完整r2，未拼接r1。

## 可复现材料

- [正式指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/metrics.csv)
- [逐批次](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/cohorts.csv)
- [逐请求](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/requests.csv)
- [GPU分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/components.csv)
- [每rank分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/components_per_rank.csv)
- [完整分析](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/analysis/qwen35b_h100_prefill_optimized_r2_20261027/results.json)
- [冻结启动manifest](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/runs/qwen35b_h100_prefill_optimized_r2_20261027/launch_manifest.json)

## 独立定点复测与数值诊断

B8/Q4096主矩阵的5轮样本波动较大，独立补测每配置10轮。以下均为清缓存、profiler关闭的正式样本，未与主矩阵合并。

| Layout | Input tok/s ↑ | Sample standard deviation |
|---|---:|---:|
| EP2 | 64,788.8 | 2,228.5 |
| PP2 native | 77,814.0 | 98.3 |
| PP2 early | 77,926.9 | 110.9 |

提前提交相对原生PP仅 +0.145%；相对EP为 +20.28%。结合主矩阵其他点，未观察到可确认的Prefill提前发送收益。

该补测wrapper随后在**非计时**的Q4096原生PP logprob诊断发生CUDA OOM，故保留failed与5/9完成状态。上表三组完整计时均在失败前完成，已独立重算吞吐并验证零缓存、输入/输出长度、提前发送计数、冻结源、模型缓存使用及清理；没有补填未完成点。

短输入数值检查改为单独的PP诊断：静态显存比例0.60，为logprob临时张量保留空间；正式性能实验仍为0.94。本诊断没有任何性能样本，使用同权重、精度、输入及原生计算路径，与此前EP诊断比较。

| Workload | Comparison | Changed first tokens / comparisons | Max shared-top10 logprob delta | All changed predictions present in both top10 |
|---|---|---:|---:|---|
| b32_q128_repeated | EP_vs_PP_native | 4/64 | 5.080088 | False |
| b32_q128_repeated | PP_native_vs_early | 0/64 | 0.000000 | True |
| b32_q512_repeated | EP_vs_PP_native | 6/64 | 6.160327 | False |
| b32_q512_repeated | PP_native_vs_early | 0/64 | 0.000000 | True |

B32的Q128、Q512分别有2/32、3/32个请求的首token跨布局不同，两次冷重放均重现；共享top10候选的最大logprob差为5.08、6.16，且不同预测并不都出现在另一布局的top10中。因此这不是已有证据可忽略的近似并列，不能直接归因于BF16。PP原生/提前发送的top10 logprob则逐项相同，单独数值诊断的所有首token也都复现了各自正式性能实验。

这是原生布局之间的数值差异检查，不能仅凭同一模型名或BF16精度认定等价。短输入EP/PP的逐请求差异、对应候选概率及诊断输出能否重现正式性能输出，均保存在followups.json和numerical_details.csv；在原因明确前，短输入的端到端差距应视为该实现的观测值。Q4096的重复及多主题主实验首token跨布局一致，B8/Q4096独立复测也一致；本次仅验证第一个输出token。

## 本轮可以得出的结论

- 正式计时中PP吞吐均高于EP；Q4096重复/多主题输入的首token跨布局一致，B32吞吐分别高17.6%/19.0%。短输入数值等价尚未成立，不能据此宣布胜负切换边界。
- PP提前activation发送在Prefill上未见稳定收益；Decode专用下一槽位lookahead不适用于本轮Prefill。
- PP Expert GEMM仍更慢，Attention累计时间接近；EP每层的dispatch/combine及完整执行/调度路径共同影响端到端结果。
- 部署判断应分阶段：本轮长Prefill可优先评估原生PP；短输入还需先解决原生EP/PP数值差异，不能把本轮结果写成质量已完全等价的普遍结论。

- [独立补测与数值审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/report/followups.json)
- [数值汇总](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/report/numerical_summary.csv)
- [数值明细](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_prefill_ep_pp_optimized_20261027/report/numerical_details.csv)

## Existing-trace check: communication/computation overlap

复查B32、P=0、Q4096重复文本的两张GPU独立Prefill trace，没有启动新GPU实验。每张卡640次实际Prefill层调用，重新归因的GEMM与通信均值与主报告吻合。

| GPU kernels / card / actual Prefill layer | µs |
|---|---:|
| Dispatch | 324.82 |
| Routed Expert GEMM | 484.73 |
| Combine | 259.22 |
| Dispatch+combine 与 routed GEMM 的时间交集 | 0.00 |
| Dispatch+combine 与 Attention 的时间交集 | 0.00 |
| Dispatch+combine 与 shared expert 的时间交集 | 3.98 |

Shared Expert本身约46.74 µs/card/layer；现有shared-expert overlap只与约0.68%的通信GPU时间重叠。这是GPU时间轴交集，不等于已经确认的端到端可节省时间。通信口径继承主报告的dispatch/combine作用域内kernel，并非纯NVLink传输线时间。

同一批token有 Dispatch(A) → GEMM(A) → Combine(A) 的依赖。可利用独立请求微批：Dispatch(B)与GEMM(A)重叠，Combine(A)与GEMM(B)重叠；也可用另一批的Attention覆盖通信。优先保留足够大的Expert输入形状，避免为重叠切得过细而损失GEMM效率。

忽略启动/排空、所有竞争及分块开销，仅把通信和GEMM看作理想独立流水资源，稳态工作间隔的乐观模型为 max(584.04,484.73)=584.04 µs，相比两项串行和1068.77 µs有484.73 µs的理想覆盖空间。这不是单批延迟、实际two-batch预测或E2E加速估计。Normal DeepEP通信使用SM，需与GEMM分配SM/带宽；分块后的两项时间也须重新测量。

当前实验two-batch overlap关闭。源码的Qwen shared-expert overlap与TBO是两种机制；现有TBO操作策略列出了Qwen3Moe等支持路径，当前Qwen3.5/3.6混合Full/GDN Attention的forward不在该Qwen3Moe操作分解路径中，不能据一个开关认定适配完成。此前暂停的TBO适配本次没有恢复。

因此主报告的PP吞吐优势是当前运行配置的结果，尚不能当作充分优化过通信/计算重叠的EP上限。完整两rank时间交集、CUDA stream与源文件SHA256见[overlap_results.json](./overlap_results.json)。
