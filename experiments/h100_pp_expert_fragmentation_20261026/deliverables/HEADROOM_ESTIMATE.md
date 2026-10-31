# PP2 Expert 优化空间估算：NVIDIA GEMM 原则与现有实测

本页是离线计算，没有启动新GPU实验。将用户的“GLM best practice”理解为 NVIDIA GEMM 矩阵乘优化指南。沿用现有 Qwen3.6-35B-A3B BF16；仅讨论本地 B4 的 Decode，不使用 GLM 模型或更低精度的性能替换当前模型。

结论：当前 Expert GEMM 57.62µs/实际层调用，建议下一工程目标40µs，35µs为更进取目标。活跃权重每次从HBM读取一遍、达到峰值带宽的理想模型约21.41µs。端到端吞吐达到当前EP2附近具有可能性，现有计算不足以保证胜出。

## 计算依据

本模型hidden=2048，每个Expert intermediate=512，SwiGLU含gate/up/down三个矩阵。每个Expert的BF16权重为：

`3 × 2048 × 512 × 2 = 6,291,456 bytes = 6 MiB`。

最终r5真实路由样本40层，本地4 token、top8，共32次配对，平均11.40个不同的活跃Expert。活跃权重共`11.40 × 6 MiB = 71.72 MB`。H100 SXM80GB官方峰值HBM带宽3.35TB/s，因此理想读取时间：

`71.72 MB / 3.35 TB/s = 21.41 µs`。

有效计算量为`6 × 32 × 2048 × 512 = 0.201 GFLOP`；即使按约989TF/s的dense BF16计算，算力项仅0.204µs。权重主导的算术强度约2.81FLOP/byte，远低于GPU的计算/带宽比例。因此不能直接用Tensor Core峰值推算加速倍数。

这是理想HBM模型：不含元数据、激活搬运、tile浪费及其他执行成本；L2驻留情况也会影响读取量，重复单层的热缓存microbenchmark不是同一个HBM条件。它不是对实际kernel时延的严格通用下界。

工程目标采用我们的假设：有效HBM达到峰值的60%–75%，另有5–10µs执行开销，可得33.5–45.7µs，故取35–45µs作为待实测的目标范围。这个效率比例和开销范围不是NVIDIA公布的保证值。已有单层T4小tile测试约33–47µs提供实现线索，不能直接替代完整模型的57.62µs。

## 与当前EP/PP对照

Expert单位µs/GPU/实际层调用；TPOT单位ms；吞吐为当前8请求、缓存4096、新增1、输出128的output token/s。测量和估算明确分列。

| 配置 | 类型 | Expert GEMM ↓ | TPOT ↓ | 吞吐 ↑ |
|---|---|---:|---:|---:|
| EP2 current | 实测 | 59.56 | 7.946 | 863.2 |
| PP2 native | 实测 | 94.19 | 11.176 | 666.8 |
| PP2 current | 实测 | 57.62 | 9.091 | 805.7 |
| PP2 target45 | 估算 | 45.00 | 8.586 | 848.3 |
| PP2 target40 | 估算 | 40.00 | 8.386 | 866.6 |
| PP2 target35 | 估算 | 35.00 | 8.186 | 885.6 |
| PP2 ideal HBM model | 估算 | 21.41 | 7.643 | 941.8 |

## 端到端换算假设

PP2每个stage20层，两个B4槽位；每个全局Decode wave，每卡执行约40次Expert调用。在流水调度、TTFT、Attention和其他成本不变且这些Expert收益均能作用于流水关键路径的假设下：

`T_cohort(target) = 1271.16ms − 127 × 40 × (57.6156 − target_Expert_us)/1000`。

`吞吐 = 8 × 128 × 1000 / T_cohort(target)`。

这不是把不同GPU的profile时间直接相加。它是固定调度的反事实模型，40来自每卡两个槽位各20层；重叠、等待、负载不平衡改变时，换算也会改变。回算原生PP时，该模型得到1456.98ms，实测为1535.71ms，误差-5.13%；原生到优化同时改变了其他kernel/布局/轨迹，说明不能把这个模型当作已验证预测器。40µs目标对应的吞吐优势小于该误差量级。

在该模型中，当前128输出测试的吞吐追平EP2需要Expert约≤40.91µs；TPOT追平EP2则需要≤29.00µs。两者不同，是因为PP这组数据的TTFT更低；延长生成后，首token差异会逐渐摊薄。

因此，把40µs作为下一目标比较合理：对应Expert比当前再快约30.6%，完整模型粗算约867 tok/s，接近当前EP2的863。达到35µs粗算886 tok/s；即使达到21.4µs的理想HBM模型，完整模型也只是约942 tok/s，并不是整体获得2.7倍加速。EP自身也有优化空间，此处只与已测EP对比。现有BF16不同执行路径的生成差异与质量边界继续适用。

## 来源

- [NVIDIA GEMM指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：算术强度、内存/算力限制、tile大小与并行度取舍。
- [NVIDIA H100官方规格](https://www.nvidia.com/en-us/data-center/h100/)：SXM80GB、3.35TB/s；BF161979TF/s带稀疏标记，dense约989TF/s。
- [原始完整报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/REPORT.md)；[计算参数和来源哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/headroom_estimate.json)；[对照CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_expert_fragmentation_20261026/deliverables/headroom_comparison.csv)。
