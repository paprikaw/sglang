# PP lookahead 在更大 workload 上的 EP / PP 对照

每组请求总数为 B8、B16、B32；PP 有两个流水槽位，每槽分别为4、8、16个请求。另有B32多主题输入。完整 Qwen3.6-35B-A3B BF16，在同一 H100178 的NVLink GPU0/1顺序测量。EP2为Attention DP2/TP1、MoE EP2；PP2为两级、每级20层、TP1/EP1。

Decode每个请求为4096个真实GPU缓存token，加1个新输入token，生成128个token。每配置5个正式批次，关闭profiler。PP原生调度和lookahead按正反顺序交替，切换后先做不计时预热。EP和两种PP均使用相同的Expert小tile内核家族（本地T≤16）；EP原生shared-expert overlap开启，TBO关闭。

lookahead使用此前最佳policy9组合：提前提交activation、按消息类型分开的NCCL通道、提前准备metadata，以及在旧输出CPU处理前发射下一槽位的真实forward。本轮扩展固定batch大小，并与相同分组的原生PP调度对照。

吞吐为整批输出token数除以首请求开始至末请求最后token的墙钟时间；TPOT为每请求首token至最后token的时间除以127，再对请求平均。TTFT包含请求发送与排队；E2E为请求开始至最后token。固定PP分组的等待包含在计时内。

## Decode 性能

| Workload | EP2 tok/s ↑ | PP2 native tok/s ↑ | PP2 lookahead tok/s ↑ | Lookahead vs native PP | Lookahead PP vs EP |
|---|---:|---:|---:|---:|---:|
| b8_repeated | 891.2 | 817.6 | 1053.0 | +28.79% | +18.17% |
| b16_repeated | 1542.7 | 1313.0 | 1662.9 | +26.65% | +7.79% |
| b32_repeated | 2572.0 | 2132.4 | 2580.1 | +21.00% | +0.32% |
| b32_diverse | 2285.5 | 1942.9 | 2305.9 | +18.69% | +0.89% |

| Workload | EP2 TPOT ms ↓ | PP2 native TPOT ms ↓ | PP2 lookahead TPOT ms ↓ |
|---|---:|---:|---:|
| b8_repeated | 7.666 | 9.165 | 6.970 |
| b16_repeated | 8.909 | 11.438 | 8.865 |
| b32_repeated | 10.957 | 14.124 | 11.508 |
| b32_diverse | 12.534 | 15.613 | 12.998 |

提前发射相对同一内核、同一分组的原生PP调度，吞吐变化为+18.69%至+28.79%。优化PP在4/4组的吞吐高于EP。这些是本轮重新测量的对照；此前仅Expert内核的结果保留为独立历史实验。

b32_repeated：优化PP整批吞吐比EP高0.32%，但TPOT高5.03%。PP的平均TTFT比EP短77.39ms，解释了这两个指标的不同方向。

b32_diverse：优化PP整批吞吐比EP高0.89%，但TPOT高3.71%。PP的平均TTFT比EP短77.74ms，解释了这两个指标的不同方向。

| Workload | Layout | Mean TTFT ms | Mean request E2E ms |
|---|---|---:|---:|
| b8_repeated | EP2 | 174.85 | 1148.42 |
| b8_repeated | PP2 native | 85.65 | 1249.55 |
| b8_repeated | PP2 lookahead | 84.65 | 969.78 |
| b16_repeated | EP2 | 194.42 | 1325.91 |
| b16_repeated | PP2 native | 102.68 | 1555.30 |
| b16_repeated | PP2 lookahead | 101.64 | 1227.55 |
| b32_repeated | EP2 | 197.31 | 1588.89 |
| b32_repeated | PP2 native | 120.32 | 1914.06 |
| b32_repeated | PP2 lookahead | 119.92 | 1581.48 |
| b32_diverse | EP2 | 196.90 | 1788.66 |
| b32_diverse | PP2 native | 118.41 | 2101.23 |
| b32_diverse | PP2 lookahead | 119.16 | 1769.92 |

整批128-token吞吐同时受到首token成本和最晚完成请求影响，不能直接用1/平均TPOT替代。

![Decode comparison](workload_comparison.png)

## 原因：Graph之间的调度等待

独立profile选择每rank第21–36个匹配本地batch大小的真实Decode Graph。PP通过序列长度总和与出现次序匹配两个stage，检查因果顺序，再回到同一流水槽位。between-Graph包含调度、反馈与通信等待，不能解释为纯链路传输时间；组件核函数累计时间不等于TPOT。

| Workload | PP policy | Model Graphs ms | Between-Graph ms | Same-slot cycle ms |
|---|---|---:|---:|---:|
| b8_repeated | native | 6.084 | 4.821 | 10.905 |
| b8_repeated | lookahead | 6.073 | 0.742 | 6.814 |
| b16_repeated | native | 7.192 | 4.948 | 12.141 |
| b16_repeated | lookahead | 7.201 | 0.772 | 7.973 |
| b32_repeated | native | 9.206 | 5.271 | 14.477 |
| b32_repeated | lookahead | 9.209 | 0.943 | 10.152 |
| b32_diverse | native | 11.749 | 5.012 | 16.761 |
| b32_diverse | lookahead | 11.754 | 0.673 | 12.427 |

![Scheduling decomposition](scheduling_decomposition.png)

## Expert / Attention / Dispatch + Combine

均为两个rank的每实际层调用GPU时间平均，单位µs；Expert只计routed GEMM，Attention只计core kernel，不含projection。EP/PP分别每rank40/20层。EP通信与共享计算可重叠，不将这些列相加为TPOT。

| Workload | Layout | Expert GEMM µs | Attention µs | Dispatch + Combine µs |
|---|---|---:|---:|---:|
| b8_repeated | EP2 | 53.24 | 12.80 | 33.64 |
| b8_repeated | PP2 native | 66.20 | 12.35 | 0.00 |
| b8_repeated | PP2 lookahead | 66.18 | 12.26 | 0.00 |
| b16_repeated | EP2 | 63.98 | 18.83 | 34.89 |
| b16_repeated | PP2 native | 82.44 | 18.41 | 0.00 |
| b16_repeated | PP2 lookahead | 82.47 | 18.46 | 0.00 |
| b32_repeated | EP2 | 94.03 | 32.07 | 39.10 |
| b32_repeated | PP2 native | 114.52 | 31.63 | 0.00 |
| b32_repeated | PP2 lookahead | 114.53 | 31.62 | 0.00 |
| b32_diverse | EP2 | 139.55 | 32.20 | 39.84 |
| b32_diverse | PP2 native | 177.98 | 31.61 | 0.00 |
| b32_diverse | PP2 lookahead | 177.93 | 31.61 | 0.00 |

## 冷 Prefill 控制

每请求4096个新输入token→1个输出，无缓存。lookahead只用于Decode，Prefill保持原生调度。

| Workload | Layout | Input tok/s ↑ | TTFT ms | E2E ms |
|---|---|---:|---:|---:|
| b8_repeated | ep2 | 64208.8 | 396.03 | 396.03 |
| b16_repeated | ep2 | 68258.1 | 635.72 | 635.72 |
| b32_repeated | ep2 | 72589.4 | 1062.61 | 1062.61 |
| b32_diverse | ep2 | 73986.1 | 1043.60 | 1043.60 |
| b8_repeated | pp2 | 74615.7 | 279.97 | 279.97 |
| b16_repeated | pp2 | 81278.4 | 456.07 | 456.07 |
| b32_repeated | pp2 | 83565.9 | 845.30 | 845.30 |
| b32_diverse | pp2 | 83480.6 | 843.40 | 843.40 |

## 生效与正确性

- 每个正式lookahead批次，两rank实际提前发射计数均增加，生产/消费匹配，pending为空。
- PP同一workload所有正式批次的完整128-token输出逐位置一致；两stage真实payload的有序hash匹配，实际分组与EP偶/奇请求分配一致。
- 每请求精确检查GPU缓存4096、输出128，以及SSE事件1–128；16项CPU调度、边界与回执读取检查通过。
- 固定batch、两槽、ignoreEOS的Decode原型；这些测量支持四个所测workload，不代表动态到达场景。
- 所有冻结输入、Guard模型缓存postcheck、owned cleanup与lease release检查通过。

前次r1在读取尚未写完的wire回执JSON时中止，已验证清理并保留原始证据。本报告正式指标全部来自补上文件稳定性屏障后的完整重跑，不拼接r1的性能数据。

## 可复现材料

- [正式指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/metrics.csv)；[逐批次](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/cohorts.csv)；[逐请求](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/requests.csv)。
- [分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/components.csv)；[profile cycles](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/profile_cycles.csv)；[完整分析](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/results.json)。
- [冻结运行](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/runs/lookahead_r2)；[启动manifest](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/runs/lookahead_r2/launch_manifest.json)。
- 图中误差线为5个整批吞吐样本的标准差；所有指标的样本范围和原始来源SHA256保留在分析中。
