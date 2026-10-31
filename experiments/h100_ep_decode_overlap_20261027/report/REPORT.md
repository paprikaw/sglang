# H100 Decode: EP communication / Expert overlap

## Measured outcome

本轮四组Decode均未获益：两分块EP重叠相对同轮原生EP吞吐降低19.7%–20.1%。
GPU时间线实测隐藏了26.6%–37.1%的Dispatch/Combine kernel时间；该比例的分母是新分块配置的通信时间。
b32_repeated串行拆分对照：原生2566.8、串行两分块1983.7、重叠两分块2057.6 output tok/s。重叠相对串行拆分变化+3.7%，但未追回拆分损失。
同一workload的每层Expert GEMM从94.10升到152.09 µs/卡，Dispatch+Combine从39.19升到74.29 µs/卡；与GEMM的实际交集为25.29 µs/卡/层。
这支持在当前小token Decode下保留原生整批EP。新路径把每层逻辑通信组数从1增为2、Expert GEMM从2个增为4个，减小每次执行粒度；时序重叠不足以抵消这版实现的开销。本结论限于当前两分块实现与测得的四组workload。

完整 Qwen3.6-35B-A3B，BF16 权重/激活/KV、FP32 GDN；H100178同一NVLink GPU0/1。沿用前一轮B8/B16/B32重复输入与B32多主题输入；每请求4096个真实GPU缓存token，新增1 token、生成128 token。EP为Attention DP2/TP1、MoE EP2；PP为两个20层阶段、两个B/2固定流水槽，开启policy9提前发射。

各配置用独立进程，在启动时固定策略并捕获CUDA Graph。Attention/GDN保持整批；MoE把各rank的token分成两组，沿用相同的小批次Triton Expert内核家族和DeepEP低延迟BF16通信；两组各有独立handle。所有布局保留原有shared-expert overlap，full-model two-batch overlap关闭。Decode没有沿用Prefill的DeepGEMM SM预算设置。

原生EP保留recv-hook的send/receive两阶段；串行和重叠分块都使用async-event的合并通信。因此串行/重叠对照隔离的是重叠调度收益；原生/分块差异同时包含分块和通信接口模式变化。每层实际Dispatch和Combine均各有2个kernel；原生是两阶段，分块是每块1个kernel。Expert GEMM则从每层2个变为4个。

每配置3个不计时完整批次预热、5个profiler关闭的正式批次。PP另外先沿用原生Decode的多形状预热，以建立提前发射所需的消息格式；这些批次不计时。前缀由本进程真实模型生成并留在GPU；无CPU KV换入。独立profile从每rank第21个匹配实际本地batch的Decode Graph开始取16个，统计每次实际层调用的GPU时间。不同进程的采样轮次不混为同一批次；没有把Graph分项相加作为TPOT。

| Workload | Configuration | Output tok/s ↑ | Sample SD | TPOT ms ↓ | TTFT ms | E2E ms |
|---|---|---:|---:|---:|---:|---:|
| b8_repeated | EP2 native | 888.8 | 4.8 | 7.680 | 176.06 | 1151.47 |
| b8_repeated | EP2 overlap | 714.1 | 5.7 | 9.818 | 186.44 | 1433.31 |
| b8_repeated | PP2 lookahead | 1,053.4 | 0.9 | 6.967 | 84.71 | 969.49 |
| b16_repeated | EP2 native | 1,551.7 | 3.8 | 8.867 | 192.12 | 1318.28 |
| b16_repeated | EP2 overlap | 1,239.5 | 7.6 | 11.469 | 194.17 | 1650.70 |
| b16_repeated | PP2 lookahead | 1,659.2 | 15.5 | 8.882 | 102.31 | 1230.30 |
| b32_repeated | EP2 native | 2,566.8 | 4.5 | 10.964 | 199.97 | 1592.35 |
| b32_repeated | EP2 serial split | 1,983.7 | 11.7 | 14.781 | 184.22 | 2061.46 |
| b32_repeated | EP2 overlap | 2,057.6 | 16.4 | 14.074 | 199.86 | 1987.23 |
| b32_repeated | PP2 lookahead | 2,582.0 | 2.9 | 11.510 | 118.43 | 1580.23 |
| b32_diverse | EP2 native | 2,285.1 | 1.8 | 12.527 | 198.02 | 1789.00 |
| b32_diverse | EP2 overlap | 1,826.5 | 5.8 | 16.047 | 201.03 | 2238.95 |
| b32_diverse | PP2 lookahead | 2,307.2 | 1.4 | 12.993 | 118.90 | 1769.07 |

| Workload | EP overlap vs native throughput | TPOT change |
|---|---:|---:|
| b8_repeated | -19.65% | +27.83% |
| b16_repeated | -20.12% | +29.34% |
| b32_repeated | -19.84% | +28.37% |
| b32_diverse | -20.07% | +28.09% |

## Expert, Attention and communication

单位：µs/卡/实际层调用；Expert仅routed GEMM，Attention仅core。通信kernel包含等待；交集按同一GPU实际时间轴计算。双分块的一层包括两组通信和两组Expert计算，仍除以完整模型的40层，不把两组错误当作80层。

| Workload | Configuration | Expert GEMM µs | Attention µs | Dispatch µs | Combine µs | Comm ∩ GEMM µs | Comm overlapped |
|---|---|---:|---:|---:|---:|---:|---:|
| b8_repeated | EP2 native | 53.33 | 13.80 | 15.14 | 18.70 | 0.00 | 0.0% |
| b16_repeated | EP2 native | 67.41 | 19.87 | 16.00 | 19.29 | 0.00 | 0.0% |
| b32_repeated | EP2 native | 94.10 | 33.06 | 17.63 | 21.56 | 0.00 | 0.0% |
| b32_diverse | EP2 native | 141.65 | 33.07 | 17.52 | 22.23 | 0.00 | 0.0% |
| b32_repeated | EP2 serial split | 144.06 | 33.09 | 28.31 | 36.05 | 0.00 | 0.0% |
| b8_repeated | EP2 overlap | 86.76 | 13.78 | 27.54 | 33.84 | 16.31 | 26.6% |
| b16_repeated | EP2 overlap | 115.04 | 19.81 | 29.27 | 40.29 | 23.13 | 33.3% |
| b32_repeated | EP2 overlap | 152.09 | 33.11 | 29.60 | 44.68 | 25.29 | 34.0% |
| b32_diverse | EP2 overlap | 209.19 | 33.22 | 29.57 | 52.02 | 30.26 | 37.1% |
| b8_repeated | PP2 lookahead | 66.21 | 12.58 | 0.00 | 0.00 | 0.00 | 0.0% |
| b16_repeated | PP2 lookahead | 82.46 | 18.80 | 0.00 | 0.00 | 0.00 | 0.0% |
| b32_repeated | PP2 lookahead | 114.61 | 31.79 | 0.00 | 0.00 | 0.00 | 0.0% |
| b32_diverse | PP2 lookahead | 178.03 | 32.01 | 0.00 | 0.00 | 0.00 | 0.0% |

## Numerical and execution evidence

每个EP配置、每种workload都在计时外，以真实输入、路由和权重对双rank各40层比较原生、串行分块和重叠。串行/重叠要求逐元素相同；原生/分块要求有限值及relative L2≤1.5%，实际最大误差见下表。正式批次检查图重放计数增加、固定策略和handle排空、每请求4096缓存token与128输出token、SSE事件和TPOT。

| Workload | Configuration | Native/split max relative L2 | Serial/overlap exact | Across-cohort full sequence stable |
|---|---|---:|---|---|
| b8_repeated | EP2 native | 0.0 | True | False |
| b16_repeated | EP2 native | 0.0 | True | False |
| b32_repeated | EP2 native | 0.0 | True | False |
| b32_diverse | EP2 native | 0.0 | True | False |
| b32_repeated | EP2 serial split | 0.0 | True | False |
| b8_repeated | EP2 overlap | 0.0 | True | False |
| b16_repeated | EP2 overlap | 0.0 | True | False |
| b32_repeated | EP2 overlap | 0.0 | True | False |
| b32_diverse | EP2 overlap | 0.0 | True | False |
| b8_repeated | PP2 lookahead | — | — | True |
| b16_repeated | PP2 lookahead | — | — | True |
| b32_repeated | PP2 lookahead | — | — | True |
| b32_diverse | PP2 lookahead | — | — | True |

旧原生EP在四个workload上均有跨批次长输出差异，因此不能把跨批次128-token序列相同作为新调度的独立正确性门槛。新调度的检查依据是同一次输入的逐层数值对照；PP固定分组仍要求跨批次完整输出一致。EP首轮因错误使用这一门槛中止，保留原始failed状态，性能样本全部排除。PP首轮在预热阶段因缺少原生Decode通信格式而中止，没有有效正式计时；新目录补回已验证的原生多形状预热后重测。

23项CPU检查通过；所有冻结输入SHA256、完整模型/CPU权重缓存使用、Guard postcheck、实验自有进程清理和租约释放通过。

[Structured results](results.json) · [Throughput and TPOT](decode_overlap.png) · [B32 GPU kernel costs](component_costs.png)

## Pinned source

Engine commit: `0f2f1990994774472b1089d2b0682f0bf926a6b4`; H100 qualification tag: `experiment/sglang/qwen35b-decode-ep-overlap-20261027-h100-qualified`.

[Decode implementation](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/qwen35b-decode-ep-overlap-20261027/python/sglang/srt/layers/moe/decode_overlap.py)
