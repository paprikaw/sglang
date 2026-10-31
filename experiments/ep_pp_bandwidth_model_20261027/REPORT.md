# EP/PP：Expert 权重读取与 dispatch/combine 的简化模型

日期：2026-10-27。离线解析建模与既有结果复核；本次没有新启动 GPU 实验。

核心判据：**PP 重复读取权重的时间，是否小于 EP 逐层交换 token 的时间。** 模型使用逐层活跃权重，而非模型总参数量。

## 1. 公平比较的最小设置

固定两张 GPU。EP2 为 Attention DP2×TP1、Expert EP2，每卡全部 L 层；PP2 为两个一 GPU stage，每卡 L/2 层。固定总 batch B=2b，PP 用两个 b-token 微批次，考察平衡流水线的稳态整批周期。Attention、CPU、PP 边界、共享 Expert、量化排列暂不放入核心式子。这里的周期可近似稳态 Decode TPOT；不是冷启动首批流水线延迟。

设每个 routed Expert 的 gate/up/down 权重为 w=3h i s_w；每层 E 个 Expert，总权重 W=Ew。两个微批次激活的 Expert 集合为 A1、A2，大小分别 M1、M2，并集 U，交集 R=M1+M2-U。H 是每 GPU 有效 HBM 读取带宽；C 是暴露在临界路径上的每层 dispatch+combine 时间。

```text
T_EP / L = U*w/(2H) + C
T_PP / L = (M1+M2)*w/(2H)
PP 多出的权重读取时间 = R*w/(2H)
PP 更好 <=> R*w/(2H) < C
```

等式取平均且假设 Expert 均衡分片、PP stage 平衡。实际应按 rank/stage 的最大服务时间计，不能把两卡均值当作严格下界。PP 每卡常驻完整层的所有 Expert，不等于每一步都会读取不活跃 Expert 的矩阵；实际还可能重复加载、存在 L2 命中及 tile/launch 开销。

## 2. 为什么只给“激活 80%”还不够

同样是整批共激活 80% Expert：若两个微批次各激活相同的 80%，R/E=0.8，PP 的活跃权重读取是 EP 的 2 倍；若两个微批次各激活互不重叠的 40%，R=0，稳态每 GPU 读取量相同。EP 的收益来自把重复读取合到一次，而不只是“每卡只放一半 Expert”。

独立均匀 top-k 路由可用 a(b)=1-(1-k/E)^b，u=2a-a²，r=a² 估算。若整批并集 u=0.8，则 a=0.5528、r=0.3056，PP 比 EP 多读约 38.2%，而不是一律两倍。真实路由相关性很强，必须用真实集合。

当前 B8 / b4 / top8 的每个微批次最多激活 32/256=12.5%，整批最多 25%，因此“激活 80%”不是当前 Decode 的实际形状。保存的路由快照有约 10–11 个活跃 Expert/次，但并非同步记录两个 slot，不能据此声称已测得 R。

## 3. A100/H100 下的阈值

GPU 标称 HBM：A100 80GB PCIe 为 1.935 TB/s；H100 80GB SXM 为 3.35 TB/s。以下按理想一次读取估算；真实部署应把 H 换成同 shape、同 kernel 的有效带宽。

令 r=R/E：W_crit=2HC/r；W<W_crit 时 PP 更有利。先取两个微批次重复激活相同 80%（r=0.8）：

| 每层 dispatch+combine | A100：PP 有利的总 Expert 权重/层 | H100：PP 有利的总 Expert 权重/层 |
|---|---:|---:|
| 10 µs | < 48.4 MB | < 83.7 MB |
| 30 µs | < 145.1 MB | < 251.2 MB |
| 100 µs | < 483.7 MB | < 837.5 MB |

若整批共激活 80%、按独立均匀路由 r=0.3056，以上门槛约放宽 2.62 倍。例如 C=30 µs 时，A100/H100 为约 380/658 MB/层。若 HBM 有效带宽仅峰值一半，门槛也减半。更大的逐层权重或更多重复活跃 Expert 偏向 EP；更高 HBM、较大 C、较少重复活跃 Expert 偏向 PP。

等价地，PP 容许的重复活跃 Expert 数为 R_crit=2HC/w。以 35B BF16 每 Expert 6 MiB 为例：

| 每层 dispatch+combine | A100：R_crit | H100：R_crit |
|---|---:|---:|
| 10 µs | 6.2 | 10.6 |
| 30 µs | 18.5 | 31.9 |
| 100 µs | 61.5 | 106.5 |

## 4. 把通信带宽放进模型

用 C=λ+V/N。λ 包括 dispatch/combine 固定协议、启动、通知与不可隐藏等待；V 是每 GPU 两个 operation 的实际串行发送字节；N 是同路径的单向有效发送带宽。接收可与发送全双工，不能再把字节重复加一遍。若 kernel 还占 SM/争用 HBM，需将这部分暴露成本留在 C。

```text
PP 更好 <=> rW/(2H) < λ + V/N
若 rW/(2H) > λ：PP 更好 <=> N < V / (rW/(2H)-λ)
若 rW/(2H) <= λ：仅 Expert+交换模型已偏向 PP，与大消息带宽无关。
```

无目标 rank 去重时，EP2 每个 rank 的 D+C 发送量约 V=b*k*h*s_a；若先按目标 GPU 合并，可小很多。下面用 E256/top8、B52、b26、h2048、BF16 activation：并集激活 80.8%、重复比例约 31.6%，无去重 V=832 KiB。固定开销 30 µs 只是敏感性假设，不是已测得的截距。

| N：单向有效 GB/s | C，λ=0 / 30 µs | A100 W_crit，λ=0 / 30 | H100 W_crit，λ=0 / 30 |
|---|---:|---:|---:|
| 150 | 5.7 / 35.7 µs | 0.070 / 0.437 GB | 0.120 / 0.757 GB |
| 25 | 34.1 / 64.1 µs | 0.418 / 0.785 GB | 0.723 / 1.359 GB |
| 12.5 | 68.2 / 98.2 µs | 0.835 / 1.203 GB | 1.446 / 2.082 GB |

例如该假设下 N=12.5 GB/s、λ=30 µs，H100 门槛约 2.08 GB/层，35B BF16 的 1.61 GB/层偏向 PP；A100 门槛约 1.20 GB/层，同模型偏向 EP。**这是可构造的同通信条件硬件反转，不是对现有 A100/H100 实测的因果解释。** 采用目标 GPU 去重近似（κ=1）时，H100 同一例的门槛降至约 0.82 GB，选择也可能反转；所以 profiler 必须记录实际流量/通信时间。所有敏感性格点见 CSV。

实际已存双卡拓扑是 A100069 NV12、H100178 NV6。官方每 link 单方向 25 GB/s，对应聚合直连发送上界约 300/150 GB/s；实际有效带宽必须测。产品页 600/900 GB/s 是所有链路双向总和，不能直接填作该双卡发送带宽。跨节点同理，100/200/400 Gb/s 端口的线速上界是 12.5/25/50 GB/s，但不是应用实测。

## 5. 哪些模型与 setup 值得用 PP

下表仅计 routed Expert 矩阵，排除共享 Expert、Attention、量化 scale、padding；不能按 checkpoint 总参数直接排序。

| 模型/权重精度 | 每 Expert | 每层全部 routed Expert | 重复 R=32 时 PP 多读时间，A100/H100 |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B / BF16 | 6.00 MiB | 1.611 GB | 52.0 / 30.0 µs |
| Qwen3.5-122B-A10B-FP8 / FP8 | 9.00 MiB | 2.416 GB | 78.0 / 45.1 µs |
| GLM-4.7-AWQ / INT4 | 11.25 MiB | 1.887 GB | 97.5 / 56.3 µs |

R=32 是共同假设，不是各模型的已测路由。PP 最值得评估的是低 token、Expert 权重较小或量化后的 Decode，以及交换协议/跨域较慢的 setup；尤其当 R 很小，EP 能省的重复读取本来就有限。高重复活跃比例、大 Expert 矩阵、高效低延迟通信更支持 EP。单模型总权重需加 KV/runtime 后适配 L/2 层/stage；GLM-4.7-AWQ 不能仅凭逐层表格认为可在两张 80GB 上 PP2，须增加 stage 或混合并行。

Prefill token 多时，必须先检查 compute-bound：每 Expert t 个 assignment 的 AI≈2t/s_w FLOP/byte。BF16 小 t 的 Decode 容易受权重读取和 kernel 粒度限制；Prefill 的高复用可能由计算吞吐主导，此时比较 max(权重字节/H, 有效 FLOPs/F_effective)+固定开销，不能直接沿用纯带宽门槛。

## 6. 既有数据是否实现了模型给出的优势

本节先复核 shared_gemm_final_r8 的固定 B8 调度对照；第6.1节加入后续 lookahead_r2 的多负载重新测量，保留两轮来源，不混成一次实验。

以下仅列每卡实际 layer invocation 的 Expert 与交换计时。它们是服务成本诊断，不是可相加的端到端切片；跨流时 kernel 时长会重叠，C 的临界路径暴露比例还需 dependency trace/消融校准。A100 routed module 与 H100 GEMM 的范围也不同。

| 同一实验内的对照 | EP Expert | EP D+C | PP Expert | PP Expert 额外代价 |
|---|---:|---:|---:|---:|
| A100 35B 原生 | 98.28 µs | 110.60 µs | 105.47 µs | 7.19 µs |
| H100 35B 原生 | 60.82 µs | 33.16 µs | 93.01 µs | 32.20 µs |
| H100 35B 双方共享优化，PP lookahead | 53.17 µs | 33.68 µs | 66.39 µs | 13.22 µs |

当前 H100 的小形状：b4/top8 使 R≤32，理想额外权重成本最多 30.05 µs；实测 D+C kernel 预算为 33.68 µs，折算 R_crit≈35.9。若这些交换成本确实暴露，纯 HBM 模型倾向 PP。实测 GEMM 的 PP 额外代价为 13.22 µs，也小于交换预算。这表明 PP 的模型内优势已有依据，端到端能否兑现还取决于调度。

| 当前同双卡、相同通用 GEMM 优化 | TPOT ↓ | 输出吞吐 ↑ |
|---|---:|---:|
| EP2 | 7.657 ms | 890.4 tok/s |
| PP2 原调度 | 9.303 ms | 804.9 tok/s |
| PP2 + lookahead | 6.998 ms | 1048.3 tok/s |

PP 优化后比 EP 的 TPOT 低 8.6%，完整请求输出吞吐高 17.7%。吞吐包含 TTFT/队列尾部，不能由 8/TPOT 直接等同得到。此前 PP 原调度即便 Expert+交换预算较优仍然更慢：PP 额外 CPU/反馈等待占掉了收益。lookahead 主要改变这部分，未再独占一个通用 GEMM 优化。

是否达到理论极限：还不能声称达到 HBM 极限。早期 r5 的 11.4 活跃 Expert 路由样本对应一次读取 21.41 µs，彼时整模型 profile Expert 均值 57.62 µs，约为模型值 2.69 倍。但路由是样本而非逐 kernel 同步 DRAM 流量，不能把二者比值当成已测 HBM 利用率，也不能拿该样本给最新 66.4 µs trace 定利用率。

调度方面，当前 profile stage span 给出的条件流水线下界 2×max(stage0,stage1)=6.483 ms；同 profile 的周期 6.833 ms，达到约 94.9% 条件服务率；独立无 profiler TPOT 为 6.998 ms，仅作量级对照。这是“保持现有模型 kernel 不变”的调度下界，和只读 Expert 权重的 HBM 下界不同。

122B 的短诊断（每 rank 仅 1–2 Graph）也有类似取舍：EP Expert 49.34 µs/层，D+C 39.87 µs/层；PP 两阶段 Expert 为 86.88 和 76.53 µs/层。PP Expert 多付约 27–38 µs，已接近通信预算，所以 PP 的余量更小，更容易被 stage 不均衡与边界成本吃掉。该轮 EP/PP 用不同 NV6 对且调度未做 35B lookahead，不作为精确 HBM 模型校准。

## 6.1 更新的 B8/B16/B32 对照：模型能解释什么

今天新增的 lookahead_r2 对照覆盖不同 batch 与多样输入，双方均使用同一 Expert 内核家族、T≤16 门限，EP shared overlap 已开启。下面的 Expert/通信来自同一组独立 profile；TPOT 来自另一组无 profiler 正式请求。

| Workload | PP 多付的 Expert | EP D+C kernel 预算 | 预算余量 | EP / PP TPOT | PP 整批吞吐相对 EP |
|---|---:|---:|---:|---:|---:|
| b8_repeated | 12.93 µs | 33.64 µs | 20.71 µs | 7.666 / 6.970 ms | +18.17% |
| b16_repeated | 18.49 µs | 34.89 µs | 16.40 µs | 8.909 / 8.865 ms | +7.79% |
| b32_repeated | 20.51 µs | 39.10 µs | 18.59 µs | 10.957 / 11.508 ms | +0.32% |
| b32_diverse | 38.38 µs | 39.84 µs | 1.46 µs | 12.534 / 12.998 ms | +0.89% |

B32 从重复文档改成多样输入时，D+C 只增加约 0.74 µs，但 PP 相对 EP 的 Expert 代价增加约 17.9 µs，通信预算几乎被吃尽。它支持“必须结合 Expert 执行特性作决策”；仅用 token 数或网络带宽不够。B32 下 EP 的 TPOT 更低，PP 的整批输出吞吐接近持平，来自 PP 较低的 TTFT；不能以 +0.32%/+0.89% 的整批吞吐声称 PP 稳态 Decode 更好。

这还不能验证“差额全部由一次权重读取解释”：现有路由诊断取每层第一次匹配调用，包含单 token Extend/早期 Decode，与后期 Graph profile 不同步。例如 B32 diverse 保存的 EP 每卡平均活跃 28.59，而 PP 一次快照平均 14.88，无法视作同一 token 波次的集合来计算交集；那会违反 U≤M1+M2。当前应保留理论条件与实测服务时间两级判断，不能拟合一个虚假的 HBM 利用率。若要把本模型升级为可预测 advisor，只需同步采集两个微批次的 Expert 集合与对应 kernel 时间，再加实际 DRAM bytes 或同形状有效带宽。

![Workload 对模型的检验](workload_tradeoff.png)

## 7. 用作 advisor 的最小输入

1. 从模型读取 w、E、L、精度，检查单 stage 容量。
2. 对目标 b 与路由，记录两个微批次的 M1、M2、U，得到 R；不必全 sweep。
3. 在相同 backend/Graph 下，记录 T_expert_PP、T_expert_EP 与暴露 D+C；若想拟合带宽项，用两个消息尺寸估 λ、N，并保存目标 rank 去重方式。
4. 先比较 Rw/(2H_eff) 与 C 作筛选，再用 ΔT_expert<C 筛选实际 kernel，最后用一组端到端对照确认 PP 调度能否兑现。

推广到 G 个一 GPU PP stage 时，对 G 个微批次激活数 M_j、全局并集 U，重复权重代价变为 (sum M_j-U)w/(G H)。仍需另计 stage 不平衡、boundary、有限批次填充/排空与 autoregressive feedback。

## 图表与可复现数据

![HBM 与通信边界](bandwidth_boundary.png)

![当前实测对照](measured_check.png)

`model.py` 给出最小函数；`build_report.py` 从原 CSV/配置生成本报告；`thresholds.csv`、`network_scenarios.csv`、`model_network_limits.csv`、`model_geometry.csv` 与 `measured_comparison.csv` 保留全部数值。来源文件路径与 SHA256 见 `sources.json`。

官方规格/机制来源（2026-10-27 直接核对原页面；web 搜索接口故障，使用官方 HTTPS 读取）：

- [A100 官方规格](https://www.nvidia.com/en-us/data-center/a100/)：PCIe 1935 GB/s、SXM 2039 GB/s。
- [H100 官方规格](https://www.nvidia.com/en-us/data-center/h100/)：80GB SXM 3.35 TB/s。
- [NVIDIA Hopper 架构](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：每 NVLink 每方向 25 GB/s，H100 18 links / A100 12 links。
- [NVIDIA GEMM 性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：算术强度、tile 与 memory/math bound 判据。
