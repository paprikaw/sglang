# PP/EP 通信与 Expert 工作集张力：定向文献核查

日期：2026-10-29。本文回答：已有工作是否只搜索 PP/EP 配置，还是已经研究了 token 数量变化时，减少通信与 Expert 执行成本之间的取舍及最优布局切换？

本轮只读论文、作者公开代码和既有实验，没有启动 GPU 实验。以下“未找到”限于所列版本与阅读范围，不等价于证明全领域不存在。

## 判断

**对 OServe，用户的区分成立：支持异构实例与配置选择，不等于已经解释并验证 MoE 的 PP/EP 张力。** 但从更广的文献看，物理机制的几个部分、甚至将它们放进同一个理论模型，已有直接先例。因而不能将贡献写成“首次发现 PP 省通信”“首次同时考虑 token 和激活专家”，也不能仅凭增加 EP 候选或找到一个配置反转就宣称系统创新。

本轮新增的关键先例是 2025 年的 **Inference economics of language models**：§3.2 显式建模 batch 激活专家及权重读取，§3.5 将 PP 的微批、EP dispatch/combine 纳入模型；公开代码还将微批大小用于决定是否启用 EP。这个先例比“OServe 可以扩展 EP”更直接收窄机制层面的新颖性。

仍值得验证的研究问题是：**在真实 serving 引擎和固定 GPU 预算下，同一阶段内的 PP/EP 边界，如何随实际路由、每专家 token 分布、微批切分、内核分块和可隐藏通信移动？已有模型或配置容量表在哪些条件下会选错，修正是否有实际决策收益？** 这是待验证假设，目前不能承诺已存在足够的论文 gap。

## 逐篇证据

| 工作及版本 | 已经明确覆盖什么 | 与当前具体问题的距离 |
|---|---|---|
| [OServe，ICML 2026 正式版](https://proceedings.mlr.press/v306/jiang26k.html)，§4、App D/K，PDF pp3–5、15、21 | 异构 TP/PP 实例与 workload assignment 联合优化；归一化容量处理混合请求；EP 可通过 profiling 扩展 | 正文与附录未展示 batch 活跃专家、PP 微批导致的 Expert 工作集变化，以及固定资源的同阶段 PP↔EP 实测边界。不能据此断言其扩展后必然预测失败 |
| [PipeSwift v4](https://arxiv.org/html/2609.16491v4)，§5.2、6.1.2、7.5，PDF pp6–7、11–12 | 明确解释 PP 将逐层跨节点 EP 通信替换为阶段间传递；对比 EP16 与 PP2×EP8；联合优化 P/D orchestration 和 PP 微批 | 直接覆盖“PP 省通信”。微批模型含 small-GEMM 固定成本，但没有在这些章节展示以每批活跃专家/直方图为变量的 PP/EP 边界；其核心结论围绕 agentic JCT 与 P/D 平衡 |
| [TEMPO v2](https://arxiv.org/html/2608.13057v2)，§2.2、4、App B/C，PDF pp3、16 | 激活 expert replica 数、token 数、128-row tile padding 的联合成本；权重读取平台；通信感知的专家副本间 token 分派 | 直接覆盖二维图背后的关键计算机制；控制动作是 EP 内的路由分派，不是 layer PP 与 EP 的布局选择 |
| [Inference economics v1](https://arxiv.org/html/2506.04645v1)，§3.2–3.5，PDF pp9–12，另核查作者代码 | batch 激活专家期望、HBM 权重读取、PP 微批与阶段传输、EP all-to-all；代码中有随微批大小变化的 EP 使用条件 | 已经把核心要素放在同一模型中。是理论近似与成本/速度分析，不能说仅枚举配置；本轮未找到现代 serving 内核、真实路由、同预算同阶段 PP/EP 边界的受控实测验证 |
| [SMIDT，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39403)，§2.2、3–5，PDF pp2–8 | PP 低通信优势；动态 Top-K 导致阶段不均衡；模块级切分、专家复制 DPMoE 用显存换通信；联合枚举 PP/EP/TP/DP | 已经有真实 PP+EP 混合优化和通信取舍。核心变量是逐层、逐 token 的 Top-K 与阶段内存平衡；所核查实验没有隔离固定 Top-K 下 batch 活跃专家集合引起的同阶段切换 |
| [MoEFS v1](https://arxiv.org/html/2607.18631v1)，§III-B/C、V-D，PDF pp4–5、9、11 | TP/PP/EP、调度 overlap、内核结构联合搜索；解析与 timeline 成本模型；讨论小粒度通信成本与大粒度摊销的 crossover | 不是只看通信的黑盒搜索。Fig3 是单卡计算内核与 EP8 完整通信链的比较，口径不等于同 GPU 预算的 PP/EP 整模型实验；本文未给出用户所要求的完整因果链 |
| [Beyond the Buzz v1](https://arxiv.org/abs/2506.05508v1)，§3.1、Fig5，PDF pp3、5 | 模型/硬件/workload/batch 设计空间；固定 64 GPU 的 EP×PP=64；DeepSeek-R1、256K Prefill 下 PP/CPP 的 latency/throughput 权衡 | 已覆盖固定资源的 EP/PP 配置图。Fig5 固定输入长度而改变 PP，不能替代 token×活跃专家变化下的边界验证；模拟器细节不足，不能断言它忽略 Expert 成本 |
| [Moebius v1](https://arxiv.org/html/2606.26607v1)，§2.1、4.5，PDF pp3、8 | 同阶段 Decode、固定 8 H200、batch-dependent TP/EP crossover；计算与通信解释；阈值、滞回和切换成本 | “负载跨过布局边界，再在线切换”已有直接系统先例。但这里是 TP↔EP，不是 PP↔EP，不能直接迁移数值阈值或实现 |
| [HAP v1](https://arxiv.org/html/2508.19373v1)，§II-C、III-A/B/D | Attention/Expert 分开估计计算与通信，搜索混合策略，讨论转换开销 | §II-C 明确将 PP 排除出搜索空间；因此它不直接回答 PP/EP 问题，但“联合考虑 Expert 计算与通信”不是空白 |

## OServe：框架重叠与机制重叠要分开

OServe §4.2 将 replica k 处理 workload j 的容量记作 n_kj。混合工作负载用归一化容量约束表示：

`sum_j x_kj / n_kj <= 1`。

App D 的 profiling 包括单层 Prefill/Decode 在不同 TP 和 workload 下的时间，以及 PP 通信。App K 明确将 EP/SP 列为可加入的并行策略。因此准确的判断是：

- 当前正式版没有直接研究这里的 MoE 激活专家机制；这个“未覆盖”的结论有明确范围。
- 加入相同 EP/PP 候选，再为不同 batch 充分 profiling，OServe 式策略可能已经捕获一部分甚至大部分切换。没有证据支持预先宣称它必然失败。
- 真正需要测试的差异是：把请求分配到实例后，合批和微批的专家集合也发生变化，容量是否因此不再能由单类容量简单叠加预测；扩大 workload 分类或 profiling 预算后差距是否仍然存在。

这使 OServe 更适合作为一个强路由基线，而非“已经做过同一篇论文”的证据，也不是可以轻易排除的无关工作。

## 最接近物理机制的三条线

### 1. PipeSwift 已明确研究 PP 的通信好处

§5.2 直接指出 wide EP 的 Prefill 每层都可能跨节点交换，PP 则仅在 pipeline stage 边界交接激活。§6.1.2 并非忽略小微批：

`T_wave(K) ≈ (K + P − 1) × (W/K + γ)`，

其中 γ 含调度、kernel launch 和 small-GEMM 固定成本。它利用该模型选择微批数 K。§7.5 Fig8 还让 EP 和 PP 使用同一 orchestration policy，再扫描 interval，故不能将其比较简单概括为“两个不同引擎随便比较”。

但其该模型中的 γ 没有显式按每个 MoE 层、每个微批的专家并集与 token 直方图变化。潜在补充是证明：这种固定项在某些同阶段的路由/微批变化下不足以预测 PP/EP 边界。需要实测证明，不能从公式较简单直接推断失败。

### 2. TEMPO 已明确研究 Expert 权重与 padding 成本

TEMPO §2.2 将 GPU 专家时间描述为 activation floor 与 token compute 的最大值，并加入 tile 项：

`t = max(a + bG + b2 × (Ttiles − G), c + βN)`，

`Ttiles = sum_e ceil(n_e / 128)`。

这里 G 是本地实际激活的专家副本数，N 是 token–expert 行数；128 是其所测内核的 tile 大小。Fig9 保持 N 不变改变 G，已说明大 N 时该差距可以变小。因此用户之前的“大 token 时不同专家数差距缩小”具有直接先例。

论文还对权重副本做轮换以避免 L2 重用影响计时；这对我们单层图的真实整模型外推很重要。当前延迟变化不能未经硬件计数器或缓存控制就全部归因为 HBM 权重读取。

TEMPO 的新意在如何给专家副本分派 token，而非选 PP stage 数。把其成本项移植到 PP/EP 选择是合理基础，但移植本身不足以证明新颖性。

### 3. Inference economics 已把这些要素联合起来

§3.2 Eq22 在独立均匀路由假设下，令模型专家总数为 E、每 token 的 Top-K 为 k、batch token 数为 b，则激活专家数的期望为：

`A(b) = E × [1 − (1 − k/E)^b]`。

此式用于估计从 HBM 读取的参数量。它是理论期望，不是本报告对实验样本做平均。§3.5 又用 `b_micro = b / P` 建模 PP，加入 P−1 次阶段 P2P，并计入 EP dispatch/combine。

本轮核查作者公开仓库固定提交 `c2c4fa918bcbdfdcde4b82ef4ef36201f6f8a855`（2025-05-28）。`new_token_latency_seconds` 中：

- 当 `use_pp=True` 时枚举 PP，令 `micro_bs=batch_size/N_PP`；函数默认不启用 PP，但 notebook 也有显式 `use_pp=True` 的调用。不能仅凭默认参数声称它没有搜索 PP，也不能假定所有图都启用了这个搜索。
- 对每个候选 PP，`N_EP=min(N_GPU/N_PP, n_experts)`；当 `micro_bs < 2 × sparsity_factor` 时，强制 `N_EP=1`。这里 EP=1 后可以仍使用 TP，不等于选择纯 PP。
- `_ff_rw_bytes` 按微批估计触及专家的权重读取；整体内存项包含 N_PP 倍的微批重复服务；net 包括 EP all-to-all 与 PP P2P。
- 这是粗粒度理论模型，EP 并未独立遍历所有 degree；没有在这段模型中发现真实 expert histogram、逐 tile 调度、完整 pipeline timeline 或按实测 overlap 拟合的项。§3.4 明确采用不隐藏网络通信的保守近似。

所以不能说“此前无人把 batch、专家权重、PP 和 EP 放在一起”。可研究的是这些近似在当前引擎上的有效边界，以及更精确的模型是否能改变有价值的决策。

[原论文](https://arxiv.org/html/2506.04645v1) · [固定提交的作者 notebook](https://github.com/ege-erdil/inference-economics/blob/c2c4fa918bcbdfdcde4b82ef4ef36201f6f8a855/inference_economics_notebook.ipynb)。

## 和现有 H100 表格的对应

当前模型的 Top-K 固定为 8；横轴 N 是整个 EP 组的原始 token 数。对照中的 Expert 输入则是每卡收到的 token–expert 行数，两者不能混用。

在现有 35B 确定性、均衡路由及理想饱和流水的折算下，EP2 与 PP2 在相同 N 下的局部行数都是 `4N`，但每次 Expert 调用触及的专家数可能不同：

| 全组原始 N | 两方案各自局部行数 | EP2 本地激活专家 | PP2 单微批本地激活专家 |
|---:|---:|---:|---:|
| 32 | 128 | 128 | 128 |
| 64 | 256 | 128 | 256 |
| 128 | 512 | 128 | 256 |

其映射来自 `active(n,ep,total)=(8/ep)×min(n,total/8)`；PP2 每微批含 N/2 个原始 token。这里的“Expert 更多”不代表模型 Top-K 或全局有效 FLOPs 增加，而是 token 行分布到更多本地专家，可能改变权重重用、tile padding 和内核效率。

以下全部为已有 CUDA Graph 数据的 compute process 1、communication round 1、event 1，单位 ms，未计算样本平均：

| N | EP2 Expert | PP2 Expert | PP 少付的专家通信 | 省通信−额外 Expert 成本 |
|---:|---:|---:|---:|---:|
| 32 | 0.379136 | 0.379136 | 0.038720 | +0.038720 |
| 64 | 0.378688 | 0.758304 | 0.046016 | −0.333600 |
| 128 | 0.373408 | 0.761984 | 0.061696 | −0.326880 |

这些是“组件预算”反转：正值为 PP 支付额外交接/气泡等开销留下空间，**不是已经测得整模型 PP 赢**。Expert 与通信分别测量，不能假定相加就是共同执行的关键路径。N=32 两边引用同一 Expert 测量点，差值为零不是两次独立测量恰好相同。

还须保留以下边界：

- 现有通信数据是 DeepEP low-latency、全组最多 2048-token 分块；N>2048 产生多次调用。大 N 处再次出现 PP 候选优势，可能同时包含 chunk/调用数量差异，不能全部归因于 Expert 激活数。
- EP8 跨两节点，EP2/4 在节点内 NVLink；degree 与通信域同时变化，不能将差异单独归因于 EP degree。
- 单层测量没有覆盖完整 Attention/GDN、PP 交接、气泡、真实路由偏斜、共执行干扰和 overlap。
- GLM/HY3 沿用用户约束：最小 EP4，只讨论 EP8 与 PP2×EP4；不提出纯 PP 的不可行配置。

[组件表 CSV](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_tradeoff_20261029/component_estimates_event01.csv) · [映射代码](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_ep_pp_tradeoff_20261029/analyze.py)。

## 更可辩护的研究定位与最小证据

目前不建议将贡献定位为通用的异构实例池或通用自动配置搜索。更具体、可证伪的命题是：

> 在相同 GPU 预算、模型精度和服务阶段下，PP 微批改变每卡的专家工作集与 token 形状；它与通信域、内核分块和 overlap 共同决定 PP/EP 边界。我们提供受控 benchmark，并验证哪些简单模型足够、哪些需要额外的路由和内核状态才能可靠选型。

三个最小验证任务，按顺序进行：

1. **先确认同阶段的真实边界。** 选一个模型和一个 GPU 预算，在现有候选边界附近测完整层/整模型 EP 与调优 PP；固定路由、Top-K、dtype、Attention 布局、Graph、KV 协议与输出工作量。Prefill 给 EP 配适用的高吞吐后端，分别报告每种协议，避免用 LL 分块劣势制造结论。分别报告单请求延迟、整批完成时间、饱和吞吐，不能混成一个 winner。
2. **再验证原因。** 固定 N 和通信形状，改变路由专家并集与分布；固定路由改变微批数；记录每个 stage/rank 的 G、n_e、padding、HBM/L2 流量、暴露通信和空闲时间。先用 N≤2048 的点消除多 chunk 混杂，再独立研究大 N 的 chunk 选择。不能通过改变 Top-K 来代替改变 batch 活跃专家数。
3. **最后评价模型和决策价值。** 相同候选与 profiling 预算下比较：只看 N、固定小微批开销、均匀路由 roofline、TEMPO 式 G/N/tile 成本与 OServe 式容量分配。增加高预算密集 profiling 的强对照。留出整个 batch/路由/workload 组合，报告选错后多出的绝对毫秒和固定资源下 SLO goodput；在线路由计入合批反馈，迁移策略计入状态搬运与等待。

若简单模型或密集容量表已足够，结论应收窄为可复现 benchmark 与部署方法；若真实路由/调度耦合造成可重复的误判，再提出相应机制。不能通过只选择有利边界证明普遍 superiority。

## 阅读范围、附加筛查与可追溯性

- 旧文献复核：OServe 正式版 App D/K；PipeSwift v4 pp6–7、11–12；TEMPO v2 pp3、16；MoEFS v1 pp3–6、9；Beyond the Buzz v1 p5；Moebius v1 p3。结合此前已保存的对应章节笔记。本轮没有将这些全标为全文精读。
- 新归档：SMIDT 正式版 10 页，定向读 §1–5 与 §6 范围讨论；Inference economics v1 23 页，定向读 §3.2–3.5 与作者实现；HAP v1 8 页，定向核查 §II-C、III-A/B/D。SMIDT Fig4 和 Inference economics PDF p12 原文已目视核对。
- 另外筛查 [Semantic Parallelism / Sem-MoE，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html)：官方摘要已有 expert coactivation 放置与在线 request rebatching。只按摘要记录其对前一版“expert-aware 实例路由”构想的相关性，未据此判断其是否覆盖 PP/EP 边界。
- [Characterizing Parallelism Strategies，2610.05305v1](https://arxiv.org/abs/2610.05305)：官方摘要的重点是 TP/PP/hybrid 的计算、通信和 pipeline bubble，未用它作为 MoE 专家工作集机制的直接证据。
- 前一版机制规划：[MOE_INSTANCE_ROUTING_DESIGN_20261029.md](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/MOE_INSTANCE_ROUTING_DESIGN_20261029.md)。本次新增 SMIDT、Inference economics 及源码核查，进一步收窄其“潜在增量”的表述。

检索标记：`pp_ep_expert_workset_literature_audit_20261029`。
