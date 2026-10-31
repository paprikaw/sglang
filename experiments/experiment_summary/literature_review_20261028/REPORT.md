# MoE EP／PP benchmark：文献核查与可发表问题评估

日期：2026-10-28。针对现有 EP／PP、专家计算二维网格、通信域和 HiCache 实验。此次只检索、阅读与核对保存结果，没有启动 GPU 实验。

**结论：有值得推进的 benchmark 切口，但还不能凭现有几组数据断言 novelty 已经充分。** 最有希望的问题是：**同等 GPU 预算下，MoE 并行部署的排名何时会改变；忽略真实路由、流水执行或 KV 驻留状态的测试，会付出多大的选型损失？**

通用的“MoE 性能 benchmark”“EP 对比 PP”“token×专家数网格”“profiling 后自动选布局”都已有直接先例。建议把贡献放在可复用的测试任务、受控协议、可复现的排名边界与失效案例上。新的调度器或自动选择模型可以是基线或后续扩展，不是这篇 benchmark 的必要前提。

## 最接近的文献到底覆盖了什么

下表来自指定版本的相关方法和实验章节，不是仅凭摘要分类。“未报告”只指核查过的正文及实验范围，不等于整个领域没有该能力。

| 工作与版本 | 已有覆盖：不能作为首次贡献 | 对本项目的具体边界 |
|---|---|---|
| [MoE-Inference-Bench，2025 v1](https://arxiv.org/html/2508.17467v1) | §7.1/Fig13 已比较 TP、PP、EP 组合；还测模型、batch、输入输出和 top-k | 它的 top-k 变化与固定模型、固定配对行数的活跃专家变化不同；不过后一问题也已有 TEMPO 等工作 |
| [MoE-CAP，NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/file/74bd547997917a20331d6df5e6049d6a-Paper-Datasets_and_Benchmarks_Track.pdf) | 成本、准确率、性能；S-MBU/S-MFU；多引擎与硬件。附录已含动态 batch、多节点、Prefill 和突发 trace | 不能用“加上动态请求或跨节点”区分；需提供 EP／PP、KV 驻留和通信域的配对部署证据 |
| [PipeSwift，2026 v4](https://arxiv.org/html/2609.16491v4) | MoE EP／PP 的阶段取舍、流水调度和 agentic 端到端结果 | PP 节省通信、流水气泡、阶段偏好已经是已知问题；需要独立于单个系统优化的基准 |
| [TEMPO，2026 v2](https://arxiv.org/html/2608.13057v2) | token 数与激活专家数的联合成本，EP 分配及服务验证 | 现有 T×E 曲面适合作为解释性证据；不能单独作为核心新意 |
| [MoEFS，2026 v1](https://arxiv.org/html/2607.18631v1) | 校准成本模型、TP／PP／EP 搜索、kernel 和可实现约束 | 少量 profiling 加配置推荐已有先例；应测独立工作负载上的选择损失 |
| [Beyond the Buzz，2025 v1](https://arxiv.org/abs/2506.05508v1) | Fig5 已固定 64 GPU、扫描 EP×PP；另分析硬件、流量、NVLink 域与 P/D 配比 | 大量结果来自专有模拟器。公开实测可补验证，但必须指出具体边界；不能推断其模型忽略未公开的成本项 |
| [DODOCO，2026 v2](https://arxiv.org/html/2605.20982v2) | 跨架构、真实／构造输入、EP 宽度和 dispatch 尾部行为 | 随机与真实路由、通信偏斜本身不新；其 dispatch 测量不是完整在线服务 |
| [DA-MoE，2026 v3](https://arxiv.org/html/2607.23099v3) | token 规模、路由偏斜、padding、kernel 自适应 | 文中的完整 MoE-layer Graph 与服务级 TTFT／TPOT 应区分；二维 kernel 网格已有覆盖 |
| [MoE Expert Caching Evaluation，2026 v4](https://arxiv.org/html/2608.07911v4) | trace 重放、batch union、缓存策略排名和负载污染 | 研究专家权重缓存，且为 trace 模拟；不等于真实 GPU 上的 KV 恢复与服务性能 |
| [Revisiting Pipeline Parallelism，OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/hwang) | 动态 chunk、context 模型、delay scheduling 和 microbatch 的 GEMM 边界 | 其核心评估是 dense 模型；MoE 路由与 EP 的联系仍需实测，调度思想本身不新 |
| [TensorRT-LLM MoE Performance Guide](https://nvidia.github.io/TensorRT-LLM/developer-guide/moe-perf-guide.html) | 已搜索 kernel／通信／TP-EP，并把微基准排名与 InferenceX 端到端结果核对 | **明确限制 pp_size=1、cp_size=1。** 可检验的扩展是加入 PP、容量和 KV 恢复；“micro→macro”本身不新 |
| [InferenceX／AgentX 方法](https://inferencex.semianalysis.com/blog/agentic-benchmark-agent-benchmark-guide) | 多硬件、框架、recipe 记录、agentic 闭环、缓存层级与服务指标 | 多测硬件、真实 trace 或可复现性都不足以单独区分；需形成专门检验 EP／PP 排名边界的受控任务 |

当前最需要正面对照的是 MoE-Inference-Bench、MoE-CAP、PipeSwift、Beyond the Buzz 和 TensorRT-LLM 的方法。TEMPO／DA-MoE 主要约束专家成本相关的新意。正式论文必须保留这些重叠，不能用“相关工作没有考虑”笼统带过。

## 建议收窄到三个可验证的问题

**RQ1：同等 GPU 预算下，哪种布局在什么条件下更好？**

比较优化且通过正确性资格的 EP、PP、多个小 EP 副本，补少量可行的 TP 对照。保持模型、精度、输入、质量要求和物理 GPU 集合一致，改变 batch、每请求缓存前缀 P、新增输入 Q、生成长度 O 及实际驻留层级。输出吞吐与 TTFT／TPOT／goodput 的边界图，不预设 PP 或 EP 总是获胜。

固定预算是实验公平性的必要条件，已有工作也采用过。新增价值必须来自它在实际引擎上揭示了此前基准未暴露、且能跨模型复现的排名变化。

**RQ2：哪些简化测法会导致错误选择？**

分别测试仅用总 token／FLOPs、仅用独立通信时间、只用 warm 算子曲线、只用短上下文、忽略缓存驻留、只测默认 PP 等规则。对照完整服务的实测结果，记录规则在哪些区域失效，以及错误布局损失多少。已有 NVIDIA 指南已经验证过 MoE kernel 排名与服务的关联，因此本项目需明确聚焦它排除的 PP 及 KV／容量边界。

指标可用 `selection_regret = 1 − goodput(chosen) / goodput(best_qualified)`。各配置必须使用同一 SLO；全部不满足 SLO 的点标记 infeasible，不能做除零或把失败点删掉。没有新选择器也可以用这些规则说明基准的用途。

**RQ3：排名变化来自哪里，能否迁移？**

在少量关键边界点采集实际 token–expert 配对、每专家行数、padding、PP 各微批的专家交集、dispatch/combine 暴露时间、stage 等待和 KV 恢复。用配对开关实验验证机制，并在第二类模型／硬件上保留独立验证点。

算子、通信和整机协议必须兼容。存在 overlap 时不能直接把独立 kernel 时间和通信时间相加。逻辑激活权重字节不等于实测 HBM 读取量；同样，CPU 前缀 token 命中率不等于搬运字节占比。

## 现有实验已经提供的证据

| 已有结果 | 对 benchmark 的用途 | 目前不能支持的结论 |
|---|---|---|
| [HY3 A100 T×E 曲面与留出验证](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/hy3_expert_surface_20261022_r2/summary.md)：MAPE 36.71%→8.18%，最坏 46.74% | 说明只按工作量估计专家计算会失准；可复用为算子测试 | 单层 warm Graph，不是整模型成本模型；不能把误差下降当成部署选型准确率 |
| [35B H100 PP lookahead](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_lookahead_workloads_20261027/report/REPORT.md)：B8 比 native PP 快 28.79%，比 EP 快 18.17%；B32 吞吐接近 | 说明 baseline 调优状态能改变对 EP／PP 的判断 | 旧的 P4096 资格不能自动延伸到最新长前缀／HiCache 负载 |
| [A100 DeepEP EP2→EP4](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_a100_20261027/REPORT.md) 与 [H100 对照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/deepep_ep2_ep4_h100_20261027/REPORT.md) | 提供跨通信域边界的动机与局部证据 | 低延迟通信 API 的 microbenchmark 不能直接代表 Prefill 原生 backend 或整模型加速 |
| [最新五种四卡布局＋HiCache](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_five_layouts_hicache_20261028/SUMMARY.md) | 已有同预算、多布局、容量／驻留相关的排名变化；最适合作为论文动机 | 仍是合成重复前缀，尚无生产 trace、全负载 PP 正确性资格和跨架构泛化 |

一个需要先排查长尾、再考虑用作论文动机的例子如下。**同一 4×H100 NVLink、Qwen3.6-35B-A3B BF16、B64；每请求 P=130,304、Q=1、O=128；正式点各三轮。** 数值为 Decode 输出 tokens/s。

| 重放协议 | EP4 | PP4 | 该协议内的排名 |
|---|---:|---:|---|
| 自然缓存重放；两者本轮 CPU 前缀命中均为 0% | 2,292 | 1,658 | EP4 更快 |
| 受控 CPU 重放；两者 CPU 前缀命中均为 100% | 1,230 | 1,669 | PP4 更快 |

**逐轮复核后的修正：本例目前更适合作为长尾诊断案例。** EP4 CPU 重放三轮分别为 1584／1564／541 tokens/s；最后一轮有一个请求到 14.325 秒才收到首 token，63 个其他请求约 5.1 秒完成。按三轮吞吐中位数比较，PP4 领先约 6.7%；原均值显示的约 35.8% 被慢轮明显放大。慢轮仍保留，不把它未经解释地删除。PP4 的两种模式实际均选择 native、microbatch=16，已经排除本例的 PP 配置切换。

EP4 两种模式下 TPOT 都更低；CPU 模式的 PP4 优势主要表现为首 token 等待更短，不能据此说其 Expert kernel 更快。不同缓存历史和新增 branch token 仍会改变生成与路由，且慢请求延迟进入执行的根因未确定。CPU→GPU 恢复在计时内，批前写入 fence 和主动 GPU 驱逐在计时外。要先定位请求接入／恢复／调度等待，再确认缓存状态对排名的因果作用。[逐轮分析与重算数据](CACHE_REVERSAL_ANALYSIS.md)。

这一例子优于把所有旧表拼成一个总排名，但一个模型、一个节点上的反转还不足以完成 benchmark 论文。旧的未开 HiCache 的 24.1×结果不适用于此轮；原 BENCHMARK.md 的 34/132 也不是截至今天的完整覆盖率。

## 需要补什么：按优先级而不是按笛卡尔积扫点

1. **先完成正确性资格。** 最新五布局报告明确排除了未通过输出校验的 PP2 lookahead，以及长上下文 slot 复用出错的 tail-local 路径。短 Prefill 的已有数值差异也需定性。资格失败记录为 failed／unsupported，正式排名只收合格点。“最新优化 PP”应写成具体版本、开关和已验证负载范围。
2. **冻结一套可运行协议。** 固定输入清单、模型／引擎／native 库版本、完整 Attention TP/DP 与 MoE EP/PP、物理拓扑、kernel/Graph/overlap、KV/GDN 内存池、候选调优预算和计时边界。已有 manifest 是基础，还需一键 runner、资格检查和统一结果 schema。优先重用现有日志。
3. **保留两类缓存测试。** 一类比较真实部署可用容量与自然命中；另一类固定缓存驻留及恢复规则以解释机制。每请求保留 P/Q/O 和请求组成，分别统计 Prefill 与 Decode，再加入完整服务结果。
4. **补第二类架构和边界迁移。** 35B 与 122B 的规模差异不自动等于两类架构。选一个传统 Transformer MoE 与现有混合架构对照，在 A100/H100 及跨 NVLink 域的少量决定性点复验。第二引擎先做代表性子集，以判断现象是否只属于某个 SGLang patch。无需一开始跑所有组合。
5. **加入真实到达与 SLO。** 在边界附近做少量动态 batch、混合输入输出、自然结束的 trace。报告 P50/P95/P99 TTFT/ITL、TPOT 和满足 SLO 的 goodput。固定 O 的服务实验仍保留为控制组；不要以单一峰值 tokens/s 代替服务结论。
6. **做测试方法本身的消融。** 冻结规则后使用未参与调优的工作负载；比较默认／调优 PP、随机／真实路由、warm／受控驻留、单阶段／完整服务得到的排名。只有确认测法改变了结论，才写成 benchmark 的主要发现。

跨硬件需同时保留“各硬件最佳可用软件栈”和“相同可用路径的控制比较”，不能把 FP8 native／fallback、不同 tile 或 Graph 差异全部归因于 GPU。统计上使用独立重复和误差区间；当前三轮内 1%–2% 的差距不宜当成稳定胜负。

建议最终的四张主图：同预算布局的吞吐—延迟边界、按 P/缓存状态划分的排名图、决定性点的关键路径分解、简化选型规则在留出工作负载上的 regret。T×E 曲面作为其中的解释图或附录。不要为了覆盖率机械补满旧 132 点。

## 是否足以推进成论文

| 候选定位 | 当前判断 |
|---|---|
| 再做一个通用 MoE benchmark／比较 EP 与 PP | 已有工作很接近，gap 偏弱 |
| T×E 二维成本＋自动选配置 | TEMPO、DA-MoE、MoEFS 等重叠大，单独不充分 |
| 固定资源下，优化 EP／PP／副本在路由、通信域与 KV 驻留交互下的可复现排名边界 | **值得推进；当前证据有动机价值，但仍需资格、泛化和方法失效实验** |

可用的工作标题：**Benchmarking MoE Parallelism under Routing, Topology, and KV-Cache Constraints**。

推进标准应是：至少找到少数可复现、影响实际部署选择的边界，并证明简化测试会稳定选错、损失显著；同时发布别人能运行的 harness 和可审计数据。如果所有“反转”最后都来自错误实现、未匹配配置或单一模型，benchmark 的科学贡献会明显变弱。若完成上述验证，就有比“多测了一些配置”更清晰的论文论点。这里是基于已核查文献的判断，不是新颖性或录用保证。

## 检索范围与归档

检索截至 2026-10-28，结合用户引用的三段实验对话、当前实验 registry／报告、已有 Papers 阅读库、官方 arXiv／会议正文及项目文档。定向核查 10 篇论文和 2 组工业基准／方法来源；新归档 5 篇 PDF。相关方法与实验章节经过阅读，两个直接 benchmark 的关键图表经过原图检查；不标记为全部逐页精读。

检索主题包括 MoE inference benchmark、EP/PP deployment、activated experts/token cost、routing skew、cache replay，以及 microbenchmark-to-serving validation。Consensus 搜索额度已耗尽，此轮使用可访问的官方论文、会议信息与项目方法完成核查；这不是穷尽式系统综述。预印本与正式发表状态已分别记录。MoE-CAP 采用 NeurIPS 正式版及其 canonical arXiv 2412.07067；另一个重复投稿编号的撤回不代表该论文撤稿。

[逐篇范围与可排序对照表](literature_matrix.csv) · [版本、PDF 来源及 SHA256](sources.json)。PDF 与公开文献笔记归入 Papers；未公开实验数据与本项目选题判断仅保留在本实验报告目录。

该报告提出后续测试设计，没有运行上述补实验，也没有声称现有文献的方法在本项目上已经实测失败。

归档核验：10篇PDF的 SHA256 核验与笔记全文搜索检查通过；公开文献资料已同步至 Papers 的 origin/main，提交 `55ba937934616cf472b5efa236e1a05be921794d`。本地阅读器服务拒绝连接，因此未生成可验证的阅读对话链接，5篇新增论文的 Zotero 导入也未完成；PDF和笔记本身已保存并可搜索。详见 [验证记录](validation.json) 与 [逐项阅读服务检查](reading_verification.json)。
