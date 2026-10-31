# EP8 与 PP2×EP4：通信节省、启动过渡和平均 TPOT

## 结论

这轮 PP 的整批输出吞吐高 7.30%，平均端到端延迟低 8.96 ms；平均 TPOT 仅高 0.379 ms/token。此前“PP 仍更慢”必须限定为平均 TPOT，不能泛指整批性能。首轮 Prefill→Verify 过渡解释了这一小差额；没有证据表明 PP 每轮都多出同样大的 bubble。

## 条件与计时边界

065／167，8×A10080GB，同型号 ConnectX-6 Dx100Gb/s。GLM-4.7-AWQ／Humming INT4、BF16 KV；EP8 为 Attention TP4×DP2／Expert EP8；PP2×EP4 为两个真实 B4 slot、目标层50／42，尾部执行 MTP。EAGLE top1／steps2／Verify Q3，local Graph B4，固定 pair NVLink＋pair外/跨节点RDMA native9dc34887…，PCIe threshold0，无CE。

全局B8、逻辑P4096、Q1、输出64。计时前复用语义目标＋草稿 Bank，native CPU4032／GPU0。因页尾64 token未保留，首个 forward 实际为每请求65 token。Bank导入、权重加载、预热和Graph捕获排除；native HiCache H2D、排队、首次残余Prefill和后续Decode包含。不能把本组当成完全GPU热缓存的纯稳态Decode。

## 未插桩正式结果

每布局两个正式cohort；延迟取请求均值后再对cohort等权平均。吞吐为每cohort512输出tokens／整批wall，再取两次均值。

| 配置 | 输出tok/s ↑ | 整批wall s ↓ | TTFT ms ↓ | TPOT ms/token ↓ | 平均E2E ms ↓ |
|---|---:|---:|---:|---:|---:|
| EP8 | 131.395 | 3.8969 | 548.645 | 46.894 | 3502.977 |
| PP2×EP4 | 140.990 | 3.6315 | 515.776 | 47.274 | 3494.013 |

整批wall由最晚完成的请求决定，不等于请求平均E2E；两者并不矛盾。两次cohort不是置信区间，不能泛化为所有workload的胜负。

## 小幅 TPOT 劣势来自哪里

`TPOT=(最后输出时间−首输出时间)/63`。首个SSE块严格为1 token。逐请求从原始事件重算：

| PP减EP8，对平均TPOT的贡献 | ms/token |
|---|---:|
| 首个token后到下一次输出的过渡 | +1.0145 |
| 随后直到最后输出的时间 | -0.6351 |
| 总差额 | +0.3795 |

另一个等价恒等式：`ΔTPOT=(-8.9633−(-32.8690))/63=0.3795ms/token`。PP首token早 32.87ms、最后token仅早 8.96ms，首尾间窗口反而略长。

正式数据中，首token后的首间隔平均为EP8 109.35ms、PP 173.27ms。PP槽A约234ms、槽B约112ms；EP约109ms。中段SSE周期EP 113.11ms、PP 112.36ms，差异很小。中段按预先固定的首尾各两个间隔排除规则计算，不是GPU忙碌率。

## 时间线直接观察

PP head rank0同一GPU时钟：Prefill A为0–200.79ms，Prefill B为202.02–403.66ms，第一次Verify A为411.24–460.37ms，第一次Verify B直到589.08ms才开始。两次Verify之间128.71ms没有这些模型forward；不能据此声称整张GPU完全空闲。

head本地CPU在microstep10000提交Verify A后，进入另一槽B的反馈收取／预处理API，414.40–582.58ms，共168.18ms。该API包含等尾阶段结果、通信、预处理以及与本地GPU的重叠，不是168ms纯网卡传输。尾阶段的记录显示，启动顺序仍为Prefill A→Prefill B→Verify A；槽A进入Decode时另一槽的残余Prefill尚未清空。这与双槽结果按序退休的源码依赖吻合。后续反馈API约49–51ms大部分与本地49–52ms Graph重叠，不能再次全额相加。

因此，主要是短输出下的流水线启动／Prefill-Decode过渡和反馈依赖，不是未预热，也不是每次Verify都多129ms。相同shape已预热；正式两次cohort和独立诊断均复现首轮长间隔。

## 为什么通信节省没有全部变为 TPOT 节省

历史同Bank组件profile保留：完整目标Verify路径的DeepEP D+C为26.71→17.93ms（−32.9%），加TP后内部通信为33.76→24.39ms（少9.37ms）。这是有诊断同步的组件数据，未包含PP边界、返回、调度，也不是每输出token指标；不能直接从未插桩TPOT中相减。

本轮无逐轮fence时间线中，同一请求组两次Verify启动间，PP每个stage要轮流执行两个slot的Graph；EP每张卡执行自己的本地B4。每GPU／每round的诊断均值如下：

| 布局/阶段 | 目标Graph ms | 草稿Graph ms | 这些Graph之外 ms | 周期 ms |
|---|---:|---:|---:|---:|
| EP8 | 107.981 | 5.133 | 3.728 | 116.842 |
| PP stage 0 | 104.079 | 0.000 | 8.644 | 112.723 |
| PP stage 1 | 91.904 | 9.198 | 11.779 | 112.881 |

PP暴露的Graph外间隔多约5–8ms/round，支持反馈、阶段边界和调度依赖会吃掉部分Graph内节省。这里只能定位到这些Graph外，不把全部残差命名为GPU bubble或纯NIC延迟。不同stage的周期不能相加，Graph/forward/batch嵌套区间也不能相加。

诊断整批wall相对首正式cohort：EP8+2.34%、PP+0.23%；EP诊断中保留了长Graph样本，没有删掉它来选胜者。图中116.84/112.72/112.88ms是诊断均值，不能替换正式SSE中段113.11/112.36ms。

## MTP与有效性限制

首轮下一次输出在EP推进3tokens，在PP推进1token；输出事件间隔不是单个Verify内核成本。每布局的生产重复、独立时间线和既有同布局参照均通过输出一致性检查；跨EP/PP严格token等价尚未完成资格验证，不能据此声称无损MTP或纯transport同轨迹因果对照。

原配对run的旧EP诊断因flow请求未进入固定DP路由而失败，原始failed记录保留，正式数据未被改写。修复flow请求的路由、观测和DP证明后，仅重跑EP，不重跑PP。选用的PP与新EP均通过全八卡Prefill/Verify覆盖、结果验证和清理；新EP完整Guard postcheck通过，租约释放。CPU权重／语义Bank保持原样，没有新GPU预约。

## 下一步应优先优化什么

优先解耦启动残余Prefill与已就绪Decode槽的反馈退休，并缩短Graph外的结果处理／阶段依赖；对两个slot继续维持MTP尾部负担均衡。只降低DeepEP内核时间不足以保证更低平均TPOT。增加输出长度能摊薄启动项，但属于不同workload，不能冒充本次修复或结果。

## 文件

- [时间线与周期图](startup_causality.png)，[PDF](startup_causality.pdf)
- [正式指标CSV](production_metrics.csv)，[含SHA256的原始事件重算](causal_evidence.json)，[验证与来源](validation.json)
- PP源目录：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_ep8_startup_causality_20261023/runs/paired_r1`
- 修正EP源目录：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_pp_ep8_startup_ep8_followup_20261023/runs/ep8_r1`
- 历史通信组件的独立证据：`../glm47_communication_evidence_20261023/REPORT.md`

所有归档均留在当前workspace；没有推送GitHub、发布网站或拷贝权重／KV／语义文本。
