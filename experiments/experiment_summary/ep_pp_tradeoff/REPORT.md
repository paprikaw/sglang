# Expert 计算与 EP／PP 通信权衡：实验总结

更新：2026-10-30。本文汇总组件核对与 35B 整模型部署边界验证。[返回全部实验的统一索引](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)。

**最新固定配置对照：** 按用户最新指示，先筛选EP4与单卡／PP在不同P、D工况是否互有胜负。已有固定Decode P4096在N8偏单卡／PP、N64偏EP4，另一节点五轮反向复测成立；冷Prefill与混合流量中的EP4也已补入并列表。本步不以实现真实P/D分离为前置条件。[固定P／D完整对照](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/ep4_fixed_phase_comparison_20261030/REPORT.md)。

**研究口径补充：** 单卡与纯PP属于每层专家不做EP分片的同一类别；PP是否胜过单卡不是该类别与EP动态配比的必要条件。下文冷混合负载测的是整请求路由，尚未测真实P/D分离。新目标须分别验证P/D分离的净收益，以及布局类别动态配比的额外收益，见[范围修正与对照设计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_disaggregation_scope_20261030/REPORT.md)。

## 1. 已有哪些数据

| 内容 | 覆盖范围 | 报告与数据 |
|---|---|---|
| H100 单层 Expert 二维表 | GLM-4.7、HY3、35B、122B；计算行数 T × 活跃专家数 E；两份独立单卡测量，1584 格，每格 30 次事件 | [完整报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_four_model_expert_surface_20261029/REPORT.md) · [绝对计时 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/expert_all_events.csv) |
| H100 专家通信二维表 | 四个模型的激活尺寸；EP2／4／8 × 全组原始 token 总数 N=8–32768；624 个条件，两轮、每点 30 次事件 | [完整报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/h100_ep_token_comm_surface_20261029/REPORT.md) · [Dispatch／Combine 绝对计时 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/communication_all_critical_events.csv) |
| EP／PP 组件推导及核对 | 130 个布局条件；GLM／HY3 最小 EP4。属于查表估算，未独立测量完整布局 | [逐条件核对 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/component_mapping_audit.csv) · [历史推导数值 CSV](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/component_event01.csv) |
| 整模型 P／D 性能反转核查 | 35B／122B 等已有同协议结果；区分完整批次时间与首 token 之后的生成时间 | [证据报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/phase_reversal_audit_20261029/REPORT.md) |
| 35B 四卡、五种部署 | EP4、2×EP2、4×单卡、2×PP2、PP4；Prefill／Decode、两种缓存协议 | [三轮平均绝对时间与原始轮次](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/four_gpu_layout_retrieval_20261029/REPORT.md) |
| 新增：35B 小 batch 整模型 | 两节点并行，各自同节点 EP4 对照；N8／16／32，P4096/Q1/O128、GPU 热缓存，每点五轮平均 | [优化 2×PP2、4×单卡与原生 PP4](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/small_batch_layout_verify_20261029/REPORT.md) |
| 新增：35B 单节点 Decode 动态配比必要性 | N8–128、P256–131072、多主题输入；补齐 EP2／单卡强基线，关键点五轮反向复测 | [最佳布局、PP 增量、绝对均值与失败审计](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md) |
| 新增：35B Prefill／Decode 混合配比机会 | 冷请求64条、两档到达速率、三种P/D请求比例；两节点四卡布局、PP提前发送与同质请求隔离对照 | [135组有效三轮均值、SLO对照与完整来源](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md) |

组件表保留每个原始事件和轮次；四卡五布局报告按三轮算术平均值比较。两者的统计口径分别保留，时间均为绝对 ms。

## 2. EP4／PP4 的比较口径修正

**PP4 的每个 stage 持有其负责层的全部专家；EP4 的每张卡持有每层四分之一的专家。驻留专家数与当前激活专家数不同。**

以 35B／122B 的 256 个专家、top-8 为例，同一批 N=32 个原始 token，在本实验的均衡受控路由下：

| 项目 | EP4 每卡 | PP4 每个 stage 的负责层 |
|---|---:|---:|
| 应驻留专家数 | 64 | 256 |
| 处理同一整批时的 token–expert 行数 T | 64 | 256 |
| 本次受控路由的活跃专家数 E | 64 | 256 |

旧图把 PP4 拆成四个微批，每批只有 8 个原始 token，再按理想满流水归一化。它给 PP 查的也是 `(T=64,E=64)`，所以截图的两个 **0.187008 ms 来自同一个测量格**，不是分别测 EP4、PP4 得到的时间。

对同一整批单层，PP 对应的查表格应为 `(T=256,E=256)`；35B 的 Graph、计算测量 1、事件 1 为 **0.758304 ms**。但源扫描的权重张量始终驻留 256 个专家，**EP4 只驻留 64 个专家的实际分片尚未单独测量**；上述查表数值不能称为已完成的 EP4／PP4 实测对照，也不能直接推出整模型快慢。

全部 130 个推导条件已用 CPU 路由枚举核对。35B／122B 的纯 PP2／4／8 对照，在 N=8、16、32 共 18 个条件中复用了相同查表格。相等依赖特定路由及微批假设，不能用 N32→64 的变化证明真实模型存在布局切换边界。

## 3. 目前能得出的结论

- **组件数据已具备，真实布局的切换边界还未证明。** 原来的“省通信时间 − 多算 Expert 时间”只是受控路由、理想满流水下的预算，还缺 PP 交接、气泡和实际 overlap。
- **35B 已有 Decode 生成延迟随 batch 反转的证据。** 两卡、每请求 GPU 缓存 P4096、新增 Q1、输出 O128：B8 偏 PP2，B32 偏 EP2；包含首 token 的整批指标没有相同的反转，详见证据报告。
- **四卡选型应包含多副本。** EP4、2×EP2、4×单卡、2×PP2、PP4 均已有结果；缓存状态和恢复成本会影响排序，不能只比较 EP4 与 PP4。

测量边界：Expert 的 T 是单卡 token–expert 配对行数，通信的 N 是整个 EP 组的原始 token 数。EP2／4 在单节点 NVLink 内，EP8 跨两节点；通信全部为 BF16、DeepEP LL，大于 2048 token 时分块，未测 overlap。Expert 权重精度为 GLM AWQ、HY3／35B BF16、122B FP8。通信长等待样本与缺测均保留，不把不同协议的数据混成一个排名。

### 为什么整模型 EP4 占优，而旧图的小 token 区域偏 PP？

先对齐全组 token 口径。四卡五布局实验是全局 B64、Decode Q1；若 64 个请求均处于活跃 Decode，一轮对应全组 N64。EP4 每个 Attention DP rank 的 16 个源 token 不能当成预测图的全组 N16。实际微批、排队和收尾仍可能改变每次 forward 的活跃工作量，需要逐步日志才能严格匹配。

| 全组 N | 原图 PP 额外开销预算 ms | 该图实际表达的含义 |
|---:|---:|---|
| 8 | 0.047232 | 在受控路由和理想满流水下，还有这笔预算支付遗漏成本 |
| 16 | 0.050176 | 同上，尚非 PP4 端到端更快的证据 |
| 32 | 0.048064 | 同上，两侧还复用了同一 Expert 测量格 |
| 64 | -0.130336 | 该组件估算本身已偏 EP4 |

上述预算固定为35B、Graph、计算测量1、通信轮次1、事件1的每层归一化值，不是跨轮均值或完整模型时延。四卡自然重放、P8256、B64/Q1/O128的三轮平均整批时间为EP4 **1497.77 ms**、PP4 **2501.19 ms**；这组结果并没有在名义N64口径上与原图的方向相反。它也不能验证图中尚未单独对齐的N8／16／32区域。

正预算不是充分的胜出条件：PP还有阶段交接、下一步反馈和流水等待；真实路由可能改变微批的专家重复工作；EP真正可省的是关键路径上暴露的通信，而不是全部孤立通信计时。原扫描也未控制真实EP驻留分片，且未覆盖全部PP适配路径。因此当前只能列出待核对的遗漏项，不能把整模型差额直接归因到其中某一项。归一化可用于理想流水服务量，但不能直接替代有限批次、带缓存恢复的整批时间。

四卡HiCache结果只比较通过资格的native／early-send及微批候选，未采用该负载上未通过资格的lookahead；不能视为所有PP优化的上限。本次后续验证已固定 GPU 缓存、全组 N8／16／32，并记录实际微批与 Graph 资格计数，结果见下一节。它仍不能把“通信预算为正”直接等同于所有 PP 布局端到端更快。

### 新增整模型验证：有 TPOT 反转，但整请求的双向收益较弱

35B、完整 BF16、每请求 P4096/Q1/O128、无 CPU KV 恢复。两个四卡 H100 节点并行，各自使用本节点 EP4 对照；下表是 011 节点每点五轮的请求算术平均。

| 全组 N | EP4 平均 TPOT ms/token | 2×PP2 lookahead 平均 TPOT ms/token | 平均请求 E2E：EP4 / 2×PP2 ms |
|---:|---:|---:|---:|
| 8 | 6.106 | 5.484 | 962.21 / 789.67 |
| 16 | 6.757 | 6.711 | 1040.92 / 944.59 |
| 32 | 7.785 | 8.629 | 1172.41 / 1192.84 |

N8 的 PP2 TPOT 比 EP4 低约 10.2%；N32 的 EP4 TPOT 比 PP2 低约 9.8%。**同一 Decode 阶段内有反转，幅度约 10%，而非很大的双向优势。** 包含 TTFT 后，N32 的 EP4 请求 E2E 只低约 1.7%；不能用 TPOT 反转替代整请求收益。

171 节点中，四个单卡副本在 N8 的 TPOT 比同节点 EP4 低约 22.8%，N32 两者只差约 0.9%；原生 PP4 三个点均较慢。因此四卡候选必须包含单卡副本，不能只比较 EP4 和 PP4。跨节点不合并求平均或直接视为因果对照。

新增 B2 的 PP2 静态消息与 lookahead 形状逐请求通过原生固定分组输出核对，并验证四个 stage 实际执行与退休；75 个正式 cohort 全部保留，两个运行的 Guard、权重使用和进程清理检查均通过。尚未测布局交替复测、动态路由或切换成本；也没有在其他模型、上下文与硬件上证实相同幅度。[完整 Markdown、逐轮数据及资格证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/small_batch_layout_verify_20261029/REPORT.md)。

### 2026-10-30：加入更强基线后，35B 尚不支持 PP／EP 动态配比主线

主扫描的16个 Decode 工况里，优化 PP2 没有取得最低平均 TPOT。P4096/N8：EP4 6.122 ms、PP2 5.495 ms，四单卡更低至4.752 ms；P256/N128：EP4 10.630 ms、PP2 15.402 ms，EP4 延迟低约31%。低并发 PP 对 EP 的优势被单卡副本覆盖，高并发反差更大但偏向 EP。

P131072/N32 有小幅整请求收益：PP2 E2E2181.57 ms，EP4 2284.38 ms，低4.5%；TPOT15.293/14.936 ms，PP 稍慢。独立节点五轮反向复测确认该现象。同内存档共同可行工况的免费逐批选型代理，最高平均TPOT改进7.46%，去掉PP不改变结果；这不是在线混合池的收益上界。

本轮建议暂缓35B单节点Decode的PP／EP配比控制器，保留部署边界benchmark。122条性能记录、388个正式cohort均分协议保存；主扫描末尾单卡dense失败，已完成点独立审计，未把失败当作慢值。单卡N128的GDN临时显存峰值尚未解决。完整绝对数、五轮复测和范围限制见[必要性验证报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/decode_pool_necessity_20261030/REPORT.md)。

### 2026-10-30：混入 Prefill 后，首 token 有取舍，尚未证明配比的独有收益

新增完整P＋D冷请求到达实验，每候选固定四张H100；011／171两个节点独立主扫描。每轮64请求，P类Q8192/O16、D类Q256/O256，三种请求比例、16／32请求每秒；没有P→D状态迁移或在线扩缩容。135组有效测量、每组三轮，旧Decode与CPU缓存协议只作背景证据。

011的P较多、32请求/秒工况：4×单卡平均TTFT／E2E为385.85／1719.50 ms；提前发送、微批16的2×PP2为231.46／2347.99 ms。PP首token更早，整请求反而更慢；两节点六档混合负载均由四单卡取得最低平均E2E。原生PP微批8同工况E2E1735.80 ms，因此增大微批不是所有指标一起改善。

TTFT≤200 ms、TPOT≤20 ms时，该工况EP2＋PP2按请求类型分流的达标吞吐为2.907请求/秒；四单卡分成P类／D类两组达到3.638请求/秒。两者都是16/64请求达标；此处收益可由请求隔离解释，不能单归于PP／EP张力。另有约1%的弱优势点及171未做隔离补测的门槛点，均保留，未宣称所有场景无收益。

建议保留TTFT／生成节奏／E2E的不同目标，先研究请求隔离与容量共享。当前35B证据不足以启动PP／EP配比控制器；不排除长缓存恢复、更大模型或其他拓扑。完整冷Prefill表、微批资格、到达偏差整点复测与SLO敏感性见[混合负载报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/REPORT.md)。

## 4. 后续补充

优先补正确驻留专家数下的 EP／PP 单层对照、真实路由和微批重复工作、PP 交接与气泡，再用同 GPU 预算验证整模型的 P／D 区间。新结果在本文追加日期和来源，并加入[实验总览](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/README.md)；已有原始快照保持不变。

[本轮完整核对记录](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/REPORT.md) · [来源与哈希](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/moe_ep_pp_tradeoff_topic_20261029_v3/provenance.json) · [回归 Benchmark 清单](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/BENCHMARK.md)
