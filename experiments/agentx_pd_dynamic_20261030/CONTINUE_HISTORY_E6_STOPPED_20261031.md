# 当前继续执行入口（2026-10-31 06:45 Melbourne）

目标仍active：修复技术问题，用两个既有H100节点共8卡的AgentX实验验证真正P/D分离是否有用，以及EP与EP=1/PP动态配比是否有额外收益，最后交付索引Markdown总结。不要因基线完成就结束目标。所有角色HiCache ON，每GPU32 decimal GB CPU预算，GPU KV按.95原生自动容量；同episode不flush，会话/DP亲和保留。不新预约、不重复询问节点授权。不启动子agent。

## 当前活动进程（已核实）

- **E6 P/D `eight_gpu_pd_ep4_checkpoint_e6_r2` 已启动**。主队列exec **5006**、`handoff_single_then_pd_e6_r1.json`，stage successor_guard_running；Guard PID **274917** / start_ticks **383000270**。C8 r1正式回放中，0/6完成；需实时读progress。
- E6 CPU watcher exec **60426**、PID **271840**、`e6_analysis_watcher_r1.json`，已接管STATUS.json。它的已加载旧代码只更新per-run分析，E6总表可手动 `python analyze_e6_controls.py`；未来watcher读取的新代码会自动更新总表。
- 后续控制等待器exec **72248**、PID **58677** / ticks **382768480**，`wait_for_e6_controls_r1.json`。等待上述修复run全部完成、Guard/清理通过、其watcher finished后，创建 `handoff_e6_controls_r1.json`，顺序执行下方控制1、2。
- 控制CPU watcher等待器exec **77844**、PID **67270**，`e6_controls_watcher_wait_r1.json`；队列创建后exec为 `e6_controls_analysis_watcher_r1.json`。
- 默认队列控制等待器exec **6332**、PID **192997** / ticks **382906105**，`wait_for_e6_native_queue_control_r1.json`，等待前两控制整组通过后创建 `handoff_e6_native_queue_control_r1.json`。wrapper V2支持上游队列尚未创建，并检测上游失败。
- 默认队列CPU watcher等待器exec **86406**、PID **202623**，`e6_native_queue_watcher_wait_r1.json`，之后是 `e6_native_queue_analysis_watcher_r1.json`。
- **已终止，不要再poll**：single Guard exec75146 exit0；direct watcher56674 exit0；PP6927 exit0；旧PD队列21316 exit1；旧watcher57476 exit0。旧目录不重启。

任一前项失败，后续门槛停止。只处理自己拥有的进程，不删外国lease、不杀其他任务。当前GPU等待不是blocked；本轮有基线完成、E6实际启动与资格检查的新证据，属于progress。

## E6已经通过和仍待验证的部分

源码 `.engine-worktrees/agentx-gdn-pd-hicache-r6-20261031`，commit `fc014e3553c4d234f38f3088a3158c0333e64830`，tag `serving/sglang/qwen35b/agentx-gdn-pd-candidate-r6-20261031`。只改MambaComponent及压力单测；16原生CPU检查通过，原错误5项行为失败复现。非INT8可选未完成快照无可驱逐槽位时延期，保留活跃state、KV/prefix锁和有效跟踪深度，完成仍可捐赠；必需分配仍严格。源码冻结，勿编辑。

约06:33 E6在两个节点真实启动、六条串行输出与E5参考一致、全rank冷/缓存复用预热全部通过。P每rank GDN80/KV2225472，D每rank GDN80/KV2195008；全部8个rank HiCache=true/32GB/.95。**C128压力路径、全部6轮、后检查和清理仍待，不能称GPU压力bug已修复。** 日志marker `MAMBA_CACHE_CHECKPOINT_DEFERRED` 每组件只警告一次，不是累计次数。详见 DECODE_CHECKPOINT_PRESSURE.md。

E5旧PD slots8目录完成C8三轮后，C128 r1在可选GDN快照分配时失败；未有完整C128。DP3最后GDN79/80、active2、transfer22、prealloc7，KV51%。整组未验收，清理/GPU idle/进程退出全通过；不能当P/D性能差。它与更早默认32接收预留导致GDN80→144的启动OOM不同。

## 后续已冻结的同E6控制（均尚未启动）

1. `eight_gpu_pp2_pd_controls_checkpoint_e6_r1`：col_pp2、pd_pp2、pd_pp2_ep4，C128各三轮，共9轮；GDN80等效/receive8，PP微批16/policy0，261项冻结哈希通过。保持P侧PP2时对比D侧PP2和EP4，并有同源码合并PP2基线。
2. `eight_gpu_ep2_split26_checkpoint_e6_r1`：col_ep2与pd_ep2 2P6D，C128各三轮，共6轮；64请求/80状态/receive8，260哈希通过。6个Decode缓存位置的副本偏置已修，8项CPU检查；一个D tile与P同节点，两个D tile跨节点，保留拓扑边界。
3. `eight_gpu_ep4_native_queue32_checkpoint_e6_r1`：col_ep4与pd_ep4，C128各三轮，共6轮；GDN160/receive32/active16，(16+32)*3=144<=160，260哈希通过。保留默认接收预留且启动前显式预算状态，避免用receive8低估P/D，并检查更强固定EP4。它同时调整GDN与队列，不作纯队列因果归因。说明 GDN_QUEUE_CONTROL.md；通用自动预算器未声称修复。

四个E6 run共27轮；目前修复run刚开始，控制都在队列。不要继续扩大配置搜索，先完成这些有明确理由的控制。详 E6_RUN_PLAN.json。

未启动废弃准备：E6 main r1缺参考，DO_NOT_LAUNCH；PP pair r1缺显式micro16，DO_NOT_LAUNCH；PP pair r2已被含同源col_pp2的controls替代；旧E5 V14 P/D候选不直接启动。准备包不是测量。

## E5同协议基线已验收（18轮，三轮算术平均）

| 配置/负载 | TTFT ms | 请求TPOT ms/token | E2E ms | root s | 压缩wall s |
|---|---:|---:|---:|---:|---:|
| 2×EP4 C8 | 431.710 | 9.640 | 3542.973 | 277.341 | 443.421 |
| 4×PP2 C8 | 277.152 | 7.137 | 3266.097 | 275.731 | 441.653 |
| 8×单卡 C8 | 348.458 | 4.990 | 2558.706 | 244.118 | 397.420 |
| 2×EP4 C128 | 2271.319 | 73.435 | 22624.015 | 790.365 | 1168.206 |
| 4×PP2 C128 | 1188.733 | 39.419 | 13128.898 | 554.648 | 835.796 |
| 8×单卡 C128 | 27305.923 | 18.662 | 33312.386 | 1012.758 | 1449.092 |

单卡全部6/6、后检查passed、清理/GPU idle/model exits全通过。C128间隔加权TPOT13.81688；host命中平均1257、回退平均8、cached44698.5575；实际GPU KV总2712832（011每卡338816，171每卡339392）。PP2完整模型去重KV15083904、EP4 17806080。首token前延迟占单卡高负载大部分；不能仅凭host计数独立归因缓存或排队。

所有8家族C8/C128请求E2E均PP2低于EP4，每家族3/3方向一致。严格1000ms/10ms SLO下C128 ready-goodput EP4 .3445、single .2175、PP2 .1369；1000/15则PP2 .5946、EP4 .4214、single .2278。完整网格保留，不挑目标证明动态。Family表已经包含单卡，48个家族点/48对比；事后标签不能当线上特征。

E5源码 `37e0828c642ce4f3795c4c4394f4ce913bcd1a35`，PP HiCache全局/局部层号修复27 CPU、48/48真实host恢复输出通过。E5与E6数据不混成同源码均值。

## 分析、报告、剩余科学判断

主报告已压缩重写：`../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md`，统一索引README已更新18轮基线完成。旧详细进度保留同目录 `PROGRESS_HISTORY_BEFORE_E5_BASELINES_COMPLETE.md`。

- E5自动最终表 `matrix_analysis/REPORT.md`、COMPARISONS、FAMILY_TRADEOFFS、STRICT_SLO_OVERLAP；18轮已验收与失败PD三轮分开。
- E6 `analyze_e6_controls.py` -> `e6_analysis/REPORT.md`：4个run按run/profile分组，显示GDN预算/receive8或32，验收、root/SLO/HiCache、压力日志；28个指标与已有EP4/PP2四个完整episode重算一致（e6_metric_reconciliation.txt）。新watcher自动调用；当前修复run可手动刷新总表。
- 每run分析 `analyze_cluster.py` / `analyze_exposure.py` / `analyze_tradeoffs.py --run ...`。完成队列自动调用；不把forward本地batch均值当同步全组batch，不把PD客户端cache字段当传输字节。
- 输出一致程度 `analyze_output_agreement.py` / output_agreement_e5：受控串行/host恢复通过不等于全部动态batch token一致。冷请求也有首token差异，未用logits控制归因数值或缓存；每轮源长度相同、下一请求按trace hash重建，不拼接模型输出。新刷新exec50174须poll，原13配对episode将补齐至15。
- 当前正式运行只有sticky分配；cache_queue未校准，见 POLICY_AUDIT.md。只有稳定布局互补仍存在，再测固定异构池在线路由，并给同质池相同缓存/队列策略；计真实状态/缓存/排队/batch影响。固定池分流不等于动态增减实例/权重重分片。

## 协议限制不能丢

8源家族171调用，C12816隔离副本2736调用；复制不增加多样性。synthetic hash tokens恢复前缀而非真实语义/专家路由。原数据有约40h间隔，单位正确；V3全局空闲压缩约144000s，分支return偏移与官方不同。所有wall/root/goodput是压缩派生回放，不是生产时间/官方分数。资料PROTOCOL_AUDIT、WORKLOAD_COVERAGE、TIMING_AUDIT。

V4独立时间修正已准备：agentx_branch_timing.py、agentx_replay_v4.py、test_branch_timing.py，9项CPU检查通过，对固定官方3方法AST+2helpers及24图、实际线程issue-before-complete/join核对；无GPU测量。不能静默混入V3。先完成当前同协议控制，再根据关键候选决定独立时间协议敏感性验证；对官方/真实任务的强结论需要更强证据。

仍未完成：真实P/D高负载验收、强固定P:D/异构对照、条件触发的动态收益/成本验证、最终证据审计和报告归档。不要update_goal complete；当前并非blocked。
