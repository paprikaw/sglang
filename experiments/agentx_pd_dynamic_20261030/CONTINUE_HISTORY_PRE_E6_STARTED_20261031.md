# 当前继续执行入口（2026-10-31 05:55 Melbourne）

目标仍 active：修复技术问题，用两个既有 H100 节点共 8 卡验证 AgentX 派生负载中 P/D 分离、EP 与 EP=1/PP 动态配比的必要性，最后交付索引 Markdown 报告。全部角色 HiCache 开启、GPU KV 原生自动容量、会话与 DP 亲和；同一 episode 内不清缓存。不新预约、不重复问 GPU 授权。

## 活动进程

- 单卡基线 `eight_gpu_col_single_durable_kv40_r3`，exec **75146**，Guard PID **37948** / start_ticks **382417862**；C8 三轮及C128前两轮完成，C128 r3运行（5/6）。E5 源码和输入已冻结，勿改。
- CPU direct watcher exec **56674**，PID **64406**，`matrix_analysis_watcher_direct_r1.json`；更新 E5 主矩阵和 STATUS.json。它退出前不要让另一 watcher 同时覆盖 STATUS。
- 后继队列 exec **5006**，PID **226367** / start_ticks **382628700**，`handoff_single_then_pd_e6_r1.json`；严格等上述 single owner 退出、全部完成、Guard 后检查和清理通过，再启动 **`eight_gpu_pd_ep4_checkpoint_e6_r2`**。260 个冻结哈希通过，六条串行参考完整。队列失败即停止。
- E6独立CPU分析 watcher exec **60426**，PID **271840**，`e6_analysis_watcher_r1.json`；等待direct watcher结束后才接管STATUS，E6性能不混入E5主矩阵。
- E6 2P6D 备用准备：exec **97215** 已退出0，260项哈希全部通过，日志 `prepare_ep2_split26_checkpoint_e6_r1.log`，目录 `eight_gpu_ep2_split26_checkpoint_e6_r1`。仅 C128×3，col_ep2 与 pd_ep2 两个 arm；同 E6、64/80、slots8、V3，全8卡。已作为后续控制队列第二项，必须等前项验收。
- E6后续控制已排入条件队列：exec **72248**，PID **58677**/start_ticks **382768480**，`wait_for_e6_controls_r1.json`；等待修复队列完整验收及其watcher finished，再创建 `handoff_e6_controls_r1.json`。先 `eight_gpu_pp2_pd_controls_checkpoint_e6_r1`（col_pp2、pd_pp2、pd_pp2_ep4，C128×3，共9轮，微批16、policy0，261哈希），再 `eight_gpu_ep2_split26_checkpoint_e6_r1`（col_ep2/pd_ep2 2:6，C128×3，共6轮，260哈希）。均为E6同源控制，未启动。
- 控制队列CPU watcher等待器 exec **77844**，PID **67270**，`e6_controls_watcher_wait_r1.json`；队列出现后接入 `e6_controls_analysis_watcher_r1.json`。详见 E6_RUN_PLAN.json。
- PP pair r1漏显式微批16，仅准备且标DO_NOT_LAUNCH；r2完整准备但被包含同E6 col_pp2基线的controls目录取代，不在队列。
- 已终止：PP exec6927 exit0；旧 PD 队列21316 exit1；旧 watcher57476 exit0。勿继续 poll/re启旧目录。

## 修复与科学状态

E5 PP HiCache 层号修复：commit `37e0828c642ce4f3795c4c4394f4ce913bcd1a35`，27 CPU、48/48 GPU 恢复输出验证。

E5 P/D slots8 目录在 C8 三轮后，C128 r1 因可选 GDN 快照分配失败退出；整组未验收、无完整 C128。清理/GPU idle/退出全通过。源日志 DP3 GDN79/80，active2、transfer22、prealloc7，KV51%。不是之前的启动 OOM，也不能当 P/D 性能劣势。

新 E6 worktree `.engine-worktrees/agentx-gdn-pd-hicache-r6-20261031`，commit `fc014e3553c4d234f38f3088a3158c0333e64830`，tag `serving/sglang/qwen35b/agentx-gdn-pd-candidate-r6-20261031`。非 INT8 可选未完成快照无可驱逐槽位时延期，保留活跃 state、KV、prefix 锁和有效跟踪深度；完成可捐赠旧快照；必需分配仍严格。16 原生 CPU 检查通过（修复前5行为失败）。GPU 尚未验证。详见 DECODE_CHECKPOINT_PRESSURE.md。r1 缺串行参考仅准备，DO_NOT_LAUNCH；使用 V16 r2。

## 已验收 E5 数据（每行三轮算术平均）

| 配置/负载 | TTFT ms | 请求TPOT ms/token | E2E ms | root s | 压缩wall s |
|---|---:|---:|---:|---:|---:|
| 2×EP4 C8 | 431.710 | 9.640 | 3542.973 | 277.341 | 443.421 |
| 2×EP4 C128 | 2271.319 | 73.435 | 22624.015 | 790.365 | 1168.206 |
| 4×PP2 C8 | 277.152 | 7.137 | 3266.097 | 275.731 | 441.653 |
| 4×PP2 C128 | 1188.733 | 39.419 | 13128.898 | 554.648 | 835.796 |

上述两个 run 六轮、后检查与清理都通过。所有8家族在C8/C128的请求E2E均PP2更低，每家族三轮方向一致。严格SLO1000ms/10ms下C128 EP4 goodput .3445、PP2 .1369，但放宽TPOT15则EP4 .4214、PP2 .5946；完整网格保留，不能挑目标宣称动态收益。逐请求8208对中 EP-only1013、PP-only150、both193；事后标签不是线上路由特征。

单卡 C8 的3轮完整、全run待完成：TTFT348.458、TPOT4.990、E2E2558.706、root244.118、wall397.420。E5失败PD的C8三轮TTFT572.037、TPOT6.923、E2E3590.808、root282.578、wall451.801，整组未验收。

## 重要边界和剩余工作

1. E6 P/D GPU正确性、全rank预热、C128压力路径与6轮完成/后检查/清理必须验证。E5主分析严格 commit，不能直接将E6混入；先用 analyze_cluster/exposure/tradeoffs 的 --run 独立报告。E6与E5比较需显式版本边界及适当同源控制。
2. 4P4D不能代表最优P:D；新E6的2P6D与colEP2可检验高负载比例，旧E5 V14 PD候选暂不启动。
3. 若稳定互补仍存在，才测实际混合池在线路由；给同质池相同缓存/队列策略。当前cache_queue未校准，见 POLICY_AUDIT.md。固定池路由不等于动态增减实例/权重重分片，不用无成本oracle证明收益。
4. V3为自定义有限闭环压缩回放：原数据有约40h间隔，时间单位正确；C8压缩约144600s。branch return偏移还与官方不同。所有wall/root/goodput都按压缩协议，不是生产时间/官方分数。
5. V4时间协议已独立准备：agentx_branch_timing.py、agentx_replay_v4.py、test_branch_timing.py，branch_timing_tests_v2.log 9通过；执行固定官方3方法AST+2helpers对24图、真实线程issue-before-complete/join。无GPU测量，不与V3合并；目前先验证压力修复。
6. 数据为8源家族171调用，C12816副本2736调用，不增加多样性；合成token还原前缀而非语义/真实专家路由。详见 PROTOCOL_AUDIT、WORKLOAD_COVERAGE、TIMING_AUDIT。

主报告 `../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md` 已索引。最终交付前补齐结论、版本与所有验收边界；目标尚不能 complete。更早运行证据保留于 CONTINUE_HISTORY_PRE_E6_QUEUE_20261031.md。

单卡 C128 r1完整2736请求：TTFT26731.6498ms、TPOT18.75016、E2E32721.4663ms、wall1422.8856s、cached46646.8772 token/request、1269次host命中、8次回退；仅1/3不是最终均值。1000/10 HTTP goodput .2270，需使用含dispatch lag的主SLO表再比较。

新增只读 OUTPUT_AGREEMENT 核查：output_agreement_e5/OUTPUT_AGREEMENT.md。定时动态batch输出并非逐请求一致；不等同受控串行/host恢复正确性验证。无缓存请求也有首token差异，尚未用logits归因数值；不要直接宣称缓存错误或全部逐token正确。每轮源长度仍相同，后续输入按hash重建而非拼本轮输出。analyze_output_agreement.py可加入后续E6数据；现为已完成E5配对样本的描述，不是通过门槛。

2026-10-31约06:03新增 `analyze_e6_controls.py`，独立汇总三个E6运行的接受状态、各arm/负载三轮均值、root/SLO、HiCache就绪核对和压力延期日志；不把E5混入。首轮尚无E6数据。28个已有EP4/PP2 C8/C128指标与原矩阵重算一致，记录 e6_metric_reconciliation.txt。未来启动的controls watcher会自动更新 e6_analysis/REPORT.md；当前已加载的E6主watcher旧代码只更新per-run分析，期间可手动运行新聚合器。

单卡C128 r2已完整：TTFT27406.7633ms、TPOT18.57010、E2E33418.4808ms、wall1473.7414s、cached44801.1696 token/request、1260次host命中、9次回退。两轮暂均值TTFT27069.2065、TPOT18.66013、E2E33069.9735、wall1448.3135；仍不是三轮最终值。8个单卡实际GPU KV总容量2,712,832 token（011每卡338816；171每卡339392），来自主矩阵capacity.csv，不是模拟值。

2026-10-31 06:17：单卡C128 r3仍运行，5/6；exec75146与queue5006已再次确认live。当前goal turn主要为verified wait，同时补必要默认接收队列控制，未新增GPU测量。

**新增且最后一项容量控制**：`eight_gpu_ep4_native_queue32_checkpoint_e6_r1`，prepare exec26757已exit0，260冻结哈希通过。E6、col_ep4/pd_ep4、C128各3轮，GDN160/每DP接收32/活跃16，(16+32)×3=144≤160，双方HiCache及原生KV自动容量不变。不是纯队列归因实验，是避免低估P/D并检查更强固定EP4的合理容量配置。说明GDN_QUEUE_CONTROL.md。
- 新条件队列exec **6332**，PID **192997** / ticks **382906105**，`wait_for_e6_native_queue_control_r1.json`；`run_after_verified_queue_v2.py`支持尚未创建的上游队列文件和失败检测。等 `handoff_e6_controls_r1.json`完整通过以及其watcher finished后，创建 `handoff_e6_native_queue_control_r1.json`，最后启动此6轮run。任何上游失败就停止。
- 新CPU watcher等待器exec **86406**，PID **202623**，`e6_native_queue_watcher_wait_r1.json`；之后变为 `e6_native_queue_analysis_watcher_r1.json`。
- E6聚合器已包括4个run，按run分别分组、显示GDN预算和receive8/32；绝不合并两种容量。0个GPU样本/0验收，目前仅准备/排队。完整计划E6_RUN_PLAN.json。
- 容量控制需要真实ready、完整结果和清理才能称通过；通用自动预算器仍未声称完全修复。无需继续扩大配置搜索，先跑完这组已说明原因的控制，再判断动态机制。
