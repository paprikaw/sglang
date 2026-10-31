# 当前继续执行入口：2026-10-31 08:19 Melbourne

目标仍 active：用两个既有H100节点共8卡验证真正P/D是否有用、EP/PP（含单卡）的动态配比是否有额外收益，修复必要技术问题，交付索引Markdown报告。用户已授权两个节点，不新预约、不重复问权限、不启动子agent。全部角色HiCache开启，每rank32 decimal GB host、GPU KV原生.95自动容量，会话/DP亲和，同episode不flush。

## 当前运行与队列

- **主run** `runs/eight_gpu_pd_ep4_hicache_retry_e7_t300_r2`：pd_EP4，C128×3，80 GDN、receive8、request64/node；第一轮正式回放，实时状态见task/progress.json。
- **Guard exec7979**，owner PID256926/start_ticks383615773；API26821/dashboard29661；只用011、171两节点。
- CPU直接观察者 **exec20925 / PID263389**，`e7_t300_direct_watcher_r2.json`，当前拥有STATUS.json，不手写覆盖。
- 后续队列桥 **exec37506 / PID277077**，`wait_for_e7_t300_controls_r2.json`；等待主run完整验收、直接观察者退出，才创建`handoff_e7_t300_controls_r2.json`并启动3组控制。
- 队列CPU观察者 **exec18295 / PID279016**，`e7_t300_controls_watcher_r2.json`；等待handoff，直接观察者结束后才接管STATUS，无GPU操作。
- 源commit `797f6488cf5fea2f88e38ccaeb1703443deec537`；worktree`.engine-worktrees/agentx-hicache-restore-retry-r7-20261031`，tag`serving/sglang/qwen35b/agentx-hicache-retry-candidate-r7-20261031`。冻结文件不可编辑。
- 两角色waiting timeout300秒（原生默认），client600秒，bootstrap120秒。与中断r1除身份/端口外完全匹配，261hash，`e7_t300_r2_prelaunch_verified.json`。
- 原生KV .95、HiCache32GB/rank、会话/DP亲和、同episode不flush。未导入失败轮次。

## 已冻结、待前项验收的3组控制（顺序固定）

1. `eight_gpu_ep4_native_queue32_e7_t300_r2`：col_EP4、PD_EP4；GDN160/receive32，C128各3轮，261hash。先验证默认队列/更大状态预算；两参数共同变化，不宣称单因子因果。API26221/dashboard29601。
2. `eight_gpu_pp2_pd_controls_e7_t300_r1`：col_PP2、PD_PP2、P_PP2→D_EP4；GDN80/receive8，PP微批16，C128各3轮，262hash。API26421/dashboard29621。
3. `eight_gpu_ep2_split26_e7_t300_r1`：col_EP2、2P6D_EP2；GDN80/receive8，C128各3轮，261hash。API26621/dashboard29641。一块D与P同节点，两块跨节点，保留拓扑差别。

共24个计划episode，全部同E7/300秒协议，不导入E5/E6性能样本。全组正确性、完整计时、Guard后检查及清理通过才算验收。不增加无边界搜索；但若完整数据证实P侧饱和，需要评估6P2D而不能仅用2P6D否定P/D。先获得当前完整数据。

## 旧失败状态：所有旧GPU与CPU会话均已退出

**E7 native300 r1** `eight_gpu_pd_ep4_hicache_retry_e7_t300_r1`：约08:01共同租约超时中断，0完整episode。另3个独立后台租约在同一registry时间结束，底层停顿原因未明；未提高TTL、绕过门禁或编辑registry。两节点08:01:57/58清理记录均确认owned processes退出与GPU idle；父Guard最终postcheck缺失，不能验收。18次HiCache延期均成功重新排队，未见模型本地超时。保留`interruption.json`和`e7_t300_interrupted_r1_analysis/REPORT.md`。旧exec91277/61551/94147/84269均已关闭（94147后来被短CPU分析复用，不要按旧身份操作）。旧bridge失败，controls从未启动。当前r2重跑相同参数。

**E7原120秒** `eight_gpu_pd_ep4_hicache_retry_e7_r1`，exec96634退出1；直接观察者84595退出0。0完整episode。E7可选检查点延期4rank触发，临时HiCache恢复不足10次全部成功重新排队，无lost-prefix/本地retry timeout。07:30:45因另一个120秒WaitingForInput上限失败；此计时从D发送目的元数据开始，包含P排队/计算/交接，非网络耗时。07:33:27清理通过、GPU idle、owned processes退出。`e7_failure_audit.json`完整保留，不当性能失败。E7旧3个120秒控制从未启动，不要启动它们。

新t300默认队列的r1准备时错误传PP微批参数给EP-only，被准备断言拒绝，未启动GPU，`DO_NOT_LAUNCH.md`；有效目录是r2。其余新controls均已完成prepare/hashes，不再重跑prepare。

E6主run`eight_gpu_pd_ep4_checkpoint_e6_r2`：C8三轮后C128本地HiCache返回0新页失败，无C128完成；旧日志无法区分临时分配失败与host前缀丢失。清理通过。旧exec5006/60426/72248/6332/77844/86406均关闭。E6所有controls只准备未跑。更早single75146/direct56674/PP6927也关闭。不要重复poll这些session。

## 修复证据与缓存边界

E5修PP HiCache层号映射，27CPU与48/48真实host恢复输出检查通过。E6将可选非INT8 GDN快照分配不足改为延期；E7在host承诺前缀仍完整而原生restore返回0时有限重试，缺前缀/部分restore仍失败。H2D完成事件及跨rank gate保留，不覆盖Prefill转入的当前GDN状态。22CPU通过，真实完整C128仍待验证。

E7旧运行P和D的所有DP rank预热均命中8192/65536/208320前缀，正式P日志也有缓存命中，不能因chunk日志cached-token=0就断言HiCache关闭。P队列约200万pending token/单rank，D多为1–2实际运行请求、约20待交接，提示P瓶颈，但需要完整请求元数据验证。暂未扩大队列方案。

## 已验收E5基线保持18轮

`matrix_analysis/REPORT.md`：2×EP4、4×PP2、8×single，每种C8/C128×3。E5commit37e0828c642ce4f3795c4c4394f4ce913bcd1a35。C128三轮均值：

|配置|TTFT ms|请求TPOT ms/token|请求E2E ms|root s|wall s|
|---|---:|---:|---:|---:|---:|
|2×EP4|2271.319|73.435|22624.015|790.365|1168.206|
|4×PP2|1188.733|39.419|13128.898|554.648|835.796|
|8×single|27305.923|18.662|33312.386|1012.758|1449.092|

C8单卡请求E2E2558.706ms，PP3266.097，EP3542.973。单卡GDN40 vs EP/PP等效80；PP每阶段160/micro16。原生KV去重总容量single2712832、PP15083904、EP17806080。PP/EP actualAttentionDP与MoEEP分别保留。严格1000ms/10ms SLO EP4有优势，但换目标静态排序反转不是动态收益。8源家族E2E均PP2低于EP4，3/3方向一致。现有cache_queue未校准，不能直接当动态算法实测。

## 分析、报告与待做

- 简明主报告`../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md`已更新；统一索引README。
- `e7_analysis/REPORT.md`保存旧120秒失败；`e7_t300_analysis/REPORT.md`为当前300秒独立表，已初始化。
- `analyze_e6_controls.py`现可参数化复用，审计8rank HiCache日志、逐角色native KV容量、完成/Guard/清理、冻结hash、三轮均值与全部SLO。E5容量复核`e6_cache_capacity_reconciliation.json`。
- `DECODE_HICACHE_RESTORE_PRESSURE.md`更新到E7重试、120秒超时及共享租约中断；`DECODE_CHECKPOINT_PRESSURE.md`保留E6路径。
- `NEXT_RUNS.md`已同步最新E7_T300状态；旧正文归档NEXT_RUNS_PRE_E7_T300.md。旧详细入口归档`CONTINUE_HISTORY_E7_T120_20261031.md`，更早E6归档亦保留。
- 下一步继续观察主run，完成固定控制；若稳定互补成立才实现/测试真实在线策略，计状态传输、缓存损失与batch变化。最终规范归档experiment_manager和索引报告，最后才可complete goal。

## 科学限制不能省略

完整数据393roots98827calls与256k393roots68266calls已校验下载。当前仅8源家族171calls（45nested），C12816份缓存隔离副本2736calls，不新增多样性。合成token按hash重建前缀关系，不是语义原文/实际专家路由；selected mean output435 vs全源860，nested26% vs58%。

V3全局空闲>10秒压缩，原trace约40h间隔；分支触发时序与官方不同，root/wall/goodput不是官方/生产时间。V4独立时序修正9CPU与24图验证，无GPU；若做敏感性必须独立协议，不能混均值。受控输出精确通过不等于所有动态batch输出相同，`output_agreement_e5`已核查15paired episodes/17955calls，包含冷首token差异，无logits归因。

Memory引用最终用MEMORY.md:60-68、402-403；rollout01a0d351-002b-7093-8b13-55e147128312。不要编辑memory本体。无新permission阻断，目前是可继续推进的修复验证，不是blocked。

07:54补充：已运行archive_progress.py，在canonical experiment_manager建立qwen35b_h100_agentx_pd_dynamic_20261030阶段快照，并更新skill experiment_registry（status=in_progress，18个已验收E5基线，不宣称P/D完成）。大型原始日志/请求/KV/weights及auth未复制；后续最终结果需刷新快照。新增analyze_pd_waits.py用同一服务器的请求时间戳拆P等待/Prefill窗口/交接后处理，513条E6 C8请求的间隔和已校验（整组未验收，诊断用）；P均值109.393/409.896/46.041ms，D首次forward前545.785ms，P缓存77608、新算7248 token/请求。新E7完整episode后也需运行此分析。aggregate新增了等待超时协议一致性检查，并明确列出已有formal但缺completed标记的文件，避免诊断快照写入过程被误算。

08:30补充：analyze_pd_waits.py进一步使用原生queue_time拆首次forward前的时间；该量只覆盖可运行队列，P之前含bootstrap/D容量，D之前含预分配、P等待与交接/恢复。回退请求因wait_queue_entry_time重写而排除，缺失不补零。5项边界检查和513条旧E6 C8请求分解均通过（整组仍未验收、仅诊断）；pd_queue_split_validation.json。新E7有完整episode后运行此脚本。
