# 当前继续执行入口：2026-10-31 09:50 Melbourne

目标active：修复技术问题，两个既有H100节点8卡上验证AgentX派生P/D与EP/PP动态配比，完成索引Markdown报告。不得因失败或基线完成就标complete；不启动子agent，不新预约，不重复问权限。HiCache每角色每rank开启，CPU32 decimal GB/rank，GPU KV原生.95，会话/DP亲和，同episode不flush。

## 活跃进程

- 主run `runs/eight_gpu_pd_ep4_e7_t300_b7200_r1`，pd_EP4/C128×3；GDN80、receive8、request64/node、4P4D。
- Guard **exec36830 / PID248699 / start_ticks383925844**；API27021/dashboard29681，011+171。
- 直接观察者 **exec42475 / PID254244**，`e7_t300_b7200_direct_watcher_r1.json`，当前拥有STATUS.json，不手写覆盖。
- 旧V3桥exec56615/PID272892和observer7666/PID279976已取消退出0，三组控制均未launch，handoff不存在。不要poll旧session。
- **新V4桥 exec90774 / PID186639 / ticks384192100**，`wait_for_e7_v4_controls_r1.json`；等待主run全部三轮验收后创建`handoff_e7_v4_controls_r1.json`。
- **新V4观察者 exec60668 / PID192495**，`e7_v4_controls_watcher_r1.json`；主observer结束后才接管STATUS，调用`analyze_e7_v4_controls.py`。
- 当前整体计划`E7_V4_RUN_PLAN.json`；`E7_T300_B7200_RUN_PLAN.json`保留V3主run3轮，未启动controls已标superseded。
- 主source commit `797f6488cf5fea2f88e38ccaeb1703443deec537`，worktree`.engine-worktrees/agentx-hicache-restore-retry-r7-20261031`，tag`serving/sglang/qwen35b/agentx-hicache-retry-candidate-r7-20261031`。主run264frozenhash；V4四组263/264/263/263hash、source clean。冻结源码/prepare_v19/v20/episode_capture/helper不得编辑。

## 已冻结的新V4控制队列

主run全部三轮、资格与Guard后检查/清理通过才启动；失败停止。当前主run是V3压力验证，与新V4性能组分开。四组V4共24episodes，不导入旧性能。

1. `eight_gpu_ep4_queue32_e7_v4_b7200_r1`：col_EP4+PD_EP4，GDN160/receive32，6轮。API27821/dashboard29761。
2. `eight_gpu_pp2_pd_e7_v4_b7200_r1`：col_PP2、PD_PP2、P_PP2→D_EP4，GDN80/receive8、PPmicro16，9轮。API28021/dashboard29781。
3. `eight_gpu_ep2_split62_e7_v4_b7200_r1`：col_EP2+EP2 6P2D，GDN80/receive8，6轮。API28221/dashboard29801。一P实例与D同171，另两个P实例在011跨节点。
4. `eight_gpu_col_single_e7_v4_b7200_r1`：8×single，GDN40，3轮。API28421/dashboard29821。

相同V4replayer/helper/runner/capture/graph哈希、300秒交接+7200秒整轮上限；`e7_v4_cohort_preflight.json`通过。前三组node_configs/endpoints/容量/预热与对应V3准备规范化后完全相同，仅运行ID与端口不同；runner完全相同。V3未启动控制保留，不再调度。

`validate_v4_bundle.py`16checks通过（9分支+4异常捕获+3前缀/route/cycle），实际冻结模块。两次完整枚举import被Python依赖/插件初始化问题阻止，日志保留；最后直接执行原始native枚举/基类AST及三个分支方法/两个helper，没有新增依赖。`v4_bundle_tests_native_ast.log`及首V4run的`v4_cpu_validation.json`。校验24候选图；仍非完整官方harness。V4根重叠SPAWN在credit issue，其他SPAWN按branch-relative在return，FORK等待return，join保留；原V3错误引入40h source gap并压缩，影响并发，因此最终关键对照统一V4。仍保留finite-root admission/hash-token/global idle cap，非官方评分。

## 刚修的整轮上限

旧 `eight_gpu_pd_ep4_hicache_retry_e7_t300_r2` **08:55:12**被回放器2400秒整轮上限中断，0完整episode。P和D均2692个HTTP200（D是stream开始，不是完成请求数）；HiCache19次延期19次成功，无本地restore/transfer timeout证据。cleanup通过、GPUidle、ownedprocesses退出，Guard postcheck因实验未完成failed，lease released。`episode_deadline_failure_audit.json`。

旧Guard7979退出1；直接20925退出0；桥37506退出1；queueobserver18295退出0；旧controls均未启动。不要poll旧会话。更旧native300r1共享租约中断及120秒交接上限失败保留在归档，两者也0完整C128。

`prepare_cluster_v19.py`包装v18：整轮7200秒，不改变引擎/源图/到达/长度/缓存/合批；HTTP socket等待仍600，交接300。`episode_capture.py`成功不写额外文件；失败保存已收集行到partial.r*.json，永不写完成标记/算均值，强制kill仍可能无法保存。8项CPU检查通过，包括旧DAG/前缀/路由、正常路径无写盘、请求失败、有限deadline、诊断写失败保留原异常。`episode_capture_tests.log`、`episode_bound_replay_delta.diff`、`EPISODE_TIMEOUT_AUDIT.md`。

## 分析与下一步

- `analyze_e7_b7200_controls.py`现在仅汇总V3压力主run；旧未启动控制移出本组但完整保留准备文件。`analyze_e7_v4_controls.py`→`e7_v4_b7200_analysis/REPORT.md`独立4组。聚合器校验分支policy/helperhash/实际请求issue字段、整轮上限、完成marker及partial排除。09:52时V3压力组1/3完整episode、V4为0/24；取最新状态。
- `analyze_pd_waits.py`读取完整episode做P收到→forward→P结束→完成；D收到→forward。原生queue_time只覆盖可运行队列，用于进一步拆分；retraction会重写时间戳，因此该拆分排除回退请求，缺失不补零。5边界检查与513个E6 C8完整请求核对通过（整组未验收，仅诊断）。`pd_queue_split_validation.json`。
- `pd_owner_load.csv`按实际instance/AttentionDP统计请求、input、P新算token、排队；不是Expert rank/GPU利用率。已校验每角色请求总数513。新增wrapper自动生成P/D等待分析；直接观察者不调用wrapper，需要第一轮完整后手工调用新wrapper。
- 新完整episode前不报性能均值；先验证主run和固定对照，若同一目标稳定互补才实现真实在线策略。同质池使用相同缓存/队列策略，计KV/GDN交接与cache损失，不把事后oracle拼接当收益。
- 当前无blocking需要用户；先持续观察。完整确认后更新主Markdown、canonical归档，最后才complete goal。

## 既有已验收E5基线18轮

两H100节点8卡Qwen3.6-35B-A3B BF16；C8/C128各3轮、2×EP4/4×PP2/8×single。E5source37e0828c642ce4f3795c4c4394f4ce913bcd1a35。C128算术平均：

|配置|TTFT ms|请求TPOT ms/token|请求E2E ms|root s|wall s|
|---|---:|---:|---:|---:|---:|
|2×EP4|2271.319|73.435|22624.015|790.365|1168.206|
|4×PP2|1188.733|39.419|13128.898|554.648|835.796|
|8×single|27305.923|18.662|33312.386|1012.759|1449.092|

C8single2558.706ms、PP3266.097、EP3542.973 E2E。singleGDN40，EP/PP等效80，PP每阶段160/micro16。去重GPUKV single2712832、PP15083904、EP17806080。严格1000ms/10msSLO EP有优势，改变SLO后静态排序反转不是动态收益。8源家族E2E PP均低于EP，3/3同方向。

## 报告、归档、限制

主报告`../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md`，统一索引README。canonical `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_agentx_pd_dynamic_20261030`，archive_progress.py与skill registry；仍in_progress、18已验收基线。

8源家族171calls(45nested)，C12816份缓存隔离副本2736calls。完整数据393roots98827calls；256k68266calls。hash-token前缀非语义文本/真实Expert路由。V3全局空闲压缩>10秒，分支时序与官方不同；root/wall/goodput不是官方AgentX/生产指标。V4有16CPU/24图核对，四组队列已准备，尚无GPU结果。计时输出不保证逐token相同，受控检查通过不代表全部batch相同；没有logits归因。

E5PP hostrestore修复27CPU/48of48GPU；E7 HiCache压力修复22CPU；本次C128首轮2736calls完整，整组仍需另外两轮及postcheck/cleanup验收。过往失败/过程详见`CONTINUE_HISTORY_E7_B2400_20261031.md`及更早归档。不重复读所有历史。

最终唯一memory block用MEMORY.md:60-68、402-403，rollout01a0d351-002b-7093-8b13-55e147128312；不要编辑memory本体。

09:23补充：共享completed_episode_files.py统一analyze_cluster/exposure/tradeoffs的完成标记门槛；未标记的损坏JSON和partial不会读入均值，failed marker拒绝，空报告清除旧派生CSV。completion_gate_validation.json有9次脚本集成执行通过；completion_gate_baseline_refresh.json复算18个E5完整episode，27CSV哈希全部未变。所有历史dashboard_spec已确认这些分析文件不是冻结服务输入。

新增analyze_forward_batches.py分析已有ModelRunner计数：PP只计stage0，EP检查AttentionDP后逐rank；纯DECODE本地forward按调用数加权，三轮算术平均。C128 EP4=5.622，PP2=5.036，single=2.212；C8约1。这不是时间加权并发、全EP组tokens或Expert行数。matrix_analysis/FORWARD_BATCHES.md及原始hist/rank/trial CSV保存；18episode source audits。Python dispatch_output漏掉Graph重放，因此LL计数0不能推断没用LL/后端比例；ModelRunner及DecodeCudaGraphRunner source已核对。新E7_b7200 wrapper会自动生成forward batch分析。

09:50补充：forward batch来源按澄清后的analyzer重新生成，18episode三份mean CSV哈希全部不变；matrix_analysis/forward_batch_sources.json已刷新analyzer与audit哈希。completion_gate相关核对保留。当前手写文档是快照，STATUS由observer负责。

09:52首个完整C128压力episode：V3 E7/300/7200 `formal.r1.json`、`completed.r1.json`、8rank forwards存在且全部聚合资格检查通过；当前r2。首轮TTFT57812.94ms，请求TPOT11.816ms，E2E62792.78ms，root1628.505s，wall2485.712s（比旧2400上限长85.712s）。P收到→forward54988.997ms，before-ready14801.021/readyqueue40187.976；Pforward→prefill结束2678.520ms，P结束→完成203.824ms；D收到→forward57758.523ms，不能与P相加。Pcached38223.626/new46632.515 tokens。D本地纯Decode平均batch2.327，4owners；P无纯Decode forward。整组accepted=false；这是第一轮诊断，不拿它当三轮均值或V4结果。

新V4observer PID192495 start_ticks384194405；bridge PID186639 start_ticks384192100。exec90774/60668均live waiting，handoff尚无；旧桥56615/7666退出0。主Guard36830与直接observer42475继续，不重启。16CPU全部通过、4组263/264/263/263hash一致性校验通过；固定runnerhash与旧V3相同。分析器已改按policy显示V3/V4标签，完整点取三轮；partial与未完成marker排除。主Markdown及统一索引已更新首轮诊断和V4计划，canonical archive_progress.py现同时保存V3压力与V4两个独立cohort。

10:05本轮新增有效离线证据：analyze_cache_owner_pressure.py完成19个完整episode核对（18已验收E5+1个未整组验收E7）。逐conversation单owner、请求数、root数、cached+new=input等通过。E7 R1 P四rank各32根/468根调用/50038272根输入，root新算10044544/28385152/41188992/31136576，子调用119/216/311/218；readyqueue均值8715/41065/57002/47150ms。根初始输入均衡，后续缓存命中和实际新算仍明显不同；仅相关诊断，不能归因EP计算或宣称路由收益。E5 C128各布局root数与root输入均匀；所有逐位置绝对值及CV的三轮均值在各run analysis/CACHE_OWNER_PRESSURE.md和CSV。所有实际测量/原始均值未变。

CACHE_ROUTING_DIAGNOSIS.md已链接主报告；cache_owner_first_episode_audit.json验证P device+host=cached、无回退，并保存active日志被检查前缀的长度/hash。P无MAMBA_CACHE_CHECKPOINT_DEFERRED记录，D有4个首次记录，不能由D记录推P原因或频率。新E7 V3/V4 wrapper自动运行owner诊断（仅适用于当前sticky_hash固定控制，未来动态策略不可原封不动使用single-owner断言）。保持V4队列不变；在线阶段必须给同质池同样缓存/队列策略，区分同质路由与混合布局的增量。当前主run仍r2，后续控制等待三轮和清理，勿重启活跃Guard。

10:32更新：权限现为danger-full-access/never，所有exec_command禁止sandbox_permissions；functions store agentx_e7_b7200_monitor已去除此字段。用户中断了等待工具，但主Guard未停，已确认原4个PID/ticks全部存活。主run现r3，formal/completed/forwards r2齐备；2/3完整，5472calls，整组仍accepted=false。analyze_e7_b7200_controls.py已刷新两轮metrics/forward/owner/waits，所有门槛通过。

R2：TTFT65827.280ms，请求TPOT10.506048ms，E2E70296.757ms，root1788.906556s，wall2451.283925s，无回退。Preadyqueue46492.560751ms，before-ready16235.633767ms，forward→end3007.021819ms；cached34038.362573/new50817.777778 tokens；D纯Decode平均batch2.096694（568575 forward）。两轮TTFT均值61820.108150ms，TPOT11.161133490ms，E2E66544.770873ms，Preadyqueue43340.268395ms。R2四rank仍各32roots/468rootcalls/50038272根输入；root新算25240832/28450752/39082560/29473472，最高/最低约1.55，而R1约4.10，不能声称该幅度稳定。长P等待重复出现；R2 D更快但E2E更慢，仍非PD或动态效益的充分对照。

主报告改为两轮独立绝对值表；CACHE_ROUTING_DIAGNOSIS.md新增R2复核。V3压力与V4控制依旧分组；计划快照2/3、V4仍0/24。后续仍由bridge90774/PID186639等待三轮及post/cleanup，再经Guard按已冻结V4四组执行。未重启或新增GPU配置。

## 2026-10-31 compact report and qualification completion

- New user request: compact indexed Markdown for synthetic P/D, Expert/communication, AgentX E2E. Written to ../experiment_summary/EP_PP_EXPERIMENTS_COMPACT.md; canonical snapshot moe_ep_pp_experiments_compact_20261031. Rechecked143 means/451 rounds and21 accepted AgentX episodes.
- Direct E7 V3 pressure run eight_gpu_pd_ep4_e7_t300_b7200_r1 is complete3/3,8208 calls,Guard postcheck+cleanup passed. Sessions36830 and42475 exited0. Rebuilt analyze_e7_b7200_controls.py outputs: mean TTFT60765.864ms,requestTPOT11.169622ms,E2E65526.386ms,wall2409.734s; P readyqueue42.661s. No matching-source colocated control; do not claim PD effect.
- Old V4 interleaved control waiter90774/PID186639 was cancelled BEFORE successor admission per user steering,observer60668/PID192495 finished_without_queue; handoff_e7_v4_controls_r1.json absent,0 V4 GPU episodes. Original prepared bundles are immutable and retained. New ordering: colocated layouts first,then fixed P:D ratios with P single/PP2,D single/EP4 and cache-capacity exceptions; dynamic only after same-objective complementarity. New ratio matrix has not been prepared or launched. Do not resume old cancelled queue.
