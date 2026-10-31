# 当前继续执行入口：2026-10-31 09:10 Melbourne

目标active：修复技术问题，两个既有H100节点8卡上验证AgentX派生P/D与EP/PP动态配比，完成索引Markdown报告。不得因失败或基线完成就标complete；不启动子agent，不新预约，不重复问权限。HiCache每角色每rank开启，CPU32 decimal GB/rank，GPU KV原生.95，会话/DP亲和，同episode不flush。

## 活跃进程

- 主run `runs/eight_gpu_pd_ep4_e7_t300_b7200_r1`，pd_EP4/C128×3；GDN80、receive8、request64/node、4P4D。
- Guard **exec36830 / PID248699 / start_ticks383925844**；API27021/dashboard29681，011+171。
- 直接观察者 **exec42475 / PID254244**，`e7_t300_b7200_direct_watcher_r1.json`，当前拥有STATUS.json，不手写覆盖。
- 后续桥 **exec56615 / PID272892**，`wait_for_e7_b7200_controls_r1.json`，成功后创建`handoff_e7_b7200_controls_r1.json`。
- 队列观察者 **exec7666 / PID279976**，`e7_b7200_controls_watcher_r1.json`，主观察者退出后才接管STATUS，调用`analyze_e7_b7200_controls.py`。
- `E7_T300_B7200_RUN_PLAN.json`当前计划；`E7_T300_RUN_PLAN.json`已标superseded。
- 主source commit `797f6488cf5fea2f88e38ccaeb1703443deec537`，worktree`.engine-worktrees/agentx-hicache-restore-retry-r7-20261031`，tag`serving/sglang/qwen35b/agentx-hicache-retry-candidate-r7-20261031`。264frozenhash主run通过，source clean。冻结源码/prepare_v19/episode_capture不得编辑。

## 已冻结的新控制队列

主run全部三轮、资格与Guard后检查/清理通过才启动；失败停止。共24episodes，不导入旧性能。

1. `eight_gpu_ep4_queue32_e7_t300_b7200_r1`：col_EP4+PD_EP4，GDN160/receive32，6轮，264hash。API27221/dashboard29701。
2. `eight_gpu_pp2_pd_e7_t300_b7200_r1`：col_PP2、PD_PP2、P_PP2→D_EP4，GDN80/receive8、PPmicro16，9轮，265hash。API27421/dashboard29721。
3. `eight_gpu_ep2_split62_e7_t300_b7200_r1`：col_EP2+EP2 6P2D，GDN80/receive8，6轮，264hash。API27621/dashboard29741。一P实例与D同171节点，另两个P实例在011跨节点。之前反复P长队列，优先检验增加P；只是实验选择，不是已证实瓶颈。旧2P6D只准备未启动。

相同replayer/helper/graph哈希、300秒交接+7200秒整轮上限；`e7_b7200_cohort_preflight.json`。单卡作为D尚未实测，负结论必须限定已测布局。

## 刚修的整轮上限

旧 `eight_gpu_pd_ep4_hicache_retry_e7_t300_r2` **08:55:12**被回放器2400秒整轮上限中断，0完整episode。P和D均2692个HTTP200（D是stream开始，不是完成请求数）；HiCache19次延期19次成功，无本地restore/transfer timeout证据。cleanup通过、GPUidle、ownedprocesses退出，Guard postcheck因实验未完成failed，lease released。`episode_deadline_failure_audit.json`。

旧Guard7979退出1；直接20925退出0；桥37506退出1；queueobserver18295退出0；旧controls均未启动。不要poll旧会话。更旧native300r1共享租约中断及120秒交接上限失败保留在归档，两者也0完整C128。

`prepare_cluster_v19.py`包装v18：整轮7200秒，不改变引擎/源图/到达/长度/缓存/合批；HTTP socket等待仍600，交接300。`episode_capture.py`成功不写额外文件；失败保存已收集行到partial.r*.json，永不写完成标记/算均值，强制kill仍可能无法保存。8项CPU检查通过，包括旧DAG/前缀/路由、正常路径无写盘、请求失败、有限deadline、诊断写失败保留原异常。`episode_capture_tests.log`、`episode_bound_replay_delta.diff`、`EPISODE_TIMEOUT_AUDIT.md`。

## 分析与下一步

- 新cohort `analyze_e7_b7200_controls.py` → `e7_t300_b7200_analysis/REPORT.md`。旧t300分析保留失败状态。聚合器新增整轮上限一致性、partial路径排除。
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

8源家族171calls(45nested)，C12816份缓存隔离副本2736calls。完整数据393roots98827calls；256k68266calls。hash-token前缀非语义文本/真实Expert路由。V3全局空闲压缩>10秒，分支时序与官方不同；root/wall/goodput不是官方AgentX/生产指标。V4有9CPU/24图核对，无GPU。计时输出不保证逐token相同，受控检查通过不代表全部batch相同；没有logits归因。

E5PP hostrestore修复27CPU/48of48GPU；E7 HiCache压力修复22CPU与19成功真实retry，但完整C128仍待验收。过往失败/过程详见`CONTINUE_HISTORY_E7_B2400_20261031.md`及更早归档。不重复读所有历史。

最终唯一memory block用MEMORY.md:60-68、402-403，rollout01a0d351-002b-7093-8b13-55e147128312；不要编辑memory本体。

09:23补充：共享completed_episode_files.py统一analyze_cluster/exposure/tradeoffs的完成标记门槛；未标记的损坏JSON和partial不会读入均值，failed marker拒绝，空报告清除旧派生CSV。completion_gate_validation.json有9次脚本集成执行通过；completion_gate_baseline_refresh.json复算18个E5完整episode，27CSV哈希全部未变。所有历史dashboard_spec已确认这些分析文件不是冻结服务输入。

新增analyze_forward_batches.py分析已有ModelRunner计数：PP只计stage0，EP检查AttentionDP后逐rank；纯DECODE本地forward按调用数加权，三轮算术平均。C128 EP4=5.622，PP2=5.036，single=2.212；C8约1。这不是时间加权并发、全EP组tokens或Expert行数。matrix_analysis/FORWARD_BATCHES.md及原始hist/rank/trial CSV保存；18episode source audits。Python dispatch_output漏掉Graph重放，因此LL计数0不能推断没用LL/后端比例；ModelRunner及DecodeCudaGraphRunner source已核对。新E7_b7200 wrapper会自动生成forward batch分析。
