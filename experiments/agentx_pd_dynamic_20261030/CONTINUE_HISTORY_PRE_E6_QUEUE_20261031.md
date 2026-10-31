# 当前继续执行入口（2026-10-31）

目标尚未完成。用户要求修复兼容性问题，并用两节点8张H100的AgentX实验检验P/D分离与EP/EP=1动态分配，最终形成索引Markdown报告。所有角色HiCache开启，会话内保留GPU KV和DP亲和。不要新预约或重复询问GPU许可。

## 最新进度（2026-10-31约05:00，覆盖下方较早记录）

P/D `eight_gpu_pd_ep4_durable_slots8_r4`在完成C8三轮后、C128 r1开始时失败。Decode DP3在`process_prebuilt → maybe_cache_unfinished_req → MambaComponent.prepare_for_caching_req → _alloc_mamba_slot`触发`AssertionError: Can not alloc mamba cache`。之前日志为GDN79/80、active2、transfer22、prealloc7；不是启动OOM。C8三轮完整均值TTFT572.0373ms、请求TPOT6.922716、interval6.944185、E2E3590.8083ms、root282.5784s、wall451.8011s，host命中/回退0。C128无完整正式样本，整组Guard后检查failed；原目录保留。清理passed、GPU idle true、registry released。

队列exec21316退出1，watcher57476退出0，都是terminal；不要继续poll旧handle或重启旧PD目录。单卡独立基线已在确认清理、250项冻结哈希和canonical backfill后启动：`eight_gpu_col_single_durable_kv40_r3`，**exec75146**，Guard PID37948/start_ticks382417862，dashboard29261，8GPU两节点，E5原样。新CPU watcher为`watch_direct_run_v1.py` / `matrix_analysis_watcher_direct_r1.json`，exec号待补。单卡和E5冻结源码均不得修改。

正定位GDN压力修复：新隔离worktree `.engine-worktrees/agentx-gdn-pd-hicache-r6-20261031`由E5创建，创建exec96159待poll。思路是未完成请求的可选缓存快照在无可驱逐槽位时安全延期，保留活跃state与已有KV/锁，完成时仍能缓存；当前尚未实施/验证，不得声称修复完成。读取native unified `effective_cache_len<=0`路径支持不插入并保留prefix_indices，但cleanup是否应保留`mamba_last_track_seqlen`需验证。不要用降低C128负载掩盖问题。

独立时间协议准备取得CPU进展：新增`agentx_branch_timing.py`和standalone `agentx_replay_v4.py`，保留V3所有运行不动。V4按官方branch start对齐return触发，root源重叠SPAWN按parent credit-issued事件触发；保持turn delay/precedence/join。显式拒绝pre-session分支（有限根会话准入不等于官方once-per-phase规则）。`test_branch_timing.py`执行固定官方源的3个方法AST和2个timestamp helper，24个catalog graph对齐；9项CPU检查全部通过，包括实际线程回放的先issue子分支及join。`branch_timing_tests_v1.log`是缺orjson的import失败，保留；v2为9passed。没有安装依赖，没有声称完整官方harness等价，没有GPU新协议数据或V15准备包。暂搁置该准备，先修当前PD崩溃。

## 较早进度（约04:40）

本轮有实际进展，目标仍active，不属于blocked。P/D C8已完成2/3，当前r3；整个PD run为2/6，随后C128三轮。两轮暂均值TTFT577.0322ms、请求TPOT6.91900、间隔加权6.94179、E2E3594.7618ms、root283.0066s、wall452.4193s；host命中/回退均0。整组未完成，不纳入accepted布局比较。队列exec21316和watcher57476均仍活跃，watcher最新14个完成episode；接下来single的队列与Guard后检查门槛不变。GPU等待期间完成了家族与严格SLO配对分析，见下方，最终报告尚未完成。

PP2 `eight_gpu_col_pp2_durable_m16_r3`已完成6/6，exec6927退出0，Guard后检查、清理和GPU idle均通过。C128三轮均值TTFT1188.7327ms、请求TPOT39.41901、间隔加权27.46638、E2E13128.8977ms、root554.6485s、wall835.7956s；无回退。`COMPARISONS.md`已纳入与EP4的正式差值。

同一队列exec21316已启动`eight_gpu_pd_ep4_durable_slots8_r4`，Guard PID134829/start_ticks382203294。两节点ready、实际GDN容量检查、输出一致性和全rank预热均通过，约04:21进入C8 r1。P/D双方所有rank的HiCache=true/32GB/.95；D接收预留8、GDN80，原生GPU KV每rank2,195,008；P每rank2,225,472。启动OOM绕过已在GPU验证，但性能和整组后检查尚未完成。后继single不变，watcher57476继续。以下“正在运行”段为较早记录，应以上述最新状态为准。

## 较早运行记录

新增`analyze_family_tradeoffs.py`由`compare_static_matrix.py`自动调用，仅分析完整验收三轮的配置。12轮家族分解的请求数及加权TTFT/E2E/cached已与主分析逐项对齐。结果在`matrix_analysis/FAMILY_TRADEOFFS.md`：C8和C128所有8家族请求E2E均PP2更低、每家族3/3方向一致；但完整预设SLO网格出现例外，C128 TPOT≤10且TTFT≤500/1000/2000ms时EP4达标rps更高。例如1000/10下EP4 .3445 vs PP2 .1369；1000/15下EP4 .4214 vs PP2 .5946。保留全部目标，不据此宣称动态收益；两侧strict目标通过率低，单卡/PD仍待。家族静态最优不能拼成动态oracle，数据属于同混合回放相互干扰的事后分解。上述分析脚本和文档均不在当前PD或single的冻结输入中。

同样新增`analyze_strict_slo_overlap.py`并挂在compare脚本后，输出`STRICT_SLO_OVERLAP.md/json`。C128 8208次配对请求（3轮合计）在1000ms/10ms目标下：EP-only1013、PP-only150、both193、neither6852。EP-only平均输入160248.9、输出253.6，EP/PP未缓存1172.1/1156.2，TPOT8.227/12.569ms；after-first期间HTTP在途均值17.37/9.42，并非GPU batch。严格SLO总体通过比例14.69%/4.18%。这是一项事后分析，不能将结果标签、实际输出长度直接变成线上路由；动态收益尚未验证。脚本额外校验配对SLO比例与主矩阵一致。

P/D C8 r1已完成171请求：TTFT593.6319ms、请求TPOT6.92926、间隔加权6.95010、E2E3614.9766ms、wall454.6809s，host命中和回退均0；约04:29进入r2。仅首轮进度，仍待C8/C128各三轮及整体后检查。watcher已见13个完成episode。

- `eight_gpu_col_ep4_durable_r3`已完成6/6，Guard exec33575退出0，后检查及清理/GPU idle通过。四个导入样本和两轮补测均已验收；C128 r3为2736请求、1147.274s。
- 当前`eight_gpu_col_pp2_durable_m16_r3`：Guard exec6927，PID68844/start_ticks381794957；HiCache、原生正确性和全rank预热通过，C8三轮及C128 r1/r2已完成（5/6），C128 r3运行（约2026-10-31 04:03开始）。两个节点各2个PP2实例，ready中全部HiCache=true/32 GB/mem_fraction=.95。
- 新后继队列：exec21316，`handoff_pp_then_pd_single_e5.json`，PID99032。等待上述PP2精确owner，随后依次`eight_gpu_pd_ep4_durable_slots8_r4`、`eight_gpu_col_single_durable_kv40_r3`。成功/清理/Guard/哈希门槛不变。旧exec36547/旧队列已因P/D启动OOM退出，不能继续使用。
- CPU分析watcher：exec57476，`watch_matrix_analysis_v3.py`，PID107104，状态`matrix_analysis_watcher_v3.json`；读取新队列与完成标记更新主分析、直接比较表和STATUS。旧watcher10173已退出。
- 主报告：`../experiment_summary/agentx_pd_dynamic_20261030/REPORT.md`。

E5源码仍是`37e0828c642ce4f3795c4c4394f4ce913bcd1a35`，PP恢复48/48输出通过。引擎未变，新V13只修正计时外通知与结果保存；2/4/8位置V12路由与V11相同。四份导入样本保留`measurement_origin`和哈希，不算新测量。

## 新发现与修正

`eight_gpu_pd_ep4_durable_r3`在Decode启动时OOM，0个正式样本；清理和GPU idle通过。默认每DP worker 32个接收预留与16个活跃请求，使原生GDN池由80扩大至144，而KV预算只按80扣除。新增V14配置使用8个接收预留，(16+8)×3=72≤80；E5、活跃请求上限、HiCache、.95静态预算和KV自动容量不变。这是配置绕过，不是通用预算器修复。4项执行原生构造函数的CPU检查通过；GPU验证将在PP2完成后进行。`DECODE_SLOT_BUDGET.md`记录源码、日志和边界。新的P/D准备包254项哈希全部通过。

V14的runner V9在ready后检查真实GDN槽位数量和接收槽位设置；不改计时热路径。新准备包和测试日志已经冻结，禁止覆盖重跑测试日志或改源码。

旧`eight_gpu_col_ep4_balanced_r2`在C128第二轮完成回放后，结束计数快照超时。011四rank停在epoch9，171已到10；旧流程未先保存HTTP结果，所以该轮无可审计性能数据。旧目录保持failed，清理和GPU idle通过。旧队列53887及watcher60374停止，不能继续使用旧队列。

V13使用`snapshot_transport.py`的node-local tmpfs控制字节，011协调器本地写、跨节点SSH确认写171，计时外读取快照；3项故障注入检查、两节点256次CPU通知和跨节点命令通过。先写`replay_completed.rN.json`再采集快照，成功后改名正式数据。详细见SNAPSHOT_RECOVERY.md。

新的运行不得修改冻结源码／准备脚本／原始样本。发现问题用新版本、新目录，不重写失败证据。

## 已有数据

新同协议EP4 C8三轮：TTFT431.7099ms，请求平均TPOT9.63956ms/token，间隔加权TPOT7.15695，E2E3542.9728ms，整轮443.4209s，根会话平均277.3409s，host命中0、回退0。

C128第一轮：2736请求，TTFT2183.8408ms，请求平均TPOT73.8790，间隔加权46.6726，E2E22473.3266ms，整轮1142.8971s，根会话平均783.9178s，182次host命中、0回退。单轮，不是最终均值。按rank/forward计数的Decode批次均值5.18–6.37，不能当同步全组batch或有效输出吞吐。

C128现在已有完整三轮均值：TTFT2271.3191ms，请求平均TPOT73.43463ms/token，间隔加权46.81802，E2E22624.0153ms，根会话790.3646s，整轮1168.2055s；平均179次host命中、0回退。三轮wall分别1142.8971、1214.4455、1147.2740s。主报告和matrix已更新，首轮数字仅保留作为历史分解。

PP2 C8 r1（仅进度，未满三轮/未完成全run后检查）：171请求，TTFT313.9200ms，TPOT7.14545ms/token，E2E3308.2044ms，wall448.1422s。暂不据单轮排跨布局胜负。

PP2 C8现有完整三轮均值（全run后检查仍待C128完成）：TTFT277.1517ms，TPOT7.13663ms/token，间隔加权6.87558，E2E3266.0972ms，root275.7307s，wall441.6533s，host命中/回退均0。请求E2E较EP4少276.88ms，root仅少1.61s、wall仅少1.77s，尚无强整体加速。只读`analyze_root_exposure.py`已由主分析自动调用；输出ROOT_EXPOSURE.md。C8根会话的无已就绪未完成请求时段EP4为201.712s、PP2为205.862s，含工具/依赖等待与回放轮询，不能当纯工具时间或oracle上限。

**新时间口径审计见TIMING_AUDIT.md。** 原数据第340行d876...的flat请求t=145004.210秒，loader乘1000、当前回放除1000正确。C8每轮跳过约144600秒（40h）原始空闲，C128约143900秒；主分析现在导出idle_compressed_s。墙钟/root/SLO goodput都是压缩回放指标，不能当真实40h会话完成时间。还核对到当前post分支按父完成+源时间差触发，官方父返回路径按child相对branch start偏移，且官方支持父发出时的源重叠。所选10条触发关系一条重叠32ms；d876大间隔当前post延迟144992s、官方父返回相对偏移0。当前公平比较限于冻结自定义协议，并不等价于官方runtime。没有修改运行中输入或调度。若最终结论要推广官方AgentX/真实会话，需独立协议敏感性验证，不能静默合并。先完成现有同协议数据，结合请求延迟/完整SLO网格判断，勿仅凭压缩wall判动态不必要。

PP2 C128 r1已完成2736请求：TTFT1154.5103ms，TPOT39.78504ms/token，E2E13309.6965ms，wall842.3060s。仅首轮、不是最终均值。相比EP4 r1，PP2平均cached75350.27/fresh9505.87 token、60次host命中4323072token；EP4 cached71409.10/fresh13447.04、182次host命中10982592token；均0回退。PP2各stage平均local Decode batch4.95–5.15，两阶段重复不能相加；EP4为5.18–6.37。当前方向PP更好，但有真实缓存工作量差异，不能直接归因于Expert/通信。

PP2 C128 r2也已完成：TTFT1169.4971ms，TPOT39.72995，E2E13024.0854ms，wall840.4614s。两轮均值TTFT1162.0037ms，TPOT39.75750，E2E13166.8910ms，wall841.3837s；仍为2/3进度。r3继续，完成后Gate自动进入P/D slots8新目录。EP4六轮完整、PP2五轮完整；整体目标未完成、动态和真P/D正式对照仍待。

后续CPU分析可检查父子前缀在固定会话分配下的局部性：当前sticky是按conversation，非整root家族，PP4个缓存owner与EP8个owner可能有不同的父子前缀重用机会。这是待分解因素，不是已证明原因。若做动态池，应给予强固定布局同等缓存感知/亲和策略，避免把更好的路由当成EP/PP互补。尚未实现新的prefix-locality分析或family-sticky GPU实验。

上述prefix-locality CPU分析现已完成：`analyze_prefix_locality.py`由主分析自动调用。C128 EP4三轮同owner历史hash前缀77798.36、任意owner77849.45，额外51.09；PP2两轮同owner77805.66、任意owner77849.45，额外43.79。实测cached分别71318.24和75256.58。首token与响应结束两种历史加入时刻结果相同；只是未考虑驱逐/驻留/GDN/GPU实际lookup时刻的hash机会，不是缓存性能oracle。静态跨owner前缀机会小，不足以单独解释约3900token的命中差，不应立即追加family-sticky大规模GPU实验。实际GPU KV按全模型token去重后EP4总17806080，PP2总15083904；PP容量更小，不能以总容量大解释优势。EP4 C128 HTTP在途均值三轮53.80/53.96/51.15，PP2两轮43.23/42.40；不是实际GPU batch。

C128期间一次12秒只读抽样，GPU利用率92–96%，回放client37.6%单核；不代表全程，也不证明全部时间是GEMM。报告保留原HTTP和包含client dispatch lag的边界。新增根会话平均指标包括全部已捕获子代理响应及工具等待，沿用全局空闲压缩，不是官方分数。

## 后续科学判断

主矩阵完成后仍须判断强固定P:D比例和异构P/D，而非直接宣布动态必要。
- 2P6D＋colEP2候选已准备V14 `eight_gpu_ep2_split26_durable_slots8_r6`，未启动，254项冻结哈希通过。修复6个Decode缓存位置的副本周期偏置，8项CPU检查通过；Decode接收8槽位可容纳于GDN80。旧V13及更早P/D包不再启动。
- 固定异构P4(两个PP2)→D4(EP4)候选V14 `eight_gpu_pd_pp2_ep4_durable_slots8_m16_r3`，未启动，255项冻结哈希通过。实际KV＋GDN交接正确性资格检查通过后才计性能。
- 只有真实互补仍存在，再测固定混合池在线路由，给予同质池相同缓存/队列策略；实例数量或角色改变须计实际成本，不能用无成本oracle之和当收益。
- 若证据不支持收益，准确报告限定负结论；不能从8个派生家族推广全部AgentX，也不能把固定实例池分流说成原地重分片。

不要启动旧32/40、未完整预热、E4或旧共享mmap通知的准备包。准备文件不等于GPU测量。

动态策略见POLICY_AUDIT.md：当前正式运行仅sticky分配，cache_queue未用于结论。现有统一fresh-token惩罚、HTTP在途作为batch、未扣生成进度等都需要在动态测试前校准；不要把探索策略失败当作机制不必要的证明。

新增只读`compare_static_matrix.py`：在主分析更新后运行，输出`matrix_analysis/COMPARISONS.md`及`comparisons.json`。只纳入完整三轮且Guard/清理通过的配置，报告绝对均值、右减左差值与全部三轮差值；不把16份副本当作独立家族构造置信区间。目前尚无两种验收配置，空表是预期行为。该脚本不在GPU热路径，也没有改动冻结实验。
