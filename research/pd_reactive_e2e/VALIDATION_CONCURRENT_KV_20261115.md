# Concurrent managed-KV Decode: bounded A100 validation

Runtime and harness source: `e00f2688b01f427efd33689a53a6648c7ff825f2`, immutable tag `experiment/sglang/pd-kv-concurrency-20261115-r5-frozen`. This records a bounded minimum and one finite workflow pair; it does not qualify arbitrary generation equivalence or production performance. PR #16 remains draft and depends on the capacity path in PR #15.

**Change and CPU evidence.** Whole P→D chain admission is no longer one per D instance. KV reservations include all in-flight PP microbatches. Native KV allocation failures create pressure; request slots, transfer handshakes and recurrent-state limits do not. Only unallocated receivers with unpublished write destinations may wait through a pool-address refresh; published writes must drain. Empty stages relay activations to keep concurrent PP rings aligned. The focused CPU suite passed 185 tests covering accounting, waiter boundaries, receiver republishing, scheduler routing and multi-slot/replica routing. The archived log is `focused_cpu_tests.log`.

**Minimum GPU evidence.** Complete Qwen3.5-122B-A10B-FP8, A100 80GB, fixed P=PP2 and D `24,0,24 → 16,16,16 → 24,0,24`, TP1/EP1. HiCache, Graph, CPU-prepacked weights, offload, NIXL and Mooncake remain enabled. The launch requested a nominal PP microbatch cap of one and a running limit of four. Post-hoc logs show native P/D merges can nevertheless form batch two; see the correction below. Native KV pressure triggers growth. Capacity is 8192→12288→8192 tokens; physical KV is 240→336→240 MiB. Two overlapping 512-token streams exactly match stabilized cache-matched PP2 references; overlap is 5.596 seconds. One up/down control sample is 2.091/1.069 seconds. CPU HiCache restores 4608 prefix tokens and matches a 32-token reference. A separate process allocates about 64 GiB and performs GEMM on the retired GPU while Decode continues. Guard postchecks and cleanup pass. A metadata/activation relay worker remains resident; this is not full worker exit.

**Finite pair.** Same AgentX 1.0.7 finite request protocol, dependencies, tool waits and HiCache; two warm tasks and two cold tasks arriving at 45/46 seconds from BurstGPT, client concurrency ten. P remains PP2 on node167; D starts on GPUs0/2 of node168. In-place adds GPU1 to PP3; replication launches a PP2 on GPUs1/3 from the existing CPU weight bank. Both use native KV waiting for the same reactive policy and one cycle. Each completes all 76 calls and four trees: 3,071,334 input and 12,964 output tokens. Workset, workload, policy, token work and production-engine hashes match. Completion audits, post-scaling output checks, both actions and Guard cleanup pass. P token-weighted GPU+CPU cache hit is 96.13% in both. In-place has up to three observed requests advancing output; replica routes 46/30 calls to D0/D1.

**Observed performance, in-place/replica.** Mean request E2E 25.611/27.214 seconds (5.9% lower in-place), TTFT 17.184/22.379 seconds, TPOT 53.960/31.206 ms, p95 E2E 58.413/56.000 seconds. Mean task E2E 515.010/545.470 seconds. Worker residency from formal start through post-shrink check is 3289.864/4104.001 GPU-seconds, including startup and standby (19.8% lower). Preparation through cleanup gives 4389.045/4635.614 GPU-seconds (5.3% lower); whole-node reservations are separate. Frozen TTFT≤15s and TPOT≤150ms joint attainment is 37/76 and 29/76, both below 90%. These are descriptive single-run means, not savings at a satisfied SLO.

**Transition and comparison limits.** Long-context in-place up/down actions take 4.501/4.210 seconds, maximum final fences 1.783/2.415 seconds. Fresh replica startup is 107.911 seconds; this is not a prestarted GPU replica baseline. In-place preparation and relay residency are counted. KV quotas (65536→98304 in-place, a second 65536 pool for replica) do not establish physical VRAM exhaustion. CPU host backing peaks at 128,114,319,360/128,267,993,088 bytes (0.12% difference), and initial CPU budgets differ. Thus `workflow_pair_passed=true` and `fully_qualified_pair=false`.

**Numerical boundary.** The two migrating long requests and specified CPU restore pass exact checks. Fresh PP3 long requests still differ from PP2 references. One short static/concurrent check also lacks matched P-cache history. Neither is promoted to general correctness. Failed r1/r4, prelaunch registry mismatch r2 and interrupted startup r3 remain separate. Further model-quality, repeated/multi-cycle performance and full-worker-exit qualification is open. No broader sweep was run.

**Evidence.** Persistent [report](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/CONCURRENT_KV_20261115.md) links raw runs and the paired analysis. Sanitized snapshot: `/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_pd_kv_concurrency_20261115`. Raw logs/private runtime configuration remain local; this document includes no credentials.


**Post-hoc correction.** The concurrent launch captured only B1 graphs, but native P/D merges prebuilt requests by max_running_requests rather than pp_max_micro_batch_size. PP3 recorded 9457 Graph replays and 1145 Eager Decode calls. Its sampled B2 records are all Eager. Excluding the sole request crossing expansion leaves paired mean TPOT 53.993/31.332 ms, so the one pause does not explain the gap. Capture coverage is being expanded to 1/2/4; no optimized GPU result is claimed yet. The old flag must not be described as an enforced actual batch limit.
# Performance candidate: Graph frontier and online KV catchup

The next candidate captures Decode batches 1/2/4 in both comparison arms. Native P/D can merge prebuilt requests beyond the nominal PP microbatch cap, so a cap of one never proved that actual execution stayed at batch one. The prior PP3 arm recorded 1,145 Eager Decode calls alongside 9,457 Graph replays per rank; sampled batch-two logs were Eager. This is a concrete coverage defect, not a measured attribution of the entire TPOT gap.

Migration now publishes the quiescent initial KV base with a scheduler-stream event, advances bounded NIXL copies between scheduler slots including idle periods, and waits for catchup before arming the final fence. Later asynchronous allocations remain unpublished until a forward completion. The existing staging-byte cap and 1,024-row wire bound remain enforced; timeout or a peer catchup failure aborts before the fence. This aims to move the large cached-prefix copy out of the final pause without retaining candidate GPU weights.

276 focused CPU tests passed, including real launch-argument coverage, allocator epochs, unpublished loadback rows, idle progress, catchup timeout, and consensus ordering. GPU request/cache correctness, migration pause, physical reclamation, and end-to-end TPOT for this candidate remain unvalidated.

The bd807961af E2E run completed 76 formal calls with the same 3,071,334 input and 12,964 output tokens as the old pair. Mean TPOT fell from 53.960 to 32.687 ms and mean E2E from 25.611 to 19.131 s. Guard postcheck, cleanup, round trip, and post-shrink output passed. Grow/shrink final fences were still 689.1/892.4 ms, and grow preparation rose to 8.18 s (6.07 s online catchup). This does not meet the pause objective; the same-source new replica control remains pending.

The next candidate keeps a ready KV frontier instead of repeatedly sorting live rows, sends retirement notifications only for acknowledged incarnations, uses bounded 1,024-row retire packets, and hashes complete CPU metadata as framed arrays with radix-split-independent token-prefix hashes. GDN transfer includes all allocated/cache-held recurrent slots and padding slot zero, rather than undefined contents of free slots. The authoritative allocator-map consensus and all coverage checks remain. 325 focused CPU tests passed; the old all-GDN spill fixture now explicitly allocates all rows it claims meaningful. GPU validation of these changes is pending.


R2 source `8b7cffa3a7` passed the bounded minimum: grow/shrink final fences 189.96/94.44 ms, control 2.071/0.982 s, both migrating outputs and CPU restore match, physical KV 240→336→240 MiB and independent 64 GiB reuse pass. All 39 sampled B2 logs use Graph; Guard postcheck/cleanup/release passed. The subsequent E2E attempt failed at the initial P/D smoke with an incomplete stream, before formal requests or migration. Its failure artifacts and cleanup remain. A harness-only diagnostic now reports stream framing/status and the concurrent Prefill outcome rather than hiding that exception; production engine code is unchanged from R2. This is not an E2E performance result.

Harness diagnostic follow-up: five stream/error unit tests passed. A whole harness-directory collection was unavailable in the engine venv because it lacks aiperf; this was not a runtime test failure. Focused router/lifecycle/cleanup/Graph and diagnostic checks use the engine environment. Raw collection failure and focused result logs are retained in the next run.


# R4 completed, prepared P/D registration candidate

R4 (`42debf1d2b5a`, production engine identical to R2) completed all 76 formal calls, round trip, post-shrink check, Guard postcheck/owned cleanup/release. Mean TPOT 32.407 ms, E2E 19.518 s, TTFT 14.263 s; grow/shrink final fences 469.1/469.8 ms, prepare maxima 1.969/1.105 s. The original 31.206 ms replica is historical; a new same-source replica is pending. The tens-of-ms goal is unmet.

The next candidate preregisters already mapped target KV physical pages in background preparation. Source registrations and published write destinations stay unchanged until P/D writers drain. The fence deregisters only retired regions; commit registers missing GDN/auxiliary regions against the actual descriptor. Abort deregisters every attempted addition before prepared backing may unmap; uncertain partial deregistration fails closed. Numeric phase counts/bytes/seconds and a timed final refresh expose remaining cost. Legacy unmanaged/synchronous paths retain their existing registration behavior.

179 focused CPU tests passed, including registration lifecycle, live-writer gating, descriptor drift, partial errors, rollback ordering, exact prepared VMM ranges and existing KV/coordinator/transport tests. A fixture initially lacked serving_capacity (2 failures) and a subsequent invocation named a nonexistent test file; both failed receipts remain in /tmp and will accompany the next run. GPU validation of the candidate remains pending. No same-source replica result or broader qualification is claimed.


# R5 output rejection and native PP microbatch-cap fix

R5 minimum finished the capacity round trip, CPU restore and independent 64 GiB reuse, but both migrating 512-token outputs differed from stabilized serial references. Final fences 134.4/79.6 ms are failed-qualification diagnostics, not accepted performance. Guard postcheck failed and the lease released after owned cleanup. Serial repeats and cross-run static PP3 outputs remain identical. The first migrating lane's logprob difference begins after expansion returns and batch-two merging starts. Prior R2 overlap samples were B1 despite two in-flight requests; R5 samples include B2. This suggests a batch-shape boundary but does not prove registration correctness by itself.

Native P/D prebuilt merging now honors pp_max_micro_batch_size per PP slot. A full slot leaves requests in the waiting queue for another empty slot; global multi-request concurrency is retained, with no forced one-request-per-instance fallback. 71 focused admission/registration/capacity tests passed, including the actual native admission method for full/empty slots, configurable caps, legacy PP1 and unset-cap behavior. Seven stream diagnostics/timing tests passed. The E2E adapter additionally retains eight largest upstream SSE content-event gaps per request; these are chunk observations, not necessarily individual token intervals. The actual prepared adapter instruments and compiles successfully. GPU acceptance remains pending; do not advance to E2E while migrating-output checks fail.


# R6 rejection; actual cap is four, fixed-shape Graph diagnostic

R6 retained the R5 output failure (final fences141/84 ms), with CPU restore/physical reclaim/independent reuse passing and Guard postcheck rejecting the run after owned cleanup/release. Effective launch inspection corrected the earlier causal claim: optimized R1+ runs already declared pp_micro_batch_size=4 and max_running_requests=4. Honoring the PP cap is a valid parameter-consistency fix but did not alter this configuration. Do not attribute the observed mismatch or performance to that fix.

Serial reference/repeat and static PP3 outputs remain identical across R2/R5/R6. Prefill's returned first-token logprob is identical too. Passing R2 migrating-overlap samples were B1, failing R5/R6 overlaps include B2. A diagnostic CLI now supports a single capture size4, retaining actual four-request admission and batch caps and padding B1/B2 to the same graph shape. Default1/2/4 behavior remains available. This is a controlled numerical diagnostic, not a claim of better performance or a relaxation of output gates. Sixteen related CPU checks passed; GPU pending. The client event-gap audit remains unchanged.

# R7 rejected; direct committed-state diagnostic

R7 fixed B4/padding did not resolve the failure: migrating outputs [true,false], CPU restore false, serial repeats [true,true]. Final fences137.1/82.6ms are rejected diagnostics. Guard exit1/postcheck failed and lease released; no E2E was launched. Raw failure and safe analysis remain in the continuous report.

The next opt-in minimum diagnostic hashes the actual source KV and allocated recurrent rows (including padding slot0) at the drained fence. Existing control-group consensus carries only row indices, tensor geometry and digests. After model/cache commit and Graph selection, before serving resumes, every target hashes its committed buffers and checks the source-owner records, including retained layers. Any mismatch fails closed after commit. The private empty-state startup Graph walk is outside this runtime audit. Hashing cost excludes this run from performance evidence. Default1/2/4 Graph and four-request limits remain; no output gate is relaxed. 165 focused CPU checks passed, including single-row corruption, NaN/signed-zero byte comparison, GDN reindexing, capacity growth, free-row exclusion, stale/missing ownership, mixed-rank opt-in, and post-commit fail-closed behavior. GPU validation is pending; first 98-check and final 165-check logs are archived separately, not summed.

R8 GPU diagnostic completed: all three ranks passed both committed-state byte audits. Expansion verified96 KV/GDN tensors totaling734,183,424 logical bytes; shrink verified96 tensors totaling153,649,152 bytes, including retained layers. Both migrating outputs still differ (first token indices283/51, zero-based); serial repeats and CPU restore match. This establishes byte preservation only at these drained fences, not arbitrary cache history, Graph bindings or mathematical batch invariance. Audit-inclusive pauses1.117/0.221s are not performance results. Guard postcheck rejected the output mismatch; owned cleanup and lease release completed.

The next minimum adds one startup MoE operator probe using actual loaded Marlin weights: keep one anchor input/routes/gate weights fixed; compare B1 against B2 with identical or shifted companion routes, and repeat each fixed batch. No attention, cache or PP migration participates, and only scalar error/count metrics are retained. This tests a numerical hypothesis without asserting it is the root cause. The regular runtime state audit and strict generation gates remain. 172 focused CPU checks passed, including fixed-anchor/composition controls and refusing diagnostics in performance runs; GPU validation is pending. The default probe flag is off and the regular Marlin argument values/output are preserved.

## 2026-11-16：R9结束与非零算术隔离

R9（`b1263047d1c0`）两条迁移输出仍不匹配串行参考，第4／280个token开始不同；三rank扩缩状态内容摘要、CPU恢复和独立64 GiB复用通过。Guard后检查拒绝，owned清理和release完成。启动Expert单个输入点无差异，不将它扩大为所有非零输入或层的算术验收。

下一最小轮增加显式非零输入：Expert使用三档幅度；dense Marlin按几何选择加载权重，比B1／B2／B4；GDN packed使用每次重新克隆的私有状态，比同一行输出及递归状态。只保存范数／误差，普通启动不安装这些钩子，诊断标志禁止进入端到端性能轮。29项相关CPU检查通过；GPU待验，不累计为新的全套回归资格。

## 2026-11-16：Marlin归约边界与R11候选

R10两条迁移生成、缓存内容、CPU恢复、独立64 GiB复用及Guard清理通过，但静态PP2合批对照仍有一条不匹配。21个非零算子记录中，GPU1／2的Expert在只改变companion路由时出现6组可重复差异；dense与GDN所测输出／状态相等。不是缓存复制证据，也尚未证明它解释全部模型差异。

候选以`SGLANG_MARLIN_MOE_CANONICAL_K=1`显式启用：Marlin MoE的每条计算条带按完整K归约对齐，不随活跃专家块数改变K切分，并禁止原子输出归约。默认关闭；P／D同时启用以保持算术路径一致，JIT specialization独立。还将PP2／PP3静态合批输出纳入最小轮门槛，禁止只靠迁移请求碰巧匹配通过。61项相关CPU检查通过；本地只做CPU检查，CUDA编译和算术／模型验收在已有A100节点的Guard租约内执行。R11 GPU待验。

**R11 在原生编译阶段失败，未开始生成验证。** 编译器指出设备函数定义漏传 `canonical_k`，与已更新的声明／主机调用不一致；Guard拒绝本轮，P／D归属清理及release通过。不存在可报告的新暂停或TPOT。下一R12仅补全设备函数参数，继续同一严格验证，不修改请求、缓存或输出门槛。

R12定向CPU回归61项通过，原生编译与GPU计算资格待Guard最小轮；CPU结果不替代CUDA编译证据。

**R12 数值修正与参考审计。** whole-K原生内核已在已有A100编译，三卡×三档非零Expert输入的同／异同伴路由均逐元素一致。两条迁移512-token输出、静态PP3、状态内容审计、CPU恢复与独立64 GiB复用通过，但静态PP2仍一条不匹配，严格门槛拒绝、owned清理与release通过。本轮含字节审计，暂停不作性能。进一步核对发现，短串行参考的P缓存命中均为0，合批均为1984，首token的Prefill logprob已经不同；先前“连续两次输出相同”错误地接受了冷参考。这不是匹配缓存历史下的纯Decode合批对照，也不证明所有差异都来自缓存。

下一R13要求参考实际命中GPU／CPU缓存、缓存计数与输出连续稳定，并逐项验证串行、静态PP2／PP3和迁移请求的Prefill缓存计数一致。保留全部原输出要求，不更改R12失败结论。引擎与whole-K内核不再改变，只修正参考预热协议。

R13定向CPU回归83项通过，覆盖重复冷输出拒绝、GPU／CPU缓存来源差异、无效计数与全部输出／缓存门槛。GPU最小轮待完成。

**R13匹配暖参考及短端到端已完成（2026-11-16）。** 源 `1ed1b4bad7350bf78a68903bd2866af672a4e1ab`／tag `experiment/sglang/pd-kv-performance-20261116-r13`，83项CPU检查。最小轮四组输出与四组Prefill缓存门槛全通过，扩缩三rank状态内容审计、CPU恢复与独立64 GiB复用通过；物理KV240→336→240 MiB。

关闭探针后的原地端到端完成76／76请求，3,071,334输入与12,964输出token，请求身份及工作量与R4完全相同。平均TPOT31.975 ms、请求E2E19.229秒、TTFT14.066秒；p95 TPOT46.697 ms／E2E37.627秒。最终扩／缩暂停238.890／315.152 ms，控制调用2.349／1.616秒，完整操作2.827／2.103秒；后台准备最长2.015／1.153秒。Prefill GPU＋CPU命中96.13%，48／76达标，服务区间常驻worker2686.504 GPU秒。缩后核对、Guard postcheck、owned清理与release通过。相对R4暂停缩短，TPOT仍约32ms；同源双实例尚未补测，不作显著TPOT改善或同SLO省资源结论。

正式76请求均记录上游SSE内容事件：扩容控制窗口内最大间隔325.498 ms，包含网络／调度及下一次内容发送，不等同逐token间隔或纯fence；缩容发生在正式流结束后，没有在途内容间隔样本。一次有限流／一周期不等于一般正确性或多周期资格。几十毫秒暂停目标仍未达到。详细原值见 `analysis/kvcached_perf_e2e_r13.json` 与 `analysis/kvcached_perf_e2e_streams_r13.json`（位于持续实验目录）。

**R14 候选：减少最终交接内的控制与注册工作。** HiCache先退役旧PP控制发送，再全组确认是否有I/O；只有全组空闲才跳过新事件轮，远端仍忙时所有rank继续参与。记录发送等待、提交、事件处理和CPU状态同步分项。已映射目标GDN在后台预注册，持久P/D元数据保持注册，旧写入地址仍需排空后才注销；失败先清理新注册再允许撤销目标存储，最终地址必须与预备集合一致。147项相关CPU检查通过；尚无R14 GPU正确性或性能结论，沿用R13全部严格门槛。

**R14最小与关闭探针的短端到端完成。** 源c7b4d44eec69，147项CPU检查；最小轮四输出／四缓存、KV／GDN内容、CPU恢复与独立64GiB复用全通过。76／76正式请求、3071334输入与12964输出token，完整工作量签名匹配R13；平均TPOT31.620ms、E2E19.167秒、TTFT14.040秒，47／76达标。扩／缩最终fence218.361／304.090ms，控制调用7.683／1.579秒，完整操作7.850／2.051秒；扩容在线KV追赶等待5.306秒，后台准备不能视为在线停顿。提交新增区域注册0，历史CPU载荷复制0；扩容SSE内容事件最大间隔304.892ms，不等同逐token或纯fence。Guard后检查、归属清理与release通过；没有达到几十毫秒目标，同源副本仍待补。

**R15：CPU控制发送的完成推进。** 当前PyTorch2.13.0+cu129的CPU复现确认：对端已收到128条Gloo发送，但发送者的native is_completed在调用wait前均为false；原PP Decode非阻塞轮询会保留旧句柄。改由进程级单CPU线程等待已经发出的发送，不新增通信或CUDA操作；调度线程继续发送接收器所需共识，在后续slot回收future，缓冲保持至真实完成，异常保留并传播，fence仍完整排空。真实CPU复现中128条均能在调度线程wait前完成。84项定向CPU检查通过，覆盖不阻塞共识、完成回收、缓冲生命周期、失败传播与常规缓存分派／睡眠生命周期。一次误纳入完整缓存GPU矩阵的CPU运行在无加速器环境失败并中止，日志保留，不声称全矩阵通过。尚无R15 GPU或端到端性能资格。持续报告与复现脚本在原实验目录analysis中；本轮不同时修改KV摘要算法。

**R15关闭探针的短端到端完成。** c36b1f23777f，76／76请求及完整工作量签名匹配R14；TPOT18.756ms、请求E2E15.551秒、TTFT12.406秒，联合SLO61／76。扩／缩最终220.136／169.234ms，完整操作2.767／1.573秒，扩容最大SSE内容间隔278.443ms。Guard后检查、归属清理及release通过。TPOT比R14单轮低40.7%，扩容暂停未改善，目标仍未达到。当前主要暴露成本为最慢rank注册注销74.837ms、canonical元数据15.028／36.564ms及KV收尾约33ms；CPU归约中的等候不能与其他rank这些开销相加。

**R16：完整KV覆盖摘要的固定宽度编码。** 将排序slot与incarnation编码为完整64位小端向量，加版本域与行数后做SHA256；两端同源，任一行或高位代次变化仍需被拒绝，非法浮点／负值／溢出显式拒绝。发送端同一快照只计算一次，单独记录signature_seconds。86项CPU检查通过，含独立struct构造的wire预期、顺序不变、64位高位保留、同数量错误槽位／代次拒绝，及原有真实tensor／allocator的伪NIXL复制、pin／free／abort与HiCache发送完成检查。没有改P/D注册时序、输出门槛或缓存预算。独立CPU微测58,300行有序17.265→2.659ms，乱序38.400→3.902ms；这不是GPU或端到端性能资格。GPU严格最小轮待运行。
