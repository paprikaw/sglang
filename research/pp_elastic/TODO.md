# PP 原地伸缩＋HiCache：实现与实验清单

## 122B／A100：P/D 下的最小扩缩容（2026-11-14）

- [ ] 主 agent 在 `codex/qwen122b-pd-elastic-20261114` 实施。固定 P2，D 以 `24,0,24→16,16,16→24,0,24` 改变层归属，使用 FP8 Marlin、Eager、同步 CPU-spill 和 HiCache。已补 D 调度／交接围栏、KV/GDN 注册刷新及量化运行态迁移；代码与 CPU 检查不代表 GPU 验收。最小实测脚本为 `pd_elastic_smoke.py`，要求同一请求跨两次转换、输出与同暖参考一致、CPU HiCache 实际恢复、独立进程复用退役 GPU，并记录残留控制 worker 显存。请求交接在转换时暂缓；P 伸缩、在途 RDMA、Graph 和完整 worker 退出不在本轮资格内。

  11月14日继续：按层 GDN 传输已允许合法的零层目标，并保留对应字节数检查。首次完整多 token P/D 路径在转换前卡住；栈显示空闲槽等待上一激活完成，阻塞下游控制消息共识。将弹性 P/D 空闲槽的激活完成等待移到三轮共识之后，保留发送缓冲区所有权；资格范围显式限制一个活跃请求、微批上限1。142项相关CPU检查通过。GPU往返、HiCache恢复和独立复用仍待验证，不能据此宣布打通。

  r6完成2048-token未切换参考，但扩容围栏的None队列标记被旧列表传输函数拒绝。补None序列化并明确区别空列表共识；145项相关CPU检查通过，含真实CPU wire函数的三个往返。r6原失败及重复取消打断控制器收尾的记录保留；两节点原始cleanup与独立GPU进程检查均为空。脚本现保护收尾不再被重复取消打断、提前保存失败证据，后续最小往返改为512-token以缩短调试时间；不将其当作2048-token性能复测。

  r7完成512-token参考并排空请求／P/D队列，实际停在HiCache fence：rank0等待未被消费的异步缓存消息，rank1／2已进入迁移共识。新增P/D专用轮询次数对齐，让落后接收端补齐已有轮次后再等待所有发送完成；不在领先rank新增消息。169项相关CPU检查通过，含三个多线程、不对称缓存消息的排空复现。r7通过owned stop协议退出并正常清理／release，GPU迁移与复用仍未验收。

  r8实测rank1／2各补一轮HiCache轮询后完成PP2→PP3拓扑提交，控制面完整8.583秒、fence8.360秒；仅一次诊断，非性能均值。随后D传输队列保留旧HiCache对象，旧controller清理后恢复轮询报错，未证明请求连续或正确性。发布缓存时补齐D预分配／传输队列引用，保留队列与socket身份；170项相关CPU检查通过，含真实Decode恢复轮询的旧controller退役复现。r8原失败、owned cleanup及正常release保留，仍需完整GPU往返验收。

  r9首次完成两个方向的拓扑提交，同一RID生成512个token、无retraction或重新Prefill；缩回后独立进程在中间卡实际分配／写入64GiB并计算通过，metadata worker仍驻留。严格输出验收失败：第55个token开始不同，但首个token的概率在迁移前已不同；D参考命中512、切换组命中576，缓存起点没有对齐。原失败与cleanup通过记录保留，不授予生成正确性或CPU恢复资格。下一轮先预热完整576前缀，增加不切换布局的重复检查，冻结相同P/D缓存起点后再执行往返。

  r10对齐D的576-token暖起点后，未切换布局的两次生成仍从第7个token分叉，P侧首token概率也不同；因此未执行任何迁移，原失败、正常清理及release保留。下一候选开启兼容Triton／radix的原生确定性推理，额外记录P侧复用GDN检查点的内容指纹，以区分缓存变动和相同输入的执行差异；保持HiCache和严格逐token检查，不把诊断当性能实验。

  r11开启原生确定性推理仍在静态重复中失败，未执行迁移；P的两个rank在三次请求中命中的GDN检查点内容各自完全一致。下一候选增加临时同步的逐层激活指纹，定位首次偏离的算子；这种诊断会改变执行时序，若其本身消除差异，不能据此宣称原执行路径通过，必须去掉后重新验证。

  r12同步逐层检查仍在静态重复失败，没有迁移；rank0首次偏离在layer2输出，输出残差相同、前两层一致。原输入指纹漏了keyword hidden_states，因此不把它当作完整输入一致证明。下一候选只额外复算layer2的真实Expert调用，固定输入、路由和权重，区分Expert算子重复性与路由／其他子模块变化。所有临时同步和复算必须在正式资格重试前移除。

  r13诊断包装器漏了原生dispatch_output关键字参数，启动后首次前向报TypeError，尚无Expert复算结果；按owned stop清理并正常release。修正参数名、补齐keyword输入指纹后，先用真实CPU Tensor检查包装器的原生关键字调用和重复计算接口，再重试硬件。

  r14真实layer2 Expert以相同输入／路由／权重连续执行两次，8／14个元素不同，最大绝对差1.53e-5／3.05e-5；静态输出仍失败、没有迁移。下一候选在原生确定性模式下固定Marlin的专家内token排列，避免原子插入改变split-K分组；182项相关CPU检查通过，含12项完整route／padding语义对照。GPU复算是否消除差异仍待验证，尚未将此推断认定为根因。

更新：2026-11-13。本页是本项目的统一进度入口；任务完成后更新这里，不另外维护相互冲突的清单。勾选表示该条写明的范围已有证据，并不表示全部功能已验证。

## 当前：减少迁移、最终暂停及推理干扰

- [ ] **候选：同节点 CPU HiCache 原位复用，交接独立于实例伸缩。** 用户澄清仍按新层归属管理，CPU 整理／同步不阻塞推理恢复；同节点优先复用共享底层存储，只改按层视图、槽位映射和读写／回收权限。当前池从本地 GPU 层数构造，切换会新建并复制，不会自动共享。确需换 CPU 池时再后台按层同步，过渡按需读取只是兼容路径。补共享内存生命周期、kernel stride、GDN 及在途 I/O 资格，保持 HiCache 与候选权重 offload。见[简要设计](DISTRIBUTED_HICACHE.md)。仅完成调研，代码及 GPU 验证未开始。

- [x] **2026-11-13：拆解后台准备卡顿与缩容190 ms，CPU批量及旧KV基础复制有界通过。** 第一组同源对照，CPU批量处理使扩／缩最终暂停113.3／184.4→104.7／138.0 ms；复用同一冻结交接的精确核验，省去保留层重复扫描。第二组同源且两边均启用CPU批量，仅切换旧GPU KV基础复制，缩容最终暂停136.0→87.0 ms，完整转换0.885→0.844秒，最终待补2103→55行、复制等待50.5→6.1 ms，接收确认仍约5.9 ms。扩容103.8→106.1 ms，无额外提速；准备期扩／缩最大token间隔74.6／131.2→72.7／135.7 ms，缩容平均11.73→11.85 ms，未改善。时间线定位到各阶段先后的调度线程CPU快照，而非把整段后台权重加载算成暂停。1181项完整CPU检查及两组各模式2048-token暖输出／CPU前缀loadback／Graph replay／清理通过；四次各138项冻结输入匹配，每次含清理均小于5分钟。HiCache、CPU权重缓存及候选权重offload保持；源为a0b42990d36e／67df645bc2b8，两组不混成重复均值。旧67.5 ms属于在线KV补丁收尾，非GDN独占。见[分项报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_batch_io_20261113/summary.md)和[实现说明](HICACHE_BACKGROUND.md)。

- [x] **2026-11-13：目标CPU池与HiCache稳定行后台预搬，有界验收通过。** 同源A100／35B BF16／B1，各模式一次正式往返，扩／缩暂停157.9／193.6→115.2／189.6 ms；客户端跨切换间隔183.7／207.3→141.0／203.2 ms。准备期间最大token间隔25.0／22.5→76.4／67.8 ms，完整扩／缩1.115／0.725→1.336／0.912秒。每方向117,309,440 B稳定CPU前缀预搬、最后CPU数据补传0 B。1162项完整CPU检查、两个方向各2048输出及两模式逐token一致、实际CPU前缀恢复、Graph replay和正常清理通过；候选GPU物理权重仍为0。初始CPU快照和最终内容核验仍消耗调度／暂停时间，缩容另有最新GDN／KV收尾；不宣称无中断、整体吞吐提高或整卡归还。 代际结束清除临时快照／传输引用，新代重置计数；历史、HiCache及CPU权重缓存保留。源码 b5467d3a432d，138项冻结输入各自一致。见[机制说明](HICACHE_BACKGROUND.md)及[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_background_20261113/summary.md)。
- [ ] **进一步降低准备期与缩容干扰。** 初始CPU独立快照仍阻塞调度，缩容各rank先后约33–37 ms；最终CPU内容核验仍约40 ms。后续候选是有写入版本保护的分块快照、块间让出调度，或可靠跟踪CPU行写入以减少全量内容扫描；这些尚未实现。最新GDN／KV仍需交接，不能以后台字节已搬完推断暂停消失。

- [x] **2026-11-12：减少最终264 ms暂停，有限GPU对照通过。** 冻结`7dda6e42a012`，1130项完整CPU检查通过。同源／同A100节点／35B BF16／B1，旧→fast-copy最终扩／缩暂停253.351／296.300→159.983／195.916 ms（少36.9%／33.9%），客户端跨已记录fence的相邻token间隔281.795／310.296→184.816／210.630 ms。目标接收行导入111.0→20.7 ms，Graph选择仍约2 ms；连续行直接复制、快照只留一份，新私有池不做无用undo，保留源至发布、失败丢弃目标。CPU前缀快照／传输／保留行字节数相同，HiCache容量不变，候选GPU权重offload，缩回释放20.37890625 GiB。每方向2048输出一致、实际Graph replay、runtime capture=0、CPU loadback、136项冻结输入及清理／release通过。每模式一次正式往返；初步未开客户端计时的一轮单独保留，不混算。整卡归还、更多模型／batch及独立review未扩展。[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_fence_copy_20261112/summary.md) · [机制说明](HICACHE_FENCE_COPY.md)。

- [x] **2026-11-12：拆分单GPU装载，CPU提前打包及有限GPU对照通过。** 冻结`20d7e1970e98`，1117项完整CPU检查通过。原生路径单独诊断为10,223次H2D、调用内3.882秒、剩余host路径1.840秒；含pageable等待与诊断成本，不能当纯DMA／纯打包。正式同源／同节点／无诊断对照，13层单GPU装载4.584→0.896秒、H2D变218次；完整扩／缩5.139／3.025→1.221／0.833秒，最后暂停264／264 ms。CPU准备前移至启动，接收卡对应分配6.026秒＋打包5.843秒；三worker额外CPU张量40.739 GiB，每worker上限24 GiB，联合Guard预留92 GiB，HiCache容量不变。两策略每方向2048输出token一致，实际Graph replay、运行时capture=0、CPU loadback及135项冻结输入各自通过；缩回中间GPU物理权重为0、驱动可用显存增加20.37890625 GiB，无GPU候选权重复用。首次Gemma派生buffer拒绝已通过独立CPU owner修正，失败与清理记录保留。每策略一次往返、35B／B1／native TP1／EP1 Triton范围；整卡归还与独立review未扩展。[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_cpu_prepacked_weights_20261112/summary.md)。

- [x] **2026-11-12：目标GPU直接从已有CPU权重缓存恢复，有限验收通过。** 冻结`521b14cb725e`，完整CPU回归1084项全过；069节点35B BF16／B1／HiCache／NIXL／预热Graph的PP2→PP3→PP2通过。同源同节点旧路径完整扩／缩20.103／20.885秒，直接CPU bank为5.089／2.978秒；最后交接242／258 ms，客户端跨fence相邻token间隔269／273 ms。无GPU导出或Gloo权重传输，每方向约21.87 GB真实恢复；缩回中间GPU模型参数及物理权重为0，释放20.37890625 GiB。两种路径每方向2048输出token一致，运行时capture=0、实际Graph replay和CPU前缀loadback通过；134／131个输入、清理及正常release通过。保留首轮GDN别名拒绝及其CPU复现，按Parameter身份修正，未放宽地址／覆盖检查。每策略一次往返，不是重复均值；固定native TP1／EP1 Triton，冷暖生成等价、更多batch／模型及整卡归还未扩展，无独立reviewer。见[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_direct_cpu_cache_weights_20261111/summary.md)。

- [ ] **2026-11-11：加速真实搬权重，保持非驻留要求。** 在 `codex/pp-weight-host-pipeline-20261111` 将逐张量pageable中转改为256 MiB有界pinned分块，经独立CUDA流异步D2H／H2D，CPU／Gloo传输，直接写入目标已准入权重页；不缓存候选GPU权重。冻结 `fcefe8604c65` 的1036项完整CPU回归通过（120.55秒），含12项新协议检查。同版本A100 legacy对照通过：完整扩／缩19.758／20.824秒；缩容源D2H约10.007秒、发送及等待10.235秒，目标H2D各1.026／0.861秒。pinned首个启动被Guard因另一正常实验占两台A100而拒绝，未启动模型；后续候选未运行。保留失败和准备记录，[PR #10](https://github.com/paprikaw/sglang/pull/10)保持Draft；新路径GPU正确性及提速尚未验证。下一次从新ID和源标签继续，不覆盖旧配置。见[设计与结果](WEIGHT_STAGING.md)。

- [x] **2026-11-11 用户新增底线：候选布局权重不能常驻 GPU。** 默认 `offload` 将Graph稳定地址与权重物理页拆开，释放退出层及多余输出权重，接班时真实搬权重。冻结 `26bdee39146f`，35B／A100／B1／HiCache的两次独立PP2→PP3→PP2往返通过；每方向实际搬21.87 GB，完整扩／缩均值19.98／20.65秒，最后暂停239／285 ms，运行时capture=0且实际Graph replay。缩回后中间GPU模型参数及物理权重页均为0，每轮驱动可用显存增加20.37890625 GiB。暖缓存连续性、在线KV及CPU loadback、监督清理／正常release、129个输入核对通过；最终1024项完整CPU回归通过（114.57秒）。当前Gloo权重搬运是主要瓶颈。旧约0.3秒仅为resident诊断；KV／GDN／Graph工作区和worker仍占资源，整个GPU归还未完成。见[设计与结果](NONRESIDENT_GRAPH_WEIGHTS.md)。

下面三项保留历史resident实验记录，不代表满足当前权重释放底线。

- [x] **主agent直接实施：**`b59993a97f28`复用启动已常驻的不可变候选权重，保留recopy对照和版本／地址检查，补齐准备与交接分项计时。1000项CPU通过；A100同版本每策略一轮，扩／缩完整迁移20.470／20.955→0.691／0.727秒，权重传输每方向减少21,871,498,496字节。暖缓存连续性、Graph replay、在线KV及最终CPU loadback和正常清理通过；尚无重复均值。见[优化记录](MIGRATION_OPTIMIZATION.md)。
- [x] **主agent直接实施：**`ed2e82c5eda8`新增opt-in PyTorch pinned HiCache分配与启动候选预热，默认mmap保留。1008项完整CPU通过；同一A100节点两次独立启动均通过PP2↔PP3暖缓存连续性、运行时capture=0／实际replay、在线KV及CPU loadback。完整迁移扩／缩均值0.293／0.321秒，最终暂停均值246／271 ms；3进程采样pinned内存5.781 GiB，非总CPU峰值。两轮清理／正常release、127个输入哈希核对通过；仅35B／B1／native／L2 HiCache范围。见[优化记录](MIGRATION_OPTIMIZATION.md)。
- [x] **主agent直接运行：**相同请求、缓存、Graph／NIXL及节点，原路径与完整优化各两次独立启动、每次往返，另保留同版本仅复用权重单轮。完整迁移扩／缩均值20.459／20.891→0.293／0.321秒，最终暂停634／655→246／271 ms，平均TPOT14.307／14.905→8.692／8.733 ms；各轮输出间隔、E2E和阶段原值已归档。N=2只作描述性重复，TPOT／E2E含进入目标布局更早的效果，更多负载与严格峰值仍未测。[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_migration_optimization_20261111/summary.md)。

**当前决定。** 保留已完成有限正确性及单次 overhead 观测的同步基线；下一阶段联合研究稳定地址的 KV 池扩缩、有限 CUDA Graph 预捕获与持续 KV patch，目标是减少推理干扰。2026-11-09 已完成独立调研；用户随后要求本轮由主 agent 自己实施，不启动 sub-agent。实施前先固定共同源码、兼容依赖、层分组和状态地址约束；主协调器维护项目分支与本页。当前实现与本次验收汇总见[基线状态](BASELINE_ACCEPTANCE_20261109.md)。

## 本轮实施：主 agent 顺序完成

- [x] **2026-11-11：集成审查及候选检查完成。** 主 agent 直接审查 Graph 地址与状态生命周期、在线 patch 槽位保护、NIXL 应用确认及提交失败边界，未发现阻止有限研究基线合入的功能性问题。修正本次改动的风格检查与 CPU 用例选择；992项组合回归全部通过、零失败／错误／跳过，113.580秒，本次Python改动的isort／Black／Ruff全部通过。全仓CI仍有原基线欠账，不能称为全绿。通过PR #4先合项目，再经总PR合共同基线；合入状态另行记录，独立审查和未测范围不扩大。见[审查记录](REVIEW_20261111.md)。
- [x] **项目合入完成。** PR #4已以`69dc304d8072`合入`codex/pp-hicache-autoscaling`；项目合并树与992项检查的候选`4d7638b01ac2`完全一致，没有组合冲突。主协调器通过[总PR #5](https://github.com/paprikaw/sglang/pull/5)合回共同基线；GitHub合并记录及冻结标签记录最终提交，后续优化基于该共同分支。

- [x] **2026-11-11：启动预捕获与运行时选择的代码与CPU检查。** 声明有限PP布局，保留原生权重／KV／GDN、独立backend和Graph工作区；运行时拒绝未知布局或地址变化，禁止重新捕获。546项CPU回归通过，见归档 `prepared-graph-20261111/cpu-r5.xml`；先前测试路径／空CUDA设备变量导致的收集失败保留原记录。[使用与边界](PREPARED_DECODE_GRAPHS.md)。
- [x] **新路径A100 PP2↔PP3有限验收通过。** 冻结 `c5aa853480`，启动捕获本地runner数量2／1／2，运行时capture=0；扩容各目标阶段replay1028次，缩回两个保留阶段各1090次、空阶段0次，活跃代Decode eager=0。两个方向各2048-token输出与当前版本同进程暖缓存参考一致，无retraction／重新Prefill，最终CPU前缀实际loadback。Graph选择1.733–2.596 ms；完整fence扩／缩0.604／0.688秒，后台prepare仍19.799／20.438秒。初始Graph后额外启动准备约3.9秒，常驻allocated显存合计增加21.513 GiB（新增阶段20.816 GiB）。完整监督178.903秒内正常清理并release，127个冻结输入一致。546项完整CPU回归后，启动NIXL隔离修正的81项相关检查通过，不累加。新验收类别只证明同版本暖缓存执行连续性，旧状态审查与冷暖生成等价未扩大；B1、单轮，无独立reviewer。首轮预算入口错误未启动模型，失败原样保留。见[本轮报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/summary.md)。
- [ ] 后续边界：实际B2／B4、减少KV／GDN及工作区常驻内存、额外峰值／逐token间隔的独立测量、严格配对性能比较、KV VMM／HiCache／GDN组合及GPU故障恢复。权重物理页释放已由本页首项验收；worker退出仍按用户要求暂缓处理，没有归还工作GPU的结论。

- [x] **2026-11-11：整模型 Decode CUDA Graph 接入通过。** 冻结 `01bc3770e3`，A100 native／HiCache／NIXL PP2→PP3→PP2在最终排空时重建目标Graph并经全组确认恢复。两个方向各2048-token输出与同暖512边界参考一致，没有retraction／重新Prefill，最终CPU前缀实际loadback。PP3三个阶段各replay1013次；缩回PP2后两个保留阶段各1095次，退出阶段inactive／0次；活跃代Decode eager为0。完整CPU回归508项通过，协议延长后73项harness复查通过，数量不累加。完整监督含清理177.895秒，Guard／owned cleanup／正常release通过。控制面最终交接3.137／2.261秒，包含重建；未宣称暂停收益、全部布局预捕获或VMM组合资格。worker常驻按用户要求后置。见[验证报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_nixl_decode_graph_20261111/summary.md)及[接入说明](DECODE_GRAPH_INTEGRATION.md)。

保留失败：GPU r1因旧共享Graph池ID退役未清除而触发PyTorch `use_count`断言；显式清理backend并重置两个池ID后修复。r2捕获成功，但1024-token请求在准备期间结束、目标replay=0，严格验收保持失败。r1／r2分别132.268／121.676秒内正常清理并release；r3延长至2048输出、4096 context后通过，未放宽活跃请求、输出或replay检查。

- [x] 用户确认所有候选GPU启动时建立并预热NIXL agent；已实现固定GPU双缓冲、独立进度与目标应用确认，接入native／HiCache持续KV patch及最终fence。已隔离锁定NIXL 1.5.2；487项elastic_pp CPU回归通过，包含实际allocator及模拟字节传输，覆盖写入确认、复用、改写、多接收端、错误payload和取消失败。首次回归发现异常处理线程退出，修复后全过；原失败保留于归档`nixl-live-20261110/cpu-suite-r1.log`，通过证据为`cpu-suite-r2.xml`。本条只验收代码和CPU协议，GPU正确性见下一条，性能及组合资格留在未完成项。
- [x] 真实A100 native／eager／HiCache PP2→PP3→PP2的有限正确性通过，冻结`152de4332f`。三个候选端点各完成两条通道预热，每卡注册64 MiB；扩／缩目标在最终暂停前实际应用1320／1318行KV，行数按来源／接收端累计，不是全局原始token数。两个方向各256-token输出与同暖512缓存边界参考一致，没有retraction或重新Prefill，最终CPU前缀实际loadback。完整监督含清理176.008秒，Guard postcheck及正常release通过；同节点日志选择CUDA IPC。首轮bytes／str端点名不匹配失败及83.513秒内正常清理的证据保留；修正后14项NIXL检查通过，与先前487项不累加。端点常驻不算转换期间归还GPU；Graph／VMM组合、真实GPU故障回退、跨节点RDMA及暂停收益仍未验收。见[本轮报告](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/kv-live-foundation-20261109/nixl-gpu-r2/REPORT.md)。

- [x] 接入真实 PP 前向与标准 token／paged allocator，保护复制中的槽位复用与重写。仅限 native／eager／background＋overlap 的在线 KV CPU shadow；最终 fence 补齐余量后复用跨 rank 传输，GDN／HiCache仍在fence交接。512项CPU检查全过；冻结 `e7c9cb1264a9` 的A100 64-token页PP2→PP3→PP2通过同暖512边界连续性和CPU前缀loadback。扩容两个源rank各在线完成662行，缩容源rank完成659行；它们是各rank的槽位行计数，不是全局输入token数。整轮含清理171.795秒，正常release。r1／r2失败保留于[验证记录](VALIDATION.md)。不据此宣称网络 overlap、Graph 接入或性能收益。

从冻结基线 `ce563696401c` 建立 `codex/pp-kv-live-foundation-20261109`，本轮不使用子 agent。[PR #4](https://github.com/paprikaw/sglang/pull/4)已在审查后合入项目，共同基线整合见总PR #5。底层原语GPU冻结源码为 `8d7c21241c`，本轮native在线shadow为 `e7c9cb1264a9`；两条执行路径分开验收。旧分工仅表示模块边界，不是正在运行的任务。

- [x] 实现 KVCacheD 按层映射协议、映射账本、部分失败重试及页面 pin；保存可重放依赖补丁、Python／原生版本检查和 stacking 边界检查。27 项依赖 CPU 检查通过；补丁在干净上游索引重放得到完全相同的源码树。见[实现与边界](KV_LAYER_MAPPING.md)及依赖锁。真实 GPU 页面验证不包含在此勾选中。
- [x] 修复 SGLang import／abort／cleanup 在校验 generation 前触碰物理页面的问题，并在 map 前检查槽位和 payload。旧基线可复现 3 个故障，新代码 769 项引擎 CPU 回归通过；保留层地址／数值仅在 CPU 适配器验证，不能据此推定真实 GPU 验收。
- [ ] 接通 HiCache、GDN 和 Attention 的完整布局约束，并验证实际 OOM／迁移失败回退。底层 GPU 的 map／unmap、保留地址和数据及页面 pin 拒绝已通过；这不替代整模型新后端兼容性或归还工作 GPU，现有 hybrid HiCache 的 native-only 保护不撤除。
- [x] 建立有限 Graph 预捕获库与持续 KV 行追补的实验模块，本轮75项CPU检查通过（25项新模块＋50项生命周期／参考资格回归）。Graph按完整tensor地址／shape／stride匹配，eager回退同样保护页面；整模型Graph尚未接入。Patch的native CPU shadow已在本轮接入scheduler，跨rank在线传输已在本轮接入NIXL，GDN增量仍待实现。见[接口与接入顺序](GRAPH_KV_PATCH.md)。
- [x] A100 `gpu-r1` 完成底层 GPU 检查：退役组驱动可见显存增加8 MiB，保留地址和数据；2次 Graph capture／2次 replay、1次地址变化 fallback及1次未映射拒绝；patch为2轮／4行，包含新增行及槽位复用。它是小规模单 GPU 检查，尚未证明完整模型、跨 rank 或在线复制的性能。
- [x] 同一租约内，开启 HiCache 的 native KV 整模型 PP2→PP3→PP2 往返通过同暖512边界的生成连续性和 CPU 前缀 loadback；整轮含清理186.509秒，Guard postcheck、清理及正常release通过。Graph在整模型中仍关闭，cold/warm等价和转换期间GPU归还未通过资格。见[验证说明](VALIDATION.md)。

## 当前：先把基线做正确

- [x] 整理 EP／PP／P/D／HiCache 共同代码基线，完成有限范围的静态 CPU 与 A100 回归。证据见 [VALIDATION.md](VALIDATION.md)；这些结果不替代动态转换验证。
- [x] 实现 resident PP 层布局往返、活跃 KV／GDN／请求进度及 CPU 前缀的转移。已有 PP2→PP3→PP2 同一暖缓存边界的连续性检查通过（r11，`c0a1a1ecb0de`）。这不证明冷／暖缓存生成等价，也不证明空阶段进程退出、GPU 已归还。
- [x] 实现三种基础策略：同步暂停、后台预搬不可变权重、暂停期间分块复制。默认基线的KV／GDN仍在最终 fence 同步；新opt-in CPU shadow的接入与验收单列在本轮条目，分块复制本身不等于在线增量同步。
- [x] 实现 CPU spill／目标池重建路径，以及 GDN 引用与清理流程修复；相关 CPU 检查及独立审查已有记录。CPU spill 验收版本为 `5fa14eae1df0`，见 [Runtime PR #2](https://github.com/paprikaw/sglang/pull/2)。
- [x] 验收同步 CPU spill 在真实 GPU 上的成功往返：r14（`5fa14eae1df0`）观察到源 KV／GDN backing 释放、目标恢复、同暖缓存边界的生成连续性及切换后的 CPU 前缀实际 loadback，清理和租约正常结束；完整监督时长 163.294 秒。独立审计 `independent_r14_audit.json`（SHA `6afd282252d8`）通过；原始结果 SHA `ff753f536a37e`。这不包含冷／暖生成等价、严格峰值比较、性能收益或转换期间归还工作 GPU。
- [ ] **B 运行、独立审查：**补齐受影响故障路径的实际 GPU 回退验证；已有 CPU 回退检查不替代真实 OOM／迁移失败场景。background、batched 的 GPU 资格也不能由 sync 的成功推定。
- [x] **主 agent：**Runtime PR #2 已以 `3a72d5781835` 合入项目分支，Cache PR #3 已在此前合入；候选组合742项CPU检查通过，独立审查未发现阻止有界 naive 基线合入的问题。引擎、测试及适配 tree 与已测候选一致。见[本次验收摘要](BASELINE_ACCEPTANCE_20261109.md)。
- [x] **主 agent：**总 PR #1 已以 `5542d19d90d8` 合入共同基线 `codex/pp-elastic-integration`；主线合并后29项定向CPU检查再次全过。冻结标签为 `baseline/sglang/pp-hicache-naive-20261109`，最终提交与远端核对保存在本次验收产物；引擎、测试及适配与742项检查的候选一致。未完成的低干扰机制与硬件资格继续保留未勾选。

## 当前：测清楚转换 overhead

- [x] 完成一次最小配对 overhead 观测：r15（`8374b0276d19`）在同暖512边界、输入及硬件下，每方向各1条无转换参考与1次 sync/cpu_spill 转换；完整监督186.775秒，正常清理／release。扩／缩 E2E 差分别 +4570.757／+3215.168 ms，各 rank 平均 fence 3389.554／3356.288 ms（非重复实验平均）。差值含目标拓扑速度及控制影响，TTFT在触发前；不作为孤立切换成本或统计性能结论。报告 `gpu-hicache-r15/REPORT.md`，原始结果 SHA `25dea5a4d093`。
- [ ] **B 运行、独立审查：**分别记录总转换时间、推理暂停／token 间隔、权重准备、KV／GDN 备份与恢复、传输字节、峰值 GPU／CPU 内存及请求 TTFT／TPOT。报告绝对 ms 和可比重复的平均值，保留原始样本；未分离的阶段标明未测，不由总时长倒推出原因。 r15已有E2E／TTFT／平均TPOT、各rank coordinator prepare／fence／total及host字节；隔离的权重／传输／恢复耗时、最大token gap、严格峰值和可比重复平均仍未测，因此本条保留未完成。
- [ ] **B 运行、独立审查：**在共同正确性条件下比较 sync、background、batched，说明后台准备隐藏了哪些开销。启动／预热和转换期间的开销分开报告，完整运行时长另列；正确性 smoke 的单次耗时不直接当作稳定性能收益。

## 已完成调研：下一阶段的接口与机制选择

- [x] 主agent核对NIXL官方API／UCX、现有SGLang NIXL P/D代码和依赖安装状态，形成[NIXL迁移方案](NIXL_KV_TRANSFER_DESIGN.md)。建议推理保留NCCL，KV迁移使用独立NIXL＋UCX和固定GPU双缓冲；这是当时的调研证据；本轮已另行隔离安装1.5.2并接入代码，运行资格见顶部当前任务，不能由本条调研推出。
- [x] 主 agent 核对 workbench vLLM/PipeLive、SGLang runtime `8374b0276d19` 及 KVCacheD 工作区，形成研究库 `LOW_INTERRUPTION_PP_REVIEW_20261109.md` 和问题簿 Q25。建议优先 KVCacheD 稳定虚拟地址，保留旧块指针内核作备选；发现按层物理映射 API 不匹配、GDN backing 重建和 stacking 边界限制。仅完成调研，无新增引擎改动或 GPU 验收。
- [x] **主 agent：**固定兼容的 KVCacheD 依赖、能力检查及 stacking 几何检查，底层A100探针确认物理页映射／回收、保留地址与数值及pin拒绝。范围和证据见上方按层映射条目；完整模型与HiCache／Graph组合仍未接通，不能由本条推定归还工作GPU。
- [ ] **主 agent：**补充提前准备 Graph 的实际B2／B4请求和逐token停顿测量。启动库已接入整模型，捕获档位1／2／4，实际B1的PP2↔PP3、无运行时capture、目标replay及暖缓存连续性已在上方验收；原生存储与工作区的常驻增量已记录。VMM容量机制、更多batch及性能收益仍未验收，详见[启动预捕获说明](PREPARED_DECODE_GRAPHS.md)。
- [ ] **B：**预初始化并预热候选布局的真实通信路径，验证路由版本、通信顺序和缓冲生命周期。
- [ ] **A：**选择固定且满足 attention 池层序号边界的 stacking 粒度。显式 preclaim 须与层所有权一致，并计入物理显存；不默认打开全局自动预分配。

## 下一步一：持续 KV patch，缩短推理中断

- [x] **主 agent：**固定隔离NIXL 1.5.2／UCX依赖，native／HiCache以固定GPU双缓冲实现基础复制、持续增量及目标应用确认；A100同节点GPU注册、CUDA IPC协议和整模型连续性已有顶部条目证据。推理保留NCCL，CPU shadow保留作对照。跨节点RDMA、完整Graph／VMM组合仍需各自资格，见[NIXL实施方案](NIXL_KV_TRANSFER_DESIGN.md)。
- [ ] **主 agent：**目标暂停前实际应用已在上述GPU往返观察到；CPU协议已覆盖槽位复用／改写、晚到代、错误payload、取消和不能确认完成时保留内存。继续补真实GPU请求新增／结束、迟到包、取消、缓冲回收及正常PP通信顺序的故障注入；追不平时限速／推迟。提交前失败保留源，不承诺提交后任意回滚。
- [ ] **主 agent：**最后只排空在途批次、补齐剩余KV和最新GDN／请求进度，再共同发布布局；HiCache保持开启，CPU-only前缀不全部恢复GPU。把完整模型Graph选择及稳定地址容量机制接入同一交接流程，代码、GPU正确性和性能分别验收。
- [ ] **主 agent：**用相同输入、缓存、源／目标及资源条件，对照同步基线和CPU shadow，记录逐token间隔、平均TPOT、尾部停顿、整请求耗时、搬运字节及峰值。必须证明准备期间继续生成、切换后连续输出和实际暂停缩短；单次正确性或脚本总时长不作为性能结论。

## 下一步二：GPU KV 池扩缩，降低转换显存需求

- [ ] **A：**核查并选取可行实现：复用 KVCacheD／VMM，或移植此前支持非连续 GPU KV 块寻址的内核／分配器。先检查与 SGLang Attention、HiCache 和 GDN 的兼容性，不要求同时实现两套。
- [ ] **A：**实现池容量与层归属变化时的按需增减和状态保留，明确 active KV、可回收缓存及 CPU 备份的处理方式。当前 native tensor pool 的重建／搬迁是对照；paged token 索引分配不等于物理显存在线扩缩。
- [ ] **B 运行、独立审查；A 提供物理账本：**测峰值显存、实际释放字节、复制量、转换时间和稳态 Attention 性能，检查非连续寻址或 VMM 的额外成本。先分别评估内存机制与 KV patch，再验证组合收益。

## 后续应用

- [ ] 真正退出不再需要的 worker 并归还 GPU；当前 resident 空阶段不计为释放资源。
- [ ] 接入 P／D 独立伸缩和状态交接，再以短动态负载验证 GPU 时间与延迟收益。复杂控制、更多模型和更多布局按前述结果追加。

**更新约定。** 每个 agent 交付前更新自己的任务条目，写明结果、版本／证据和剩余限制。子分支修改本文件后随 PR 提交；尚未含本文件或由单一负责人写入时，交付具体条目补丁，由主协调器顺序落盘。失败也要更新，未通过验收不打勾。不覆盖他人的条目，不为本页更新而重复跑已有效的检查。新增任务先落在本页，再派发；详细日志留在实验归档中。
