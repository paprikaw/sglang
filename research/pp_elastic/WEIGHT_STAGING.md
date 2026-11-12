# 加速真实权重搬运

**2026-11-12新增候选：CPU提前打包。** `cpu_prepacked`在启动Graph预热时，用同一个canonical CPU bank和原生loader在独立CPU参数上完成融合／Expert切片，直接写入pinned CPU内存；不读取源GPU值、不替换正在推理的模型参数。有限候选布局准备完成后，运行时只允许命中，按整张Parameter异步H2D，再执行原生post-load处理并检查Graph地址及完整覆盖。每worker默认24 GiB上限，三worker在Guard中额外预留72 GiB，与HiCache的20 GiB预算一起准入；不扩大HiCache本身，不改canonical bank。缓存随worker退出，不是跨进程持久缓存。

**新增计时。** CPU准备拆为pinned分配、CPU原生打包；运行时分开记录H2D壁钟／CUDA迁移流跨度、后处理及完整转换。直接checkpoint对照记录H2D调用次数、调用内壁钟及剩余CPU dispatch时间；pageable拷贝调用内含驱动整理与等待，不能称为纯DMA，也不能把异步阶段任意相加。候选GPU权重仍采用offload，硬件验收必须继续检查物理释放、完整输出、实际Graph replay及最终暂停。GPU结果另记。

**首轮诊断及修正。** 带逐调用诊断的直接路径通过往返，扩容的10,223次H2D调用内壁钟3.882秒，剩余host时间1.840秒，后处理约1 ms；诊断增加了dispatch开销，正式测速改为默认关闭的显式开关。CPU预打包首次GPU在启动期被CPU-only检查拒绝：GemmaRMSNorm原生loader还写入绑定模块的gemma_weight缓冲。现在为加载器建立独立CPU owner，连同该派生buffer一起准备、计入CPU预算并H2D恢复；不放宽CPU-only检查。失败78.894秒内完成owned cleanup和正常release，135项输入一致，没有运行时性能样本。未启动的旧配置保留为not_run，新版本另建tag／run。

**2026-11-12结果。** 同一069节点、同源`521b14cb725e`、35B BF16／B1／HiCache／NIXL／预热Graph，直接CPU bank的完整扩／缩5.089／2.978秒，旧路径20.103／20.885秒，分别减少74.7%／85.7%。最后交接仍为242／258 ms，客户端跨fence相邻token间隔269／273 ms；主要缩短后台权重装载，没有消除最后同步。每方向恢复约21.87 GB；GPU导出、Gloo权重传输及候选权重复用均为0。缩回中间GPU物理权重为0，驱动可用显存增加20.37890625 GiB。两策略的2048-token输出逐一一致；每策略一次往返，未扩大到重复均值或其他模型／batch。[完整指标、来源与限制](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_direct_cpu_cache_weights_20261111/summary.md)。

**新增：直接使用已有CPU缓存。** 用户要求避免源GPU导出。显式选择`--elastic-pp-weight-staging cpu_cache`后，各目标rank直接打开初次加载使用的canonical CPU bank，只读取自己新接手的层，调用模型原生checkpoint loader将融合／切片后的值写入目标既有参数地址。源rank不发送权重，不经Gloo，也不新增常驻GPU候选副本或从GPU建立新的CPU权重快照。当前资格范围是native Qwen3.5、TP1／EP1、Triton、无量化、overlap；运行时使用background，启动Graph预热的同步转换也使用同一CPU bank。通用默认暂保留legacy，实验显式选择新路径。

**正确性边界。** 核对初次加载的cache key和完整checkpoint元数据，保持daemon pin及mmap至CUDA流结束，确认目标参数全部覆盖、对象／地址／shape／stride不变后才标记就绪。CPU bank缺失或身份变化直接失败，不回退磁盘或源GPU。Graph预热和候选物理权重页释放继续独立生效。计时区分打开缓存、原生加载（含CPU打包及H2D）、prepare、最后暂停和完整切换；原生加载时间不能解释为纯PCIe带宽。

目标层同样执行原生post-load处理（例如可选的BF16 SwiGLU交错），处理后再核对地址；不重处理活跃层。候选层映射失败时保留Graph缓存模块的原生loader，允许在原地址重试。3项新增检查已在修正前复现遗漏，原失败记录保留。

**保留的GPU发现及修正。** 首轮初次加载的三rank缓存记录通过，但启动Graph预热在覆盖检查处退出。GDN的A_log／dt_bias同时注册在外层和RadixLinearAttention，原生loader仅报告一个名字；现在按同一Parameter对象核对覆盖，不能仅凭共享地址或storage认定已加载，实际加载字节同样按对象去重。新增回归在修正前复现两个拒绝，修正后本文件22项、完整CPU 1084项通过。首轮101.123秒内正常清理并release，134项输入一致；修正后新候选和同源旧路径对照均通过，分别163.951／168.954秒内完成监督清理，134／131项输入一致。下述旧节点legacy数据仍作历史对照，不覆盖新实测。

**问题。** 原先约20秒测的不是纯CPU加载：每次重新从源GPU导出，经普通CPU内存／Gloo，再逐张量装入目标GPU。35B BF16每方向实际搬21.87 GB；这条链路的耗时不能直接当作PCIe的CPU→GPU带宽。

**本轮改动。** 增加 `--elastic-pp-weight-staging pinned`：每进程复用256 MiB页锁定CPU缓冲，打包为有界字节块，在独立迁移流异步拷贝，经过Gloo后直接写入目标参数，保持稠密转置的物理布局。缓冲在网络与CUDA读取结束后才可复用，整层完成后才标记就绪。候选层仍释放GPU物理页；不增加候选GPU权重副本。旧pageable路径用 `legacy` 保留，并补D2H、传输、H2D计时。

**验收。** 先检查跨块／混合dtype／转置、错误generation与布局拒绝、目标地址保持、失败排空和整层提交；再在同一A100节点按原PP2→PP3→PP2协议对照。比较完整切换、最后暂停、搬运分项及请求TPOT，检查暖缓存完整输出、Graph replay、非活跃权重页为0和实际释放显存。每轮含清理最多300秒；未拿到结果前不宣称提速。

CPU缓存预算另含每worker最多256 MiB暂存；当前三worker最多768 MiB，计入原20 GiB主机池预算。没有实现长期CPU权重快照、本机共享内存绕过Gloo或权重NIXL直传；若Gloo成为瓶颈，再据分项数据选择。当前范围及结果继续更新[TODO](TODO.md)。

**已取得的证据。** 冻结 `fcefe8604c65`，1036项完整CPU检查通过，120.55秒。相同源版本／A100节点的legacy对照通过暖缓存连续性、实际Graph replay、非驻留权重及正常清理；完整扩／缩19.758／20.824秒。缩容源rank1的D2H累计10.007秒，发送及等待10.235秒；两个目标rank0／2的H2D各1.026／0.861秒。前两项说明导出和中转占大头；发送／接收含对端等待，不是独立链路带宽。监督含清理194.694秒，130个冻结输入一致。

**pinned中转路径尚未验收。** 其首个GPU尝试因两台A100被另一正常前台实验占用，被Guard在模型启动前拒绝；后续pinned候选未运行。保留原记录，不能把上面的直接CPU bank结果作为pinned／Gloo路径的GPU资格。[PR #10](https://github.com/paprikaw/sglang/pull/10)记录当前范围，尚未合入共同基线。[历史中转诊断](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_pinned_weight_staging_20261111/summary.md)。
