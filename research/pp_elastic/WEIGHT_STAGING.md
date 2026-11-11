# 加速真实权重搬运

**新增：直接使用已有CPU缓存。** 用户要求避免源GPU导出。显式选择`--elastic-pp-weight-staging cpu_cache`后，各目标rank直接打开初次加载使用的canonical CPU bank，只读取自己新接手的层，调用模型原生checkpoint loader将融合／切片后的值写入目标既有参数地址。源rank不发送权重，不经Gloo，也不新增常驻GPU候选副本或从GPU建立新的CPU权重快照。当前资格范围是native Qwen3.5、TP1／EP1、Triton、无量化、overlap；运行时使用background，启动Graph预热的同步转换也使用同一CPU bank。通用默认暂保留legacy，实验显式选择新路径。

**正确性边界。** 核对初次加载的cache key和完整checkpoint元数据，保持daemon pin及mmap至CUDA流结束，确认目标参数全部覆盖、对象／地址／shape／stride不变后才标记就绪。CPU bank缺失或身份变化直接失败，不回退磁盘或源GPU。Graph预热和候选物理权重页释放继续独立生效。计时区分打开缓存、原生加载（含CPU打包及H2D）、prepare、最后暂停和完整切换；原生加载时间不能解释为纯PCIe带宽。

目标层同样执行原生post-load处理（例如可选的BF16 SwiGLU交错），处理后再核对地址；不重处理活跃层。候选层映射失败时保留Graph缓存模块的原生loader，允许在原地址重试。3项新增检查已在修正前复现遗漏，原失败记录保留。

**本次验证。** 新路径CPU检查进行中；A100完整往返待运行。下述legacy测量保留为历史对照，不是直接CPU加载的测量结果。

**问题。** 原先约20秒测的不是纯CPU加载：每次重新从源GPU导出，经普通CPU内存／Gloo，再逐张量装入目标GPU。35B BF16每方向实际搬21.87 GB；这条链路的耗时不能直接当作PCIe的CPU→GPU带宽。

**本轮改动。** 增加 `--elastic-pp-weight-staging pinned`：每进程复用256 MiB页锁定CPU缓冲，打包为有界字节块，在独立迁移流异步拷贝，经过Gloo后直接写入目标参数，保持稠密转置的物理布局。缓冲在网络与CUDA读取结束后才可复用，整层完成后才标记就绪。候选层仍释放GPU物理页；不增加候选GPU权重副本。旧pageable路径用 `legacy` 保留，并补D2H、传输、H2D计时。

**验收。** 先检查跨块／混合dtype／转置、错误generation与布局拒绝、目标地址保持、失败排空和整层提交；再在同一A100节点按原PP2→PP3→PP2协议对照。比较完整切换、最后暂停、搬运分项及请求TPOT，检查暖缓存完整输出、Graph replay、非活跃权重页为0和实际释放显存。每轮含清理最多300秒；未拿到结果前不宣称提速。

CPU缓存预算另含每worker最多256 MiB暂存；当前三worker最多768 MiB，计入原20 GiB主机池预算。没有实现长期CPU权重快照、本机共享内存绕过Gloo或权重NIXL直传；若Gloo成为瓶颈，再据分项数据选择。当前范围及结果继续更新[TODO](TODO.md)。

**已取得的证据。** 冻结 `fcefe8604c65`，1036项完整CPU检查通过，120.55秒。相同源版本／A100节点的legacy对照通过暖缓存连续性、实际Graph replay、非驻留权重及正常清理；完整扩／缩19.758／20.824秒。缩容源rank1的D2H累计10.007秒，发送及等待10.235秒；两个目标rank0／2的H2D各1.026／0.861秒。前两项说明导出和中转占大头；发送／接收含对端等待，不是独立链路带宽。监督含清理194.694秒，130个冻结输入一致。

**尚未完成。** 新路径首个GPU尝试因两台A100被另一正常前台实验占用，被Guard在模型启动前拒绝；后续候选保持未运行，不打断该实验。原记录保留，不能宣称新路径GPU正确或提速。[PR #10](https://github.com/paprikaw/sglang/pull/10)保持Draft，未合入共同基线。[专题报告与指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_pinned_weight_staging_20261111/summary.md)。
