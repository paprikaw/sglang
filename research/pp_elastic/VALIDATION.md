# 2026-11-14：CPU确认完整收发前移

## 2026-11-14：122B P/D 的 KVCacheD 容量接入，GPU 待验收

r3已通过启动 idle 容量检查，但首个 P→D 暖请求失败：Mooncake 对 KVCacheD VMM 区域注册报 `Bad address [14]`。没有请求正确性或扩缩结果；owned cleanup 与租约释放通过。候选修复在 `cuMemCreate` 显式请求 GPUDirect RDMA，并检查设备支持；原生扩展已在已预约 A100 上重编译，依赖补丁与 ABI 已固定。容量模式缺失该 ABI／环境标志时拒绝启动，注册地址注册失败立即报错。108项相关 CPU 检查通过；RDMA 修复仍待新 GPU 实验验证。

首轮 GPU r2 在启动的两个布局 Graph 预捕获后，被正常 idle 内存检查拒绝：hybrid SSM 检查仍读分配器虚拟 `size=12288`，而实际可用8192。未关闭检查；将 allocator.size 改为提交容量、另存 virtual_capacity，并在提交时刷新 frozen pool observer 与检查器。新增真实 idle invariant 及统计 owner 更新的 CPU 检查。r2 未服务正式请求、未验证容量扩缩，两个节点的 owned cleanup 已通过。

独立任务分支 `codex/qwen122b-pd-kvcached-capacity-20261114`，依赖 [P/D 接入 PR #14](https://github.com/paprikaw/sglang/pull/14)。沿用其 `2a8f799737` 中撤回未验收调度改序的执行顺序；不采用此前失败的多请求试验作基线。固定 P2，D 的容量前沿为 PP2 8192／PP3 12288 token，Graph 地址按最高容量预留。注册真实 packed K/V 行步长；HiCache、GDN、CPU 权重 offload、准备期权重恢复和 NIXL 保留。

首次相关组合165项CPU通过。扩大检查891项通过、7项异步检查因插件关闭跳过；唯一失败是旧错误文案断言（无 frontier 仍被正确拒绝），随后更新断言并复测相关控制与容量路径。KVCacheD 扩展已在已预约167节点按 A100 架构、MAX_JOBS=1 编译；使用独立流初始化新物理页。源码与完整补丁固定于 `dependencies.lock.json`。这些不代表真实GPU扩缩验收。

最小验证及后续比较见 [KVCacheD P/D 容量记录](KVCACHED_PD_CAPACITY.md)。要求真实请求用到 >8192 槽位、缩回回收物理页、HiCache CPU 恢复及独立显存复用；通过后才开展短 AgentX＋BurstGPT。


r8仅前移发送仍复现：rank2在确认send完成处等待rank0，rank0等激活send，空rank1等输出。候选进一步将三类CPU确认的接收和发送完成等待也前移，实际队列更新仍在原模型／输出之后应用，GPU发送工作仍由原对象持有；静态PP路径不变。新增有待接收确认／空环、忙／空闲、各rank的组合测试，114项相关CPU检查通过。有限流量结束并缩回后，再对相同4608-token暖输入核对8个输出。GPU试跑前未取得多请求或端到端资格。

# 2026-11-14：空阶段多请求P/D循环等待

r7保留CPU-only查询仍在正式流量开始后停滞，排除仅CUDA显存查询造成超时的解释。三个D rank的watchdog栈分别在旧激活send完成、输出元数据recv、P/D队列consensus recv；控制环经过空阶段而激活绕过它，形成循环。候选修复将弹性路径的三类CPU队列确认发送提前到任何激活／输出等待之前；仍在原位置完成发送、保留P2P缓冲区所有权，普通静态PP时序不变。新增忙／prebuilt槽测试，GPU配对资格待验证；不把此前单请求资格扩为多请求成功。

# 2026-11-14：短 reactive 试跑诊断

r6完成4608-token长Prefill启动检查、两个AgentX暖会话预热并进入正式流量，随后10秒状态查询超时；停止前Decode仍有输出，无伸缩完成。GPU进程清理与Guard缓存审计通过，正式数据不完整，不用于性能比较。新诊断候选将高频状态查询中的CUDA显存遍历移出、查询deadline改30秒，并为调度停滞保留原生watchdog栈；尚未证明查询是根因。静态PP2对照不安装PP3专用诊断通信。

# 2026-11-14：122B P/D 权重恢复与在线 Decode 重叠

r23冻结`5856c9ea8528a13decdc3393f8e782dcdfcba3d1`／tag `experiment/sglang/122b-pd-pp-eager-modules-20261114-r23`，固定P2、D PP2→PP3→PP2，HiCache／Graph／真实CPU权重offload保留。D启动预加载CUDA模块和数据，驱动确认三个rank均EAGER；与r20的`python/sglang`完全相同。新增rank1恢复38.04 GiB／1.671秒，同一RID输出12→101；准备期最大间隔34.0 ms，整轮最大177.8 ms，之前r20为1684.2 ms。扩／缩控制1.917／1.279秒、fence124.0／103.0 ms，客户端跨fence177.8／127.0 ms。每方向一次；未单独计量预加载冷启动／显存增量。

512-token同暖参考逐个一致、无重算／retraction，新P→D、CPU恢复576 token、Graph运行时capture=0及原Decode继续时独立64 GiB复用通过。退役rank物理层权重／KV为0，仍allocated1.03／reserved1.78 GiB，worker未退出。Guard后检查、CPU缓存审计、owned cleanup及release通过，未运行完整E2E。

130项相关CPU检查覆盖r20恢复完成前不得发布、查询错误及中途copy失败后排空；r20至r23引擎代码相同，不重复跑未变套件。r21／r22仅增加opt-in原生采样，CPU精确PID ptrace smoke和外部定时在父进程持GIL时采样通过；GPU两个诊断往返亦通过，但采样耗时不作性能重复。r22栈显示NIXL的首次gather卡在cudaLaunchKernel驱动路径、权重线程query等runtime mutex；r23只前移模块／数据加载后长停顿消失，支持首次加载同步解释，未逐一拆分内部驱动调用。r19原退出根因仍未定。五轮成功原件的512-token序号、511个间隔、两个fence、HiCache／Graph、目标权重／KV归零和数据哈希已复算。[简洁报告及图](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/WEIGHT_DECODE_OVERLAP.md) · [原值及SHA256](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_pd_weight_decode_overlap_20261114/overlap_evidence.json)。P侧、多请求、在途交接、固定容量外的扩展、完整E2E及整卡退出资格保持未验证。

# EP／PP／P/D 集成基线验证

## 2026-11-14：迁移开销审计与准备期长间隔定位

只新增opt-in CPU栈／心跳诊断，r19源码`54ab8e4b53722dc1c961dc0b1d7c32c08c7f0f0e`、tag `experiment/sglang/122b-pd-pp-overhead-20261114-r19`；`python/sglang`与r18完全相同。辅助函数返回／异常传播和清理的两种CPU smoke通过，未新增完整CPU套件或引擎资格。复算r16／r18成功原件的511个相邻token间隔与分项：新扩容准备期1.683秒，新增rank1的38.04 GiB H2D为1.673秒；最终fence128.9毫秒发生在稍后，不是该长间隔本身。

r19扩容控制1.946秒、fence127.6毫秒，三rank各8个200毫秒栈样本：rank1主线程在GPU输出irecv，后台线程等权重流完成，其他rank等P/D消息；CPU心跳最长约20毫秒。尚未隔离NCCL／CUDA内部因果。随后缩容准备时rank0退出、HTTP RemoteDisconnected；整轮失败，不能用于新的正确性资格、完整往返性能或稳定性结论。Guard后检查科学成功false、缓存审计及清理true，租约已释放。失败根因未定，不自动追加GPU复测。见[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/MIGRATION_OVERHEAD_AUDIT.md)及[可复算数据／SHA256](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_pd_migration_overhead_audit_20261114/overhead_audit.json)。下列r18成功资格保持原范围，不把r19失败隐藏为性能样本。

## 2026-11-14：122B P/D接入最新共享HiCache／Graph／CPU权重优化

冻结`dccbefa3c9c76ff33e9fdcd8b572a5bb62ee2e2f`，tag `experiment/sglang/122b-pd-pp-optimized-20261114-r18`，Draft PR #13。固定P2、D侧24/0/24→16/16/16→24/0/24、FP8 Marlin TP1 EP1、单请求及交接静止边界；保留HiCache与确定性模式，启用启动Graph库、非驻留CPU权重、共享CPU HiCache及后台NIXL增量KV。512-token同暖输出逐个一致、无retraction／重新Prefill，新P→D请求及实际CPU恢复576 token通过；共享底层未改变、历史CPU载荷复制0、运行期Graph capture0且有replay、非活跃权重物理页0。独立进程分配并写入64 GiB及矩阵计算时原Decode继续。

控制调用扩／缩1.940／1.262秒；最终暂停128.9／103.3毫秒，各rank协调器完成最长1.930／1.276秒；每方向一次。准备期仍有1.683秒出token间隔，不能宣称转换成本完全隐藏。启动预热约28.2秒、三rank处理后CPU权重池实际19.02／38.04／19.02 GiB另计。缩回后中间rank层权重映射0、KV0，allocated约1.03 GiB、reserved约1.78 GiB，worker未退出。活跃4→5→4卡，驻留worker5卡，Guard整节点8卡。1275项组合CPU检查、启动VMM逐层回收修复后的74项相关检查、73项保存结果复算及Guard后检查／缓存审计／清理通过；r17启动OOM原失败保留。P侧、多请求、在途交接、容量增益及完整E2E未验证。[结果与原始证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/OPTIMIZED_PP_VALIDATION.md)。

## 2026-11-14：122B A100原生P/D下的最小D侧PP扩缩

冻结 `d8c614d14e6c7844df780a8f513e16f3afad7e1e`，tag `experiment/sglang/122b-pd-pp2-pp3-roundtrip-20261114-r16`，Draft PR #13。完整Qwen3.5-122B-A10B-FP8／A100 Marlin W8A16，EP1、Eager、HiCache每rank 8 GiB、确定性模式、同步CPU spill；P固定PP2，D为24/0/24→16/16/16→24/0/24。最终移除逐层同步和Expert复算钩子后，静态重复、同一请求跨两次转换的512-token完整暖输出、新P→D交接和CPU恢复后32-token输出均与相应暖参考逐个一致，无retraction／重新Prefill。两个活跃D rank实际从CPU恢复576 token。182项相关CPU检查与28项保存结果复算通过；不扩大为冷暖生成等价、真实任务质量或长期稳定性。

缩回后独立进程在GPU1分配／写入64 GiB、矩阵乘并保持5秒，原Decode同期继续；中间rank的KV为0，allocated约1.51 GiB、reserved约2.98 GiB，控制worker与边界张量仍驻留，因此是实际显存复用资格，不是完整worker退出。扩／缩控制8.573／8.061秒，fence约8.306／7.747秒，各一次；不作为重复均值或性能收益。固定P2加上D的活跃计算卡数4→5→4，物理worker5卡，Guard整节点8卡。4183项冻结输入、缓存使用审计、owned清理及Guard postcheck通过，租约正常释放。P伸缩、在途P→D、多请求、Graph、可用KV容量增加及E2E尚未验证。r1–r14原失败与带诊断的r15成功原样保留。[精简报告与证据](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen122b_pd_elastic_roundtrip_20261114/summary.md)。

## 2026-11-13：共享CPU HiCache，有限同源对照通过

PR #10已以`8570a9153a76`合入项目。新任务[PR #11](https://github.com/paprikaw/sglang/pull/11)的冻结源码`d9a5731c1222f7b77d3a9c497da005ffd08fc94f`／tag `experiment/sglang/pp-hicache-shared-host-20261113-r2`通过1206项完整CPU检查，零失败／错误／跳过，监督139.131秒，源码未变。Black通过；两处既有import的isort欠账在基点也存在，不宣称全仓CI通过。

同A100节点、35B BF16／B1、HiCache、NIXL、CPU预打包权重、候选权重offload、启动Graph，只改变共享CPU开关，各一次PP2→PP3→PP2。私有→共享扩／缩最终暂停105.2／88.5→43.8／39.0 ms、完整转换1.342／0.879→1.007／0.610秒；客户端跨fence间隔129.6／102.0→70.6／52.4 ms，准备期最大间隔74.1／96.2→26.5／58.0 ms。共享模式各rank历史CPU快照／发送／复制均0，物理region身份贯穿往返一致；每方向2048输出与各自同进程暖参考及配对模式逐token一致，真实CPU前缀loadback、在线KV、Graph replay和runtime capture=0通过。非活跃GPU权重页为0，缩回中间GPU释放20.37890625 GiB权重显存。共享／私有142／138项冻结输入匹配，监督含清理242.341／189.582秒，owned清理、节点GPU空闲观测及正常release通过。每模式一次，不作重复均值或整体吞吐结论。

首版`e8cea6e148b1`的A100启动被额外L3 Gloo建组阻塞，没有请求或性能结果，254.904秒内清理及release通过。修正版L2-only视图不建这类组，空阶段采用共享容量；失败及此前两项CPU替身接口失败均保留。当前限定单节点TP1／EP1 native page_first/kernel L2，CPU容量固定；独立review、更多模型／并发、跨节点、冷暖生成等价及整卡归还未扩展。见[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_shared_host_20261113/summary.md)及[接入说明](SHARED_HOST_CACHE.md)。任务PR仍Draft，总PR #12尚未包括该变更。

## 2026-11-13：CPU批量HiCache及旧GPU KV基础复制

两组独立同源开关对照均为同A100节点／35B BF16／B1／TP1 EP1 Triton、HiCache／CPU预打包权重／NIXL／启动Graph／候选权重offload，每模式一次PP2→PP3→PP2、每方向2048输出。第一组冻结a0b42990d36e37459ae1e4dc3353fc75ff96b116／tag experiment/sglang/pp-hicache-batch-io-20261113-r1，1173项CPU检查，批量I/O将扩／缩最终暂停113.3／184.4→104.7／138.0 ms。第二组冻结67df645bc2b833400405d19026959f57f56f260f／tag experiment/sglang/pp-hicache-batch-io-20261113-r2，1181项CPU检查全过、零失败／错误／跳过，pytest117.830秒、监督134.252秒、源码不变。两边保留CPU批量，仅切换旧KV基础复制，缩容最终暂停136.0→87.0 ms，完整转换0.885→0.844秒，客户端跨fence间隔186.2→100.9 ms；最终待补2103→55行、复制等待50.5→6.1 ms，接收确认约5.9 ms。扩容103.8→106.1 ms，无额外提速。

精确CPU内容核验结果只在同一次冻结交接复用；初始GPU KV仅在真实HiCache读写静止后记录版本，第一forward完成事件后发布。新增异步loadback、回收复用和同槽位改写仍受版本与ready事件保护。第二组准备期最大间隔扩／缩74.6／131.2→72.7／135.7 ms，缩容平均11.73→11.85 ms，未改善；快照在三个rank上先后占调度线程33–37 ms，时钟重叠不等于隔离因果。旧67.5 ms属于在线KV补拷／确认，不是GDN本身。候选非活跃物理GPU权重为0，缩回驱动可用显存增加20.37890625 GiB；KV／GDN／Graph工作区和worker保留，整卡归还未扩展。两组输出分别与各自暖参考和模式逐token一致，CPU前缀实际恢复、实际Graph replay／运行时capture=0通过。四轮各138项冻结输入已在后续文档修改前核验，监督164.453／186.501／168.357／173.466秒正常清理并release。新路径无GPU失败，主agent实施及自行审查，无新独立reviewer；两组不混为重复样本或整体吞吐收益。见[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_batch_io_20261113/summary.md)和[统一清单](TODO.md)。

## 2026-11-13：CPU HiCache后台准备与增量交接

冻结 b5467d3a432d23be55538f78d40f38f30259a52b，tag experiment/sglang/pp-hicache-background-20261113-r3。1162项完整CPU检查通过，零失败／错误／跳过，pytest109.334秒，监督124.924秒，源码未变。同源A100／35B BF16／B1，各模式一次正式往返，扩／缩暂停157.9／193.6→115.2／189.6 ms；客户端跨切换间隔183.7／207.3→141.0／203.2 ms。准备期间最大token间隔25.0／22.5→76.4／67.8 ms，完整扩／缩1.115／0.725→1.336／0.912秒。每方向稳定CPU载荷117,309,440 B提前发送，最终CPU补传0 B，保留层不再重写；内容仍逐行核验，最新树／分配器／请求状态仍在fence取得。独立目标池使用迁移CPU组做容量同步；源池保留到发布，失败丢弃目标。代际临时引用被重置。1162项完整CPU检查、两个方向各2048输出及两模式逐token一致、实际CPU前缀恢复、Graph replay和正常清理通过；候选GPU物理权重仍为0。初始CPU快照和最终内容核验仍消耗调度／暂停时间，缩容另有最新GDN／KV收尾；不宣称无中断、整体吞吐提高或整卡归还。

两模式138项冻结输入各自一致，后台／对照监督173.578／165.782秒完成正常清理与release。首个pre-worker入口失败、首个安全回退后台往返及CPU夹具／选择器错误单列，不混入正式性能比较。主agent实施与自行审查，无新增独立reviewer。[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_background_20261113/summary.md)。

## 2026-11-12：HiCache复制与最终暂停优化

冻结`7dda6e42a012bc8e02a5424d245ff7f2ec72fc8c`／tag `experiment/sglang/pp-hicache-fast-copy-20261112-r1`。1130项完整CPU检查通过，零失败／错误／跳过，pytest135.87秒，含监督168.453秒，源码未变化。新CPU检查覆盖连续／稀疏／非连续stride、快照独立、源storage别名拒绝、共享目标回滚、私有导入失败清理；首轮专项夹具缺少layer_range，修正后34项专项通过，之后完整回归通过。

同一A100节点、35B BF16／B1／native TP1／EP1 Triton／HiCache／NIXL／prepared Graph／CPU预打包／offload，禁用与启用fast-copy各一次正式PP2→PP3→PP2通过。完整扩／缩1.205685／0.825684→1.118385／0.723485秒；fence253.351／296.300→159.983／195.916 ms，客户端跨已记录fence的相邻token间隔281.795／310.296→184.816／210.630 ms。客户端间隔含正常Decode、排空及SSE，非纯新增暂停；各rank分项并行且含等待，不相加最大值。目标接收行导入111.0→20.7 ms；初始化31.1→30.8 ms，Graph选择仍约2 ms。

每方向省去117,309,440字节目标行的undo备份，按逻辑载荷计；新目标storage与源不重叠，源保持到发布，失败丢弃目标。CPU前缀快照、发送、接收和保留行字节数逐rank一致，HiCache每rank 1 GiB、CPU权重上限24 GiB及联合Guard预算92 GiB不变。两个方向各2048输出在两模式及各自暖参考间一致，实际Graph replay、runtime capture=0、在线KV及最终CPU前缀loadback通过。非活跃物理权重为0，缩回中间GPU注册参数及物理权重为0、驱动可用显存增加20.37890625 GiB；KV／GDN、工作区和worker仍保留，整卡归还未扩展。正式新／旧监督190.686／179.790秒内清理并正常release，136项冻结输入各自一致。两边均有相同的启动freeze_gc自请求未连通警告，不影响上述验收，原日志保留。

初轮fast-copy通过正确性与清理，但未开客户端时间戳，仅保留159／205 ms服务端计时；原未启动对照及新正式ID均保留，不混入重复均值。后续仅补充prepare对HiCache源文件的自动冻结，实际GPU测量tag不变；本轮已在启动前手动补齐相同文件。更多模型／batch／量化、冷暖生成等价、跨节点RDMA和独立review未扩展。[报告与全部来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_fence_copy_20261112/summary.md)。

## 2026-11-12：CPU提前打包，有限GPU对照通过

冻结`20d7e1970e980fb2812c8c5e395e4a7c6c911be4`／tag `experiment/sglang/pp-cpu-prepacked-20261112-r3`。完整CPU清单1117项通过，零失败／错误／跳过，pytest118.44秒，含监督139.327秒，源码未变化；覆盖独立CPU参数／Gemma派生buffer、原生覆盖、只在CPU操作、预算先行、warm hit、Graph地址与同步启动共识。证据为`weight-cpu-prepacked-20261112/cpu-full-r3.*`，此前1112／1113项通过不累加。专项Ruff通过，server_args的既有F722/F841未作为本次修复范围。

同一A100节点、35B BF16／B1／TP1／EP1／Triton／HiCache／native KV／持续NIXL patch／启动Graph／offload，`cpu_cache`与`cpu_prepacked`各一次PP2→PP3→PP2通过，两侧正式测量都关闭逐调用诊断。13层单GPU装载4.584→0.896秒，完整扩／缩5.139196／3.025319→1.221063／0.833178秒；最后fence241.415／273.741→264.222／263.587 ms，客户端跨fence间隔265.995／288.075→289.187／319.701 ms。该间隔含正常Decode及SSE，不代表纯新增暂停。准备更早使接班位置改变，混合源／目标布局的请求TPOT不用于隔离干扰结论。

优化后218次H2D、壁钟0.894秒，运行时CPU pack为0；启动接收worker分配6.026秒＋pack5.843秒，总准备11.869秒。三worker额外CPU张量40.739 GiB，每worker上限24 GiB，联合Guard预留92 GiB；canonical bank和引擎HiCache容量不变。两策略每方向2048输出token逐一一致、暖参考连续性、实际Graph replay、runtime capture=0、在线KV及最终CPU前缀loadback通过。没有GPU权重导出／Gloo权重传输／候选权重复用；缩回中间GPU注册模型参数和物理权重为0，驱动可用显存增加20.37890625 GiB。两边135项冻结输入各自一致；优化／对照监督176.887／162.372秒内正常清理并release。

初次CPU预打包GPU在启动期被CPU-only检查拒绝：GemmaRMSNorm的绑定loader还写GPU派生buffer，78.894秒内清理并release，135项输入一致。修正为独立CPU owner和buffer并计入预算，不放宽CPU-only检查；原失败及两份not_run配置保留。单独诊断的10,223次拷贝／3.882秒调用内壁钟／1.840秒剩余host路径含pageable整理、等待及诊断成本，不作正式对照。更多模型／batch／量化、冷暖生成等价、跨节点RDMA、KV VMM组合与整卡归还未扩展；无独立reviewer，PR保持Draft。[专题报告、CPU及GPU来源](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_cpu_prepacked_weights_20261112/summary.md)。

## 2026-11-12：CPU缓存直接恢复，GDN参数别名修正

最终冻结`521b14cb725e0429b54999f1622250e65a7bc154`，tag `experiment/sglang/pp-direct-cpu-cache-weights-20261112-r2`。完整CPU回归1084项通过，零失败／错误／跳过，pytest 115.32秒；源码未变化。同一069节点、35B BF16／B1／native／HiCache／NIXL／prepared Graph、offload，直接CPU bank和同源legacy各一次PP2→PP3→PP2通过。完整扩／缩分别5.088962／2.978069秒与20.102884／20.884594秒；最后交接分别241.622／258.253 ms与231.269／287.113 ms。客户端跨已记录fence的相邻token到达间隔分别269.381／272.757 ms与257.468／301.757 ms，含正常Decode和传输，不是额外暂停的精确扣除值。

两个方向各2048个输出token在两种路径间逐一一致，同版本暖缓存参考、实际Graph replay、运行时capture=0、在线KV及最终CPU前缀loadback通过。直接路径无源GPU导出／Gloo权重传输／GPU候选权重复用；缩回中间GPU模型参数及物理权重页为0，驱动可用显存增加21,881,683,968字节。直接／legacy的134／131项输入一致，监督163.951／168.954秒内正常清理并release。仅该模型／batch和一次往返；接班token位置随准备速度变化，不能把请求TPOT差异全归因于较少推理干扰。冷暖生成等价、更多batch／模型／量化、跨节点RDMA及整卡归还未扩展，无独立reviewer。精简报告、CSV、配置差异及跨路径输出核对见[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_direct_cpu_cache_weights_20261111/summary.md)。

`2b77805f03c4`的1081项去重CPU检查均有通过记录：整批1072通过，9项因测试机cgroup内存余量耗尽被缓存daemon拒绝后，在相同源码独立复核全部通过，未放宽安全检查。不能称为一次零失败整批运行。

069节点已有35B BF16 canonical CPU bank，首轮在启动Graph预热时，原生loader报告名称与GDN的重复注册Parameter别名不全相等，被覆盖检查拒绝。三rank初次缓存读取、134项冻结输入、owned cleanup和正常release通过；完整监督101.123秒，没有运行时转换性能结果。按Parameter对象核对别名覆盖并去重加载字节，仍检查完整checkpoint及所有对象／地址／shape／stride，仍拒绝仅共享storage的独立Parameter。新增三项回归中，两项在修正前失败，修正后本文件22项通过。原始记录位于`weight-direct-cpu-cache-20261111/gpu-direct-069-r1-20261112`；完整回归及GPU修正候选另记，不覆盖失败记录，无独立reviewer。

## 2026-11-11：有界pinned权重暂存，CPU通过／新GPU路径待验证

任务分支 `codex/pp-weight-host-pipeline-20261111`，冻结 `fcefe8604c659180075fae0d4939dec49eb603ad`／tag `experiment/sglang/pp-pinned-weight-staging-20261111-r1`。1036项完整CPU检查通过，零失败／错误／跳过，120.55秒；新增12项包含跨块、混合dtype、转置、错误元数据拒绝、目标地址保持、失败排空与完成后标记。首次测试夹具错误保留，修正后完整回归通过。

同一A100节点的legacy对照通过完整暖缓存PP2↔PP3往返，控制面扩／缩19.758／20.824秒；缩容源D2H约10.007秒、CPU发送及等待10.235秒，目标H2D各1.026／0.861秒。监督含清理194.694秒，Guard清理／正常release与130个冻结输入核对通过。pinned尝试在模型启动前被其他正常前台占用阻止，保留未启动记录；没有新路径GPU正确性、带宽或提速结论。[设计与状态](WEIGHT_STAGING.md)。本轮主agent直接实施／审查，无独立reviewer，PR保持Draft。

## 2026-11-11：Graph预热与权重物理页释放分开验收

冻结 `26bdee39146f103716d23749b5b08f523789a12e`，tag `experiment/sglang/pp-graph-nonresident-weights-20261111-r3`。35B BF16、A100、实际B1、native KV／HiCache／NIXL，默认offload的两次独立PP2→PP3→PP2往返通过。完整扩／缩均值19.98／20.65秒，后台权重传输19.68／20.30秒，最后暂停239／285 ms；每方向实际搬21,871,498,496字节，权重复用字节为0。当前权重为Gloo的GPU→CPU→GPU路径，NIXL用于在线KV。

中间GPU缩回后的decoder物理权重页及注册模型参数均为0；每轮释放21,881,683,968字节（20.37890625 GiB），驱动可用显存同步增加相同字节数。所有rank的非活跃层物理权重页均为0。两方向2048-token完整输出与同版本同进程暖缓存参考一致，目标实际Graph replay、活跃Decode eager=0、运行时capture=0、在线KV应用及最终CPU前缀loadback通过。各轮129个冻结输入哈希一致；监督含清理177.091／178.488秒，Guard postcheck、owned cleanup与正常lease release通过。最终1024项完整CPU回归通过，零失败／错误／跳过，114.57秒；此前通过数不累加。代码由主agent自行实施与审查，没有新增独立reviewer。

这是权重显存释放资格，KV／GDN、Graph工作区和worker上下文仍保留，`gpu_released=false`。冷暖生成等价、更多模型／batch／量化、KV VMM组合、跨节点RDMA及GPU故障恢复未验证；当前只检查了35B初始化峰值。两轮仅作描述性重复。历史resident版本约0.3秒不能支撑资源节省主张。[完整报告及绝对指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_nonresident_graph_weights_20261111/summary.md)。

## 2026-11-11：候选权重复用与HiCache分配预热

任务分支 `codex/pp-migration-latency-20261111` 从共同基线 `5f530f6de33d` 派生，由主agent直接实施、测试和审查，没有新增独立reviewer。`b59993a97f28` 增加不可变候选权重复用、recopy对照、统一策略确认、在线权重更新拒绝和分项／输出事件计时；`ed2e82c5eda8` 增加opt-in PyTorch pinned HiCache及启动预热。当前源码的1008项完整CPU检查通过，零失败／错误／跳过，115.40秒；用例来自本目录精确清单，不与历史通过数累加。原字段兼容失败、CPU夹具错误及Guard启动前资源冲突均保留。

硬件／工作负载沿用下面的35B BF16、A100三resident rank、B1、native KV／64-token页、HiCache／NIXL、20/0/20→13/13/14→20/0/20及每方向2048输出。权重同版本对照每策略先各一轮：recopy完整迁移20.470／20.955秒，reuse为0.691／0.727秒；每方向少搬21,871,498,496字节。随后冻结tag `experiment/sglang/pp-pinned-host-warm-20261111-r1`，独立启动两次，reuse＋pinned完整迁移扩／缩均值0.293／0.321秒，最终暂停246／271 ms。后者两轮原值与分项见[优化记录](MIGRATION_OPTIMIZATION.md)，N=2仅为描述性重复，不称稳定尾部或广泛性能结论。

两轮新路径均满足暖缓存完整IDs、在线KV目标应用、最终CPU前缀实际loadback、运行时capture=0、目标Graph replay与活跃Decode eager=0。Graph选择最大rank为2.28–2.56 ms；完整最终暂停明显长于选择Graph。完整监督161.478／153.763秒，Guard postcheck、owned cleanup和正常lease release通过；每轮127个冻结输入结束后哈希一致。结果SHA256分别为 `ddbd62e22188b3be5f89b621071024ab4ff87187982ef6a3543ae60d23b7c3cb`、`3111d737a73c408851675672241cd06626c8571b7e86120d213741c6d749c2ad`。原始证据根目录为 `source_integration/sglang-pp-elastic-20261105/migration-optimization-20261111/`。

3进程pinned allocator保留内存合计5.781 GiB，是当前采样而非总CPU峰值；候选GPU权重、KV和Graph继续常驻。只验收实际B1、L2 HiCache、同节点NIXL／CUDA IPC；冷暖生成等价、更多模型／batch、跨节点RDMA、VMM组合及转换期间归还工作GPU均不由本次取得资格。默认mmap路径保留；pinned需显式开启。

## 2026-11-11：低干扰实现的集成审查

主 agent 直接审查任务分支 `codex/pp-kv-live-foundation-20261109`，未发现阻止已测native／HiCache／NIXL、PP2↔PP3范围合入的功能性问题；没有新增独立reviewer。当前专项574项通过，随后将历史537项基线与新专项按用例去重，**992项组合CPU检查全部通过，零失败／错误／跳过，113.580秒**，不累加通过数。精确用例见[cpu_test_nodes.txt](cpu_test_nodes.txt)；旧[cpu_test_paths.txt](cpu_test_paths.txt)仅为文件索引，部分文件含GPU测试，整文件调用不是CPU验收命令。

产物在 `source_integration/sglang-pp-elastic-20261105/review-20261111`：`cpu-r3.xml`／JSON／日志、import差异检查、依赖补丁哈希和风格检查。首次误用整个文件清单的广泛调用已中止，输出保留，不计为合格运行；r2为574项专项，r3为992项组合。本轮修正仅含import排版、一个测试的CPU注册、CPU用例清单及保护统一diff补丁的whitespace hook。本次Python改动的isort／Black／Ruff均通过；父分支全仓lint本来存在格式、可执行位及23个测试注册缺项，候选补齐一项后仍有22个，不声称GitHub CI全绿。

没有新增GPU实验。与 `c5aa853480` 的生产／运行适配差异逐项检查，import名称及别名集合相同，去除import后的AST一致；复用下面冻结GPU检查的有界结论，不称为逐字节相同源码或新的硬件验收。GPU故障恢复、VMM组合、实际B2／B4及性能优势仍未取得资格。详见[审查记录](REVIEW_20261111.md)。

## 2026-11-11：启动预捕获与运行时选择

冻结 `c5aa853480`（完整提交见[manifest](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/manifest.json)），annotated tag `experiment/sglang/pp-prepared-decode-graph-20261111-r2`。35B BF16、三张常驻A100、TP1／EP1、native／HiCache／NIXL background＋overlap，20/0/20→13/13/14→20/0/20。每方向B1、513输入／2048输出、公共暖边界512 token、64-token页，完整输出IDs与当前版本同进程暖缓存参考一致；无retraction／重新Prefill，最终CPU前缀实际loadback。当前类为 `same_process_warm_execution_continuity`，不复用已变更生产源码的旧独立审查，不授予冷暖生成等价；观测到cold／warm输出仍不同。

启动时rank0／1／2分别捕获2／1／2套本地区间，档位1／2／4。服务后capture=0；扩容每阶段实际replay1028次，缩回后两个保留阶段各1090次、空阶段0次，活跃代Decode eager=0。Graph选择1.733–2.596 ms；按每方向最晚rank记录完整fence为0.603709／0.687642秒，各rank并行不可相加，后台prepare仍19.799／20.438秒。它不是逐token最大间隔，也不是与旧3.137／2.261秒基线的配对重复性能实验。

初始Graph完成后，额外候选准备各rank3.872–3.945秒，常驻allocated增量为0.348／20.816／0.348 GiB，合计21.513 GiB。没有额外峰值上限或物理页回收结论。目标在最终fence前应用3064／2940个按来源累计的KV槽位行；同节点UCX日志为CUDA IPC。完整监督178.903秒、worker自身155.998秒，边界不同；Guard postcheck、owned cleanup及正常release通过，127个冻结输入一致，JSON稳定可见性检查通过。结果SHA256 `8baf4c3a827a10ff53e4a701438710472691f6b6104d93b81d0b1bd6ecb7f81c`。

546项完整CPU回归通过；随后启动NIXL隔离修正的81项相关检查通过，数量不累加。首次环境import、fake-args兼容、测试路径及空CUDA变量收集失败保留于归档 `prepared-graph-20261111`。gpu-r1在model启动前因auto资源参数、随后缺外层预算文件而失败，未手动清理其有界quarantine；gpu-r2改回已有受监督入口并正常通过。本轮主agent自行实现／核查，无独立reviewer；仅实际B1、单轮，worker与候选存储常驻，VMM组合、故障恢复、跨节点RDMA、性能优势及转换期间归还GPU均未验收。详见[报告与快照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/summary.md)。

## 2026-11-11：NIXL 持续迁移接通整模型 Decode Graph

**通过范围。** 冻结 `01bc3770e32254e7088d50769c16b115c89dc9f3`，annotated tag `experiment/sglang/pp-nixl-decode-graph-20261111-r3`。35B BF16、三常驻A100、TP1／EP1、native KV、HiCache、background／overlap、NIXL 1.5.2，20/0/20→13/13/14→20/0/20。两个方向各2048-token输出与同暖512边界参考完全一致，无retraction或切换后重新Prefill，最终活跃阶段实际CPU前缀loadback；不是冷暖生成等价资格。

**Graph。** 启动及最终排空时使用原生full Decode runner，Prefill仍eager。旧backend和进程共享池ID显式退役，不复用已失效池；提交目标后捕获新Graph，全组确认才恢复。扩容目标三阶段各replay1013次；缩容后rank0／2各1095次，rank1 inactive／0次，活跃代Decode eager=0。捕获档位1／2／4，实际请求只验证B1。原语预捕获库和稳定地址VMM组合不在本轮资格中。

**时间与协议。** 控制面扩／缩准备19.866／20.348秒、fence 3.137／2.261秒；各rank捕获分别1059.347／2262.343／809.463 ms，以及1156.896／0／960.974 ms。它们并行执行，不能累加；fence包含KV／GDN／HiCache和Graph，不是逐token最大间隔。暂停前目标实际应用3094／2930个按来源累计的槽位行。UCX的CUDA→CUDA非零写入日志选择CUDA IPC zero-copy，只验证同节点。没有重复实验或性能改善结论。

**检查与清理。** 完整CPU回归508项通过；延长smoke协议后73项harness复查通过，不累加。完整监督含加载及清理177.895秒，Guard postcheck、owned进程清理、正常lease release通过；135个冻结输入哈希一致，远端JSON均通过稳定可见性检查。结果SHA256 `ca56488c5b25fe141dbcbf19b8a9a96fa2fdcf2aafa0878a7a2db20a2ad798ea`。归档为 `graph-integration-20261111/gpu-r3`；[精简报告及快照](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_nixl_decode_graph_20261111/summary.md)含原始结果、CSV、审计及回执，大日志留原处。

**失败保留与边界。** r1旧共享池已无Graph引用，复用池ID触发PyTorch use_count断言；修正显式backend清理和池ID重置后，r2捕获成功，但1024-token请求在准备期间结束、目标replay=0，严格验收仍失败。r3以2048输出／4096 context保持请求跨越切换，未放宽检查。r1／r2分别132.268／121.676秒内正常清理并release。worker常驻按用户要求暂缓处理；启动布局Graph bank、VMM／GDN／HiCache组合、Prefill Graph、GPU故障恢复、跨节点RDMA、性能收益及转换期间归还工作GPU未验收。本轮没有独立reviewer，尚未合入共同主线。

## 2026-11-10：真实前向／allocator与在线CPU shadow

新增 native KV CPU shadow 路径，接口和范围见 [GRAPH_KV_PATCH](GRAPH_KV_PATCH.md)。本轮组合检查最初476项通过、1项失败；失败为原有 PP 分区参数绕过配置声明的问题，旧 HEAD 中也存在同样的两处直接赋值。已改为声明更新，并补测配置投影与 allocator resize 前置拒绝。随后40项相关检查全部通过；按最新结果去重，共478项通过、0失败／错误／跳过。它是组合证据与受影响路径复测，不是修改后重跑全部478项，也不与此前769项相加。

产物为 `kv-live-foundation-20261109/live-patch-cpu-r2/pytest.xml`、`live-patch-cpu-r3/pytest.xml` 和 `live-patch-cpu-audit.json`；初次112项定向检查保留在 `live-patch-cpu-r1/`。真实请求在线复制、GPU恢复与性能仍需另行检查，本轮没有独立 reviewer。

首个GPU候选 `b9cfdc0180` 在模型加载前被page-size限制拒绝，原启动协议固定64-token页；`live-patch-gpu-r1` 共35.628秒，清理和正常release验证通过。未改为page-size 1；已补齐标准分页 allocator 的整页生命周期、页尾追加与前向前重写保护，保持原缓存协议。随后 `live-patch-cpu-r4/pytest.xml` **483项CPU组合检查全部通过**，160.810秒，零失败／错误／跳过。这是新一次完整组合回归；早先失败保留，不重复累计通过数。新GPU资格另记。

第二个GPU候选 `57742e9842fc` 使用原64-token页，完成加载和512-token预热后，在首个转换的后台新层构建中失败。原因是进程级forward context让构建线程误判自己在推理；转换尚未提交。`live-patch-gpu-r2` 共144.885秒，owned进程清理及正常release验证通过。原始结果稳定可见，SHA256 `d26add2feee8c4897fb9b0e55364aaf7da6047ad5e012b65b0a960590e379961`。

隔离初版通过512项CPU组合检查（`live-patch-cpu-r5/pytest.xml`），但额外Dynamo检查发现ContextVar.get无法被fullgraph追踪，因此改用线程本地属性。保留同线程forward内的构建状态读取拒绝，以及嵌套／异常恢复；新增编译后切换backend的回归，防止绕过检查或冻结错误backend。修复后的检查及新GPU结果另记，不把失败批次算作在线patch验收；这里只验证同步forward作用域，不支持在作用域内await。

最终线程隔离版本的 `live-patch-cpu-r6/pytest.xml` **512项检查全部通过**，约141秒，零失败／错误／跳过；范围为elastic_pp全目录、配置声明、draft构建隔离和split attention决策，包含fullgraph下切换backend的检查。这次完整组合回归替代前一修复候选的当前资格，不累计为1024项；实际模型CUDA Graph仍未启用。

**真实GPU接入通过。** `live-patch-gpu-r3` 固定源码 `e7c9cb1264a9599121f3c8ff114bf19f798307d5` 和annotated tag `experiment/sglang/pp-online-kv-shadow-20261110-r3`。三张resident A100上的35B BF16模型使用native KV、64-token页、background／overlap、HiCache开启、Graph关闭；布局20/0/20→13/13/14→20/0/20。两次转换的在线shadow与活跃Decode连续性检查均通过，同暖512边界输出一致，最终CPU前缀实际loadback。扩容源rank0／2各在线完成662槽位行，缩容源rank1完成659行；全部shadow在fence补齐。行计数包含该rank所有待迁出Attention层，不等于全组独立token数。

整轮含清理171.795秒，Guard返回0，owned进程清理及正常lease release验证通过。结果、cleanup和supervisor均经稳定可见性检查；原始结果SHA256 `afa79db7aa34596563b57d139a9f5881a36345c4ff107944943cccf3543fed01`。精简报告在同归档的 `live-patch-gpu-r3/REPORT.md`，结果快照为 `sglang_pp_online_kv_shadow_20261110`。这是单次有界正确性验收，不证明暂停收益；网络、GDN和HiCache仍在fence，整模型Graph、VMM／HiCache组合、OOM／迁移失败GPU回退、冷暖生成等价及转换期间归还工作GPU均未取得资格。没有独立reviewer。

## 2026-11-05：静态 EP／PP／P/D 资格

2026-11-05。共同分支为 `codex/pp-elastic-integration`，基点为 R11 `b862312ce3`。运行记录和完整检查清单位于[验证 manifest](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/validation_manifest.json)。这次验证执行正确性和开关实际生效，不据此发布性能提升结论。

**CPU 回归。** 共 537 项通过，零失败、零跳过。覆盖 EP／BF16 接口、量化参数转发、PP 通信与微批、HiCache 共识、GDN 和 P/D 状态交接、CPU 权重缓存，以及 elastic PP 控制与状态迁移。结果见 [JUnit](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/final_cpu_tests.xml)，52 个测试文件列于历史文件索引 [cpu_test_paths.txt](cpu_test_paths.txt)。复跑使用锁定 Python、引擎 `python/` 的 `PYTHONPATH` 和 `/home/bxb1/conda/lib` 的 `LD_LIBRARY_PATH`，关闭 pytest 自动插件加载并显式加载 `pytest_asyncio.plugin`。CPU用例应从本页顶部的[cpu_test_nodes.txt](cpu_test_nodes.txt)读取；旧文件索引包含混有GPU测试的文件，不能整文件调用来复现CPU资格。

**真实 A100 检查。** 使用节点 `spartan-gpgpu168` 的四张 A100 80GB，模型为本地 Qwen3.6-35B-A3B，BF16 权重／KV、FP32 GDN，HiCache 开启。EP2 位于 NVLink 对 `[0,1]`，EP4 覆盖整节点；P/D 案例为 `[0,1]` 上 EP2 Prefill，向 `[2,3]` 上 PP2 Decode 交接。每个 rank 检查全部本地层的专家布局；代表 Expert 层通过真实权重与独立路由／计算参考比较，包括全部 token 路由到一个 rank 的情况。随后运行完整模型、8 个并发请求及受控缓存恢复。

**EP2、EP4 已通过。** 原生 NCCL 路径在 r6 的两个／四个 rank 上分别实际执行 144 次 Decode Graph；调度 overlap 开启。GPU 前缀被显式驱逐后，从 CPU 恢复的输出与同形状 GPU 热缓存参考一致。DeepEP normal 路径在 r8 通过真实 dispatch、Expert、combine 和完整模型检查；这个模式实际走 eager，不能把参数中未关闭 Graph 当作 Graph 已执行。对应记录为 [r6](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/gpu-r6/results.json) 和 [r8](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/gpu-r8/results.json)，其中其他布局的失败记录原样保留。

**PP2 已通过。** r9 中两个阶段各执行 276 次 Decode Graph，producer-ready 发送有实际计数。固定 8 请求、两个 4 请求微批的 lookahead 检查中，两个阶段各产生并消费 40 次提前准备，无遗留项；快速发送次数分别为 96、48。优化前后生成 token 完全一致。这个可选快速协议只验证了静态 PP2、固定合批且不返回 logprob 的场景；通用请求继续使用基础协议，不能直接打开 policy9。

**P→D 已通过。** r9 中 P 两个 rank 和 D 两个阶段均完成真实模型请求。P、D 均实际从 CPU 恢复 512 token 会话前缀，恢复前后的生成 token 一致。D 的恢复事件为 `copy_mamba_to_request=False`，没有用本地缓存覆盖 P 已传入的 GDN 状态。D 两个阶段各执行 534 次 Decode Graph。PP2 与 P/D 完整结果见 [r9](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/gpu-r9/results.json)；Guard postcheck 通过，进程清理已验证，节点租约已释放。

**修复和测试条件。** 合并后发现 BF16 DeepEP low-latency 的构造限制误挡了已有 A100 normal 适配器，现以“实际安装适配器”的显式能力标识放行，并增加默认拒绝／适配器允许的回归。测试还修复了 P/D 并发启动端口冲突和 mmap 控制文件被截断的问题。PP 快速协议检查取消其不支持的可变 logprob 消息。P/D 恢复检查先预热到相同缓存长度，避免将 GDN 不同计算形状的数值差异误判为 CPU 搬运错误。历史失败记录保留，不改写为成功。

**版本与尚未覆盖的范围。** CPU 使用 r8，引擎 `python/` 源码与 r9 完全相同；r9 的变化是测试和可选运行适配。原生 EP2／EP4 使用 r6，后续唯一引擎改动是上述 DeepEP normal 构造兼容修复，它不改变原生 NCCL 路径。各次不可移动的标签、提交及依赖在 manifest 中记录。动态 PP 扩缩、GPU 真正释放、H100、其他模型／量化路径、DeepEP LL／CE，以及 TBO、异构 TP／PP、speculation 等可选组合仍需各自的硬件资格验证；目前不能声称它们全部无 bug。

**原工作保留。** 46 个原有 worktree 的已跟踪状态和未提交源码哈希均与整合前一致。45 个历史分支已通过祖先关系纳入；33 个 DeepEP 分支作为独立依赖锁定并已有远端备份，其余 10 个分支逐项说明重复、被替代或作为参考保留的原因。见[覆盖清单](branch_coverage.json)和[保留检查](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/original_worktree_preservation.json)。

## 2026-11-09：naive PP＋HiCache 组合验收

在冻结候选 `de7808daeae6` 上，历史537节点与新增205节点去重后共742个CPU用例全部通过，零失败、错误或跳过，源码不变且无可见CUDA设备。实际合入项目的引擎、测试和适配tree与候选一致；r14／r15的有限GPU证据按相同引擎与运行适配复用，本轮未重跑GPU。独立审查与范围说明见[基线验收摘要](BASELINE_ACCEPTANCE_20261109.md)。原2026-11-05静态资格和每次失败记录保留，不自动扩大到新动态组合。

总 PR #1 合入主线后，29项配置／backend／BF16 EP定向CPU检查全部通过（5.422秒），零失败、错误和跳过；源树与742项候选一致。该重复检查不额外计作独立覆盖。冻结标签：`baseline/sglang/pp-hicache-naive-20261109`。

## 按层 KV 映射首个检查点（2026-11-09）

从 `ce563696401c` 在独立任务分支实施，尚未合入项目或共同基线。用户要求本轮不使用子 agent，因此仅由主 agent 实现及自行检查；本轮不声称完成独立审查或 GPU 验收。接口、复现方法和剩余边界见 [KV_LAYER_MAPPING.md](KV_LAYER_MAPPING.md)，状态见 [TODO](TODO.md)。

引擎 CPU 回归为 **769 项通过、0 失败、0 错误、0 跳过**，包括原先742个节点及27个新增节点，逐个去重。使用相同新增测试在旧基线复现 wrong-generation abort、cleanup、import 三处故障，**3 项均按预期失败**。全量回归后的源码审计确认运行逻辑与测试未变，唯一后续修改是缓存池类的说明文字，不重复累计检查数量。

隔离依赖源码上的协议／几何 CPU 检查为 **27 项通过、0 失败、0 错误、0 跳过**。它们使用模拟原生映射，覆盖所有权过滤、重复 map、部分失败和重试、pin／解 pin、不同 group、初始化 fence 和 stacking 范围；不替代真实 CUDA 页测试。首次 CPU 批次误包含需 CUDA 的旧分配用例，其中28项因隐藏设备／缺少原生加载条件失败；原始记录保留，未计入通过，也未修改这些用例来绕过 GPU 资格。

原生扩展在已有预约节点构建成功，随后在 `CUDA_VISIBLE_DEVICES=999` 下实际加载并检查 Python API／native ABI v1，确认可见 CUDA 设备数为0；没有为这一步启动模型或 GPU 分配测试。依赖补丁从固定上游提交重放，生成 tree `14c85a5f0701a5a7789683ee57f9cbf873ea4cc7`，与依赖检查点完全相同。补丁哈希和前置环境见 [dependencies.lock.json](dependencies.lock.json)。

原始产物位于实验归档 `source_integration/sglang-pp-elastic-20261105/kv-live-foundation-20261109/`，包含 `cpu_attempt1/`、`cpu_attempt2/`、`cpu_final/`、`engine_test_source_audit.json`、`dependency_patch_replay.json`、`native_build_final.json` 和 `native_capability_probe_final.json`。首轮登录节点缺少 `libcuda.so.1` 的加载失败也保留；最终加载证据来自预约节点。真实 GPU 地址／数值保留、物理页回收、OOM 回退、packed stride、HiCache 接入及端到端性能仍未验收。原有 hybrid HiCache 的 native-only 保护保持不变。

## Graph／持续 KV patch 初步 GPU 验证（2026-11-09）

冻结源码 `8d7c21241c`，annotated tag `experiment/sglang/pp-graph-kv-probe-20261109-r1`，依赖 tree／native binary 与前一节构建一致。新模块与运行生命周期／参考资格共75项CPU检查通过（新增模块25项，另50项回归），不与此前769项简单相加。本轮由主 agent 实施及核查，没有独立 reviewer。

**底层真实 GPU。** A100 80GB 上，32 MiB 稳定虚拟地址分成4个层组，起初只映射2组。迁移显式映射目标组；重复 map 保留原值，pin 拒绝提前回收。退役8 MiB层组后，驱动可见空闲显存实际增加8 MiB；保留层地址和数值不变，重映射后仅新页为零。两张小算子的 CUDA Graph 均预捕获并 replay，地址改变触发一次 eager fallback，未映射 backing 阻止一次执行。packed-stride K／V 行复制完成2轮、4行，包含第一轮尚待确认时发布的新增行，以及槽位复用后的新 incarnation；最终 K／V 与源一致。此项是单 GPU 原语检查，不是 Attention 后端性能、实际并行 overlap 或完整模型验证。

**HiCache 整模型回归。** 同一 Guard 租约随后在三张 resident A100 上执行 native KV 后端的35B BF16 PP2→PP3→PP2，两次活跃 Decode 切换后的输出均与同暖512边界参考一致，CPU前缀有实际loadback。Graph仍关闭；cold／warm参考生成等价仍未取得资格，工作GPU也未在转换期间归还。整轮包含清理186.509秒，Guard postcheck通过，owned进程清理和正常lease release均验证。它验证新代码共存时的同步基线，没有执行整模型 Graph bank 或在线 patch。

产物目录为上述归档下的 `gpu-r1/`，包含 `component_gpu.json`、`results.json`、`supervisor_outcome.json`、`cleanup_ack.json`、对应可见性回执及 `post_run_audit.json`；精简报告为 `REPORT.md`。底层原始结果 SHA256 `6991b2fb9ac3e02e8a84645b697fd91fec92dd55500c2908316d80798d21bbea`，整模型结果 `e6eb56810c04f57b0ecc0716a1f6c516faa2f1806a7465e88c8fb228184134b1`。真正的 OOM／迁移失败回退、HiCache／GDN稳定地址、allocator生命周期钩子、跨 rank patch、完整模型Graph及暂停收益仍待验证。

## NIXL 预初始化与在线 KV：有限整模型验收（2026-11-10）

冻结源码`152de4332fba49d3de440a28e33e88aa2e51b09f`，annotated tag `experiment/sglang/pp-nixl-online-20261110-r2`，Draft PR #4；本轮由主agent实施、运行及核查，没有独立reviewer。NIXL／nixl-cu12 1.5.2装在隔离覆盖目录，原生绑定、UCX及插件哈希见依赖锁，未改共享serving环境。

**CPU。** 首次组合回归484通过、2失败，原因是错误返回路径的异常变量未绑定，后台线程退出；修复后elastic_pp 487项全过，零失败／错误／跳过。随后修正原生metadata导入bytes与通知str的身份表示，并保护接收端scatter失败后的stream排空；不能证明完成时保留target backing，要求worker退出。修改后的14项NIXL定向检查再次全过，不与487项累加；仅模拟字节传输和CUDA排空故障，实际GPU故障恢复仍未验收。原始记录在归档`nixl-live-20261110/`，包括失败日志、`cpu-suite-r2.xml`及`cpu-api-r6.xml`。

**真实A100。** 35B BF16、native KV、64-token页、eager、HiCache开启，三个resident候选rank在启动时建立agent，各完成两条通道预热并注册64 MiB固定缓冲。20/0/20→13/13/14→20/0/20的两个方向均通过同暖512-token缓存边界的256-token生成连续性，没有retraction或重新Prefill；CPU前缀在最后实际loadback。扩／缩目标在最终暂停前累计应用1320／1318个槽位行；行数按来源／接收端累计，不是全局原始token数。UCX日志对非零CUDA写入选择`cuda_ipc/cuda` zero-copy协议，仅为同节点路径证据。Attention KV payload分别4,325,376／5,898,240字节，GDN／请求进度／HiCache历史状态仍在fence交接。

**时长与清理。** 完整监督含加载及清理176.008秒，Guard postcheck通过，owned进程清理和正常lease release均验证。各rank记录的扩容fence为852.830／852.818／852.725 ms，缩容594.697／594.689／594.665 ms。这是每rank的coordinator阶段耗时，每方向仅一个样本，不是逐token最大停顿、独立重复平均或暂停收益证据。首轮`nixl-gpu-r1`的启动身份比较失败原样保留，83.513秒内正常清理；修复后本轮通过。

[完整报告](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/kv-live-foundation-20261109/nixl-gpu-r2/REPORT.md)及同目录的`post_run_audit.json`、`supervisor_outcome.json`、`controller_results_visibility.json`保留证据；结果SHA256 `c56f58fd6505c73bf5f5388f5a63a8db8b8ffb6ef26163ae26f5031c4b0d6dfa`。后续文档更新不改变此运行源码。

**边界。** 本轮只验收native／eager／HiCache路径的有界正确性；完整模型Graph、KVCacheD／GDN／HiCache组合、真实GPU故障注入、跨节点RDMA和性能收益仍未验收。常驻空阶段不等于在转换期间归还GPU。CPU shadow保持可选正确性对照，冷暖生成等价资格不扩大。

## 2026-11-14 reactive E2E pilot candidate

The r23 single-request GPU evidence does not qualify the new four-request, 65K context profile. This task preserves the microbatch-one/native P/D/Graph/HiCache/offload/NIXL restrictions, exposes real queue and transfer counters, and requires quiescent new handoffs at each transition. The first paired AgentX/BurstGPT run is exploratory; no E2E speedup or physical GPU saving has yet been measured. Replica startup uses the admitted canonical CPU bank; standby/startup costs remain in the resource ledger.

Targeted CPU gate: 54 tests passed (reactive hysteresis/serialization, bounded P/D validation, router gating/affinity/cancellation and queue-inclusive latency). The default conda environment lacked orjson; the qualified serving Python completed the suite. This is not GPU qualification.

Pilot r1 failed before formal flow: per-process queue telemetry was installed only in the parent, not the spawned GPU schedulers. P/D static warm smoke passed; Guard cache audit and owned GPU cleanup passed, and the lease released. Fix: install telemetry at module import under the scoped worker-role environment and assert every scheduler reports it before launching AgentX. Also reserve 262144 token rows on both arms so four 55K contexts fit the frozen pool; available GPU memory in r1 was 12.58 GiB per active P rank. No dynamic KV-capacity increase is claimed.

Pilot r2 stopped before readiness: the startup PP3→PP2 overlap capacity check required 27812662635 bytes, but rank0 had 27667857408 bytes free (a 144805227-byte deficit). Guard cache audit/owned cleanup passed and lease released. Reduce the over-provisioned recurrent pool from 64 to the historical AgentX value 48 in both arms; retain 262144 KV rows and all safety margins. This changes provisioning, not the peak admission check. No formal requests or E2E result exist in r2.

Before launching r3, the frozen legacy placement header was audited: it used only source conversation_id, grouping distinct AgentX clones together. The new pilot namespaces affinity by stable x_correlation_id plus conversation_id; warmup and later turns retain the same key, and different cloned sessions do not share one mutable owner. Payloads, arrival times and dependency IDs are unchanged. This CPU-only correction is applied identically to both arms before any formal pilot traffic.

Pilot r4 reached native warmup but no formal flow: P held two long requests in waiting_queue while D waited for transfer. Source audit identified deterministic Triton truncation alignment 4096 versus the inherited 512 chunk; PrefillAdder rejects every truncated chunk. Fix both arms to chunk4096/max-prefill8192, add a pre-launch configuration guard and a 4608-token real P/D smoke before AgentX. r4 was explicitly stopped; both remote workers completed identity-verified cleanup. The coordinator's premature cleanup receipt and its later recovery are retained. Its supervisor was interrupted and lease expired, so manual postcheck could not authenticate the cache audit; do not call r4 a successful Guard run. Cleanup waiting is now cancellation-safe; future bounded stops signal only the owned controller while Guard remains live.
