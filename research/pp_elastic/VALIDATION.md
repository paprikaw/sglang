# EP／PP／P/D 集成基线验证

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
