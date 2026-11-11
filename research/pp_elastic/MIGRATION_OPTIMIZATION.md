# 迁移与切换优化记录

本轮目标是同时减少完整迁移时间、最终交接暂停和迁移期间对生成的干扰。每次改动记录原因、实际测试及负面结果；正确性条件保持不变。以已合并的 `5f530f6de33d` 为起点，任务分支 `codex/pp-migration-latency-20261111`。

**原始观察。** 已归档的启动预捕获实验中，PP2→PP3／PP3→PP2后台准备19.799／20.438秒，最终交接0.604／0.688秒。每次目标权重虽然已由启动Graph预热保留，后台仍把所有转移层经GPU→CPU→GPU重新传输。这里是单轮历史观察，没有逐token停顿或重复性能结论。原始依据见[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/summary.md)。

**优化一：直接复用已经常驻的不可变权重。** 启动仍正常加载、传输与捕获全部候选，完成后记录每层持久状态的地址、形状、类型及可用的版本计数。运行时挂接目标模块后检查这些条件，再标记目标层就绪；全组确认采用同一策略，才跳过重复传输。禁止该模式下的在线权重更新。保留 `--elastic-pp-prepared-weight-policy recopy` 作为同版本对照。它不减少已常驻候选的显存，也不省略KV／GDN／HiCache迁移。实际结果见下文。

**测量方法。** 补记准备、权重、KV patch收尾、GDN／KV交换、HiCache、模型提交、Graph选择及各次共同确认的本地耗时；它们有嵌套，不相加估算总时长。端到端另记流式输出时间与chunk大小，只有逐次单token输出才称为逐token间隔。比较相同源／目标、输入、缓存、Graph及NIXL条件下的reuse和recopy；保留每次原始值与正确性结果。未运行、未分离及失败的项目如实标记。

**优化一的第一组结果。** 冻结 `b59993a97f28`，同一A100节点，35B BF16、三常驻rank、HiCache／NIXL／启动Graph、两个方向各2048输出。recopy→reuse的扩容准备19.805→0.026秒，完整迁移20.470→0.691秒；缩容准备20.289→0.019秒，完整迁移20.955→0.727秒。每方向权重传输21,871,498,496→0字节。扩容／缩容平均TPOT分别14.261→8.906、14.930→8.967 ms；最大实际单token间隔829→683、794→715 ms。每条流有2048个单token到达事件；完整迁移按控制请求、最终暂停按coordinator分开计时。最终fence仍约0.64–0.68秒，其中约0.5秒为目标HiCache准备／导入。

两运行均保持暖缓存完整IDs、目标Graph replay、在线KV patch和最后CPU loadback检查，并正常清理／release。首次完整CPU检查988过、12个测试因新增诊断字段的旧格式断言失败；将诊断字段限制到显式profile模式后，1000项完整回归通过。原失败保留。这是同版本、同节点、每策略一轮观察，尚无重复均值；整请求差值同时包含更早进入目标布局的效果，不作为纯传输干扰的孤立成本。原始结果在 `migration-optimization-20261111/gpu-{recopy,reuse}-r1`。

**优化二：预热并复用CPU页锁定内存分配。** 默认mmap池每次需要重新注册大块CPU内存。新增显式 `--elastic-pp-hicache-allocator pytorch`，让PyTorch负责pin与缓存，避免手动重复注册／解注册。启动遍历候选布局时构造并销毁空的临时L2池，预热准确的分配形状；恢复原缓存的传输计数器。运行时仍重新建立元数据并完整搬迁有效缓存，复用的是空闲内存块。销毁时清除所有CPU tensor别名，避免被已冻结的controller循环引用继续占用。仅限L2、预捕获布局；默认mmap保留。

**优化二的重复结果。** 冻结 `ed2e82c5eda8`，tag `experiment/sglang/pp-pinned-host-warm-20261111-r1`。同一节点独立启动两轮，每轮两个方向各2048输出，其余条件沿用前组。扩容完整迁移0.2928／0.2935秒，均值0.2931秒；缩容0.3167／0.3255秒，均值0.3211秒。最终暂停扩容244.9／247.2 ms，均值246.1 ms；缩容266.3／275.3 ms，均值270.8 ms。两轮每方向的Graph运行时capture均为0，实际选择最大rank耗时2.28–2.56 ms，目标有replay、活跃代无Decode eager。暖缓存完整IDs、在线KV patch、最终CPU前缀loadback以及正常清理／release全部通过，127个冻结输入在每轮结束后均未改变。

扩容／缩容平均TPOT的两轮均值为8.692／8.733 ms；每轮最大实际单token间隔分别269.2／273.0及281.1／290.2 ms。这些最大间隔是各次请求的观测，不是P99或重复运行尾部分布。HiCache准备／导入最慢rank约119–127 ms，仍是剩余暂停的一部分；不把Graph选择时间称为完整切换时间。两轮的3进程PyTorch pinned allocator保留内存均为5.781 GiB（含使用中与缓存块），从开始服务到往返结束未增加host allocator调用次数。它是这套小容量协议下的采样值，不是总CPU峰值；原HiCache预留20 GiB保持不变。候选Graph、权重及原生KV的GPU常驻代价沿用前版，不由本次CPU优化消除。

**检查与负面结果。** 当前冻结源码1008项完整CPU回归通过，115.40秒，零失败／错误／跳过。新增检查覆盖pin所有权、退役时移除CPU别名、重复destroy、配置边界和启动时机。早先一个CPU夹具错误已修正并保留原失败；未修改GPU验收条件。`gpu-pinned-r1` 在启动前因其他正常实验占用节点被Guard拒绝，没有产生GPU性能样本；`gpu-pinned-r2/r3`才是上述两次有效运行，完整监督分别161.478／153.763秒。补测对照的第一次申请也因节点占用被拒绝，不能把它算作失败迁移或有效重复。

**完整对照与干扰诊断。** 原路径在相同节点、冻结旧版本的独立checkout补测，`gpu-recopy-r3`通过，完整监督172.767秒，127个输入一致，清理／正常release通过。原路径两轮扩／缩完整迁移均值20.459／20.891秒，最终暂停634／655 ms；完整优化对应0.293／0.321秒和246／271 ms。平均TPOT为14.307／14.905→8.692／8.733 ms。原路径扩容准备阶段的1034／1017个完整输出间隔均值19.149／19.462 ms，同序号暖缓存无切换参考8.443／8.463 ms，说明重复准备干扰源生成。优化版准备仅约26 ms、各只有3个间隔，不由它估计稳定速度。各轮绝对指标、重复均值、分项及图归档在[专题报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_migration_optimization_20261111/summary.md)。

**怎样复现。** 使用[启动Graph说明](PREPARED_DECODE_GRAPHS.md)中的有限布局，在既有native／HiCache／NIXL配置上保留 `--elastic-pp-prepared-weight-policy reuse`，另显式加入 `--elastic-pp-hicache-allocator pytorch`。相同源码可用 `recopy` 和默认mmap复查原路径；启动捕获不受reuse策略跳过。测试入口 `hicache_transition_smoke.py --prepare` 冻结源码、tag和依赖，再由 `hicache_guard_run.py --run <目录>` 通过Guard运行；profile使用 `--migration-profile`。原始目录为 `artifacts/source_integration/sglang-pp-elastic-20261105/migration-optimization-20261111/`，逐轮config、结果、输出事件和清理回执均保留，未提交大日志或密钥。

**后续选择。** 先测出复用权重后的主要耗时，再决定是否优化最终KV／GDN／HiCache交接或在线patch频率；不能由总耗时推断单个组件。worker退场和完整VMM接入仍沿用原定后续范围。
