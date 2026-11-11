# 启动时准备 Graph，运行时切换

2026-11-11范围修正：用户要求候选权重不得常驻GPU。默认offload保留Graph虚拟地址，释放退出层物理权重页，接班前真实搬权重。A100两次独立往返完整扩／缩均值19.98／20.65秒，最后暂停239／285 ms；运行时capture=0，缩回后中间GPU权重为0、释放20.37890625 GiB。实现及边界见[非驻留权重](NONRESIDENT_GRAPH_WEIGHTS.md)。历史约0.3秒仅为resident诊断，不能证明伸缩节省资源；整个GPU归还仍未完成。

**历史resident策略（冻结版本）。** 当时的候选库完成后，运行时默认复用已经验证的不可变权重，避免再次传输；仅限预捕获库，在线权重更新被拒绝。需要对照时使用 `--elastic-pp-prepared-weight-policy recopy`。L2 HiCache另可显式启用 `--elastic-pp-hicache-allocator pytorch`，在启动时预热候选的CPU页锁定分配。35B／B1／A100两个独立往返的完整迁移扩／缩均值0.293／0.321秒，最终暂停246／271 ms；运行时capture=0，Graph选择约2–3 ms。原始样本、常驻内存与范围见[迁移优化记录](MIGRATION_OPTIMIZATION.md)，不能把选择Graph耗时当作完整切换耗时。下文原结果保留其冻结版本。

**已测结果。** 冻结 `c5aa853480`，A100 native／HiCache／NIXL、实际B1的PP2→PP3→PP2通过同版本暖缓存连续性检查。运行时capture=0、目标确有replay，Graph选择1.733–2.596 ms；完整fence扩／缩0.604／0.688秒，初始Graph后候选准备约3.9秒。额外常驻allocated合计21.513 GiB，其中新增阶段20.816 GiB。完整监督178.903秒内完成并验证清理／正常release，尚无配对性能收益或工作GPU归还结论。[报告与绝对指标](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/summary.md)。

**做什么。** 在接收请求前走一遍声明的 PP 布局，预热并捕获各布局的原生 Decode Graph，再回到初始布局。运行时沿用 NIXL 持续 KV 追补和最终交接，选择已准备的 Graph；不在交接时重新捕获。HiCache 开启，候选 worker 和端点常驻。代码与 GPU 验收分开记录在 [TODO](TODO.md)。

**怎样保持有效。** 每个本地层区间保留自己的attention backend、Graph和工作区。权重保留虚拟地址及Tensor记录，退出层释放物理页，再次启用时重新映射并由正常传输填入权重；KV和GDN目前仍保留原生存储。下一次回来时覆盖迁入的活跃状态、同步保留层的GDN，不能使用旧数值。提交后核对权重、KV／GDN、请求表和Graph输入的地址、形状及步幅；不一致就停止切换，不偷偷重新捕获或改用eager。启动内部转换不发布给请求，完成后公开代数仍从0开始。

**怎样开启。** 在现有 native／overlap 弹性 PP 启动参数上增加 `--elastic-pp-graph-partitions '20,0,20;13,13,14'`，初始布局使用 `--elastic-pp-initial-partition '20,0,20'`。这是三张常驻卡上的逻辑 PP2／PP3，首尾物理 rank 固定活跃。最多声明四个布局；自动伸缩的候选必须都在此集合内。batch 捕获档位沿用 `--cuda-graph-bs-decode 1 2 4`。只支持既有 TP1／EP1、full Decode、native KV 路径，Prefill 仍 eager；不接 LoRA、TBO、PDMux、compile、speculative 或 CPU-spill Graph。

**如何验证。** `hicache_transition_smoke.py` 增加 `--prepared-graphs --cuda-graph --same-process-warm-reference`；其余 NIXL／HiCache 和受监督的 300 秒预算保持原协议。两次真实生成分别跨越 PP2→PP3 和 PP3→PP2，完整 token IDs 与同进程、同版本的暖缓存参考比较；要求无 retraction／重新 Prefill、目标确有 replay、最终 CPU 前缀实际 loadback，并要求启动捕获次数固定、运行时 capture=0。该新验收类别只证明暖缓存执行连续性，不复用已变更生产源码的旧独立状态审查，也不授予冷暖生成等价资格。旧的严格参考审查类别不改变。

**历史resident版本的代价与边界。** 多布局的权重并集、KV／GDN 和 Graph 工作区持续占用显存，缩容暂不回收这些内存。记录启动准备时间、相对初始 Graph 完成后的额外 `memory_allocated`、每布局捕获时间、运行时选择时间及完整最终交接时间。它不是 VMM，也不证明归还工作 GPU、零暂停或逐 token 最大停顿；后两项需要另外测量。先通过有限整模型检查，再讨论共享工作区或减少常驻内存。
