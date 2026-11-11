# 启动时准备 Graph，运行时切换

**做什么。** 在接收请求前走一遍声明的 PP 布局，预热并捕获各布局的原生 Decode Graph，再回到初始布局。运行时沿用 NIXL 持续 KV 追补和最终交接，选择已准备的 Graph；不在交接时重新捕获。HiCache 开启，候选 worker 和端点常驻。代码与 GPU 验收分开记录在 [TODO](TODO.md)。

**怎样保持有效。** 每个本地层区间保留自己的 attention backend、Graph 和工作区；按全局层号保留权重、KV 和 GDN 的内存。退役只移除当前执行关系，不释放 Graph 仍引用的存储。下一次回来时覆盖迁入的活跃状态、同步保留层的 GDN，不能使用缓存中的旧数值。提交后核对权重、KV／GDN、请求表和 Graph 输入的地址、形状及步幅；不一致就停止切换，不偷偷重新捕获或改用 eager。启动内部转换不发布给请求，完成后公开代数仍从 0 开始。

**怎样开启。** 在现有 native／overlap 弹性 PP 启动参数上增加 `--elastic-pp-graph-partitions '20,0,20;13,13,14'`，初始布局使用 `--elastic-pp-initial-partition '20,0,20'`。这是三张常驻卡上的逻辑 PP2／PP3，首尾物理 rank 固定活跃。最多声明四个布局；自动伸缩的候选必须都在此集合内。batch 捕获档位沿用 `--cuda-graph-bs-decode 1 2 4`。只支持既有 TP1／EP1、full Decode、native KV 路径，Prefill 仍 eager；不接 LoRA、TBO、PDMux、compile、speculative 或 CPU-spill Graph。

**如何验证。** `hicache_transition_smoke.py` 增加 `--prepared-graphs --cuda-graph --same-process-warm-reference`；其余 NIXL／HiCache 和受监督的 300 秒预算保持原协议。两次真实生成分别跨越 PP2→PP3 和 PP3→PP2，完整 token IDs 与同进程、同版本的暖缓存参考比较；要求无 retraction／重新 Prefill、目标确有 replay、最终 CPU 前缀实际 loadback，并要求启动捕获次数固定、运行时 capture=0。该新验收类别只证明暖缓存执行连续性，不复用已变更生产源码的旧独立状态审查，也不授予冷暖生成等价资格。旧的严格参考审查类别不改变。

**代价与边界。** 多布局的权重并集、KV／GDN 和 Graph 工作区持续占用显存，缩容暂不回收这些内存。记录启动准备时间、相对初始 Graph 完成后的额外 `memory_allocated`、每布局捕获时间、运行时选择时间及完整最终交接时间。它不是 VMM，也不证明归还工作 GPU、零暂停或逐 token 最大停顿；后两项需要另外测量。先通过有限整模型检查，再讨论共享工作区或减少常驻内存。
