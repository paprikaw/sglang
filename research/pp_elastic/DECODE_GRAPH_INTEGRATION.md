# 持续 KV 迁移中的 Decode CUDA Graph

2026-11-11 更新：已开始实现可选的[启动预捕获与运行时选择](PREPARED_DECODE_GRAPHS.md)，通过声明有限候选布局并保留原生存储来复用 Graph；本页后文保留已验证的交接时重建基线。新路径的实际验收以 TODO／VALIDATION 为准，不能将下面的旧结果当作新路径通过。

2026-11-11。用户要求先接通 Graph，worker 常驻暂不处理。本轮由主 agent 实施，接在 [NIXL 基线](NIXL_KV_TRANSFER_DESIGN.md) 上，进度统一记录在 [TODO](TODO.md)。

**先接通正确路径。** 启动时捕获初始非空阶段的 Decode Graph；旧布局继续用 Graph 推理并通过 NIXL 发送 KV。最后排空全部在途请求／通信，提交目标权重、KV／GDN、HiCache 和路由，销毁旧 Graph，使用 SGLang 原生 runner 捕获目标 Graph。所有 rank 确认后才恢复推理。空阶段不捕获；再次回到旧布局也重新捕获，因为 native 池地址已经变化。

**捕获不能覆盖请求。** 原生 runner 在捕获前清空索引缓冲，使用保留的 request、KV 和 GDN 槽位 0。重建前检查 request 行 0 与 GDN 映射仍为空；不允许用活跃请求预热。捕获失败不伪装为 eager 成功；目标已经提交时停止服务，不能继续使用旧代。

**旧内存池也要退役。** 删除 runner 不能保证进程级 Graph pool handle 仍有效。最终排空后显式清理旧 backend，并同时重置 SGLang 与通信分配器保存的 pool ID，让新一代创建新池；不依赖 Python 垃圾回收时机。GPU 首轮已暴露旧池引用计数归零的 PyTorch 断言，失败记录和清理证据保留，修正后另用新标签复验。

**范围。** native KV、固定 TP1／EP1、overlap backing、full Decode Graph，Prefill 仍用 eager；不接 torch.compile、speculative、Prefill Graph 或 CPU-spill Graph。HiCache 与 CPU shadow／NIXL 在线 patch 均保留。有限启动预捕获库与稳定地址池仍是后续优化，本轮明确在最终交接期间重新捕获，会增加该阶段时长。

**下一步建议：启动时准备有限布局。** 先限定 PP2／PP3 和 batch 1／2／4，在接收请求前逐布局预热并捕获；同时固定该布局权重、KV／GDN、输入输出及 workspace 的地址，保留 Graph、backend 和内存池的有效引用。当前 native 提交会更换池地址，不能只保存 Graph 就直接复用。布局之间先隔离 Graph 工作区并核算额外显存，不假定所有布局可以安全共用一个池；启动预捕获也不要求把所有布局的 KV 数据提前填满。运行时继续用 NIXL 基础复制与持续追补，最后只排空、补齐状态并共同选择目标 Graph。仍有最终交接，不承诺零暂停；验收需证明切换期间没有重新捕获、目标实际 replay、输出及 HiCache 连续，并分别记录启动时间、常驻／峰值显存和逐 token 停顿。此项是方案，整模型尚未实现。地址与共享池约束见 [PyTorch 对应版本说明](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/docs/source/notes/cuda.md#cuda-graphs)。

**如何验收。** 逐 rank、逐 generation 记录 capture 秒数、实际成功 replay 和 Decode eager 次数。`hicache_transition_smoke.py --cuda-graph --live-kv-patch --kv-patch-transport nixl --nixl-root …` 要求 PP2→PP3→PP2 每个源／目标活跃阶段都实际 replay，新增阶段生效、退出阶段无 Graph，NIXL 暂停前已有真实写入，输出与同暖缓存参考一致且没有重新 Prefill。`--active-output-tokens` 可在冻结前选择 256–2048，超过1024时将 context 同步设为4096，避免截断；不同长度的记录分开归档。GPU r2虽已成功捕获目标Graph，但1024-token请求在后台准备完成前结束，实际 replay 为0，因此保留为未通过而不放宽验收；后续改用更长请求。

**当前证据。** 冻结 `01bc3770e32254e7088d50769c16b115c89dc9f3` 与标签 `experiment/sglang/pp-nixl-decode-graph-20261111-r3`，A100整模型往返通过：PP3三阶段各replay1013次，缩回PP2后两个保留阶段各1095次，退出阶段inactive／0次；对应活跃代Decode eager计数均为0。两个方向各2048-token输出与同暖512边界参考一致、无retraction／重新Prefill，最后CPU前缀实际loadback；仅实测B1，捕获档位为1／2／4。完整CPU回归508项通过，协议调整后73项harness复查通过，数量不累加。完整监督含清理177.895秒并正常release，最终交接为3.137／2.261秒，包含Graph重建。详细证据在 [VALIDATION](VALIDATION.md) 与[结果报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_nixl_decode_graph_20261111/summary.md)。不宣称暂停缩短、无中断或转换期间GPU已归还；worker继续常驻。
