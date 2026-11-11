# 持续 KV 迁移中的 Decode CUDA Graph

2026-11-11。用户要求先接通 Graph，worker 常驻暂不处理。本轮由主 agent 实施，接在 [NIXL 基线](NIXL_KV_TRANSFER_DESIGN.md) 上，进度统一记录在 [TODO](TODO.md)。

**先接通正确路径。** 启动时捕获初始非空阶段的 Decode Graph；旧布局继续用 Graph 推理并通过 NIXL 发送 KV。最后排空全部在途请求／通信，提交目标权重、KV／GDN、HiCache 和路由，销毁旧 Graph，使用 SGLang 原生 runner 捕获目标 Graph。所有 rank 确认后才恢复推理。空阶段不捕获；再次回到旧布局也重新捕获，因为 native 池地址已经变化。

**捕获不能覆盖请求。** 原生 runner 在捕获前清空索引缓冲，使用保留的 request、KV 和 GDN 槽位 0。重建前检查 request 行 0 与 GDN 映射仍为空；不允许用活跃请求预热。捕获失败不伪装为 eager 成功；目标已经提交时停止服务，不能继续使用旧代。

**范围。** native KV、固定 TP1／EP1、overlap backing、full Decode Graph，Prefill 仍用 eager；不接 torch.compile、speculative、Prefill Graph 或 CPU-spill Graph。HiCache 与 CPU shadow／NIXL 在线 patch 均保留。有限启动预捕获库与稳定地址池仍是后续优化，本轮明确在最终交接期间重新捕获，会增加该阶段时长。

**如何验收。** 逐 rank、逐 generation 记录 capture 秒数、实际成功 replay 和 Decode eager 次数。`hicache_transition_smoke.py --cuda-graph --live-kv-patch --kv-patch-transport nixl --nixl-root …` 要求 PP2→PP3→PP2 每个源／目标活跃阶段都实际 replay，新增阶段生效、退出阶段无 Graph，NIXL 暂停前已有真实写入，输出与同暖缓存参考一致且没有重新 Prefill。`--active-output-tokens` 可在冻结前选择 256–1024，避免 Graph 加速后请求过早结束；不同长度的记录分开归档。

**当前证据。** 首轮 elastic_pp CPU 回归 503 项通过，随后补充启动空阶段与目标重建检查；最终 CPU／GPU 结果在 [VALIDATION](VALIDATION.md) 更新。尚未据此宣称暂停缩短、无中断或 GPU 已归还。
