# Graph 预捕获与持续 KV patch

2026-11-09。由主 agent 顺序实现，属于 [TODO](TODO.md) 的低干扰阶段，接在[按层物理映射](KV_LAYER_MAPPING.md)之后。代码在 Draft PR #4，尚未接入完整模型的在线切换。

**Graph：先准备少量布局，再选择。** `graph_bank.py` 在启动或明确排空的边界捕获，激活后禁止追加 capture。布局、batch、tensor 地址、shape、stride、dtype 和 device 全部匹配才 replay；其余情况统计 eager fallback。调用方必须登记权重、KV／GDN、输入、输出及 workspace，不能只检查 KV。页面未映射时拒绝执行；replay 和 eager 都把页面 pin 保留到完成事件。捕获失败或执行失败不能直接再跑一次，以免重复写入状态。当前模块没有替换 SGLang 的 `CudaGraphRunner`，真实通信预热、完整 binding 清单和 GDN 稳定地址仍待接入。

**Patch：先复制，再追新增行，最后短暂停追平。** `kv_patch.py` 维护迁移 generation 和每个 token 槽位的 incarnation，源写入记录完成事件后发布不可变 KV 行；独立 CUDA stream 等待事件并分批复制。复制期间允许新增其他行，被复制的槽位必须延后释放／复用，源和目标物理页也保持 pin。复用的槽位形成新 incarnation，必须重新复制；最终 fence 停止源写入并有界追平。失败时先排空已提交复制，再解除保护。当前 copier 是单 GPU、按 stride 复制的实现；跨 rank 传输、allocator 分配／释放钩子、请求新增／结束和 metadata 交接仍待接入。GDN 可变状态继续使用最终同步基线。

**真实 GPU 验证分两部分。** `hicache_transition_smoke.py --kv-foundation-root … --supervised-lifecycle` 在同一受监督 A100 租约内，先用第四张卡运行独立 VMM／Graph／Patch 检查，再用前三张卡运行 HiCache 整模型 PP2→PP3→PP2 基线。前者检查实际显存释放、保留地址／数值、真实 replay、fallback 和非空增量；后者继续使用 native KV 后端。二者成功不等于 KVCacheD 已接通 hybrid HiCache，也不等于整模型持续 patch 已通过。失败和清理记录均保留，整轮包括清理最多五分钟。

**接入顺序。** 先完成上述 GPU 检查，再把行生命周期钩子接到实际 allocator 与 KV 写入，随后接跨 rank transport；Graph 侧先准备真实执行布局和完整 tensor bindings。只有真实请求能在基础复制期间继续生成、最终交接后 token 与状态一致，才比较暂停和端到端性能。保留同步基线用于回退，不预设新机制一定更快。GPU 结果及精确版本写入 [VALIDATION](VALIDATION.md)。

**本轮结果。** 75项CPU检查通过。冻结 `8d7c21241c` 的 A100 `gpu-r1` 检查通过：物理页回收8 MiB，保留地址／数值；2次capture／2次replay，实际触发fallback和未映射拒绝；patch为2轮／4行，新增行和槽位复用检查通过。随后 native HiCache 整模型往返也通过同暖缓存连续性；全程186.509秒并正常清理／release。以上仍不包括完整模型的 Graph 复用、持续 patch 或性能收益。
