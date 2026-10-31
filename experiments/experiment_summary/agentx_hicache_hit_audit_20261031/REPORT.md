# AgentX：为什么计入 HiCache 后三种布局命中率仍不同

核查日期：2026-10-31。**CPU 离线核查，未新增 GPU 实验。** 本文使用同一 E5/V3 的 18 个已验收 episode、26,163 个调用；三种布局均为 P/D 合并。模型 Qwen3.6-35B-A3B，八张 H100、节点 011/171；引擎 `37e0828c642ce4f3795c4c4394f4ce913bcd1a35`。不混入后续 E7/E8、V4 或真实 P/D 分离结果。

**结论：HiCache 已经在起作用，但当前配置并没有保证三种布局保住同一批 KV＋GDN 检查点。单卡的有效 KV 容量明显较小；EP4 对 PP2 的剩余差距则不能只用总容量解释，已定位到子请求分布与少数缓存位置上的集中损失。现有日志没有记录每次 miss 的组件原因，尚不能把全部差异归因为 KV 淘汰或 GDN 丢失。**

## 1. 确认不是 CPU 命中漏算

每个请求都验证了 `cached_tokens = details.device + details.host`。下表分母均为全部原始输入 token，总命中是 `Σcached / Σinput`，不是逐请求百分比的算术平均。每布局 C128 三轮各 2,736 请求，合计 8,208；每轮输入总量相同，因此三轮总体比例也等于三轮比例的算术平均。

| 八卡部署 | GPU 命中 | CPU 命中 | 合计 | 每请求平均新算 token | C8 有命中、C128 降为零的请求数 |
|---|---:|---:|---:|---:|---:|
| 8×单卡 | 9.32% | 43.36% | **52.68%** | 40,158 | 2,665 |
| 2×EP4 | 79.08% | 4.97% | **84.05%** | 13,538 | 723 |
| 4×PP2 | 86.75% | 1.87% | **88.62%** | 9,659 | 234 |

最后一列是三轮合计、相对于同一源请求 C8 的观测差异，不是引擎记录的“驱逐事件数”。C128 由同八个源任务复制成十六份，副本之间缓存隔离。

**C8 下，三种布局的 171 个源调用逐条命中数完全相同，三轮均为 91.4619%，CPU 命中为零。** 这支持“差异随负载压力和运行历史出现”，不能进一步证明某一种淘汰原因。

## 2. 找到的配置问题：32 GB 优先于 ratio=2

启动参数同时存在 `hicache_ratio=2.0` 和 `hicache_size=32`。冻结源码中，只要显式 size 大于零，就用 GB 容量路径，**并没有得到两倍 GPU 池大小的 CPU 缓存**。

这 32 GB 还按 GPU 两类池的字节比例拆成 **KV 与 GDN 两份**。结果如下，GB 为十进制，容量按日志而不是参数名理解；实际分配含页/槽对齐，合计可略高于 32 GB：

| 八卡部署 | 独立缓存位置数 | C128 每位置根任务 | 每位置 GPU KV token | 每位置 CPU KV token | GPU / CPU GDN 槽 | 每卡 CPU KV / GDN GB |
|---|---:|---:|---:|---:|---:|---:|
| 8×单卡 | 8 | 16 | 338,816–339,392 | 1,131,968–1,132,480 | 40 / 137 | 23.18 / 8.82 |
| 2×EP4 | 8 | 16 | 2,225,472–2,226,048 | 1,402,112–1,402,112 | 80 / 52 | 28.72 / 3.35 |
| 4×PP2 | 4 | 32 | 3,770,368–3,771,584 | 2,755,136–2,755,264 | 160 / 118 | 28.21 / 3.80 |

PP 每个 stage 的槽只存本 stage 的层，两个 stage 服务同一批请求，不能把逻辑容量再乘二；PP 每位置承担的根任务数也为另外两种的两倍。EP4 使用 Attention TP1/DP4，一个 EP4 实例内有四个独立 Attention 缓存位置。单卡 CPU 137 个 GDN 槽由冻结模型每槽字节数及分配日志唯一反推；其余两个槽数也与启动 warning 一致。完整数值见 [容量表](cache_resources.csv)。

两个直接后果：

- 单卡虽然有 32 GB HiCache，实际用于 KV 的仅约 **23.18 GB**，GPU KV 仅约 **34 万 token/位置**；CPU 能装约 **113 万 token**。CPU 的 43.36 个百分点命中证明恢复确实发生，不能理解为“HiCache 没开”。它仍不是无限容量。
- EP4 的 CPU KV 140 万小于 GPU 的 223 万，PP2 的 CPU KV 276 万小于 GPU 的 377 万。两者启动日志明确提示主存池小于设备池。

本轮是 `write_through`，GPU 数据会备份到 CPU；两级副本会重叠，**GPU 槽数＋CPU 槽数不是不重叠的前缀容量**。备份也会因主存不足触发淘汰或失败。没有启用 storage backend，CPU 之外没有第三层持久缓存来兜住被淘汰的内容。

不过，**EP4 的整组 GPU KV 容量约 1,781 万、CPU KV 约 1,122 万，均高于 PP2 的约 1,508 万、1,102 万；EP4 命中却更低。** 因而不能以“PP 的总 KV 容量更大”解释 EP4/PP2 的 4.57 个百分点差异。必须检查每个位置的请求分布、GDN 状态与调度。

## 3. 定位到的损失：少数热点位置和子请求

将每一轮、每一副本、每一源会话、每一 turn 对齐，共 8,208 组请求：

- **7,069 组命中相同**；911 组 PP2 命中更多，228 组 EP4 命中更多。
- PP2 相对 EP4 净多复用 **31,838,272 token**，即每请求 3,879 token，分摊到输入量为 4.57 个百分点。
- 其中 **13,207,232 token（41.48%）来自子请求**。明确标记的 subagent 调用，PP2 合计命中 **93.75%**，EP4 为 **79.07%**。这些 subagent 在 PP2 的 C128 中逐条都保留了 C8 的命中量，EP4 有 293 个调用从 C8 有命中降为零。
- EP4 每轮**按子请求数选出的两个最忙位置**只占八个位置的四分之一，承担约 **28.1%–28.4%** 输入，却集中了相对于 C8 的净缓存损失的 **72.5%、75.8%、74.5%**。根任务输入本身是平衡的，子会话的首次放置后保持粘性仍会形成热点。

这说明“根任务数相同”“同一对话保持粘性”还不足以保证相同的缓存压力。位置上的更多子请求还会挤占根请求共享的 KV/GDN 资源。**这是已定位的相关关系；更换路由后能改善多少，仍须对照实验验证。**

另外，历史 hash 前缀核查中，将同一任务的其他位置也算入，额外机会仅约 **44–51 token/请求**，远小于这里的 3,879 token 差距。因此，“前缀只是留在另一个实例”不足以解释主要差异。hash 历史不是实际缓存库存；不能把它与真实命中的差值全部算成驱逐损失。[历史前缀位置核查](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/matrix_analysis/PREFIX_LOCALITY.md)

## 4. 一条能直接复核的同会话例子

源族 `a3344328…`，C128 第 2 轮、副本 6，根会话；turn 从 0 开始。该会话所有调用都保持各布局内同一个位置。EP4 为 `011 / DP1`；PP2 为 `171 / PP2.r0`；单卡为 `011 / single.r2`。

| turn | 输入 token | 同源 C8 命中 | EP4 GPU＋CPU 命中 | PP2 GPU＋CPU 命中 | 单卡 GPU＋CPU 命中 | EP4 距上一轮客户端完成秒数 |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 174,784 | 170,624 | 0 | 170,624 | 0 | 2.127 |
| 9 | 178,560 | 174,784 | 0 | 174,784 | 0 | 3.367 |
| 10 | 182,400 | 178,560 | 178,560 | 178,560 | 0 | 2.441 |

turn 9 时，**同一会话仅在上一调用完成 3.367 秒后发起下一调用，仍留在同一个位置；PP2 复用 174,784 token，EP4 两层均为零**。到 turn 10，EP4 又能复用。它不是换了源请求、只算 GPU 或整段对话始终没有共同前缀。[逐请求前后文](example_history.csv) · [更多成对例子](examples.csv)

但保存的 metadata 只给最终可用前缀，没有记录当时 full-KV 命中多长、GDN 校验在哪个边界失败；因此该例不能单独判定丢的是 KV 还是状态。

## 5. GDN 是为什么“CPU 还有 KV”也可能命中不了的关键

此模型既有 Full Attention，也有 GDN/linear attention。复用到深度 L，必须同时有前 L 个 token 的 KV 和 **深度 L 对应的 recurrent state**。只剩 KV，或者只剩另一个深度的状态，都不能直接按完整前缀复用。

冻结引擎实际执行这一约束：只有所有 cache 组件都有效，才更新最佳匹配边界；GDN 可以来自 GPU 或 CPU。CPU GDN 槽不足时会单独淘汰状态，KV 可能仍留着；之后匹配会缩短，甚至归零。备份分配失败也不是“已写入完整 CPU 缓存”。

本轮 `mamba_max_states_per_path=1` 只对符合条件的 **GPU** 状态做软限制，并保留已有 CPU 备份；不能把这个参数直接认定为全部历史状态永久丢失的原因。`enable_session_radix_cache=False`，没有启用额外的跨请求 session 路径保护；正在运行的请求仍有原生 lock 保护。单卡 40 个 GPU 状态槽还使实际并发上限从请求的 16 降到 **8**，启动日志明确写出按每请求 **5 槽预算**估算；“40 槽等于保住 40 个会话”是不成立的。

本轮并没有证明“GDN 一定是主要瓶颈”：单卡 CPU GDN 槽反而更多，不能把单卡低命中仅归因为 CPU GDN 不足。EP4/PP2 也都没有发生 request retraction；单卡命中低于 C8 的 2,871 个请求中，2,864 个未发生 retraction，因此 retraction 也解释不了大部分差异。

此外，普通日志里的 `full token usage`、`mamba usage` **扣除了可淘汰缓存**，不表示全部已占用缓存。看到 usage 未到 100%，不能据此排除缓存淘汰。

## 6. 当前能下的结论，以及最小的后续验证

当前已确认：计数正确、CPU 恢复发生、容量并不等效、差异随 C128 出现、热点和子请求贡献显著、同会话整段复用确实会丢失。**尚未获得每次 miss 的精确淘汰/匹配/恢复原因，不能声称已把 4.57 个百分点完全拆解。**

下一次只需做一个 cache 对照，不必扩展布局矩阵：沿用同一 C128，保留三种原部署，显式分别控制 **CPU KV 字节数与 CPU GDN 字节数**，而非仅设置共同的 `hicache_size` 让其按 GPU 池比例拆分。容量应由本轮前缀历史与主存可用量核定，并用事件计数确认是否仍有淘汰/备份失败；不要预设某个 GB 数必然足够。GPU GDN 增大还要重新做单卡显存可行性检查。

只补四类请求级观测即可收敛原因：

1. 查询时 `full_kv_hit_length`、最终 KV＋GDN 匹配长度、恢复后可用长度。
2. 按 KV/GDN、GPU/CPU 分开的淘汰数量和节点深度。
3. 备份/恢复分配失败原因，以及待完成写回是否覆盖该前缀。
4. 查询与完成时各位置的空闲槽、可淘汰槽、活跃请求数。

先保持当前路由做容量控制；若差距仍在，再对热点子会话做分布控制。这样才能区分容量、状态管理与到达/调度历史，避免同时改多项。研究自然部署性能时保留当前结果；研究 PP/EP 计算通信差异时另列缓存控制结果，二者不能合并成同一因果结论。

## 可审计来源

- [显式 HiCache 容量覆盖 ratio](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/pool_host/base.py:151)。
- [32 GB 按 GPU KV/GDN 字节比例拆分](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:116)。
- [混合模型同时创建 KV 和 GDN 主存池](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:702)。
- [所有组件有效才更新最佳前缀边界](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:689)。
- [GDN 检查点允许来自 GPU 或 CPU](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py:141)。
- [CPU KV 不足时淘汰；无法腾出则备份失败](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/unified_radix_cache.py:1294)。
- [GDN CPU 槽不足时淘汰，仍不足则回滚](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py:838)。
- [GDN Host 淘汰可单独删除状态](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py:842)。
- [每路径 cap 只淘汰符合条件的 GPU 状态，保留已有 CPU 备份](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py:267)。
- [日志 usage 扣除了可淘汰缓存](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py:245)。
- [首次 Prefill chunk 的 GPU/CPU 命中分项](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031/python/sglang/srt/managers/schedule_batch.py:2479)。

[18 轮汇总](trials.csv) · [三布局缓存指标](full_metrics.csv) · [26,163 个请求](requests.csv) · [8,208 组成对请求](paired_requests.csv) · [各缓存位置](owner_metrics.csv) · [子请求类型](source_kind_metrics.csv) · [热点集中度](concentration.csv) · [来源哈希和边界](audit.json) · [复算脚本](analyze.py)

原始大日志保留在原 run；本目录只保存派生表与来源校验。没有修改引擎、原始结果或正在运行的实验。
