# SGLang 研究基线

2026-11-11：native／HiCache／NIXL持续迁移已接通完整Decode CUDA Graph，A100 PP2→PP3→PP2有实际replay及2048-token连续性证据。当前在最终排空时重新捕获，启动预捕获库／VMM组合未接通，worker继续常驻，见[接入说明](DECODE_GRAPH_INTEGRATION.md)。代码仍在Draft PR #4，未合入共同主线。

2026-11-09：naive PP＋HiCache 的实现范围、组合回归及合并依据见[本次基线验收](BASELINE_ACCEPTANCE_20261109.md)。低干扰机制已开始实现[按层 KV 映射依赖](KV_LAYER_MAPPING.md)及[Graph 预捕获／持续 KV patch](GRAPH_KV_PATCH.md)；真实 GPU、缓存接入和后续机制的完成状态以[TODO](TODO.md)为准。

当前 PP＋HiCache 实现、验收和后续优化统一追踪于[任务清单](TODO.md)。每个 agent 交付时更新对应条目，规则见[协作约定](WORKFLOW.md)。

后续开发统一合入 `codex/pp-elastic-integration`。上游基点沿用已做 AgentX P/D 验证的 R11，新增功能通过合并保留原分支历史。原实验 worktree、冻结标签和尚未提交的工作均保留。

**引擎中的优化。** 包含 EP overlap、BF16 Expert runner、AWQ/Humming 专家切分、PP tensor channel 和调度、HiCache/GDN 恢复及 P→D 层映射、CPU canonical/node-local 权重缓存，以及排空与恢复 PP 的控制面。冲突优先保留较新的层映射、逻辑输出回传和 PP loadback 轮询协议，未将旧协议叠加到新协议上。

**显式选择的运行适配。** `runtime_profiles/` 保存原实验的源码，`runtime_sources.json` 的 `source_sha256` 记录取档时原始哈希，`sha256` 记录基线中的当前文件；后续兼容修复有 Git 提交。A100 BF16、PP2 producer-ready/typed-channel/lookahead、AgentX 观察和缓存诊断、122B 流水均衡，以及 GLM CE Prefill 都有对应源码。它们不是全局默认补丁；运行者必须选择模型、硬件和执行路径。旧 `prepare.py` 等脚本保留历史目录假设，不能当作新基线的通用启动器直接使用。PP2 快速协议仅验证固定合批、无 logprob 的负载，不能直接用于任意请求。

**外部依赖。** DeepEP 的 CUDA/通信实现属于独立仓库，按 `dependencies.lock.json` 固定版本和原生扩展哈希，不将其 Git 根合并进 SGLang。CE Prefill 与 Decode Graph 的适用范围不同。依赖锁保存部署来源；它不表示所有组合均已通过 GPU 验证。

**实验功能。** Qwen two-batch overlap、异构 TP/PP、PP speculation 和动态 PP 重配置仍受各自 opt-in 开关及配置限制约束。保留源码不等于完成硬件资格验证。`branch_coverage.json` 说明每个历史分支如何纳入或为何以参考形式保留。

**测试。** `gpu_smoke.py` 在 canonical Experiment Guard 下运行真实 A100 35B 模型，覆盖 EP2、EP4、PP2、EP2 Prefill→PP2 Decode，启用 HiCache，计数实际 Decode Graph 执行，并检查 CPU 恢复后的输出。该脚本是回归检查，不用于发布性能结论。CPU 回归及实际硬件验证结果见项目的 `VALIDATION.md`；未覆盖的模型、通信后端和在线扩缩路径需单独验证。

**开发方式。** 从此分支创建任务分支和独立 worktree；任务完成后带测试合回此分支。实验固定 commit、annotated tag、运行适配及依赖；冻结后不移动实验标签。不删除承载历史实验或未提交工作的旧 worktree。实验结果和密钥保存在 Git 之外。

**GitHub 协作。** naive PP＋HiCache 的[总 PR #1](https://github.com/paprikaw/sglang/pull/1) 已合并。低干扰机制继续以 `codex/pp-hicache-autoscaling` 为项目集成目标；本轮用户要求由主 agent 自己实施，不启动子 agent。任务仍使用独立分支与 worktree，先审查合入项目分支，组合验证后再整体合回共同基线。按可检查的小步提交并推送，任务关系和验证状态记录在 PR 中，详见[协作约定](WORKFLOW.md)。
