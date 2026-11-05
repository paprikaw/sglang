# SGLang 研究基线

后续开发统一合入 `codex/pp-elastic-integration`。上游基点沿用已做 AgentX P/D 验证的 R11，新增功能通过合并保留原分支历史。原实验 worktree、冻结标签和尚未提交的工作均保留。

**引擎中的优化。** 包含 EP overlap、BF16 Expert runner、AWQ/Humming 专家切分、PP tensor channel 和调度、HiCache/GDN 恢复及 P→D 层映射、CPU canonical/node-local 权重缓存，以及排空与恢复 PP 的控制面。冲突优先保留较新的层映射、逻辑输出回传和 PP loadback 轮询协议，未将旧协议叠加到新协议上。

**显式选择的运行适配。** `runtime_profiles/` 保存原实验的源码，`runtime_sources.json` 的 `source_sha256` 记录取档时原始哈希，`sha256` 记录基线中的当前文件；后续兼容修复有 Git 提交。A100 BF16、PP2 producer-ready/typed-channel/lookahead、AgentX 观察和缓存诊断、122B 流水均衡，以及 GLM CE Prefill 都有对应源码。它们不是全局默认补丁；运行者必须选择模型、硬件和执行路径。旧 `prepare.py` 等脚本保留历史目录假设，不能当作新基线的通用启动器直接使用。PP2 快速协议仅验证固定合批、无 logprob 的负载，不能直接用于任意请求。

**外部依赖。** DeepEP 的 CUDA/通信实现属于独立仓库，按 `dependencies.lock.json` 固定版本和原生扩展哈希，不将其 Git 根合并进 SGLang。CE Prefill 与 Decode Graph 的适用范围不同。依赖锁保存部署来源；它不表示所有组合均已通过 GPU 验证。

**实验功能。** Qwen two-batch overlap、异构 TP/PP、PP speculation 和动态 PP 重配置仍受各自 opt-in 开关及配置限制约束。保留源码不等于完成硬件资格验证。`branch_coverage.json` 说明每个历史分支如何纳入或为何以参考形式保留。

**测试。** `gpu_smoke.py` 在 canonical Experiment Guard 下运行真实 A100 35B 模型，覆盖 EP2、EP4、PP2、EP2 Prefill→PP2 Decode，启用 HiCache，计数实际 Decode Graph 执行，并检查 CPU 恢复后的输出。该脚本是回归检查，不用于发布性能结论。CPU 回归及实际硬件验证结果见项目的 `VALIDATION.md`；未覆盖的模型、通信后端和在线扩缩路径需单独验证。

**开发方式。** 从此分支创建任务分支和独立 worktree；任务完成后带测试合回此分支。实验固定 commit、annotated tag、运行适配及依赖；冻结后不移动实验标签。不删除承载历史实验或未提交工作的旧 worktree。实验结果和密钥保存在 Git 之外。
