# EP／PP／P/D 集成基线验证

2026-11-05。共同分支为 `codex/pp-elastic-integration`，基点为 R11 `b862312ce3`。运行记录和完整检查清单位于[验证 manifest](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/validation_manifest.json)。这次验证执行正确性和开关实际生效，不据此发布性能提升结论。

**CPU 回归。** 共 537 项通过，零失败、零跳过。覆盖 EP／BF16 接口、量化参数转发、PP 通信与微批、HiCache 共识、GDN 和 P/D 状态交接、CPU 权重缓存，以及 elastic PP 控制与状态迁移。结果见 [JUnit](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/final_cpu_tests.xml)，52 个测试文件列于 [cpu_test_paths.txt](cpu_test_paths.txt)。使用锁定 Python，设置引擎 `python/` 的 `PYTHONPATH`、`/home/bxb1/conda/lib` 的 `LD_LIBRARY_PATH`，关闭 pytest 自动插件加载并显式加载 `pytest_asyncio.plugin`，将清单中的文件传给 pytest 即可复跑。

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
