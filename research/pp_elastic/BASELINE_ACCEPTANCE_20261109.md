# Naive PP＋HiCache 基线：实现状态与合并验收

2026-11-09。本页总结已经实现的范围及本次合并证据；持续进度仍只在 [TODO](TODO.md) 勾选。它是显式 opt-in 的实验基线，下一阶段的低干扰机制尚未启动。

**已实现什么。** 同一 resident 实例可以在固定 TP1／EP1 下改变 PP 层归属，完成 `(20,0,20)→(13,13,14)→(20,0,20)`。迁移保持活跃 KV／GDN、请求进度与 CPU-only HiCache 前缀，并处理 dormant rank、缓存上下文重绑和在途读写的安全边界。三个 worker 始终常驻；这不是已经把第三张工作 GPU 归还给其他任务。

**当前有哪些策略。** sync、background、batched 复用共同事务数据面。background 只预搬不可变权重，batched 在暂停期间分块复制，三者的可变 KV／GDN 仍在最终 fence 同步。native CPU spill 先备份到 CPU authority、释放源 backing、重建目标池，再恢复状态；它不是在线 KV 容量调整。发布前失败需要全 rank 确认源恢复，发布后失败停止服务。CPU spill 强制受支持的 native／sync 或 batched／Graph-off 配置；动态 P/D 及其他未支持组合继续拒绝。

**已经通过的 GPU 边界。** r14 在 `5fa14eae1df0` 完成同暖512边界的 sync/cpu_spill 扩缩往返，观察到 CUDA backing 实际释放与恢复、精确续写、零 retraction／再次 Prefill、双向 CPU 前缀搬运及最终新的 loadback。完整监督含清理163.294秒，独立审计通过。结果 SHA256 前12位 `ff753f536a37`，独立审计 `6afd282252d8`；审查者核对其17个输入哈希。冷暖生成等价仍不合格，不能把同暖边界资格扩大成任意缓存历史下的生成等价。

**开销测到什么。** r15 在 `8374b0276d19` 每方向各有一条参考与一次转换，完整监督186.775秒。扩／缩请求 E2E 差分别为4570.757／3215.168 ms，各 rank 平均 fence 为3389.554／3356.288 ms。Graph 关闭，样本每方向 n=1；E2E 差还含目标布局速度和控制影响，不是孤立暂停或统计收益。结果 SHA256 前12位 `25dea5a4d093`。逐 token 间隔、隔离阶段成本和严格峰值仍待测。

**本次组合回归。** 在独立候选提交 `de7808daeae6` 上执行742个去重后的 CPU 用例：原历史537个节点与新增205个节点。全部通过，失败／错误／跳过均为0，pytest用时106.113秒，运行前后源码不变，CUDA可见设备为0。覆盖已有 EP／PP／P-D、HiCache、Graph 缓冲、权重缓存及新增事务、容量、backend 视图、验收与清理检查；CPU检查不能代替新的GPU硬件资格。第一次收集包装器未识别 pytest 相对 test 根的路径，尚未执行测试即停止；纠正解析后复用同字节收集结果，原始设置失败单独保留。

**为什么没有重跑 GPU。** 候选、Runtime `8374b0276d19` 与 r14 的引擎 `python/` tree 均为 `35126c909ff08980928defd7d7f0ce21f2a5df7a`，运行适配 tree 均为 `69112aa58357d40daa97adc843994742dfc4a30f`。Runtime 通过 [PR #2](https://github.com/paprikaw/sglang/pull/2) 合入项目的提交 `3a72d5781835` 与候选的引擎、测试及运行适配 tree 一致；因此本轮保留既有GPU证据的有限资格，补组合CPU回归，而不宣称新跑过静态EP／PP或动态GPU。独立只读审查未发现阻止作为该范围实验基线合入的问题。

**尚未完成什么。** 真实 GPU OOM／迁移失败回退、background／batched 的 GPU 资格、持续 KV patch、在线 KV 池扩缩、候选 CUDA Graph 预捕获、真正归还工作 GPU，以及动态 P/D 接入仍未完成。静态路径保留既有显式优化，不把它们的旧GPU资格扩展为所有动态组合都已通过。

**证据与入口。** 验收产物统一位于 `baseline-acceptance-20261109/`：`metadata.json`、`nodeids.txt`、`junit.xml`、`pytest.log`、`source_equivalence.json`、`project_merge.json` 和 `independent_review.json`。历史GPU依据为 `gpu-hicache-r14/results.json` 及 `peer_review/independent_r14_audit.json`、`gpu-hicache-r15/REPORT.md` 与原始结果。它们归档在现有 source-integration 实验目录，未复制敏感请求或原始日志到仓库。合并完成状态及冻结标签见 [TODO](TODO.md)；总体路径为 Runtime／Cache → 项目分支 → [总 PR #1](https://github.com/paprikaw/sglang/pull/1) → `codex/pp-elastic-integration`。

**合并与冻结。** Runtime PR #2 以 `3a72d5781835` 合入项目，总 PR #1 以 `5542d19d90d8` 合入共同基线。主线工作区随后对受支持配置限制、Runtime backend 视图及 BF16 EP 适配复查29项CPU检查，全部通过、零失败／错误／跳过，pytest用时5.422秒；这是定向重复检查，不与742相加为独立覆盖。冻结标签为 `baseline/sglang/pp-hicache-naive-20261109`，后续任务从该标签建立分支。最终代码等价与远端核对另存 `release_manifest.json`。
