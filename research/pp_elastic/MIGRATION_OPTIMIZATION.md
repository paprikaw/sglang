# 迁移与切换优化记录

本轮目标是同时减少完整迁移时间、最终交接暂停和迁移期间对生成的干扰。每次改动记录原因、实际测试及负面结果；正确性条件保持不变。以已合并的 `5f530f6de33d` 为起点，任务分支 `codex/pp-migration-latency-20261111`。

**原始观察。** 已归档的启动预捕获实验中，PP2→PP3／PP3→PP2后台准备19.799／20.438秒，最终交接0.604／0.688秒。每次目标权重虽然已由启动Graph预热保留，后台仍把所有转移层经GPU→CPU→GPU重新传输。这里是单轮历史观察，没有逐token停顿或重复性能结论。原始依据见[报告](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_prepared_decode_graph_20261111/summary.md)。

**优化一：直接复用已经常驻的不可变权重。** 启动仍正常加载、传输与捕获全部候选，完成后记录每层持久状态的地址、形状、类型及可用的版本计数。运行时挂接目标模块后检查这些条件，再标记目标层就绪；全组确认采用同一策略，才跳过重复传输。禁止该模式下的在线权重更新。保留 `--elastic-pp-prepared-weight-policy recopy` 作为同版本对照。它不减少已常驻候选的显存，也不省略KV／GDN／HiCache迁移。目前实现中，CPU和GPU结果待记录。

**测量方法。** 补记准备、权重、KV patch收尾、GDN／KV交换、HiCache、模型提交、Graph选择及各次共同确认的本地耗时；它们有嵌套，不相加估算总时长。端到端另记流式输出时间与chunk大小，只有逐次单token输出才称为逐token间隔。比较相同源／目标、输入、缓存、Graph及NIXL条件下的reuse和recopy；保留每次原始值与正确性结果。未运行、未分离及失败的项目如实标记。

**后续选择。** 先测出复用权重后的主要耗时，再决定是否优化最终KV／GDN／HiCache交接或在线patch频率；不能由总耗时推断单个组件。worker退场和完整VMM接入仍沿用原定后续范围。
