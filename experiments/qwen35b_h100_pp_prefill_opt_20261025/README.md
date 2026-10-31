# H100 PP2 Prefill：已完成的局部优化

[完整结果、全部指标和归因](RESULTS.md) · [实验总览中的Q05](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q05.md) · [版本归档](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/qwen35b_h100_pp_prefill_optimization_20261025/manifest.json)

- 8请求、无旧缓存、各新增4096token：原生chunk8192相对4096吞吐 **+10.63%**，输出一致。
- 32请求、各GPU已缓存89,984token、各新增512token：保留chunk4096，发送就绪适配 **+4.08%**，输出一致。
- 有界合批、缩小chunk及长缓存强行放大chunk均未采用。没有Decode／TPOT提速结论；改动不自动替换共享引擎。

源码：`chunk_budget.py`、冻结运行中的`pp_ready_send.py`、`coalesce.py`；分析由`analyze.py`、`inspect_prefill.py`、`consolidate.py`重算。`profile_ready.py`在清理前等待两阶段gzip／JSON完整写盘。CPU测试11项通过。

## 原始设计记录

# H100：PP2 Prefill 调度优化

模型为 Qwen3.6-35B-A3B，BF16，完整 40 层；单节点四张 H100。
EP2 使用 GPU0/1，Attention 为两个独立副本；PP2 使用 GPU2/3，每个 stage 20 层。
复用完整 CPU 权重 Bank。请求的历史前缀由模型实际计算，正式计时前已经在 GPU 缓存，
不是合成 KV，也不把 CPU 前缀恢复时间混入测量。

## 本轮改动

只有 PP head 的输入接收过程增加可切换的、有等待上限的合批。
首次收到少于 8 个文本请求时，最多等 5 毫秒，收到 8 个即提前结束。
控制消息不等待。合批发生在向下一 stage 转发请求之前，两个 stage 收到同一组请求。
不等待已知的请求总数，不修改 Attention／Expert 内核、模型精度、路由或 DeepEP。
等待包含在 TTFT 和整批完成时间中；不把“等齐请求”偷偷放在计时之前。

## 数据点

| 请求数 | 每条已经缓存的历史 token | 每条新增 Prefill token | 输出 token | 用途 |
|---:|---:|---:|---:|---|
| 32 | 4,096 | 512 | 1 | 快速局部验证 |
| 32 | 89,984 | 512 | 1 | 复核此前长缓存、PP 吞吐较差的共同负载 |
| 8 | 0 | 4,096 | 1 | 检查普通冷 Prefill 是否退化 |

每个数据点先对精确形状预热，再对同一 PP 进程按原版／候选／候选／原版测试。
EP2 在另一对卡上补两个原版参考样本。准备缓存、预热和 profiler 不与正式样本重叠。
混合 Attention 的 GDN 状态会随新分支更新，所以每次正式样本前独立恢复相同初始缓存状态；
准备时间单独保存，不计入性能。每个正式请求必须验证历史缓存命中及实际输出长度。

短缓存点另做相同输入的逐请求输出一致性检查。
长缓存点正式测试后分别采集原版与候选的完整 Prefill 时间线。
所有运行均由 Experiment Guard 持有整节点租约和清理记录，不申请新节点。

## 入口

- [测试逻辑](optimization.py)
- [合批逻辑及适用边界](coalesce.py)
- [CPU 单元测试](test_coalesce.py)
- [冻结运行目录](runs/)

结果以运行目录中的 `analysis/REPORT.md` 为准；本文件中的方案不代表候选已获得加速。
