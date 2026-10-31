# AgentX输入与缓存协议核查

依据本地保存的官方harness提交 `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf` 和已下载256k原始数据，2026-10-30复核。

## 下轮输入是否包含本次实时生成内容

Weka loader的 `get_default_context_mode()` 与 `_resolved_context_mode()` 均返回 `DELTAS_WITH_RESPONSES`。该模式累积trace重建的预置assistant片段，**丢弃实时推理返回的assistant内容**。`UserSession.should_store_response()` 只有在 `DELTAS_WITHOUT_RESPONSES` 才返回true。

当前实验按原始hash块独立重建各轮input_ids，不把本模型生成的output_ids写入下一轮。这与上述“下一轮输入不由本次实时输出决定”的官方Weka语义一致。不能因为模式名包含WITH就误认为官方会保存live responses。

相应边界也必须保留：输入前缀的复用关系由trace重建；本次生成token不保证成为下一轮相同token前缀。因此它不是原始文本上的真实agent任务执行，也不应自动推广到真实响应反馈下的全部缓存收益。

源码：

- [模式定义](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/common/enums/enums.py:192)
- [Weka默认模式](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/dataset/loader/weka_trace.py:844)
- [实际响应保存条件](../../datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8/src/aiperf/workers/session_manager.py:151)

## Hash命名空间

重新遍历256k原始文件：393个trace全部声明 `hash_id_scope: local`。所选8个trace也均为local，块大小64。当前 `(repeat/copy namespace, graph_id, hash_id)` 键在同一父/子代理家族内共享内容，并隔离不同trace及负载放大副本；未把跨trace相同数字ID误当作共享prefix。

官方loader也对每个trace清理生成缓存并设置trace ID，子代理使用父trace的命名空间。393个trace不是一个全局共享hash空间。

## 当前实验与官方AgentX分数的范围

当前保留完整选定会话、原始长度、依赖、子代理与工具等待，以完整有限episode为测量边界；仅将全系统无推理时超过10秒的空闲压缩至10秒。它直接发送input_ids，未采用官方chat消息渲染、trajectory中段采样、会话边界priming及官方计分。因此始终标为 **AgentX派生系统实验**，不称为官方AgentX分数。

官方 `AgenticReplayStrategy` 可以从trajectory中部开始，在WARMUP回放先前请求恢复前缀，再进入PROFILE；这些语义与当前从会话开头执行的协议不同，不能把两类结果直接合并。
