# Qwen35B 的P/D交接还包含循环状态

这是根据本次模型配置和运行源码计算的字节数，**不是RDMA字节或传输耗时的实测**。

本次35B模型有30层GDN、10层完整Attention。Attention TP=1（MoE EP4配合Attention DP4），KV和卷积状态用BF16，SSM状态用FP32。当前Mooncake路径每次P→D发送一个完整循环状态槽。

| 项目 | 全模型、每请求状态大小 |
|---|---:|
| SSM：30×32×128×128×4 byte | 60.000 MiB |
| 卷积状态：30×8192×3×2 byte | 1.406 MiB |
| GDN合计 | **61.406 MiB** |
| 每个实际传输KV token：10层×K/V×2头×256维×BF16 | **20 KiB/token** |

因此，即便某次只传64个KV token，KV本体为1.25 MiB，循环状态仍为61.406 MiB；两者合计62.656 MiB，尚未计控制消息和传输粒度。这个例子不代表每次AgentX调用都只传64 token。

对本次问题的含义：**高前缀命中减少新增Prefill工作，但不会自动消除P/D的循环状态交接。** 因此，在AgentX中判断P/D是否划算，不能只看输入/输出token比例或假设只移动增量KV。

当前还不能据此把延迟差异归因于这61.406 MiB。实际KV传输范围、GDN/KV传输时间、排队和overlap都需要单独计量；P侧miss token数与D侧缓存命中也不能直接当成网络字节。EP/PP布局切换与一次普通P→D交接的成本也不同。

数值与文件哈希见 [state_transfer_geometry.json](state_transfer_geometry.json)。直接源码依据：

- [GDN状态形状](../../.engine-worktrees/agentx-gdn-pd-hicache-r3-20261030/python/sglang/srt/configs/mamba_utils.py:219)
- [Qwen模型的形状参数](../../.engine-worktrees/agentx-gdn-pd-hicache-r3-20261030/python/sglang/srt/configs/qwen3_next.py:293)
- [完整状态槽传输](../../.engine-worktrees/agentx-gdn-pd-hicache-r3-20261030/python/sglang/srt/disaggregation/mooncake/conn.py:1426)
