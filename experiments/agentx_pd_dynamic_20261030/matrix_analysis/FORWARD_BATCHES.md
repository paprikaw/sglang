# 已验收E5基线：实际本地Decode batch

18个已验收episode；每组C8/C128各三轮。使用原来保存的ModelRunner计数，没有新增GPU测量。

|8卡布局|根会话上限|独立缓存位置|本地纯Decode batch均值|
|---|---:|---:|---:|
|2×EP4|128|8|5.622|
|2×EP4|8|8|1.065|
|4×PP2|128|4|5.036|
|4×PP2|8|4|1.130|
|8×single|128|8|2.212|
|8×single|8|8|1.037|

每轮先以纯DECODE模式的本地forward调用数加权batch，再对三轮取算术平均。PP只计stage0；其单次forward batch不表示整个实例全部在途请求。EP使用独立Attention-DP rank。不含EXTEND/MIXED中的请求，不是时间加权并发或全EP组总token，更不是Expert层接收的token-expert行数。各布局状态预算与KV容量差异仍按主报告保留，不能将batch均值差异单独作为速度差异的原因。

Python dispatch_output计数不能覆盖CUDA Graph逐次重放，因此即使LL计数为0，也不能声称实际未使用LL或给出通信后端使用比例。已核对ModelRunner在Graph路径直接调用graph runner、后者调用backend.replay；未增加GPU侧计数。

[完整指标与容量](REPORT.md) · [数值表](forward_batch_means.csv) · [来源](forward_batch_sources.json)

- [eight_gpu_col_ep4_durable_r3：逐rank、原始batch计数和每轮结果](../runs/eight_gpu_col_ep4_durable_r3/analysis/FORWARD_BATCHES.md)
- [eight_gpu_col_pp2_durable_m16_r3：逐rank、原始batch计数和每轮结果](../runs/eight_gpu_col_pp2_durable_m16_r3/analysis/FORWARD_BATCHES.md)
- [eight_gpu_col_single_durable_kv40_r3：逐rank、原始batch计数和每轮结果](../runs/eight_gpu_col_single_durable_kv40_r3/analysis/FORWARD_BATCHES.md)
