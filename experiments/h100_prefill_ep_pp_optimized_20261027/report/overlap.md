## Existing-trace check: communication/computation overlap

复查B32、P=0、Q4096重复文本的两张GPU独立Prefill trace，没有启动新GPU实验。每张卡640次实际Prefill层调用，重新归因的GEMM与通信均值与主报告吻合。

| GPU kernels / card / actual Prefill layer | µs |
|---|---:|
| Dispatch | 324.82 |
| Routed Expert GEMM | 484.73 |
| Combine | 259.22 |
| Dispatch+combine 与 routed GEMM 的时间交集 | 0.00 |
| Dispatch+combine 与 Attention 的时间交集 | 0.00 |
| Dispatch+combine 与 shared expert 的时间交集 | 3.98 |

Shared Expert本身约46.74 µs/card/layer；现有shared-expert overlap只与约0.68%的通信GPU时间重叠。这是GPU时间轴交集，不等于已经确认的端到端可节省时间。通信口径继承主报告的dispatch/combine作用域内kernel，并非纯NVLink传输线时间。

同一批token有 Dispatch(A) → GEMM(A) → Combine(A) 的依赖。可利用独立请求微批：Dispatch(B)与GEMM(A)重叠，Combine(A)与GEMM(B)重叠；也可用另一批的Attention覆盖通信。优先保留足够大的Expert输入形状，避免为重叠切得过细而损失GEMM效率。

忽略启动/排空、所有竞争及分块开销，仅把通信和GEMM看作理想独立流水资源，稳态工作间隔的乐观模型为 max(584.04,484.73)=584.04 µs，相比两项串行和1068.77 µs有484.73 µs的理想覆盖空间。这不是单批延迟、实际two-batch预测或E2E加速估计。Normal DeepEP通信使用SM，需与GEMM分配SM/带宽；分块后的两项时间也须重新测量。

当前实验two-batch overlap关闭。源码的Qwen shared-expert overlap与TBO是两种机制；现有TBO操作策略列出了Qwen3Moe等支持路径，当前Qwen3.5/3.6混合Full/GDN Attention的forward不在该Qwen3Moe操作分解路径中，不能据一个开关认定适配完成。此前暂停的TBO适配本次没有恢复。

因此主报告的PP吞吐优势是当前运行配置的结果，尚不能当作充分优化过通信/计算重叠的EP上限。完整两rank时间交集、CUDA stream与源文件SHA256见[overlap_results.json](./overlap_results.json)。
