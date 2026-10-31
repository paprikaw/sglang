# GLM A100：统一节点内 Prefill chunk1024

| Workload | 配置 | 吞吐 tok/s | TTFT mean / median / P99 s | TPOT mean / median / P99 ms | E2E mean / median / P99 s |
|---|---|---:|---:|---:|---:|
| prefill_c4 | PP2 × EP4 | 1943.53 | 10.710 / 10.709 / 16.735 | — | 10.710 / 10.709 / 16.735 |
| decode_c4 | PP2 × EP4 | 101.92 | 0.413 / 0.487 / 0.487 | 38.292 / 38.275 / 38.538 | 19.981 / 19.970 / 20.092 |
| prefill_c4 | EP8 | 1322.77 | 18.254 / 18.340 / 24.734 | — | 18.254 / 18.340 / 24.734 |
| decode_c4 | EP8 | 100.06 | 0.438 / 0.521 / 0.521 | 39.045 / 38.884 / 39.660 | 20.390 / 20.391 / 20.465 |

同168/169、8×A100、GLM-4.7-AWQ、C4，Prefill8192→1／Decode128→512；每组1warmup＋1formal，全部16请求完成且cached_tokens=0。旧DeepEP/native9dc、SG870和计算后端均不变。

PP launch/global1024→local1024，EP8 launch/global2048→local1024；EP source capacity128→256。16 rank actual resolved schedule、两节点DP调整warning及正式窗口中的1024-token Prefill大小证明执行配置，而不是只读取参数。

正式窗口依JSON mtime−wall近似，serverlog只有秒级；仅取严格内窗。观察到的batch行数不是完整chunk计数，不能据此算执行速度或漏执行。startup resolved receipts和DP调整warning是主要证明，formal batch大小是辅助。

节点内chunk相同，不等于全局step相同：EP8每global step最多2048 tokens，PP1024；expert M、容量与stage工作分配随之不同。这是配置规范化比较，不是通信库优化或相同GEMM shape消融。

吞吐使用共同batch起点到最后HTTP完成的makespan，尾部写文件排除，先前客户端同步写入影响保留。mean/median/P99直接来自raw request＋SSE复算；n=4的P99仅描述本批次，不是在线tail。TTVOT和历史四行在results/metrics.csv。

PP unchanged与EP8改chunk的历史对照依然是先后部署；未做统计显著性判断。输出序列sanity单独保存，不宣称logits等价。CPU权重缓存仅缩短启动；不命中prefix/HiCache，不删除缓存。

Guard scientificpost、owned-process/GPU cleanup与全局lease释放通过。原runtime：`/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/glm47_a100_equal_local_chunk_20261010_r1`。
