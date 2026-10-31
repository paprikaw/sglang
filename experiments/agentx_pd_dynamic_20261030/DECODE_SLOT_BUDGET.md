# P/D启动显存不足：Decode接收槽位与GDN预算

2026-10-31。失败目录：`runs/eight_gpu_pd_ep4_durable_r3`，0个正式episode；清理和GPU idle通过。原失败目录保留。

## 原因与证据

E5的`HybridMambaDecodeReqToTokenPool`会将实际GDN池扩为：

`max(配置GDN槽位, (活跃请求槽位 + 接收预留槽位) × (1主状态 + 2个ping-pong状态))`。

本次EP4每个Attention-DP worker的活跃请求上限16，默认接收预留32，配置GDN槽位80，因此实际分配为`max(80, (16+32)×3)=144`。KV容量估算的`_handle_max_mamba_cache`仍按显式80槽位扣除状态内存；这64个额外槽位约为3.838 GiB/GPU，未包含在该预算中。

[Decode日志](runs/eight_gpu_pd_ep4_durable_r3/cache/spartan-gpgpu171/pd_ep4.r0.log)记录四rank实际144槽位。rank1/2在KV分配阶段报CUDA OOM：需要2.09 GiB，仅剩2.00 GiB；其余rank的KV池完成后也只剩约0.52 GiB。Prefill侧保持80槽位。这个启动失败不能计作P/D性能差。

## 当前采用的配置修正

新目录`eight_gpu_pd_ep4_durable_slots8_r4`将原生参数`--disaggregation-decode-extra-slots`显式设为8。活跃请求上限16和配置GDN槽位80保持，`(16+8)×3=72≤80`，不再触发池增长。HiCache仍为每rank 32 decimal GB，显存静态比例仍为0.95，KV保留原生自动容量估算。

这是**接收队列容量调优与绕过方式，并未修复E5通用自动预算器**。8个接收槽位的吞吐影响需由实际实验评估；它不是“32个槽位没有成本”的替代证明。主实验的全部比较仍使用E5模型与内核。

`prepare_cluster_v14.py`在启动前校验队列状态上界可容纳，`cluster_runner_v9.py`在服务ready后校验实际GDN池大小与启动参数。P/D的串行正确性、每rank预热和正式回放流程保留。

## GPU启动与正确性验证

2026-10-31约04:21，新目录已通过两节点启动、实际GDN容量检查、串行输出与参考完全一致的检查，以及全部Attention-DP rank的冷／缓存命中预热，开始正式C8回放。

- Decode四rank均为接收预留8、GDN80、HiCache开启、CPU预算32 decimal GB、静态显存比例0.95；原生GPU KV容量为每rank2,195,008 token。
- Prefill四rank同样开启HiCache，GDN80；原生GPU KV容量为每rank2,225,472 token。
- [Decode ready](runs/eight_gpu_pd_ep4_durable_slots8_r4/nodes/spartan-gpgpu171/pd_ep4/ready.json)、[Prefill ready](runs/eight_gpu_pd_ep4_durable_slots8_r4/nodes/spartan-gpgpu011/pd_ep4/ready.json)、[输出一致性](runs/eight_gpu_pd_ep4_durable_slots8_r4/evidence/pd_ep4/serial_qualification.json)、[全rank预热](runs/eight_gpu_pd_ep4_durable_slots8_r4/evidence/pd_ep4/untimed_warmup.json)。

这些证据确认启动OOM已被当前配置避开，且实测正确性检查通过；尚不代表整个负载矩阵完成，也不能据此声称接收队列8与32的吞吐相同。

## CPU验证

[4项检查](decode_slot_budget_tests.log)执行固定E5原生构造函数的AST，只替换实际GPU分配调用：复现默认32槽位导致80→144，验证8槽位维持80、PP阶段160槽位可容纳对应队列，并验证单卡40槽位不能直接复用8个接收预留。不是另一份手写公式代替原生实现。

后续 P/D 完成 C8 三轮，但 C128 出现另一处可选 GDN 缓存检查点分配失败，已清理；见[运行期间的状态压力与 E6 修复](DECODE_CHECKPOINT_PRESSURE.md)。启动容量检查的通过不能替代运行期状态所有权检查。未启动的 E5 V14 P/D 候选也需先升级并验证该修复，不能直接进入高负载比较。
