# Q02：PP2 发送就绪优化与正确性诊断

日期：2026-10-25。Prefill 在本轮已预热的受控测试中输出等价通过。

## 1. 先说明本轮测的是什么

| 术语 | 本轮准确含义 |
|---|---|
| Prefill | 同时放行 8 条请求；每条**没有已缓存上下文**，处理 4096 个新输入 token，生成 1 token；整批新输入 32768 token。每批清空原生 GPU 缓存。 |
| Decode 回归 | 8 条请求，每条先独立重建 4096 个真实模型产生、驻留 GPU 的旧缓存，然后新增输入 1 token、生成 128 token。前缀构建在计时外；每批都重新构建，不让上下文持续增长。没有 CPU HiCache 换入。 |
| PP2、两槽 | 同一节点两张 GPU，每张负责连续 24 层；每阶段 Attention TP1、Expert EP1。两槽是调度循环批次位置，不是两个模型副本。Prefill 每次每卡实际处理 1 条请求的 4096 个新 token。 |
| TTFT、TPOT、端到端 | 客户端请求放行至首输出；首输出至末输出除以输出数量减一；放行至末输出。单位分别是秒、毫秒、秒。Prefill 仅输出 1 token，TPOT 不适用。 |
| 吞吐 | 每一整批输入或输出 token 数除以整批最后完成时间，再对正式批次取均值。Prefill 单位 input token/s，Decode 单位 output token/s。 |
| 均值 / 中位数 / P99 | 延迟表每格按此顺序；Prefill 每策略 2 批共 16 请求，Decode 每策略 2 批共 16 请求。P99 为最近秩样本最大值，不是稳定服务尾延迟估计。 |
| GPU 分项 | 独立 profile，按每卡每次层调用统计 GPU 活动并集。Attention 包含投影；Routed 是被路由的专家，Shared 是共享专家。NCCL 驻留包含等待，不能当作纯 NVLink 传输时间。 |

设备固定为 spartan-gpgpu069 GPU0/1、2×A100 80GB、NV12 NVLink。Guard 登记整节点 4 GPU，实际只用两卡，不申请新节点。
完整 Qwen3.5-122B-A10B-FP8：48 层、256 路由专家、每 token top-8；36 层线性 Attention、12 层 Full Attention。
A100 运行 W8A16 Marlin、BF16 激活和 KV、Triton Attention；CUDA Graph 与 overlap schedule 均关闭。原版和候选复用同一个已经加载的完整 CPU 权重 bank 和同一个模型进程。

## 2. 改动及受控方式

原生发送函数先把张量元数据通知另一阶段；数据若还没算好，接收阶段的 NCCL kernel 可能提前启动并等待，占用计算资源。
候选在发送端当前流已经接入生产事件依赖的位置，等待 CUDA 事件完成，然后再调用**同一个原生发送函数**发布元数据。
只改变通知时机，不改模型、张量、精度、专家路由或计算内核。等待仍在请求计时内；没有把开销移到计时外。

两个策略各预热两批；正式 Prefill 顺序为**原版→候选→候选→原版**。之后分别独立采一批 profile，不计入性能。Decode 使用原版→候选→原版→候选，每批重新准备真实 GPU 前缀。记录完整生成 token IDs 和文本哈希。
代码是实验私有 overlay，**尚未默认写入共享引擎**；也没有声称能推广到其它槽数、PP 深度、模型或链路。

## 3. 请求性能

| 负载 | 策略 | 吞吐 ↑ | TTFT 秒 ↓ | TPOT 毫秒 ↓ | 端到端秒 ↓ | 正确性状态 |
|---|---|---|---|---|---|---|
| prefill | 原版 | 8160.97 | 2.3980 / 2.3970 / 4.0295 | 不适用 | 2.3980 / 2.3970 / 4.0295 | 输出一致 |
| prefill | 发送就绪候选 | 9391.74 | 2.1383 / 2.1397 / 3.4899 | 不适用 | 2.1383 / 2.1397 / 3.4899 | 输出一致 |
| decode | 原版 | 87.20 | 0.1943 / 0.1808 / 0.2394 | 90.7469 / 90.6737 / 91.6095 | 11.7192 / 11.7192 / 11.7555 | 仅诊断，输出等价未通过 |
| decode | 发送就绪候选 | 86.60 | 0.1964 / 0.1818 / 0.2416 | 91.3756 / 91.3834 / 92.3222 | 11.8011 / 11.8013 / 11.8467 | 仅诊断，输出等价未通过 |

同进程正式 Prefill，候选相对原版吞吐变化 **+15.08%**，平均 TTFT 降低 **10.83%**。是否可使用该收益，须同时看第 5 节的正确性状态。

## 4. 时间线归因：不是只看最终吞吐

每卡每层单位为毫秒；最后两列为该 trace 的 kernel 最长驻留和 Attention 与 NCCL 的时间交集。
GPU 分项存在重叠，不能把这张表相加当成请求延迟。两阶段各 8 请求×24 层，共 192 次相同本地形状的层调用。

| stage | 策略 | Attention 毫秒/层 | 其中 Marlin 投影毫秒/层 | Routed Expert 毫秒/层 | Shared Expert 毫秒/层 | 最长 NCCL 驻留毫秒 | Attention/NCCL 时间交集毫秒 |
|---|---|---|---|---|---|---|---|
| 0 | 原版 | 9.2318 | 6.3800 | 8.3115 | 0.7755 | 301.51 | 1035.91 |
| 0 | 候选 | 7.9523 | 4.9118 | 6.9984 | 0.6174 | 0.10 | 0.06 |
| 1 | 原版 | 7.8315 | 4.8379 | 6.9612 | 0.6077 | 301.10 | 0.00 |
| 1 | 候选 | 7.9022 | 4.8810 | 6.9780 | 0.6149 | 0.18 | 0.00 |

本轮第一阶段的 Attention 从 9.2318→7.9523 毫秒／层，其中投影从 6.3800→4.9118（降低 23.01%）；Routed Expert 从 8.3115→6.9984（降低 15.80%）。第二阶段投影 4.8379→4.8810，基本不变。改动主要缓解原先第一阶段的争用，而不是把所有层的数学计算做得更少。

第一阶段 NCCL kernel 活动并集从 2399.68→1.85 毫秒、与 Attention 的交集从 1035.91→0.06 毫秒；原来最长约 301.51 毫秒的 NCCL 驻留降到 0.10 毫秒。**这些长时间主要是等待，不是物理搬运变快了几千倍。** 等待改在发送端数据就绪事件之前由 CPU 承担，避免对端过早启动等待 kernel；等待仍计入请求时间。这组对照连同 profiler 关闭时的 15.08% 吞吐改善，支持“提前发布导致的 NCCL 等待干扰计算”是可优化慢点。

此次仍有 PP 顺序依赖、边界传输、每层更多专家权重等成本；不能把全部 EP／PP 差距都归因于这个问题。此轮没有重新测 EP2，也没有测试四槽候选，不能直接宣称候选在新一轮配对中胜过 EP2 或适用于四槽。

## 5. 正确性、此前失败与验收范围

| 检查 | 原版重复稳定 | 候选重复稳定 | 跨策略完整 token 一致 | 判定 |
|---|---|---|---|---|
| warmup | True | True | True | passed |
| prefill | True | True | True | passed |
| decode | False | False | False | inconclusive_original_unstable |

- r1：正式 Prefill 吞吐 8149.01→9360.71 input token/s，均值 TTFT 2.4012→2.1422 秒；64 条正式 Prefill 的文本哈希一致，但后续 Decode 哈希不一致、未采到 trace，整轮 Guard 为失败。不能拿这个中间结果当全部验收通过。
- r2：原版／候选第一次预热 Prefill 输出不一致，立即停止且 GPU 清理通过；未产出正式性能或 trace，不补写成功记录。
- r3：先保存预热、正式输出与 trace，再分别判断原版重复稳定性、候选重复稳定性和跨策略一致性。`inconclusive_original_unstable` 表示原版自身重复已变化，不能据此直接指控候选，也不能据此宣告候选正确。
- r3 的 Prefill 正式 32 条请求 token IDs 完全一致，预热 32 条也一致，独立 profile 输出亦与正式参考一致。Decode 原版 A/A 与候选 B/B 均不稳定，精确原因未定位；因此 Decode 指标只留作诊断，不当成有效提速或退化结论。
- 5 项候选 CPU 单元测试通过。r3 的 Guard、CPU 权重复用审计、实际 GPU 清理和注册释放通过。
- **验收范围仅为本轮已经预热的两槽 PP2 Prefill 性能实验，不是完整生成质量或通用调度正确性资格。** r2 预热输出差异仍保留为待查；不默认启用候选到 Decode 或其它工作负载。
- 原始失败证据不修改；数据完整性、Guard 操作状态、输出等价是三个独立判断。

## 6. 可追溯入口

- [本轮重新计算的指标](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/analysis/metrics.csv)；[逐卡 profile 表](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/analysis/profile.csv)；[校验 JSON](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/analysis/validated.json)。
- [冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/config.json)；[候选实现](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/code/pp_ready_send.py)；[冻结 benchmark](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/code/benchmark.py)。
- [正式 Prefill 正确性](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/pp2/correctness.prefill.json)；[Decode 正确性](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/pp2/correctness.decode.json)；[逐文件来源与哈希](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/results/provenance.json)；[实际 GPU 清理证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/qwen_pp_ready_send_20261025/runs/two_slots_r3/task/cleanup.json)。
- [此前未修改的 Q01](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/component_reports/Q01.md)；[优化前 Prefill 诊断](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/qwen_prefill_diagnosis/REPORT.md)。
