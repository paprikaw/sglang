# 2P6D 运行记录

更新时间：2026-10-31T07:35:22.195512+00:00；状态：**failed**；owner均衡对照完整 episode：**2/4**；另保留旧分流诊断 1 轮。

当前引擎 E8 `fa52c3cd95d2c171c9eac9a8bfdc32296aa8c75f`，包含 E7 状态恢复修复和本轮 Hybrid MHA PP 层映射修复。

## 已验证性能（每点一轮，不能估计稳定误差）

| D 配置 | C | 请求数 | episode s | 平均 TTFT ms | 平均 TPOT ms | 平均 E2E ms | P 前缀命中 | 新增 P token／请求 | 实际 D batch均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×EP2 D | 8 | 171 | 376.08 | 591.09 | 6.79 | 3585.36 | 91.46% | 7,248.1 | 1.12 |
| EP4＋EP2 D | 8 | 171 | 373.36 | 616.39 | 6.67 | 3536.62 | 91.46% | 7,248.1 | 1.12 |

未完成的 episode 不计入上表：
- EP4＋EP2 D · agentx_c128 · instance_equal_diagnostic：仅收集 825 条，`RuntimeError`；[部分响应与错误](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/results/pd_2p6d_pp2_ep4ep2/agentx_c128/sticky_hash/partial.r1.json)。不能用已完成子集估计整轮性能。
- 3×EP2 D · agentx_c128 · owner_balanced：仅收集 1268 条，`RuntimeError`；[部分响应与错误](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_3ep2_e8_v4_gdn256_20261031_control/results/pd_2p6d_pp2_3ep2/agentx_c128/sticky_hash/partial.r1.json)。不能用已完成子集估计整轮性能。
- EP4＋EP2 D · agentx_c128 · owner_balanced：仅收集 1184 条，`RuntimeError`；[部分响应与错误](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control/results/pd_2p6d_pp2_ep4ep2/agentx_c128/sticky_hash/partial.r1.json)。不能用已完成子集估计整轮性能。

## 旧分流诊断，不能充当均衡布局对照

初次EP4＋EP2按实例等概率，C128初始根会话分配为EP2每owner32、EP4每owner16。以下完整结果有效，但不与均衡主表混排；修正后仍是静态路由。

| D配置 | C | 平均TTFT ms | 平均TPOT ms | 平均E2E ms |
|---|---:|---:|---:|---:|
| EP4＋EP2 D · 实例等分 | 8 | 621.67 | 6.64 | 3517.67 |

均值为请求等权；D batch 来自正式 episode 的 CPU forward 计数，按执行步加权，不能当作时间加权并发。TTFT／E2E 主表以 HTTP 起点计算，ready 起点在 CSV 中另列。

## 启动与资格证据

- [当前运行目录](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot)；[冻结配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/config.json)；[命令与状态预算](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/preflight.json)。
- [原始结果和完成 marker](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/results)；资格结果位于运行目录 `evidence/<arm>/serial_qualification.json`，必须通过才进入性能阶段。
- [GDN／KV 实际分配](slot_allocations.csv)；[运行标记](runtime_markers.csv)；[本地／跨节点 D 实例分项](instance_metrics.csv)；[完整性能与分位数](full_metrics.csv)；[正式 forward batch分布](forward_batches.csv)。未产生的文件暂不可用。
- [当前分析与原始哈希](pilot_analysis.json)；[复算脚本](analyze_pilot.py)。

- [C128 即时诊断样本](diagnostic_snapshot.json)：EP2 曾出现 KV 约98%、GDN 约21%，同时 P 排队；包含接收预分配，仅为日志样本，不是 episode 均值或因果证明。

## 保留的失败

- 最初默认 Guard 不接受 Qwen profile，在 GPU claim 前停止，无 GPU 数据。
- E7/V4 r2：准备脚本把 ASCII控制值初始化为 binary0；首个资格请求触发 capture控制错误。两节点原生 worker清理通过；中断的 supervisor没有成功postcheck。
- E7/V4 r3：原生 PP2→EP2 第一条512-token资格请求失败，日志为 `layers_current_pp_stage=5, len(dst_k_ptrs)=0`。8个rank均成功分配256GDN槽，错误来自 MHA 层索引映射，正式性能结果0。原生 cleanup通过，Guard确认释放；科学postcheck失败按原样保留。
- E8：修复 Hybrid MHA KV全局层映射，28项CPU回归通过；实际GPU正确性和正式episode以本页当前证据为准。
- E8 首个布局 C128：native bootstrap 120s／接收等待300s超时；[首次错误与队列样本](failure_diagnostics.json)。旧实例等分会使EP2每owner负载加倍，不能仅归因于GDN或KV预算。3×EP2均匀池保持原路由，EP4＋EP2另补owner均衡对照。

这些失败不计入平均值。旧 E5/V3、E7/V3 和新 E8/V4 的性能分别归档。

## 逐次运行与清理状态

- [two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot)：`failed`；[配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/config.json)；[native清理证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_vs_3ep2_e8_v4_gdn256_20261031_pilot/task/cleanup.json)。
- [two_p_six_d_3ep2_e8_v4_gdn256_20261031_control](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_3ep2_e8_v4_gdn256_20261031_control)：`failed`；[配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_3ep2_e8_v4_gdn256_20261031_control/config.json)；[native清理证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_3ep2_e8_v4_gdn256_20261031_control/task/cleanup.json)。
- [two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control)：`failed`；[配置](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control/config.json)；[native清理证据](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control/task/cleanup.json)。

[返回简明方案](REPORT.md)；[已有C8／C128 workload与缓存分布](../agentx_workload_profile_20261031/REPORT.md)。
