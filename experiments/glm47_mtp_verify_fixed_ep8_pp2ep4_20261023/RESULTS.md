# GLM-4.7：修复 Verify Attention 后 EP8 与 PP2×EP4 对比

065／167，8×A100 80GB，同型号高速 NIC。B8、逻辑 P4096、Q1、输出64。每条实际从原生 CPU HiCache 恢复4032 token，剩余64 token 与新增Q1重算。两组均开启 MTP、修复版 Verify Attention 和 CUDA Graph。权重／Bank 导入不计时，HiCache→GPU 计时。

EP8：Attention TP4×DP2，Expert EP8。PP2×EP4：Attention TP4、Expert EP4／stage，两个 B4 slot。DeepEP 均为 pair 内 NVLink、pair 外 RDMA，未启用 PCIe CE。

## 实测性能（无 profiler）

| 配置 | cohort | 输出 tok/s | TTFT mean / median / P99 ms | TPOT mean / median / P99 ms | E2E mean / median / P99 ms |
|---|---|---:|---:|---:|---:|
| ep8_mtp_fixed | timed | 131.508 | 555.143 / 555.145 / 555.199 | 46.776 / 47.262 / 52.981 | 3502.027 / 3532.629 / 3892.971 |
| ep8_mtp_fixed | repeat | 131.662 | 558.068 / 558.072 / 558.118 | 46.701 / 47.184 / 52.866 | 3500.250 / 3530.726 / 3888.569 |
| pp2ep4_mtp_fixed | timed | 128.761 | 502.196 / 502.193 / 605.762 | 52.728 / 52.712 / 55.820 | 3824.067 / 3858.671 / 3976.181 |
| pp2ep4_mtp_fixed | repeat | 128.349 | 506.189 / 506.186 / 609.747 | 52.856 / 52.836 / 55.953 | 3836.124 / 3870.753 / 3988.976 |

P99 仅为8条请求的样本最大值。两 cohort 的差异不是置信区间。

## MTP 推进与额外验证次数

| 配置 | 每轮推进 token（接口 spec_accept_length） | draft 接受率 | 每请求 Verify 次数 |
|---|---:|---:|---:|
| ep8_mtp_fixed | 2.4583 | 0.7295 | 26.250 |
| pp2ep4_mtp_fixed | 2.4653 | 0.7303 | 26.000 |

spec_accept_length 是每轮输出推进量，不能全部解释成被接受的 draft token。

## 目标验证分段（独立诊断）

单位 ms／每卡／一次实际目标 Graph 执行；EP 是8卡均值，PP 分 stage 取4卡均值。

| 执行范围 | Attention | Routed Expert | Shared Expert | Dense FFN | Dispatch+Combine | TP/NCCL | 其他 | 重叠 | Graph |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep8_mtp_fixed all_ranks | 17.547 | 9.470 | 15.356 | 0.238 | 26.714 | 7.043 | 26.083 | 0.000 | 102.452 |
| pp2ep4_mtp_fixed stage0 | 8.750 | 5.104 | 7.252 | 0.291 | 8.573 | 3.249 | 12.528 | 0.278 | 46.025 |
| pp2ep4_mtp_fixed stage1 | 8.771 | 5.293 | 8.150 | 0.000 | 9.432 | 3.245 | 13.819 | 0.000 | 48.710 |

**不能把 stage 均值相加当作 TPOT**：Graph 不包含完整的跨 stage 激活交接、草稿／接受结果反馈和主机调度；这些执行会与另一 slot 部分重叠。profile 的完成 fence 也改变了节奏，因此诊断 host gap 不能标成生产流水线气泡。

## 与原 Verify 实现的历史同口径数据

| 配置 | 原 Verify 输出 tok/s（两cohort均值） | 修复后 | 原 Verify TPOT ms | 修复后 |
|---|---:|---:|---:|---:|
| ep8_mtp_fixed | 96.502 | 131.585 | 69.721 | 46.739 |
| pp2ep4_mtp_fixed | 88.284 | 128.555 | 77.182 | 52.792 |

旧数据仅作同 Bank／同 workload 历史参照，不是本轮重跑的原实现控制组。

## 差距的归因

1. **节点内 Expert 交换优势仍存在。** Dispatch＋Combine 约为 EP8 0.300 ms／MoE层，PP stage0／1 0.199／0.205 ms。后者低约三分之一。这是整体布局优势的证据，并非只改变链路、保持 peer 数和路由域不变的纯网络因果实验。
2. **Attention 修复对两边均生效。** 用两段均值相加仅观察逻辑全层工作量，PP Attention 17.52 ms，与 EP 17.55 ms 接近；Expert 合计 26.09 对 25.06 ms，PP 略高。这个相加值不是实际 PP cycle 或 TPOT。
3. **本轮排除了较低接受率这一解释。** PP 每轮推进2.465 token，EP为2.458；PP平均每请求26次Verify，EP26.25次。PP并没有因为接受率低而执行更多轮。旧版中存在的接受率差异不能直接沿用于本轮。
4. **差异更多出现在输出推进节奏。** 无profiler生产测量中，各请求输出更新间隔的中位数再取均值，EP约112.47 ms，PP约125.61 ms。独立诊断中同一slot的目标调用起点间隔中位数也为EP约119.9 ms、PP约132.4–132.6 ms；但该诊断有完成fence，不能换算成生产气泡比例。结合源码，stage激活交接、slot复用等待、接受结果／draft反馈和CPU调度是优先检查的环节；当前证据没有把这些项逐个定量拆开。
5. **吞吐差2.3%和平均TPOT差13%并不矛盾。** 吞吐按整批512个输出token除以最晚完成时间；TPOT是每条请求首尾token间隔除以63后取均值。PP两slot首token错开，EP各请求的后续推进也不完全一样。
6. **这不是无损MTP的最终公平排名。** 两边的语义输入和缓存相同，但输出逐token等价未通过，后续路由也可能随输出改变。现有证据支持路径分析；需先修复输出一致性，再作严格生产性能定论。

## 数值与实验边界

- ep8_mtp_fixed：同配置两cohort逐token可重复=True；与历史同布局 MTP-off 的输出完全一致 7/8。
- pp2ep4_mtp_fixed：同配置两cohort逐token可重复=True；与历史同布局 MTP-off 的输出完全一致 1/8。
- 本轮 EP／PP 输出完全相同：False。吞吐提升不证明 speculative decoding 无损。
- 历史原Verify的 ep8_mtp_on 与同布局MTP-off逐token一致为 2/8；原路径也未达到严格等价，不能把全部差异归为本次Attention优化新引入。
- 历史原Verify的 pp2ep4_mtp_on 与同布局MTP-off逐token一致为 2/8；原路径也未达到严格等价，不能把全部差异归为本次Attention优化新引入。
- Verify 优化保持默认关闭，属于受限形状的实验实现；保留所有不一致证据。
- 本 workload 远未用满 KV 容量，不能据此否定 PP 在容量受限时的收益。
- 两组性能、profile、进程退出与 GPU 清理均通过。

## 原始证据

- 本轮：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1`
- 历史性能：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3`
- 历史分段：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1`

`summary.json` 保存全部来源路径／SHA256、逐cohort mean/median/P99、分段、MTP接受率、输出一致性与限定条件；原始输出只保留在源结果中。
