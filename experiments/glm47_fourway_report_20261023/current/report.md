# GLM-4.7-AWQ：EP8 与 PP2×EP4，MTP 开／关

065／167，8×A100。B8，P4096，原生 CPU 缓存4032／GPU0，Q1→64；首次 forward 为65 token。计时包含原生 HiCache→GPU，排除权重加载、源 Bank 导入和 profiler。

## 两轮平均

下表为两轮均值 ± 半极差；不是置信区间。

| 配置 | 输出 tok/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| ep8_mtp_on | 96.502 ± 0.838 | 553.127 ± 3.627 | 69.721 ± 0.757 | 4945.539 ± 44.086 |
| ep8_mtp_off | 84.939 ± 0.137 | 533.294 ± 5.554 | 87.213 ± 0.066 | 6027.734 ± 9.701 |
| pp2ep4_mtp_on | 88.284 ± 0.056 | 501.549 ± 0.139 | 77.182 ± 0.050 | 5364.026 ± 3.310 |
| pp2ep4_mtp_off | 83.527 ± 0.136 | 496.253 ± 1.311 | 89.057 ± 0.137 | 6106.836 ± 9.935 |

## 每轮完整延迟

| 配置 | Cohort | tok/s | TTFT mean / median / P99 ms | TPOT mean / median / P99 ms | E2E mean / median / P99 ms |
|---|---|---:|---:|---:|---:|
| ep8_mtp_on | timed | 97.340 | 556.754 / 556.759 / 556.809 | 68.963 / 69.647 / 74.651 | 4901.452 / 4944.525 / 5259.805 |
| ep8_mtp_on | repeat | 95.664 | 549.500 / 549.504 / 549.601 | 70.478 / 71.233 / 76.228 | 4989.625 / 5037.156 / 5351.935 |
| ep8_mtp_off | timed | 84.802 | 538.848 / 538.851 / 538.900 | 87.279 / 87.279 / 87.281 | 6037.435 / 6037.437 / 6037.525 |
| ep8_mtp_off | repeat | 85.076 | 527.740 / 527.743 / 527.789 | 87.148 / 87.148 / 87.149 | 6018.033 / 6018.039 / 6018.123 |
| pp2ep4_mtp_on | timed | 88.341 | 501.410 / 501.406 / 604.841 | 77.132 / 78.295 / 82.394 | 5360.715 / 5389.328 / 5795.621 |
| pp2ep4_mtp_on | repeat | 88.228 | 501.688 / 501.686 / 605.311 | 77.233 / 78.400 / 82.499 | 5367.336 / 5396.337 / 5802.697 |
| pp2ep4_mtp_off | timed | 83.663 | 494.942 / 494.938 / 593.933 | 88.920 / 88.920 / 90.131 | 6096.901 / 6096.925 / 6119.721 |
| pp2ep4_mtp_off | repeat | 83.391 | 497.564 / 497.562 / 596.168 | 89.194 / 89.194 / 90.396 | 6116.771 / 6116.770 / 6139.669 |

## MTP 原生接受统计

| 配置／轮 | Verify 总数 | 正确 draft／提议 draft | 汇总接受率 | 汇总输出／Verify | 请求平均接受率 | 请求平均 length |
|---|---:|---:|---:|---:|---:|---:|
| ep8_mtp_on / timed | 206 | 304 / 412 | 73.786% | 2.4854 | 74.481% | 2.4985 |
| ep8_mtp_on / repeat | 206 | 304 / 412 | 73.786% | 2.4854 | 74.481% | 2.4985 |
| pp2ep4_mtp_on / timed | 213 | 298 / 426 | 69.953% | 2.4038 | 70.328% | 2.4115 |
| pp2ep4_mtp_on / repeat | 213 | 298 / 426 | 69.953% | 2.4038 | 70.328% | 2.4115 |

严格 draft 接受率分母为 Verify×2，不含 bonus；length 使用 completion_tokens／Verify。汇总比值与请求比值平均不同，末轮64-token截断也不能被忽略。

## 完整跨臂输出校验

每臂自身两轮8/8完全一致。下表跨臂按 request_index 匹配，未假设模式／拓扑等价。

| 两臂 | Cohort | 首 token 相同 | 完整64-token ID与文本相同 | 首分歧位置（从0开始；—为相同） |
|---|---|---:|---:|---|
| ep8_mtp_on / ep8_mtp_off | timed | 8/8 | 2/8 | —, 7, 25, 61, —, 23, 7, 46 |
| ep8_mtp_on / ep8_mtp_off | repeat | 8/8 | 2/8 | —, 7, 25, 61, —, 23, 7, 46 |
| ep8_mtp_on / pp2ep4_mtp_on | timed | 8/8 | 2/8 | 6, 5, —, —, 37, 23, 7, 48 |
| ep8_mtp_on / pp2ep4_mtp_on | repeat | 8/8 | 2/8 | 6, 5, —, —, 37, 23, 7, 48 |
| ep8_mtp_on / pp2ep4_mtp_off | timed | 8/8 | 3/8 | —, 34, 25, —, 37, 23, 7, — |
| ep8_mtp_on / pp2ep4_mtp_off | repeat | 8/8 | 3/8 | —, 34, 25, —, 37, 23, 7, — |
| ep8_mtp_off / pp2ep4_mtp_on | timed | 8/8 | 0/8 | 6, 5, 25, 61, 37, 56, 19, 46 |
| ep8_mtp_off / pp2ep4_mtp_on | repeat | 8/8 | 0/8 | 6, 5, 25, 61, 37, 56, 19, 46 |
| ep8_mtp_off / pp2ep4_mtp_off | timed | 8/8 | 2/8 | —, 7, —, 61, 37, 56, 23, 46 |
| ep8_mtp_off / pp2ep4_mtp_off | repeat | 8/8 | 2/8 | —, 7, —, 61, 37, 56, 23, 46 |
| pp2ep4_mtp_on / pp2ep4_mtp_off | timed | 8/8 | 2/8 | 6, 5, 25, —, 51, —, 19, 48 |
| pp2ep4_mtp_on / pp2ep4_mtp_off | repeat | 8/8 | 2/8 | 6, 5, 25, —, 51, —, 19, 48 |

## 独立 CUDA profile

严格有效 profile：0/4；状态：`pending_or_failed`。未通过完整 validator 的数值不展示。

EP8 每 rank 覆盖92层 target；PP 每 stage 为46层，MTP draft只在 tail。下列单位是本地 B4 微批次周期，不能解释为请求 TPOT，也不对不同 stage／rank 的最大值求和。Profiler可能增加CPU／GPU空闲和调度间隔；GPU idle不能直接称为pipeline bubble。

- `ep8_mtp_on`：not_measured；数值=null。
- `ep8_mtp_off`：not_measured；数值=null。
- `pp2ep4_mtp_on`：not_measured；数值=null。
- `pp2ep4_mtp_off`：not_measured；数值=null。

## 解释边界

- Same semantic inputs, but full output histories differ between arms. These are configuration measurements, not an equal-token-history PP/communication overhead experiment.
- Within-arm exact repeat parity is checked. Different target logits or actual GPU KV at first divergence were not captured here; numerical or state root cause remains unproven.
- All production metrics exclude profiler, weight loading and source-bank import; include native CPU HiCache H2D and initial 65-token tail forward.
- P99 is the maximum of eight request samples, not a robust tail estimate. Two same-process cohorts are not independent run confidence intervals.
- Profile activity is diagnostic only. EP target covers 92 layers; each PP stage covers46 target layers and only tail runs the MTP draft. Never sum stage/rank maxima into TPOT.
- Per-local-B4-cycle CUDA work is not per-request TPOT. PP alternates two request slots; output normalization needs actual committed-token receipts.
- Profiler can inflate CPU/GPU idle and scheduling cadence. Traced GPU idle is not proof of pipeline bubble; component fractions are not a direct partition of unprofiled request TPOT.
- Missing, partial or failed profiles remain null/pending; unknown activity is never redistributed into Attention, Expert or communication.

## 来源

- 性能源：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3`
- Profile源：`/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/glm47_semantic_bank_fourway_20261023_v7/runs/profile_followup_r2`
- `summary.json` 保留来源路径、SHA256、每轮统计、完整校验与已有 validator 的结果；`performance.csv` 为八个未 profile cohort。
