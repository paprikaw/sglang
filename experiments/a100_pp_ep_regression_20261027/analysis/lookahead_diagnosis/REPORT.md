# Why lookahead helps H100 Graph Decode more than this A100 eager run

All rows below use Qwen3.6-35B-A3B BF16, PP2 with two slots, GPU-hot prefix4096 + new1 -> output128; the global batch is split equally. Timing uses five profiler-off cohorts. Inputs match, but GPU architecture, Attention backend, Expert backend and execution mode differ. These runs do not isolate a hardware-only effect.

| Execution path | Workload | PP0 tok/s | PP9 tok/s | Change | TPOT PP0 -> PP9 (ms) |
|---|---|---:|---:|---:|---:|
| H100 CUDA Graph | b8_repeated | 817.63 | 1053.05 | +28.79% | 9.16 -> 6.97 |
| H100 CUDA Graph | b32_repeated | 2132.41 | 2580.13 | +21.00% | 14.12 -> 11.51 |
| H100 CUDA Graph | b32_diverse | 1942.88 | 2305.93 | +18.69% | 15.61 -> 13.00 |
| A100 eager | b8_repeated | 123.39 | 122.46 | -0.75% | 64.30 -> 64.82 |
| A100 eager | b32_repeated | 451.83 | 453.47 | +0.36% | 69.94 -> 69.71 |
| A100 eager | b32_diverse | 451.97 | 455.28 | +0.73% | 69.93 -> 69.42 |

## What policy 9 changes

Policy 9 advances independent next-slot Decode preparation/launch, sends ready activations early, and uses typed communication channels. The end-to-end effect belongs to this combined scheduling policy; it is not an isolated ablation of a single lookahead call. It retains per-request token dependencies, model/Expert kernels, and payloads. It does not issue the next token before its own dependency is available.

A100 formal cohorts recorded ahead_deltas=[248,248] in every PP9 cohort and passed PP0/PP9 output agreement. The optimization ran; the formal throughput benefit remained within 1%.

## Where the gaps appear

Existing H100 B8 profiler-on samples: model Graph spans sum to 6.084 -> 6.073 ms per token cycle; forward handoff plus feedback intervals fall from 4.821 -> 0.742 ms. This identifies scheduling/feedback exposure without a material model-time change.

A100 eager traces were reanalyzed offline with the original late-16-forward window after dropping 20 forwards. The table averages rank0/rank1 forward measurements. GPU span includes gaps between correlated GPU operations. GPU operation union excludes those gaps; it is not an SM utilization measurement.

| Workload | Policy | CPU forward duration (ms/stage) | GPU forward span (ms/stage) | GPU operation union (ms/stage) | Within-span gap (ms/stage) |
|---|---:|---:|---:|---:|---:|
| decode_b8_repeated | 0 | 43.178 | 43.260 | 5.553 | 37.707 |
| decode_b8_repeated | 9 | 42.294 | 42.402 | 5.568 | 36.833 |
| decode_b32_repeated | 0 | 43.043 | 43.178 | 8.197 | 34.981 |
| decode_b32_repeated | 9 | 43.313 | 43.344 | 8.181 | 35.163 |

The A100 trace supports prioritizing work inside eager forward: host kernel issuing, preparation and synchronization. It does not separately establish how much each cause contributes. Current policy 9 does not change that per-layer execution path. Some stage waiting is reduced, while the large within-forward gaps persist.

Profiler-on envelopes are diagnostic and inflated relative to profiler-off request latency. Their reductions must not be substituted into measured TPOT. Eager forward envelopes include independent preparation; signed intervals in the derived cycle CSV may be slightly negative, so those intervals are not pure idle time or network wire latency.

## When lookahead should pay off

- Short stage forwards with exposed CPU retirement, scheduling or communication-control delays on the token critical path; CUDA Graph or better kernels can make this regime more visible.
- Independent ready slots with enough sustained Decode work to run while another slot retires its output.
- An optimization that advances actual GPU execution, with stage balance and dependency windows that permit overlap.
- Little remaining gain when the limiting time lies inside eager forwards, Expert weight reads/GEMMs, Attention, required bulk transfer, or an already well-filled pipeline.

For removable critical-path fraction f, ideal throughput speedup is bounded by 1/(1-f). A 20% gain requires removing at least 16.7% of baseline cycle time; 30% requires 23.1%. Not all GPU idle intervals are removable by lookahead. Larger batch alone is not a trigger: H100 B8 gains more than B32, while both A100 batches show little change.

Current implementation targets eligible Decode windows (no speculation/grammar/stopping boundary; sufficient active slots and safe GDN/KV ownership). Prefill needs its own dependency and scheduling analysis.

## Sources

- [A100 formal metrics](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/a10035_nvlink_r4/metrics.csv)
- [H100 formal metrics](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/metrics.csv)
- [H100 profiled cycles](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/analysis/lookahead_r2/profile_summary.csv)
- [A100 offline diagnosis](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/lookahead_diagnosis/result.json)
- [A100 derived per-cycle rows](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/analysis/lookahead_diagnosis/profile_cycles.csv)
- [Diagnosis code](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/a100_pp_ep_regression_20261027/lookahead_diagnosis.py)
- [Scheduling code](/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/gap_schedule.py)
