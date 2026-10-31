# 122B H100 PP2 refinement

2×H100 80GB, node178 GPU0/1 NVLink; full Qwen3.5-122B-A10B FP8. B32 repeated Decode: GPUcached4096 + new1 → output128. EP AttentionDP2/TP1; PP two fixed B16 slots. Both retain native FP8/DeepGEMM and the common Decode prefix-state lock optimization.

| Configuration | Decode tok/s ↑ | Stdev | TPOT ms ↓ | vs original PP9 | vs fresh EP2 |
|---|---:|---:|---:|---:|---:|
| PP2 24+24 + policy9 | 1593.46 | 0.77 | 18.888 | +0.00% | -9.06% |
| PP2 24+24 + tail-local feedback | 1600.53 | 1.09 | 18.793 | +0.44% | -8.65% |
| PP2 25+23 + policy9 | 1594.40 | 0.21 | 18.865 | +0.06% | -9.00% |
| PP2 25+23 + tail-local feedback | 1600.88 | 0.98 | 18.771 | +0.47% | -8.63% |
| EP2 native optimized | 1752.15 | 15.99 | 16.145 | +9.96% | +0.00% |

Tail-local retirement gives a reproducible small improvement (0.44% on 24+24, 0.41% on 25+23). Repartitioning gives no material additional gain. Keep the 24+24 policy10 candidate; this workload still favors the native optimized EP2.

At matching generated positions, PP Expert GEMM is 164.24 µs/layer. EP Expert GEMM is 120.42 µs plus 45.06 µs Dispatch/Combine; 14.82 µs of communication overlaps other computation. PP's apparent component advantage is therefore very small before considering overlap. Profiled model graph time is PP 16.084 ms versus EP 15.427 ms per matched local work amount.

## Correction to the previous component comparison

The previous EP slice used sequence-length sums 65888–66128, while PP used 65728–65840. The selected generated positions did not overlap. Thus the old 147 µs PP versus roughly 170 µs EP GEMM+communication comparison mixes generation windows.

| Identical position window (original traces) | EP GEMM µs/layer | EP Dispatch+Combine µs/layer | PP GEMM µs/layer | EP graph ms/generation | PP two local graphs ms/generation |
|---|---:|---:|---:|---:|---:|
| earlier_pp_window | 110.55 | 45.16 | 147.34 | 14.974 | 15.263 |
| later_common_window | 117.96 | 45.24 | 156.97 | 15.347 | 15.729 |

The aligned original profiles show PP slightly slower inside the model graphs. Communication kernel duration also overlaps other EP compute, so GEMM+communication sums do not equal an exposed critical-path cost.

## PP-specific changes

1. **Tail-local retirement (policy10):** keep policy9 early activation send/lookahead, but let the tail consume its own sampled-token tensor. The head still receives the real tail output; it stops echoing the same tensor back. Producer events, two-slot ownership and send-buffer lifetimes remain enforced. Expert/Attention kernels and captured model graphs are unchanged.
2. **Measured-time stage partition:** test 25+23 instead of 24+24 layers, moving GDN layer24 to the head while retaining six full-Attention layers per stage. The tail also owns the output head. P2P channels are initialized before Prefill temporary allocations; KV capacity is unchanged.

## Paired scheduling evidence

- pp_feedback_r3: policy10 vs policy9 +0.444% throughput, 5/5 positive pairs; pair changes: +0.485%, +0.343%, +0.459%, +0.476%, +0.458%.
- pp_balance_r2: policy10 vs policy9 +0.407% throughput, 5/5 positive pairs; pair changes: +0.443%, +0.298%, +0.482%, +0.372%, +0.437%.

Each scheduling comparison uses one model process and unchanged CUDA Graphs, three full warmups per policy, five counterbalanced profiler-off pairs, exact 128-token output agreement against native policy0, exact GPU-prefix cache counts and drained ownership receipts. Stage-partition and EP comparisons are fresh sequential deployments; they are not within-process pairs.

## Profiles at matching generated positions

| Run / policy | Rank | Layers | Two local graphs ms | Between graphs ms | Cycle ms |
|---|---:|---:|---:|---:|---:|
| pp_feedback_r3 / 9 | 0 | 24 | 15.757 | 1.843 | 17.600 |
| pp_feedback_r3 / 9 | 1 | 24 | 16.420 | 1.235 | 17.656 |
| pp_feedback_r3 / 10 | 0 | 24 | 15.717 | 1.752 | 17.469 |
| pp_feedback_r3 / 10 | 1 | 24 | 16.451 | 1.129 | 17.580 |
| pp_balance_r2 / 10 | 0 | 25 | 16.397 | 1.118 | 17.514 |
| pp_balance_r2 / 10 | 1 | 23 | 15.797 | 1.773 | 17.570 |

EP profiles cover one 48-layer graph per generation; PP profiles cover two local stage graphs (both B16 slots). Selected sequence-length sums are matched. Profile-on cycles explain mechanisms and are kept separate from full profiler-off TPOT.

| Fresh EP2 rank | Graph ms | Gap ms | Cycle ms | Dispatch + Combine µs/layer | Comm ∩ all compute µs/layer |
|---|---:|---:|---:|---:|---:|
| 0 | 15.427 | 0.100 | 15.526 | 40.26 | 14.69 |
| 1 | 15.427 | 0.099 | 15.526 | 49.86 | 14.96 |

## Reproduction and attempts

All accepted runs passed frozen-input hashes, native full-model FP8/DeepGEMM qualification, canonical CPU weight-cache loader receipts, exact workload checks, owned GPU cleanup and Guard release. Source and result hashes are preserved with each run. No CPU KV restore was used.

- `ep_control_r1`: complete; included.
- `pp_balance_r1`: failed; excluded.
- `pp_balance_r2`: complete; included.
- `pp_feedback_r1`: admission_failed_before_GPU_claim; excluded.
- `pp_feedback_r2`: failed; excluded.
- `pp_feedback_r3`: complete; included.
