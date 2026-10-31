# GLM-4.7-AWQ: EP8 versus optimized PP2×EP4

Status: **MEASURED_SAME_INPUTS_NOT_TOKEN_EQUIVALENT**

Completed native measurements: **True**. Strict cross-topology token equivalence: **False**.

P4096 · B8 · Q1 → 64 output tokens · semantic MTP · 8×A100 · same nodes. Native CPU HiCache starts with 4032 token hits/request and GPU prefix empty; the first forward includes 64 recomputed prefix tokens.

| Cohort | Configuration | Output tok/s | TTFT mean / median / P99 (s) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (s) |
|---|---|---:|---:|---:|---:|
| timed | EP8 | 89.88 | 0.664 / 0.664 / 0.884 | 72.296 / 72.656 / 78.626 | 5.218 / 5.085 / 5.696 |
| timed | PP2×EP4 optimized | 88.83 | 0.501 / 0.501 / 0.605 | 76.702 / 77.848 / 81.880 | 5.333 / 5.361 / 5.764 |
| repeat | EP8 | 91.17 | 0.618 / 0.618 / 0.822 | 71.740 / 72.100 / 77.814 | 5.138 / 5.005 / 5.616 |
| repeat | PP2×EP4 optimized | 89.09 | 0.505 / 0.505 / 0.609 | 76.378 / 77.536 / 81.555 | 5.316 / 5.345 / 5.747 |

## MTP

| Cohort | Configuration | Accept length mean / median / P99 | Verify count mean / median / P99 | Total Verifies |
|---|---|---:|---:|---:|
| timed | ep8 | 2.482 / 2.560 / 2.667 | 25.875 / 25.000 / 29.000 | 207 |
| timed | pp2ep4 | 2.411 / 2.370 / 2.667 | 26.625 / 27.000 / 29.000 | 213 |
| repeat | ep8 | 2.482 / 2.560 / 2.667 | 25.875 / 25.000 / 29.000 | 207 |
| repeat | pp2ep4 | 2.411 / 2.370 / 2.667 | 26.625 / 27.000 / 29.000 | 213 |

## Qualification

Cross-topology output-ID/text parity: {"timed": false, "repeat": false}.

| Cohort | EP8 group1 − group0 mean TTFT (ms, common client clock) | EP8 rank4 − rank0 host first-forward timestamps (ms, uncalibrated node clocks) |
|---|---:|---:|
| timed | 440.659 | 434.641 |
| repeat | 407.510 | 403.377 |

Both completed measurements are valid as same-input configuration performance. Different output histories and MTP acceptance mean this is not a pure equal-history PP/communication overhead comparison.

- timed: exact cross-topology output-ID/text parity failed; not an equal-token-history overhead comparison
- repeat: exact cross-topology output-ID/text parity failed; not an equal-token-history overhead comparison

## Boundaries

- Bounded P4096 B8 Q1 to 64 semantic MTP Decode only; not general model-quality proof
- Actual native HiCache prefix hit is 4032; 64 prefix tokens are recomputed, so first forward processes 65 new tokens
- TTFT includes native HiCache H2D, queueing and first-forward work; model loading, semantic Prefill, warmup and witness are excluded
- TPOT = (last client token time - first client token time) / 63, not mean Verify-step latency
- P99 is maximum of eight requests; two sequential cohorts in one server are not independent-run confidence intervals
- PP has separately qualified bounded untimed witness evidence; EP8 has no new untimed CPU oracle cohort
- Humming loaded-kernel invariant metadata is not a per-launch trace
- Exact output parity does not prove exact logits, GPU KV values, or non-speculative target parity
- MTP acceptance/Verify counts are reported, not required equal; a timing difference alone is not communication-only attribution
- Transport is NVLink within pairs and RDMA outside pairs, not Copy Engine PCIe optimization
- EP8 uses explicit per-RID DP routing and local B4 admission buffering; PP uses two fixed B4 slots

Full per-request MTP metrics and source SHA256 receipts are in [comparison.json](analysis/paired_r1/comparison.json).

[Metric CSV](analysis/paired_r1/metrics.csv) · [Artifact visibility receipts](analysis/paired_r1/artifact_visibility.json)
