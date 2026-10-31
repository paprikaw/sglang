# PP2 × EP4 semantic microbatch A/B

Same source revision, full canonical CPU weight banks, real semantic Prefill KV.
HiCache CPU-to-GPU load is timed; model loading, Prefill and warmup are not.

| Metric | Baseline | Candidate |
|---|---:|---:|
| TTFT (s) | 0.595 | 0.441 |
| TPOT (ms) | 82.041 | 47.004 |
| End-to-end (s) | 5.764 | 3.402 |
| Draft acceptance (%) | 70.731 | 71.615 |
| Global output tok/s | 80.005 | 139.484 |

Profile cohorts are separate. P99 from eight requests is the sample maximum.

Correctness: correctness_unconfirmed
Performance data are retained even on mismatch; no validated speedup claim until greedy parity passes.
