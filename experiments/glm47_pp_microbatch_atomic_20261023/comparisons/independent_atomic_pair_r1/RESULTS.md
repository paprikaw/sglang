# PP2 × EP4 semantic microbatch A/B

Same source revision, full canonical CPU weight banks, real semantic Prefill KV.
HiCache CPU-to-GPU load is timed; model loading, Prefill and warmup are not.

| Metric | Baseline | Candidate |
|---|---:|---:|
| TTFT (s) | 0.560 | 0.441 |
| TPOT (ms) | 80.075 | 45.785 |
| End-to-end (s) | 5.605 | 3.326 |
| Draft acceptance (%) | 70.745 | 73.094 |
| Global output tok/s | 86.686 | 143.084 |

Profile cohorts are separate. P99 from eight requests is the sample maximum.

Correctness: correctness_unconfirmed
Performance data are retained even on mismatch; no validated speedup claim until greedy parity passes.
