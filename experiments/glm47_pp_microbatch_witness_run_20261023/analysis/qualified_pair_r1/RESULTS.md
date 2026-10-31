# PP2 × EP4: bounded semantic microbatch comparison

Qualified only for this measured workload; no general correctness or model-quality claim.

GLM-4.7-AWQ; 065/167; 8×A100 80GB; Attention TP4, Expert EP4, PP2; two fixed B4 slots.
B8; semantic P4096 + Q1 → 64 tokens. Each cohort starts with native HiCache CPU P4032 and GPU P0; first forward has 65 tokens.
Timing includes HiCache H2D, excludes model loading, semantic Prefill and warmup. Two timed cohorts run before the two separate untimed witness cohorts; zero witness clones/readbacks and no CUPTI during timing.

| Configuration | Cohort | TTFT mean / median / P99 (s) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (s) | output tok/s |
|---|---|---:|---:|---:|---:|
| baseline | timed | 0.648 / 0.648 / 0.776 | 138.623 / 141.527 / 145.604 | 9.382 / 9.485 / 9.949 | 51.462 |
| baseline | repeat | 0.646 / 0.646 / 0.773 | 138.635 / 141.533 / 145.631 | 9.380 / 9.483 / 9.948 | 51.467 |
| candidate | timed | 0.501 / 0.501 / 0.605 | 76.702 / 77.848 / 81.880 | 5.333 / 5.361 / 5.764 | 88.829 |
| candidate | repeat | 0.505 / 0.505 / 0.609 | 76.378 / 77.536 / 81.555 | 5.316 / 5.345 / 5.747 | 89.087 |

P99 is nearest-rank over eight requests (the sample maximum). Two cohorts do not establish run-to-run statistical confidence.

All run Guard posts and owned GPU cleanup passed; all eight ranks have native Graph, atomic schedule, deterministic and loaded-Humming invariant receipts. All saved 64-token outputs/text agree across both arms and all four cohorts. Every bounded witness oracle and exact repeat/cross-TP/cross-arm comparison passed.

Humming receipts prove registered dense/grouped-masked cubin invariants, not each actual launch. Witnesses cover only the first 16 Verifies per slot and do not independently prove target logits or actual KV-value equality. The default microbatch flag remains unchanged.

timed: output throughput ×1.726; mean TPOT ×1.807 speedup (44.7% reduction).
repeat: output throughput ×1.731; mean TPOT ×1.815 speedup (44.9% reduction).
