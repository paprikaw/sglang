# PP2 MTP microbatch investigation

Engine: `f415858311c0d7646b368b30b64b371dca3427e4`, local tag
`serving/sglang/glm47-awq/pp-mtp-microbatch-20261023`.

The opt-in scheduler submits the next independent slot's activation before
retiring the other slot's feedback, uses a separate feedback process group,
and owns copies of graph output tensors until their send completes. Its scope
is PP2, two slots, single-layer EAGLE/topk1, no DP Attention or elastic PP.
The original path remains the default.

## First real-model pair: diagnostic, not correctness-qualified

Same 065/167, 8 A100, GLM-4.7-AWQ, Attention TP4/Expert EP4 per stage,
native EAGLE 2/1/3, two B4 slots, full Decode Graph. B8, P4096/Q1,
64 generated tokens. Actual native HiCache hit is 4032 tokens/request;
the remaining 64 prefix tokens plus Q1 enter the initial forward.
All KV comes from actual semantic model Prefill. Model loading, semantic
Prefill and warmup are excluded; native CPU-to-GPU restoration is timed.
DeepEP uses NVLink within pairs and RDMA outside pairs, not the CE PCIe path.

| Request-mean metric | Original scheduler | New microbatch scheduler |
|---|---:|---:|
| TTFT, s | 0.594818 | 0.440735 |
| TPOT, ms | 82.041350 | 47.003533 |
| End-to-end, s | 5.763520 | 3.402064 |
| Global output tok/s | 80.004945 | 139.484164 |
| Draft acceptance | 70.7314% | 71.6151% |

Both runs completed, all eight workers replayed Graphs, slot/RID audits and
Guard postchecks/cleanup passed. However, strict generated-token/text parity
fails for five of eight requests. These numbers are retained as observations,
not a validated speedup. Even the original scheduler's production and later
profile cohorts differ on six of eight requests with identical input tokens;
the new scheduler's two cohorts differ on five. Thus divergence alone does
not identify a newly introduced bug. Arrival order and batch composition were
not controlled in this pair. Quantized numerical sensitivity remains a
hypothesis, not an established cause.

## Scheduling evidence, separate profile cohorts

Some traces lose their first marker. No missing marker was inferred or
reindexed. Only complete local traces are used: original tail rank 5 and new
tail rank 4, peers in the same TP group. This is not a globally synchronized
cross-host overlap measurement or a sum that equals request TPOT.

| Tail-stage profile metric | Original | New |
|---|---:|---:|
| Target Verify envelope, ms | 41.782 | 42.368 |
| Attention, ms | 21.888 | 21.809 |
| Routed Expert, ms | 3.628 | 3.860 |
| Shared/dense Expert, ms | 2.248 | 2.241 |
| Dispatch + Combine, ms | 8.570 | 8.413 |
| Local PP kernel union | 40.75% | 4.08% |
| Same-cohort Verify recurrence, ms | 188.623 | 108.227 |
| PP kernels in interior window | 198 | 198 |
| PP kernel median, ms | 0.03152 | 0.03406 |
| PP kernel P99, ms | 38.90605 | 0.32259 |
| PP kernels longer than 1 ms | 9 | 0 |

The ordinary small-transfer duration did not improve. Approximately 38 ms
waiting tails disappear while A/E envelopes remain similar. This supports the
proposed scheduling mechanism; it does not replace output correctness checks.

## Artifacts and next controlled check

- `comparisons/p4096_pair_r1/comparison.json`: exact metrics and both outputs.
- `offline_audits/p4096_pair_r1.json`: observed slots, row order and cross-rank
  submissions, plus output parity.
- `analysis/results/p4096_pair_local_timeline.json`: valid/invalid trace ranks,
  local category unions and PP kernel duration distributions.
- Fresh harness `../glm47_pp_microbatch_atomic_20261023` controls semantic
  Prefill and Decode admission with one ordered native batched request, and
  adds a second unprofiled cold-HiCache repeat to test within-mode stability.
  Original source, runtimes, run specifications and raw evidence stay frozen.
