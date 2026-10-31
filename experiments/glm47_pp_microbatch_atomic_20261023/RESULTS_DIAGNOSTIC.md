# PP2 MTP microbatch diagnostic — 2026-10-23

## Status

The opt-in scheduler is implemented at engine commit
`f415858311c0d7646b368b30b64b371dca3427e4` in the dedicated worktree
`../../.engine-worktrees/glm47-pp-spec-microbatch-20261023`.
`SGLANG_PP_SPEC_MICROBATCH=1` enables it; the default remains disabled.
The feature is **not yet correctness-qualified for general use**. Both the
unchanged scheduling path and the candidate fail exact same-process output
repeatability. Do not use the timing difference below as a validated speedup.

## Controlled protocol

- Same booked A100 80 GB nodes 065/167, 8 GPUs; PP2, Attention TP4 and Expert EP4
  on each stage. Full canonical CPU weight banks were reused, not rebuilt.
- GLM-4.7-AWQ; EAGLE steps=2, topk=1, draft tokens=3. Native full CUDA Graph
  replay at B4/width12 was proven on every rank.
- One ordered native HTTP B8 payload creates two stable B4 request groups.
  Both production arms use groups `{0,1,2,3}` and `{4,5,6,7}`.
- Each request has logical P4096, Q1, and 64 output tokens. Actual semantic
  model Prefill supplies the KV: HiCache holds 4032 prefix tokens and the first
  measured forward handles the 64-token uncached tail plus Q1.
  **No synthetic KV** is used.
- Prefill, warmup, and model loading are excluded. Before each timed cohort,
  native HiCache is CPU-ready and GPU-empty. HiCache-to-GPU loading is included
  in TTFT. Two unprofiled production cohorts run in each process; CUPTI is only
  enabled for a separate profile cohort.
- Transport is NVLink within GPU pairs and RDMA outside those pairs. CE PCIe
  transport was not enabled or changed by this scheduler experiment.

## Observations, not correctness-qualified performance claims

Request means in the first production cohort:

| Metric | Original scheduling | Opt-in microbatch |
|---|---:|---:|
| TTFT | 0.5600 s | 0.4413 s |
| TPOT | 80.0754 ms | 45.7852 ms |
| End-to-end latency | 5.6048 s | 3.3258 s |
| Output throughput | 86.6858 token/s | 143.0845 token/s |

Authoritative metrics remain in the per-run JSON and the independent pair
finalizer; this table is a diagnostic snapshot, not an accepted benchmark row.

## What changed in the measured schedule

Both arms have valid selected-window profile markers on all eight ranks.
For the same tail rank (rank4), the selected ten-Verify window shrank from
895.366 to 528.576 ms. Its PP NCCL kernel union occupied 40.35% versus 4.28%
of that local window. These kernels include transfer and waiting; the trace
does not separate pure wire time from GPU polling.

| Rank4 PP kernel statistic | Original | Microbatch |
|---|---:|---:|
| Kernels intersecting the selected window | 198 | 198 |
| Median | 0.03014 ms | 0.03142 ms |
| P99 | 38.7526 ms | 0.4460 ms |
| Maximum | 38.7641 ms | 0.4916 ms |
| Count over 1 ms | 9 | 0 |

For matching request groups, summing the two stages' local Verify work gives
82.984 versus 83.462 ms. Attention is 43.658 versus 43.659 ms; routed Expert
7.324 versus 7.285 ms; shared Expert 4.394 versus 4.396 ms; DeepEP dispatch and
combine 16.227 versus 16.260 ms. This sum is a work comparison, **not** a
globally synchronized latency or an overlap percentage. The near-identical
compute and typical transfer costs, alongside disappearance of long PP waits,
support the scheduling-wait explanation. Some head-stage waiting remains.

## Correctness boundary

Independent offline audits passed every rank's ordered RID, logical microstep,
stable B4 grouping, and monotonic retirement checks. Repeated cohorts may swap
the two physical slot numbers as a whole; requests do not migrate between
groups. Both original and candidate produced identical repeated token IDs and
text for only 3/8 requests. Cross-arm equality was 2/8. Consequently the strict
finalizer reports `correctness_unconfirmed`. The cause of output variability
is unresolved; it must not be attributed to harmless numerical effects without
additional evidence.

An independent, frozen deterministic-mode control is being run in
`../glm47_pp_microbatch_deterministic_20261023`; both arms enable the same native
deterministic mode. Its results are a separate protocol and must not overwrite
this cohort.

## Evidence

- `comparisons/independent_atomic_pair_r1/RESULTS.md`
- `offline_audits/independent_atomic_pair_r1.json`
- `offline_audits/independent_atomic_repeat_r1.json`
- `analysis/results/atomic_pair_local_timeline.json`
- `runs/atomic_base_r1/task/cleanup.json`
- `runs/atomic_micro_r1/task/cleanup.json`

Both runs passed owned GPU/model-process cleanup. The persistent CPU weight
banks remain available for reuse. No new node was booked.
