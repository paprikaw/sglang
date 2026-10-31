# External Event diagnostic follow-up (v8)

This is a profile-only follow-up of the four valid, exactly repeated production
arms in `../glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3`.
No production latency or throughput is remeasured or overwritten.

- Same pinned engine `9e48f876a8eb2d7460317bc713bfa8680d78aea5`, immutable v2
  semantic Bank, model, transport, local B4 Graphs and B8 requests.
- Same P4096 (native CPU4032/GPU0 at admission), Q1, 64 outputs. A short untimed
  8-output warmup runs before reset and a fresh Bank import for the formal cohort.
- No torch profiler/CUPTI collector, startup calibration or marker kernels.
- `FullCudaGraphBackend.capture_one` registers Events only in the actual capture,
  not the two native warmups. Target/draft, backend, shape and capture generation
  own distinct retained external timing Events. Every scope invocation has its
  own pair; nested categories remain nested, not separately added.
- Each actual Graph replay, including every repeated draft replay, completes its
  outer Graph end Event once and reads all timestamps before returning. No
  component-level fence is inserted. This deliberately changes diagnostic cadence.
- Exactly 12 actual target forwards per rank are required. Actual draft replay
  multiplicity, layer coverage and CPU-committed output advances are saved.
  Draft `target_ordinal` means the most recently observed target; it does not
  establish a complete Verify/draft cycle or imply three output tokens.
- Attention includes QKV/output projection, RoPE, KV store and attention. Routed
  Expert encloses both grouped GEMMs and activation. Shared Expert is separate.
  DeepEP spans begin before dispatch_a/combine_a and end after dispatch_b receive
  hook/combine_b current-stream rejoin. TP scopes can nest within A/E scopes.
- PP boundary Events bound the actual current-stream send/receive API dependency
  or enqueue span. An async send is not network completion. These are not NIC
  transfer times and are not added to a Graph's component partition.

These are **GPU stream spans**, including dependency waits and Event node
costs. They are not kernel busy times. Uncovered Graph span is not GPU idle.
Interval unions/overlap must be explicit; no cross-host clock addition or sum
of per-rank maxima is allowed. Do not reconstruct unprofiled TPOT by adding
these diagnostic values. Keep the original v5 production metrics separate.

Qualification is the separately guarded actual DeepEP external-Event probe:
`../deepep_graph_external_events_20261023/runs/external_events_r1` (8 ranks,
three changed-payload Graph replays, validated identity combine, cleanup PASS).
Preparation SHA-binds its result, config and frozen source specification. This
qualifies the mechanism, not full-model success before actual collection.
