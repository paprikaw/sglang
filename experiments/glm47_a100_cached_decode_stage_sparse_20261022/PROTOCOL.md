# Matched stable Decode component profile

Supplementary profile of the accepted 2026-10-22 GLM-4.7-AWQ A100 EP8 versus PP2×EP4 experiment. Original 128-output-token latency results remain unchanged.

- Exact booked nodes: 065 / 167, 8 A100; reuse required complete CPU weight banks, on-miss fail.
- Workload: B8 independent cached prefixes, P=81920 per request, Q=1. Short 32-output-token cohorts limit trace size; this is not a replacement end-to-end benchmark.
- Original engine, quantization, native DeepEP, TP/DP/EP partition and disabled Graph/overlap remain unchanged. DeepEP uses pair NVLink plus RDMA outside pairs; no CE or new PCIe policy is substituted.
- Fill synthetic native CPU HiCache once per arm; warm all prefixes into GPU, then run an unprofiled hot-prefix control and a profiled hot-prefix cohort. Assert every formal request reports device=81920, host=0. Exclude first HiCache H2D, loading and warmup from component attribution.
- Detail EP8 B4 Decode ordinals 9-12 on all ranks. Detail PP B4 ordinals 17-24 on both stages, representing four iterations of each of two slots. Assert actual metadata and all-rank receipts.
- Preallocated CUDA events are read only after requests complete. No inserted per-component synchronizations or barriers. CPU-scoped Kineto launch correlation identifies actual device activity; unknown and overlapping intervals remain explicit.
- Compare profile/control TPOT to expose profiler perturbation. Rank/stage GPU activity composition is not a fraction of HTTP TPOT. Report PP boundary separately; do not sum concurrent stages, independent GPU clocks, or nested scopes. Communication kernel activity can include polling/waiting and is not pure wire time.
- Full-node Guard registration, cache joint budget, frozen source identities, lease and owned-process cleanup remain mandatory.

## Low-overhead confirmation

The preceding full Kineto run inflated EP8 control TPOT from 192.9 to 270.2 ms, and selected forwards even more. Its communication fractions must not represent the baseline. This arm uses only deferred preallocated CUDA events at coarse Attention, FFN, DeepEP and TP/PP API boundaries. It disables operator traces, record_function scopes, recursive tensor-shape capture and fine-grained projection/router hooks. No synchronizations are added inside the cohort. Report same-stream API execution/wait partitions, with control/profile and selected/nonselected-forward perturbation checks; these are not pure kernel or wire time.

## Final sparse estimator

Coarse all-layer events inflated the four detailed EP8 steps by 24%, although whole-cohort TPOT changed only 2%. This version rotates layer samples across otherwise identical stable steps: EP8 ordinals5-27 sample 4 layers each, covering all92 once per rank; PP ordinals9-54 sample2 local layers per microbatch, covering all46 stage layers twice, once for each slot. Sum each unique layer mean within a rank, then average ranks; never sum rank clocks or stage cycles. Compare selected/unannotated same-cohort forward intervals and whole-cohort control/profile TPOT. Report these as estimated per-layer-body execution/wait shares, not precise E2E or wire fractions.
