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
