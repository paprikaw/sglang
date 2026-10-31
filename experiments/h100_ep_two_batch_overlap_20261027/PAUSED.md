# Paused at the user's request — 2026-10-27

The user redirected work to validation of the existing Expert GEMM optimization
across larger workloads. Do not resume TBO compatibility development or launch
its GPU experiments without a new request.

The uncommitted source is preserved in
`.engine-worktrees/qwen35-ep2-tbo-20261027`. Seven CPU metadata tests and Python
syntax checks passed; no implementation GPU run was launched. The historical
raw-TBO-flag startup failure remains separate in
`h100_ep_native_overlap_20261027/runs/native_tbo_on_r3`.

Before any future GPU test, resolve the Qwen3.5 full-attention fast-RoPE
`position_sin` / `position_cos` cache lifetime across the two child batches.
Later attention layers reuse that cache, so interleaving children may otherwise
read the other child's positions. Real hybrid/GDN state and logit checks are
also outstanding. This implementation is not qualified for performance use.

The active replacement study is
`experiments/h100_expert_workload_validation_20261027` and uses the original
frozen engine, with TBO disabled.
