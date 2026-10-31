# Bounded collector lifecycle qualification

No model or Attention/Expert kernels. One existing A100 node is leased as a
whole four-GPU node; actual workers use GPU0 and GPU1. Existing weight/KV banks
are neither loaded nor modified.

Two independent fresh torchrun processes run sequentially: baseline explicitly
removes both flags; candidate supplies `TEARDOWN_CUPTI=0` and
`DISABLE_CUPTI_LAZY_REINIT=1` before Torch import. Each sends and verifies real
BF16 NCCL P2P activations (20,480 / 40,960 / 61,440 elements), records two
calibration warmups on rank0 only, and manually captures/replays the real P2P
exchange after each pair. Rank1 follows the same communication schedule without
the calibration profilers, giving a same-process-generation control.

After all three captures, both ranks collect (1) CPU+CUDA bootstrap with one
priming fill and three correlated 100,000-cycle marker sleeps and (2) a CUDA-only
session containing an actual captured NCCL graph replay and the same markers.
Every session reports raw GPU event, runtime launch, graph launch, NCCL kernel,
and marker correlation counts. No missing events are converted into guessed
signatures. A diagnostic worker can finish successfully with missing GPU
events: `passed` denotes payload correctness and cleanup, not collector health.

The two modes each have a 120-second deadline. The remote process-tree owner
checks cancellation/EOF, records exact descendant cleanup, and verifies GPU
release before changing modes. Preparation is CPU-only and validates the
existing narrow `communication_only` / `nccl_pp_boundary` Guard contract.
