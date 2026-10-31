# Deterministic-control PP2 MTP correctness diagnostic — NOT RUN

This is a conditional correctness control after ordered atomic B8 admission did
not make the legacy baseline repeatable. It is not another kernel optimization.

Both arms pin the unchanged engine commit
`f415858311c0d7646b368b30b64b371dca3427e4` and add the identical native
`--enable-deterministic-inference` flag. The only A/B difference remains
`SGLANG_PP_SPEC_MICROBATCH=0/1`. No GPU run has been launched from this preparation.

- Existing A100 nodes 065/167, 8 GPUs; full canonical CPU weight bank reuse-only.
- PP2×EP4, two ordered B4 slots, B8 P4096 Q1→64 production + same-process repeat.
- Real semantic input corpus copied byte-for-byte from the atomic diagnostic;
  actual model Prefill KV, native HiCache CPU-ready/GPU-empty before timing.
  Cached P4032 and initial forward 65 tokens; synthetic KV remains forbidden.
- Same EAGLE steps2/topk1/draft3, native full CUDA Graph B4/width12, Humming AWQ.
- Same NVLink pairs + RDMA outside pairs (PCIe CE remains off).
- Untimed Prefill/warmup, profiler off during both production cohorts. Separate
  12-Verify profile preserves idle CUPTI bootstrap and all-rank receipts.
- Native deterministic Triton is source-compatible with TARGET_VERIFY and radix
  HiCache. It fixes KV split size and uses unified extend attention. CUDA TP pins
  NCCL allreduce tree and eight channels, disables custom/symmetric-memory reduce.
- Each initialized scheduler rank must prove resolved deterministic mode, actual
  Triton backend's deterministic state, and NCCL algorithm/channel environment;
  declaration alone does not pass. Third-party Humming/DeepEP determinism is not
  guaranteed, so exact same-process repeats AND cross-arm greedy output parity
  remain mandatory before any validated speedup claim.

Do not edit an arm's frozen inputs after preparation. New changes require fresh
run IDs/specs. Guard admission, weight proof and owned cleanup remain unchanged.
