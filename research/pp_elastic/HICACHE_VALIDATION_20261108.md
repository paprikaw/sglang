# HiCache resident PP transition validation, 2026-11-08

This is a correctness check for one Qwen3.6-35B-A3B BF16 model with native KV and
FP32 GDN state. Tensor and expert parallel sizes are both one. Three resident
A100 workers on booked node spartan-gpgpu168 keep HiCache enabled while the
planned global-layer partitions change from 20/0/20 to 13/13/14 and back. The
zero-layer worker remains a resident GPU process. CUDA Graph is explicitly
disabled. This is neither a GPU-return measurement nor a pause benchmark.

The overlap reference and CPU-spill path share the transports and coordinator.
Overlap retains both generations and does not establish GPU-capacity relief.
CPU spill snapshots full KV/GDN backing to CPU, releases device owners, rebuilds
target backing and restores retained state. It has CPU recovery and staged
receive tests; physical CUDA release and real request recovery remain unverified.
Background currently moves immutable weights before the fence; mutable state
still moves at the fence. No GPU comparison of the three pause policies exists.

The first three attempts failed before requests: integer HiCache-size parsing,
strict actual-layer weight qualification, then native paged allocator live-slot
support. Those failures remain preserved. In r4, strict canonical loading passed
for all three expected ranks and the actual PagedTokenToKVPoolAllocator was
identified. HiCache initialization then stalled before HTTP readiness or any
request. The 165-second worker deadline produced a failed result with
166.714 seconds including owned cleanup. Guard verified cleanup and released its
normal-priority lease. No transition or output-equivalence claim follows.

The fixed-size HiCache host constructors enter a pipeline-group MIN reduction
for FULL and MAMBA capacity. The zero-layer logical branches skipped those
reductions. Cache checkpoint 3f41063832 now makes each resident rank participate
in the same order, with neutral int64 values for the zero-layer rank. CPU tests
exercise physical MHA/Mamba host constructors and logical pools, aligned
capacities, the all-logical fallback and ratio sizing. A fresh tagged GPU run is
required to validate that correction on the actual startup path.

r5 used source 03317bcccb816dc8dcbff79926f5fa2fed6e6470 with immutable tag
experiment/sglang/pp-hicache-transition-20261108-r5. All three ranks completed
HiCache initialization, and HTTP startup completed. The first internal-state
inspection then failed because the diagnostic hook used the old Scheduler
pp_rank attribute instead of ps.pp_rank. No correctness request or transition
completed. The failed run took 112.434 seconds including verified cleanup;
Guard released its lease. The hook now uses the actual parallel-state API, and
43 CPU acceptance/scheduler checks pass, including installed wrappers for all
three resident ranks and failure paths. This correction requires a fresh GPU
attempt and does not change the failed r5 classification.

The merged checkpoint passed 197 CPU checks in 48.97 seconds across 15 files,
including multi-generation three-rank KV/GDN backing reconstruction, host
transactions, allocator live rows, partial OOM ownership and retry, publication
failstop, bounded CPU-spill wire chunks and adversarial evidence validation.
The [CPU archive](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/hicache-runtime-combined-20261108-r5)
contains the exact file list and JUnit. CPU execution does not qualify CUDA
fences, owner release or GPU OOM retry.

The short real-model acceptance requires actual Decode state at both generation
fences, positive CPU-only prefix payload bytes in each generation, unchanged
active output IDs and fresh final CPU prefix loadback on every final active rank.
For CPU spill it also requires positive detached bytes and a measured CUDA
allocated-byte decrease on every source-active rank. All loading, warmup,
requests, transitions and verified cleanup must finish within 300 seconds.

Raw r4 evidence remains at
[the immutable run directory](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/gpu-hicache-r4).
Its source is 1c4823bb198d4026bf8aab8734c338fb705304f2 and immutable tag
experiment/sglang/pp-hicache-transition-20261108-r4. The result and cleanup
visibility receipts were verified before classification. Guard model-cache
postcheck passed while scientific acceptance failed. Raw attempts are never
rewritten as success after a source fix.

The grouped [experiment snapshot](/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/sglang_pp_hicache_transition_20261108/manifest.json)
records exact sources, launch manifests, stable result checksums and cleanup
proof for r4 and r5. Earlier failed attempts remain at their original paths.
