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
capacities, the all-logical fallback and ratio sizing. r5 subsequently validated
completion of these constructors on all three actual GPU workers.

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

r6 was prepared and tagged at f6d1fe27a0 but never launched. Cross-review found
that a not-yet-idle demotion can return no probe or an old probe; demotion now
retries until the same command has three positive acknowledgements and fresh,
completed probes. Requests now use 512 common prefix tokens plus one fixed
continuation token: the real Req API reserves the final input token, so a
512-token input could not prove a full 512-token loadback. The strict loadback
threshold remains 512. Both prefix and active references are collected after
warmup at the same match boundary; original cold outputs are also retained.

The new continuation evidence joins each response meta_info.id to the same
request at all three resident fences. It requires zero cumulative retractions,
zero Prefill launches after the first Decode, and an actual post-fence Decode
launch on each target-active rank. Counts cover active model-launch batches;
zero-layer passthrough is excluded. Per-request records contain only IDs and
integers, with no retained request or tensor owners. The repaired harness and
scheduler boundaries have 57 passing CPU checks, including real cache-probe
hook retries, the actual Req prefix cutoff, actual ForwardMode values and
adversarial retraction/re-prefill/request-ID evidence. None of these CPU changes
is a dynamic pass.


r7 used clean 4c331bcfa24940ee751a7958bcd88309b9aaecb8 and immutable tag
experiment/sglang/pp-hicache-transition-20261108-r7. Four cold/warmed reference
requests completed. Both warmed requests reused 512 GPU prefix tokens and
reported zero retractions. The initial CPU demotion then rejected incomplete
rank evidence: normal internal-state IPC returns one reply per DP replica,
while the check correctly requires all three resident PP workers. Zero
transitions occurred. The run failed in 87.423 seconds including verified
owned-process/GPU cleanup; Guard released its lease. Stable result and cleanup
visibility receipts were verified. This is a request-stage failure, with no
dynamic recovery or output-equivalence claim.

The correction is confined to the private single-node PP3/TP1/DP1 acceptance
hook. Non-first PP workers send only get/set internal-state replies through
independent diagnostic sockets, and the same two tokenizer communicators wait
for three replies. Inference and topology-control replies keep their usual DP
semantics. A handler collective would deadlock because PP forwards the request
after the local handler returns. The repaired hook has 66 passing acceptance
and scheduler CPU checks. These execute the real scheduler IPC constructor,
ZeroMQ/msgpack, tokenizer communicator initialization/dispatcher and graceful
resource-release method; two replies cannot complete a three-rank request.
Socket construction/publication failures close the new context even when the
exception is retained. Independent review found no blocker in this narrow
scope. Abnormal-process cleanup still requires Guard evidence on each real run.

An independent run recovered the exact historical 537 CPU node IDs across 52
files and passed every selected case, with zero failures, errors or skips.
Source remained clean at 4c331bcfa249 before and after, with zero visible CUDA
devices. The [historical CPU archive](/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/hicache-runtime-historical-537-20261108)
contains raw JUnit, selection, command/environment and source audit. This is a
separate suite from the 197 focused checks. Initial whole-file collection
attempts were incomplete and are preserved separately; they are not baseline
passes. The next real attempt requires a new immutable r8 SHA/tag.
