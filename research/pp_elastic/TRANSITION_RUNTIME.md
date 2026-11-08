# PP + HiCache transition runtime checkpoint

The three explicit policies share the model/KV transports and one coordinator.
`sync` drains before host/active-state snapshots, capacity admission and target
allocation, then copies KV as one source-sized batch. `batched` uses the same
fence and bounded transport chunks. `background` allocates unpublished target
backing and pre-copies immutable weights through CPU/Gloo while the old pipeline
executes; the final fence still snapshots all mutable KV/GDN and host prefixes.
It does not claim background KV pre-copy or measured minimum pause.

Native peak admission accounts for incoming weights, full new attention pools,
and the entire replacement recurrent pool when its layer IDs change, plus a
workspace reserve. The existing source pools remain physically resident until
commit and cleanup. CPU snapshots alone therefore do not reduce the GPU peak;
this checkpoint requires source-plus-target capacity and does not implement
CPU-spill/reduced-pool recovery. A zero-layer rank remains a resident process,
not a released GPU. KVCacheD retains its existing backend admission rather than
using logical VMM tensor sizes as physical allocation estimates.

A host transaction snapshots after draining, stages a target controller/cache,
and publishes only after model/KV commit. Prepare/copy failures retain the source;
a failure after model/KV publication is terminal for serving and cannot promise
source rollback. Retired host resources are released only after publication.
Dormant rank cache-node handles may be rebuilt from a canonical active tree,
but request token progress and allocator identity must match before rebinding.

This remains a draft integration checkpoint. The Qwen hybrid HiCache guard permits only native L2 cache mode with page-first
layout and kernel I/O; storage, session-radix and unsupported recurrent modes
remain rejected. CPU checks qualify this narrow configuration only; dynamic
GPU state recovery remains unvalidated. Existing static PP + HiCache runs do not qualify
this dynamic path. The short real-model acceptance must include startup, warmup,
PP2 -> PP3 -> PP2, active KV/GDN/progress, CPU-only prefix loadback, and cleanup
within 300 seconds; successful server startup alone is not acceptance.
