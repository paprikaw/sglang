# PP + HiCache transition runtime checkpoint

The three explicit policies share the model/KV transports and one coordinator.
`sync` drains before host/active-state snapshots, capacity admission and target
allocation, then copies KV as one source-sized batch. `batched` uses the same
fence and bounded transport chunks. `background` allocates unpublished target
backing and pre-copies immutable weights through CPU/Gloo while the old pipeline
executes; the final fence still snapshots all mutable KV/GDN and host prefixes.
It does not claim background KV pre-copy or measured minimum pause.

The default `--elastic-pp-transition-backing overlap` retains source pools until
commit and cleanup. Native peak admission includes incoming weights, new
attention pools, a complete replacement recurrent pool when layer IDs change,
and workspace reserve. CPU snapshots alone do not reduce this GPU peak.

The opt-in `cpu_spill` backing contract is wired through the same coordinator
for `sync` and `batched`: drain host I/O, snapshot mutable state, detach source
device owners, collectively confirm release, admit against physical free memory,
allocate target backing, restore retained state and transfer moved layers. It
requires the cache branch's spill/alias APIs and fails closed when they are
absent. Native eager execution is required; background preparation still needs
source-plus-target capacity. Admission after spill counts all target KV/GDN,
including retained layers whose source device owners were released. The runtime
records CUDA allocated/reserved/free bytes around release. The short acceptance
rejects logical detach without a measured decrease in allocated device bytes.
This is not token-capacity resizing or measured OOM relief. A zero-layer rank
remains a resident process, not a released GPU. KVCacheD retains its existing
backend admission rather than using logical VMM sizes as physical allocations.

A host transaction snapshots after draining, stages a target controller/cache,
and publishes only after model/KV commit. Prepare/copy failures retain the source;
a failure after model/KV publication is terminal for serving and cannot promise
source rollback. A pre-publication failure restores source backing and device
aliases on every resident rank; any unverified restore prevents serving and
retry. Partial target allocations are included in abort. Successful rollback
resumes from the collectively agreed PP ring boundary. CPU source authority and
retired host resources are released only after publication.
Dormant rank cache-node handles may be rebuilt from a canonical active tree,
but request token progress and allocator identity must match before rebinding.

This remains a draft integration checkpoint. The Qwen hybrid HiCache guard permits only native L2 cache mode with page-first
layout and kernel I/O; storage, session-radix and unsupported recurrent modes
remain rejected. CPU checks qualify this narrow configuration only; dynamic
GPU state recovery remains unvalidated. The first three short attempts failed
during startup (CLI host-size type, owned-layer qualification, and native paged
allocator live-slot discovery); none reached a transition. Native paged
allocator support is now merged, with the owned-layer qualification preserved.
Existing static PP + HiCache runs do not qualify
this dynamic path. The short real-model acceptance must include startup, warmup,
PP2 -> PP3 -> PP2, active KV/GDN/progress, CPU-only prefix loadback, and cleanup
within 300 seconds; successful server startup alone is not acceptance.
