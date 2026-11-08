# PP + HiCache transition runtime checkpoint

Dynamic acceptance is still blocked. r9 at aa93b5188c8a completed one
publication and uninterrupted active Decode relative to its warmed reference,
then stalled during the shrink host-cache preparation. The failed run took
176.100 seconds with verified cleanup and a released Guard lease. Cold and warm
gen0 requests with identical input and greedy settings already diverged at
output index six; warmed-reference agreement alone does not qualify generation.
The private acceptance now rejects unequal cold/warmed references before any
transition. The static r10 cold256/warm8 diagnostic has completed in 85.345 seconds with
verified cleanup and a released lease. Its raw result remains failed because
the observer counted Decode CoW no-ops. Separately classified offline tensors
show checkpoint/CoW/kernel-input bitwise preservation across 30 GDN layers;
cold/warm output still diverges. Continuous FP32 state at512 is unobserved,
and no numerical tolerance or dynamic pass is established.

The three explicit policies share the model/KV transports and one coordinator.
`sync` drains before host/active-state snapshots, capacity admission and target
allocation, then copies KV as one source-sized batch. `batched` uses the same
fence and bounded transport chunks. `background` allocates unpublished target
backing and pre-copies immutable weights through CPU/Gloo while the old pipeline
executes; the final fence still snapshots all mutable KV/GDN and host prefixes.
It does not claim background KV pre-copy or measured minimum pause.
CPU-spill sync uses the shared bounded wire chunks to limit temporary source
payloads. A strategy comparison still requires fixed backing, resources and
request/cache state; these CPU checkpoints do not measure pause differences.

The default `--elastic-pp-transition-backing overlap` retains source pools until
commit and cleanup. Native peak admission includes incoming weights, new
attention pools, a complete replacement recurrent pool when layer IDs change,
and workspace reserve. CPU snapshots alone do not reduce this GPU peak.

The opt-in `cpu_spill` backing contract is wired through the same coordinator
for `sync` and `batched`: drain host I/O, snapshot mutable state, detach source
device owners, collectively confirm release, admit against physical free memory,
allocate target backing, restore retained state and transfer moved layers. It
includes the cache branch's spill/alias and receive-staging APIs, publication
context rebinding (10b46fea07), and dormant host sizing (a99c6cf681). Spill
receives are validated and copied to CPU staging before the
wire buffers are released; this adds per-rank host admission and byte accounting,
not an atomic node-wide RAM reservation or a strict long-window peak proof.
Overlap mode still accumulates incoming GPU rows until import. Native
eager execution is required; background preparation still needs
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
Failed weight construction drops traceback references across exception chains
before source restoration, including partial constructors with loader cycles.
Dormant rank cache-node handles may be rebuilt from a canonical active tree,
but request token progress and allocator identity must match before rebinding.

This remains a draft integration checkpoint. The Qwen hybrid HiCache guard permits only native L2 cache mode with page-first
layout and kernel I/O; storage, session-radix and unsupported recurrent modes
remain rejected. CPU checks qualify this narrow configuration only; dynamic
GPU state recovery remains unvalidated. The first three short attempts failed
during startup (CLI host-size type, owned-layer qualification, and native paged
allocator live-slot discovery); none reached a transition. The fourth attempt
passed strict three-rank canonical loading and actual paged-allocator discovery,
then timed out in HiCache initialization before requests. It completed owned
cleanup in 166.714 seconds; Guard released its lease. The fixed-size host-pool
MIN collective was skipped by the zero-layer logical pool. The merged fix makes
all resident ranks join FULL then MAMBA reductions, using neutral values on
logical pools. The fifth attempt completed HiCache initialization on all ranks,
then failed before requests because the internal-state diagnostic hook used the
obsolete scheduler rank attribute. That hook now uses `scheduler.ps.pp_rank`,
with 43 installed-hook and scheduler checks passing. r5 completed verified
cleanup in 112.434 seconds and Guard released its lease. A freshly tagged
attempt remains necessary. Native paged
allocator support is now merged, with the owned-layer qualification preserved.
Existing static PP + HiCache runs do not qualify
this dynamic path. The short real-model acceptance must include startup, warmup,
PP2 -> PP3 -> PP2, active KV/GDN/progress, CPU-only prefix loadback, and cleanup
within 300 seconds; successful server startup alone is not acceptance.

The 2026-11-08 combined checkpoint passed 197 CPU checks across 15 focused files
in 48.97 seconds. This covers multi-generation three-rank native KV/GDN backing
rebuilds, host transactions, allocator fences, source recovery, OOM exception
ownership and adversarial acceptance evidence. The exact file list and JUnit
are preserved at
`/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/hicache-runtime-combined-20261108-r5`.
CPU checks do not prove CUDA ownership release or a successful dynamic GPU run.

The r6 source/tag was prepared but not launched. Pre-launch review corrected
fresh all-rank idle-demotion acknowledgement and the actual request prefix
cutoff. The next protocol uses 512 cached prefix tokens plus one fixed query
token and requires fresh loadback of all 512. Warm references and original cold
outputs are retained. Exact request IDs, cumulative retractions and active
model-launch modes now distinguish continued Decode from re-prefill; 57 focused
harness/scheduler checks pass. This does not qualify the pending GPU run.


r7 completed four cold/warmed requests at clean `4c331bcfa249`, then rejected
DP-only diagnostic evidence before CPU demotion; no transition occurred. Its
87.423-second failure has verified cleanup and a released Guard lease. Private
PP3/TP1/DP1 diagnostic replies now collect each resident worker without changing
general DP control semantics. The real IPC/communicator/resource-release boundary
and existing acceptance checks pass 66 CPU cases. An independent exact historical
537-case CPU baseline also passed at clean `4c331bcfa249`; these are distinct
suites. See the validation report and its linked JUnit/source archives. Actual
dynamic GPU preservation and CPU-spill CUDA release still require acceptance.

r8 at clean `36069aeca07c` reached a real generation-one commit on all three
ranks and moved 95,289,344 host-prefix bytes to the newly active rank. PP3
Decode resumed, but request completion later used the batch-result processor's
retired tree and failed on its cleared request-pool reference. The full run
failed in 160.804 seconds with verified cleanup and a released lease. Its
original result lacks transition records because the harness waited for the
active request before recording control success; committed engine metrics are
preserved separately. Control success is now saved immediately and continuation
is verified separately. A real worker-loop failure regression plus the existing
acceptance/IPC/scheduler checks pass 67 cases. Saved state is limited to the
diagnostics required for acceptance. The cache context/session fix was merged
before r9; 232 focused and a separate exact historical 537 CPU selection passed
at clean aa93b5188c8a. The first-generation completion fault did not recur in r9.

The shrink dormant constructor formerly skipped the fixed-size FULL/MAMBA MIN
reductions, advancing to an all-gather while physical ranks waited in MIN.
a99c6cf681 uses the same logical host-sizing helper as initial construction.
The same three-process Gloo reproduction now completes two MIN operations and
the following all-gather on every rank during initial, expand and shrink
construction. It uses real host constructors with a stubbed I/O controller and
zero CUDA devices. Separate stage logs distinguish KV/GDN bulk, delta and import
from host exchange and host preparation/import. GPU shrink, full-cycle request
equivalence and CPU-spill CUDA owner release remain unqualified.

The private CoW observer now applies the real method's complete target-worker,
hybrid-pool and forward-mode guards before recording an actual initial warm
Extend. It still delegates every original call exactly once. Real-method CPU
checks retain stale Decode indices and verify no false observation; the current
four-file selection passes 103 checks with no failures or skips. This changes
future diagnostics only. r10's original failed result is immutable, and its
offline classification is explicitly separate. Source/tag at launch were local
and frozen; remote synchronization remained pending after a bounded transport
push timed out. No further GPU run or strategy sweep has been launched. The next authorized
private acceptance class is exact continuity at the same 512-token warmed
boundary, with cold/warm generation qualification permanently false. Its fixed
r10 raw/classification/independent state evidence and production state-path
hashes are checked before preparation and again by the worker. It grants no
tolerance and preserves all existing active-state, host-transfer and final
loadback assertions. The five-file CPU selection passes 125 checks without
failures or skips; the actual 30-layer receipt also validates. The next run is
one sync/overlap grow/shrink with a 225-second work deadline and 75 seconds
reserved for cleanup, retaining the 300-second total bound.
