# Elastic PP port: vLLM contract to SGLang

This branch ports the existing dynamic-pipeline research implementation to
SGLang without introducing Ray or replacing SGLang's native PP event loop.
The source of truth is the live vLLM implementation in
`/home/bxb1/.codex/worktrees/b9e7/vllm` together with its migration and
persistent-data-plane tests.

## Non-negotiable invariants

1. The process/NCCL group covers the maximum resident candidate set and is
   created once.  Scaling changes a generation-stamped logical route; it does
   not rebuild the communicator on the serving critical path.
2. A topology is a complete, non-overlapping layer partition.  A batch carries
   one topology generation and every hop uses that batch-local generation.
3. Weight preparation and the bulk KV copy happen while the source topology is
   still serving.  A scheduler fence admits no new source-generation work,
   drains all resident ranks, applies the final KV delta, and then commits the
   target route atomically.
4. Logical route commit is separate from physical cleanup.  Migrated-out
   layers, KV views, and physical pages are released after at least one target
   batch has retired and after their CUDA/NCCL lifetime events complete.
5. KV migration copies only live request slots.  Virtual block/slot identity is
   preserved across ranks; physical-page placement is backend-private.
6. KVCacheD capacity is page/group aligned and is calculated from measured
   per-rank weight bytes, non-layer weights, runtime workspace, and exact KV
   geometry.  Logical block capacity and physical page resizing are separate
   operations.
7. A failed prepare or copy leaves the source generation usable.  A committed
   generation never rolls back in place; recovery creates a newer generation.

## Live implementation map

| Concern | vLLM reference | SGLang integration target |
| --- | --- | --- |
| Topology generation and active ranks | `vllm/v1/engine/dynamic_core.py` | `sglang/srt/elastic_pp/topology.py`; scheduler-owned control state |
| Long-lived dynamic PP data plane | `vllm/v1/executor/dynamic_ray_distributed_executor.py`, `dynamic_utils.py` | existing `scheduler_pp_mixin.py` ring and `GroupCoordinator`; add batch-local peers, do not add Ray |
| Scheduler cut and post-cut route publish | `DynamicEngineCore._commit_autoscaling_topology_after_sync` | PP scheduler control message ordered with microbatches |
| Dynamic layer materialization | `dynamic_gpu_model_runner.py::{add_layers,remove_layers}` | model layer factory plus model-loader partial weight filter |
| Direct weight migration | owner-to-owner worker transfer | `elastic_pp/layer_transport.py` over a dedicated duplicate of the maximal PP group |
| Sparse live-slot KV copy and catch-up | `dynamic_kv_synchronizer.py` and `KVPatchMeta` | SGLang token-to-KV pool transfer adapter plus per-generation delta log |
| KVCacheD VMM/page lifetime | `vllm/kvcached_integration.py` | optional SGLang KV storage backend; no hard dependency for default serving |
| Autoscaling policy | `autoscaling_policy.py` | `elastic_pp/autoscaling.py` plus the tokenizer-side controller; decisions reuse the manual transaction path |

## Existing SGLang machinery retained

- `SchedulerPPMixin.event_loop_pp` remains the steady-state scheduling loop.
- Its forward stream, copy stream, CUDA events, last-rank output queue, and
  asynchronous tensor sends remain the overlap implementation.
- `GroupCoordinator` remains the physical communicator owner.
- HiCache/FlexKV/LMCache behavior remains unchanged unless the explicit
  KVCacheD backend is selected.
- Elastic-EP is not reused as PP policy, but its maximal-group/joiner control
  patterns can be reused where their semantics match.

## Current boundary

The implementation now supports real changes in the number of active logical
pipeline stages while the maximal process/NCCL group remains resident.  For
example, a four-rank group may switch between logical PP2
`40,0,0,40` and logical PP4 `20,20,20,20`; zero-layer middle ranks are bypassed
by generation-local peers.

The physical first and last ranks must still remain active because embedding,
final norm, and language-head modules are materialized according to startup
endpoint roles.  A transition such as `80,0,0,0 -> 40,40,0,0` therefore remains
out of scope until endpoint-role materialization lands.  This also means the
current autoscaler changes logical GPU use but does not terminate or return the
warm resident processes to an external cluster resource manager.

## Landed mechanism pieces

- The logical route is generation-stamped and may skip resident zero-layer
  middle ranks while retaining the fixed maximal process group.
- The first checkpoint load records every weight tensor and then calibrates
  actual resident parameter and buffer bytes across ranks.
- Unquantized Llama decoder layers have a transactional residency lifecycle:
  target-only modules are prepared and loaded before commit, while retired
  modules remain resident until CUDA-safe cleanup.
- Weight migration uses an independent PP-shaped NCCL/Gloo communicator.  Its
  manifest validates the generation, source, destination, and exact half-open
  layer interval before installing runtime tensors.
- Weight receives write directly into prepared parameter storage, avoiding a
  second full-size GPU tensor allocation during scale-up.
- The standard unquantized NHD MHA KV pool now has the same transactional
  prepare/commit/deferred-cleanup lifecycle.  A zero-layer resident rank can
  allocate unpublished per-layer backing and atomically publish its pointer
  tables at the scheduler fence.
- Sparse KV bulk copy snapshots all allocated token slots (including radix
  cache residents).  While it runs, the token allocator records new slot
  allocations; the final delta includes only those slots that remain live,
  including old slot ids that were freed and reused.
- KV gather, transport, and scatter run on a dedicated migration CUDA stream
  and send fixed-size slot chunks over the migration communicator.  This is a
  migration memory bound, not dynamic inference chunking.
- Background preparation publishes one common future microstep.  Every PP rank
  drains its microbatch ring at that scheduler fence, launches final-delta
  collectives on the scheduler thread in deterministic rank order, then commits
  model, KV, and route generation atomically.
- The KVCacheD backend refreshes gained layer-group mappings at the fence and
  trims incompatible reserved pages before the new topology becomes visible.
  Deferred cleanup waits on a CUDA event and stays outside the serving pause.
- The optional backlog autoscaler runs only in the central TokenizerManager. It
  polls a compact waiting/running state from every DP replica, checks replica
  topology agreement, applies high/low hysteresis, stable observations,
  post-commit residence time, retry backoff, and an optional operation budget.
  Automatic and manual transitions share one asyncio lock and the exact same
  generation-stamped reconfiguration communicator.

Hybrid, MLA, page-major, post-capture VMM, speculative-copy-table, and
quantized KV layouts remain explicit follow-up work rather than silently
taking an unsafe path.  Dynamic endpoint modules and physical process/GPU
release are also explicit follow-up work.

## Automatic-controller example

The controller is opt-in.  The ordered partitions must cover every model layer,
keep the physical endpoints non-empty, and strictly increase the number of
active ranks:

```bash
python -m sglang.launch_server \
  --pp-size 4 \
  --enable-elastic-pp \
  --elastic-pp-initial-partition 40,0,0,40 \
  --enable-elastic-pp-autoscaling \
  --elastic-pp-autoscale-partitions '40,0,0,40;20,20,20,20' \
  --elastic-pp-autoscale-up-requests 16 \
  --elastic-pp-autoscale-down-requests 0 \
  --elastic-pp-autoscale-up-stable-polls 2 \
  --elastic-pp-autoscale-down-stable-polls 4 \
  --elastic-pp-autoscale-poll-interval-s 0.5 \
  --elastic-pp-autoscale-min-residence-s 10
```

The up/down thresholds are calibration inputs, not universal defaults.  A
performance result must record the complete frontier and controller settings.

## Delivery order

1. Pure topology/transition plan and state machine, with generation and fence
   invariants covered by CPU tests.
2. Batch-local logical route over the fixed PP communicator, initially with a
   manually supplied topology and no KV ownership change.
3. On-demand model-layer and endpoint-role preparation, with exact weight-byte
   profiling recorded during the first load.
4. Native SGLang KV-pool sparse migration and final-delta catch-up.
5. Optional KVCacheD page-backed adapter and aligned capacity formula.
6. Manual PP redistribution with live requests and KVCacheD on H100.
7. Load-driven controller with hysteresis, serialized manual/automatic control,
   and automatic logical PP2-to-PP4-to-PP2 H100 validation.
