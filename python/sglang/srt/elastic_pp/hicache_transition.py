"""Fenced, lossless L2 state transfer for elastic pipeline ownership.

Host slot numbers and radix node counters are local identities. Wire payloads
use namespaced complete token prefixes and global model layer numbers instead.
The caller must retain source pools until commit and provide target pools whose
controller points at the target device pool. L3 and pending I/O fail closed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import torch

from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport


@dataclass(frozen=True)
class HostNode:
    identity: str
    component: int
    slots: torch.Tensor


def _contiguous_host_rows(slots):
    """A CPU-only ordered interval, or None for an arbitrary row gather."""
    if slots.device.type != "cpu" or slots.ndim != 1 or not slots.numel():
        return None
    start = int(slots[0])
    if start < 0:
        return None
    if slots.numel() == 1 or torch.equal(
        slots, torch.arange(start, start + slots.numel(), dtype=slots.dtype)
    ):
        return start
    return None


def _gather_host_rows(tensor, slots, *, fast_copy):
    if fast_copy and tensor.device.type == "cpu":
        start = _contiguous_host_rows(slots)
        if start is not None:
            # The wire owns this snapshot even when source rows are contiguous.
            return tensor.narrow(0, start, slots.numel()).clone()
        # index_select already allocates independent storage; a second clone
        # only doubles CPU traffic while the scheduler is fenced.
        return tensor.index_select(0, slots)
    return tensor.index_select(0, slots.to(tensor.device)).cpu().clone()


def _copy_host_rows(target, slots, source, *, fast_copy):
    if fast_copy and target.device.type == source.device.type == "cpu":
        start = _contiguous_host_rows(slots)
        if start is not None:
            target.narrow(0, start, slots.numel()).copy_(source)
            return
    target.index_copy_(0, slots.to(target.device), source.to(target.device))


def host_nodes(cache) -> dict[tuple[str, int], HostNode]:
    """Read at a scheduler fence; include CPU-only prefixes."""
    result = {}
    stack = [(cache.tree_core.root_node, ())]
    while stack:
        node, prefix = stack.pop()
        key = node.key
        if node.parent is not None:
            if key.is_bigram:
                raise NotImplementedError(
                    "elastic HiCache bigram trees are unsupported"
                )
            prefix = prefix + tuple(key.raw_token_ids())
        identity = hashlib.sha256(
            json.dumps(
                [key.extra_key, key.cache_salt, prefix], separators=(",", ":")
            ).encode()
        ).hexdigest()
        for component, data in enumerate(node.component_data):
            if data.host_value is not None:
                slots = (
                    torch.as_tensor(data.host_value, dtype=torch.int64).cpu().clone()
                )
                result[(identity, component)] = HostNode(identity, component, slots)
        stack.extend((child, prefix) for child in node.children.values())
    return result


def host_layer_views(pool, layer_ids):
    """Return global-layer -> named [host_slot, ...] views, without copying."""
    ids = tuple(int(i) for i in layer_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate global host layer identity")
    if getattr(pool, "mtp_draft_device_pools", ()):
        raise NotImplementedError("elastic HiCache draft pools are unsupported")
    if hasattr(pool, "conv_buffer"):
        tensors = {"temporal": pool.temporal_buffer}
        tensors.update({f"conv_{i}": x for i, x in enumerate(pool.conv_buffer)})
        if pool.layout not in ("page_first", "page_first_direct"):
            raise NotImplementedError("unsupported recurrent host layout")
        views = {g: {n: t[:, i] for n, t in tensors.items()} for i, g in enumerate(ids)}
        count = pool.num_mamba_layers
    elif hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        if pool.layout == "layer_first":
            views = {
                g: {"k": pool.k_buffer[i], "v": pool.v_buffer[i]}
                for i, g in enumerate(ids)
            }
        elif pool.layout == "page_first":
            views = {
                g: {"k": pool.k_buffer[:, i], "v": pool.v_buffer[:, i]}
                for i, g in enumerate(ids)
            }
        else:
            raise NotImplementedError(
                "elastic HiCache requires layer_first or page_first MHA"
            )
        count = pool.layer_num
    else:
        raise NotImplementedError(
            f"unsupported elastic HiCache pool: {type(pool).__name__}"
        )
    if len(ids) != count:
        raise ValueError("global host layer map does not match physical pool")
    return views


def assert_host_quiescent(cache, *, allow_pp_control_sends=False):
    if getattr(cache, "enable_session_radix_cache", False):
        raise NotImplementedError(
            "elastic HiCache session-radix frontier migration unsupported"
        )
    if getattr(cache, "enable_storage", False):
        raise NotImplementedError("elastic HiCache L3 migration is unsupported")
    if getattr(cache, "buffer_pipeline", None) is not None:
        raise NotImplementedError(
            "elastic HiCache buffer-only migration is unsupported"
        )
    for name in (
        "ongoing_backup",
        "ongoing_load_back",
        "ongoing_prefetch",
        "ongoing_write_through",
    ):
        if getattr(cache, name, None):
            raise RuntimeError(f"elastic HiCache requires retired {name}")
    # PP readiness packets do not read or write CPU cache rows. A scheduler
    # snapshot may leave those sends in flight; waiting here can deadlock the
    # serving ring. Fence-time callers must still retire the old communicator.
    if not allow_pp_control_sends and getattr(cache, "work_list", None):
        raise RuntimeError("elastic HiCache requires retired work_list")
    controller = getattr(cache, "cache_controller", None)
    for name in ("load_queue", "write_queue", "ack_load_queue", "ack_write_queue"):
        if getattr(controller, name, None):
            raise RuntimeError(f"elastic HiCache requires retired controller {name}")


class HiCacheTransitionAdapter:
    """One data plane shared by synchronous and pre-copy scheduling policies.

    pools maps component integer to global-layer tensor views. Target pools
    must be independent of live source pools; apply is reversible until commit.
    The adapter deliberately never resets the radix tree or request allocator.
    """

    def __init__(self, *, cache, pools, group, rank, fast_copy=False):
        self.cache, self.pools, self.group, self.rank = cache, pools, group, rank
        self.fast_copy = fast_copy
        self._staged = {}
        self._undo = []
        self._generation = None
        self._snapshot = None
        self.precopy = None
        self.metrics = {
            "snapshot_bytes": 0,
            "host_bytes_sent": 0,
            "host_bytes_received": 0,
            "peak_staged_bytes": 0,
            "private_target_undo_bytes_avoided": 0,
        }

    def snapshot_at_fence(self, transition):
        """Capture immutable CPU rows before allocating target backing."""
        assert_host_quiescent(self.cache)
        snapshot = {}
        outgoing = {
            layer
            for edge in transition.kv_transfers
            if edge.src_rank == self.rank
            for layer in range(edge.layer_range.start, edge.layer_range.end)
        }
        nodes = host_nodes(self.cache)
        if self.precopy is not None and self.precopy.batch_io:
            self.precopy.verify_at_fence(self.pools, nodes)
            self.metrics.update(self.precopy.metrics)
        for (identity, component), node in nodes.items():
            for layer, tensors in self.pools.get(component, {}).items():
                if layer not in outgoing:
                    continue
                key = (identity, component, layer)
                if self.precopy is not None:
                    original = self.precopy.records.get(key)
                    if original is not None and self.precopy.matches(
                        key, tensors, node.slots
                    ):
                        self.metrics[
                            "host_verified_unchanged_bytes"
                        ] = self.metrics.get("host_verified_unchanged_bytes", 0) + sum(
                            t.numel() * t.element_size()
                            for t in original.tensors.values()
                        )
                        continue
                snapshot[key] = {
                    name: _gather_host_rows(t, node.slots, fast_copy=self.fast_copy)
                    for name, t in tensors.items()
                }
        self._snapshot = snapshot
        self.metrics["snapshot_bytes"] = sum(
            t.numel() * t.element_size()
            for tensors in snapshot.values()
            for t in tensors.values()
        )
        return self.metrics["snapshot_bytes"]

    def prepare_transition(self, transition):
        if self._generation is not None:
            raise RuntimeError("HiCache transition already prepared")
        # This method is fence-only: even metadata must not race radix splits.
        assert_host_quiescent(self.cache)
        self._generation = transition.target.generation

    def exchange_at_fence(self, transition):
        if self._generation != transition.target.generation:
            raise RuntimeError("HiCache transition generation mismatch")
        assert_host_quiescent(self.cache)
        nodes = host_nodes(self.cache)
        receive_errors = []
        if self.precopy is not None:
            incoming = {
                layer
                for e in transition.kv_transfers
                if e.dst_rank == self.rank
                for layer in range(e.layer_range.start, e.layer_range.end)
            }
            self._staged = {
                key: record.tensors
                for key, record in self.precopy.preloaded.items()
                if key[2] in incoming and key[:2] in nodes
            }
        for edge in sorted(
            transition.kv_transfers,
            key=lambda e: (
                e.src_rank,
                e.dst_rank,
                e.layer_range.start,
                e.layer_range.end,
            ),
        ):
            if self.rank == edge.src_rank:
                payload = {"generation": self._generation, "records": []}
                for (identity, component), node in sorted(nodes.items()):
                    for layer, tensors in sorted(self.pools.get(component, {}).items()):
                        if edge.layer_range.start <= layer < edge.layer_range.end and (
                            self._snapshot is None
                            or (identity, component, layer) in self._snapshot
                        ):
                            payload["records"].append(
                                (
                                    identity,
                                    component,
                                    layer,
                                    (
                                        self._snapshot[(identity, component, layer)]
                                        if self._snapshot is not None
                                        else {
                                            name: _gather_host_rows(
                                                t,
                                                node.slots,
                                                fast_copy=self.fast_copy,
                                            )
                                            for name, t in tensors.items()
                                        }
                                    ),
                                )
                            )
                # Tensor-dict transport only accepts top-level tensors.
                manifest, wire = [], {"generation": self._generation}
                for i, (identity, component, layer, tensors) in enumerate(
                    payload["records"]
                ):
                    manifest.append((identity, component, layer, tuple(tensors)))
                    for name, tensor in tensors.items():
                        wire[f"{i}:{name}"] = tensor
                wire["manifest"] = manifest
                wire["node_manifest"] = sorted(
                    (identity, component, len(node.slots))
                    for (identity, component), node in nodes.items()
                )
                self.metrics["host_bytes_sent"] += sum(
                    t.numel() * t.element_size()
                    for t in wire.values()
                    if isinstance(t, torch.Tensor)
                )
                works = self.group.send_tensor_dict(
                    wire, dst=edge.dst_rank, async_send=True
                )
                ElasticPPLayerTransport.wait(works or ())
            elif self.rank == edge.dst_rank:
                wire = self.group.recv_tensor_dict(src=edge.src_rank)
                if wire.get("generation") != self._generation:
                    receive_errors.append("HiCache payload generation mismatch")
                    continue
                expected_nodes = sorted(
                    (identity, component, len(node.slots))
                    for (identity, component), node in nodes.items()
                )
                if wire.get("node_manifest") != expected_nodes:
                    receive_errors.append(
                        "HiCache source/target host prefix manifests differ"
                    )
                    continue
                seen = set()
                for i, (identity, component, layer, names) in enumerate(
                    wire["manifest"]
                ):
                    if not edge.layer_range.start <= layer < edge.layer_range.end:
                        receive_errors.append("HiCache layer outside transfer")
                        continue
                    key = (identity, component, layer)
                    if key in seen:
                        receive_errors.append("duplicate HiCache state")
                        continue
                    seen.add(key)
                    self._staged[key] = {name: wire[f"{i}:{name}"] for name in names}
                    self.metrics["host_bytes_received"] += sum(
                        t.numel() * t.element_size() for t in self._staged[key].values()
                    )
                self.metrics["peak_staged_bytes"] = max(
                    self.metrics["peak_staged_bytes"],
                    sum(
                        t.numel() * t.element_size()
                        for tensors in self._staged.values()
                        for t in tensors.values()
                    ),
                )

        if receive_errors:
            raise RuntimeError("; ".join(receive_errors))

    def apply_staged(
        self, transition, *, target_cache, target_pools, private_target=False
    ):
        if self._generation != transition.target.generation:
            raise RuntimeError("HiCache transition generation mismatch")
        assert_host_quiescent(target_cache)
        if private_target and target_cache is self.cache:
            raise RuntimeError("private HiCache target must not be the source cache")
        source_storage = {
            t.untyped_storage().data_ptr()
            for layers in self.pools.values()
            for tensors in layers.values()
            for t in tensors.values()
            if t.numel()
        }
        nodes = host_nodes(target_cache)
        operations = []
        for (identity, component, layer), tensors in self._staged.items():
            node = nodes.get((identity, component))
            if node is None:
                raise RuntimeError("target radix tree lacks source CPU prefix")
            views = target_pools[component][layer]
            if set(views) != set(tensors):
                raise ValueError("HiCache tensor schema mismatch")
            for name, source in tensors.items():
                target = views[name]
                slots = node.slots.to(target.device)
                if (
                    source.shape != (len(slots), *target.shape[1:])
                    or source.dtype != target.dtype
                ):
                    raise ValueError("HiCache tensor shape/dtype mismatch")
                if private_target and target.numel():
                    if target.untyped_storage().data_ptr() in source_storage:
                        raise RuntimeError("private HiCache target aliases live source")
                    if (
                        target.untyped_storage().data_ptr()
                        == source.untyped_storage().data_ptr()
                    ):
                        raise RuntimeError("private HiCache target aliases staged rows")
                preloaded = (
                    self.precopy.preloaded.get((identity, component, layer))
                    if self.precopy is not None
                    else None
                )
                reuse = (
                    preloaded is not None
                    and preloaded.tensors.get(name) is source
                    and torch.equal(preloaded.slots, node.slots)
                )
                operations.append((target, slots, source, reuse))
        # Validate everything before the first mutation; retain overwritten rows
        # until publication so abort also repairs a partially failed scatter.
        for target, slots, source, reuse in operations:
            if reuse:
                self.metrics["host_preloaded_incoming_bytes_reused"] = (
                    self.metrics.get("host_preloaded_incoming_bytes_reused", 0)
                    + source.numel() * source.element_size()
                )
                continue
            if private_target:
                # The scheduler owns this newly allocated, unpublished pool.
                # Abort discards it; no serving tensor depends on its old rows.
                self.metrics["private_target_undo_bytes_avoided"] += (
                    source.numel() * source.element_size()
                )
            else:
                self._undo.append(
                    (target, slots, target.index_select(0, slots).clone())
                )
            _copy_host_rows(target, slots, source, fast_copy=self.fast_copy)

    def commit(self, transition):
        if self._generation != transition.target.generation:
            raise RuntimeError("HiCache transition generation mismatch")
        self._undo.clear()
        self._staged.clear()
        self._snapshot = None
        self._generation = None

    def abort(self, transition):
        if self._generation not in (None, transition.target.generation):
            raise RuntimeError("HiCache transition generation mismatch")
        for target, slots, original in reversed(self._undo):
            target.index_copy_(0, slots, original)
        self._undo.clear()
        self._staged.clear()
        self._snapshot = None
        self._generation = None


def clone_host_allocator(source, target):
    """Preserve live addresses and free-list order; discard only free capacity."""
    limit = min(source.logical_size, target.logical_size)
    free = torch.as_tensor(source.free_slots, dtype=torch.int64).cpu()
    releases = [
        torch.as_tensor(v, dtype=torch.int64).cpu() for v in source.release_slots
    ]
    used = torch.ones(source.logical_size, dtype=torch.bool)
    used[free] = False
    for value in releases:
        used[value] = False
    if hasattr(source, "slot_used") and not torch.equal(used, source.slot_used.cpu()):
        raise RuntimeError("source host allocator free lists and used slots disagree")
    if used[limit:].any():
        raise RuntimeError("target host pool cannot preserve live high host slots")
    target.free_slots = free[free < target.logical_size].clone()
    target.release_slots = [
        v[v < target.logical_size].clone()
        for v in releases
        if (v < target.logical_size).any()
    ]
    target.num_release_slots = sum(len(v) for v in target.release_slots)
    if hasattr(target, "slot_used"):
        target.slot_used.zero_()
        target.slot_used[:limit].copy_(used[:limit])
    if hasattr(target, "mem_state") and hasattr(source, "mem_state"):
        target.mem_state[:limit].copy_(source.mem_state[:limit])
    if target.logical_size > source.logical_size:
        target.free_slots = torch.cat(
            (
                target.free_slots,
                torch.arange(
                    source.logical_size, target.logical_size, dtype=torch.int64
                ),
            )
        )


def project_target_device_pool(pool, *, target, generation):
    """Non-publishing device views for a replacement L2 controller.

    The tensors remain owned by the real pool's pending transaction. This view
    must never be installed in the model runner or serving token allocator.
    """
    import copy

    from sglang.srt.mem_cache.elastic_hybrid_pool import ElasticHybridLinearKVPool

    def mha_view(source, interval):
        view = copy.copy(source)
        current = source._current_elastic_pp_kv_map()
        pairs = [
            (
                current[i]
                if i in current
                else source.get_elastic_pp_pending_kv_buffer(
                    generation=generation, layer_id=i
                )
            )
            for i in range(interval.start, interval.end)
        ]
        view.start_layer, view.end_layer, view.layer_num = (
            interval.start,
            interval.end,
            interval.size,
        )
        view.k_buffer = [p[0] for p in pairs]
        view.v_buffer = [p[1] for p in pairs]
        view.k_data_ptrs, view.v_data_ptrs, view.data_ptrs, view.data_strides = (
            source._build_elastic_pp_pointer_tables(view.k_buffer, view.v_buffer)
        )
        # copy.copy must not retain source-generation descriptors/transactions.
        view._kv_buffer_descs = []
        view._kv_copy_config = None
        view._elastic_pp_pending_kv = {}
        view._elastic_pp_retired_kv = {}
        return view

    if not isinstance(pool, ElasticHybridLinearKVPool):
        return mha_view(pool, target), None
    view = copy.copy(pool)
    view.start_layer, view.end_layer = target.start, target.end
    view.full_kv_pool = mha_view(pool.full_kv_pool, pool._attention_range(target))
    # HybridLinearKVPool's mapping values are ordinal addresses into the full
    # pool; the controller expects *local* addresses into host tensor layers.
    ids = pool.elastic_pp_kv_layer_ids(target)
    view.full_attention_layer_id_mapping = {layer: i for i, layer in enumerate(ids)}
    view.full_layer_nums = len(ids)
    req = copy.copy(pool.req_to_token_pool)
    mamba = copy.copy(req.mamba_pool)
    mamba.mamba_cache = pool._pending_state
    mamba.__dict__.pop("_conv_slot_desc", None)
    mamba.__dict__.pop("_conv_fuse_ok", None)
    mamba.mamba_layer_ids = list(pool._pending_mamba_ids)
    mamba.num_mamba_layers = len(pool._pending_mamba_ids)
    req.mamba_pool = mamba
    req.mamba_map = {layer: i for i, layer in enumerate(pool._pending_mamba_ids)}
    view.mamba_pool, view.req_to_token_pool = mamba, req
    return view, req


def cache_host_views(cache, device_pool=None):
    """Bind physical host tensor indices to global model layer identities."""
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    device_pool = device_pool or cache.token_to_kv_pool_allocator.get_kvcache()
    result = {}
    full = getattr(cache, "full_kv_pool_host", None)
    if full is not None and getattr(full, "layer_num", 0):
        mapping = getattr(device_pool, "full_attention_layer_id_mapping", None)
        ids = (
            tuple(mapping)
            if mapping is not None
            else tuple(range(device_pool.start_layer, device_pool.end_layer))
        )
        result[int(ComponentType.FULL)] = host_layer_views(full, ids)
    mamba = getattr(cache, "mamba_pool_host", None)
    if mamba is not None and getattr(mamba, "num_mamba_layers", 0):
        result[int(ComponentType.MAMBA)] = host_layer_views(
            mamba, tuple(device_pool.req_to_token_pool.mamba_map)
        )
    return result


def prepare_target_host_cache(
    cache,
    *,
    device_pool,
    target,
    generation,
    fast_copy=False,
    metrics=None,
    allocation_only=False,
    prepared=None,
    preloaded=None,
    verified_preloaded=None,
):
    """Build target L2 while keeping allocation failures safe for GPU rollback."""
    try:
        return _prepare_target_host_cache(
            cache,
            device_pool=device_pool,
            target=target,
            generation=generation,
            fast_copy=fast_copy,
            metrics=metrics,
            allocation_only=allocation_only,
            prepared=prepared,
            preloaded=preloaded,
            verified_preloaded=verified_preloaded,
        )
    except BaseException as exc:
        from sglang.srt.elastic_pp.backing_spill import discard_tensor_tracebacks

        raise discard_tensor_tracebacks(exc)


def clone_cache_runtime_metadata(cache):
    """Give a target generation private collaborators pointing at its cache."""
    import copy
    import types

    from sglang.srt.session.streaming_session import StreamingSession

    clone = copy.copy(cache)
    clone.tree_core = copy.copy(cache.tree_core)
    clone.components = {
        ct: copy.copy(component) for ct, component in cache.components.items()
    }
    clone._components_tuple = tuple(clone.components.values())
    clone.tree_core.components_by_type = clone.components
    clone.tree_core.components = clone._components_tuple
    clone.session_refs = copy.copy(cache.session_refs)
    clone.session_refs.tree_core = clone.tree_core
    clone.session_refs.components = clone._components_tuple
    # copy.copy probes special attributes on the uninitialized wrapper, whose
    # __getattr__ delegates to inner. Construct it normally instead, retaining
    # logical slot handles in a private mapping for this generation.
    clone.session = StreamingSession(inner=clone)
    clone.session.slots = dict(cache.session.slots)
    for ct, component in clone.components.items():
        source = cache.components[ct]
        component.cache = clone
        component.tree_core = clone.tree_core
        # A component can cache one of its own bound methods as an eviction
        # strategy. Shallow copying that callable would retain the old owner.
        for name, value in vars(component).items():
            if isinstance(value, types.MethodType) and value.__self__ is source:
                setattr(component, name, types.MethodType(value.__func__, component))
    return clone


def prewarm_host_allocator(scheduler):
    """Warm exact candidate L2 allocation shapes before any request can run.

    No cache metadata is migrated or published. All ranks join the normal
    fixed-size host-capacity collectives, including dormant ranks. Temporary
    pools are destroyed while PyTorch keeps their registered blocks reusable.
    """
    import dataclasses
    import time

    mr = scheduler.tp_worker.model_runner
    if not scheduler.is_initializing:
        raise RuntimeError("host allocator warmup requires startup")
    if mr.server_args.elastic_pp_hicache_allocator != "pytorch":
        raise RuntimeError("host allocator warmup requires PyTorch pinned backing")
    source = scheduler.tree_cache
    if source.enable_storage:
        raise ValueError("host allocator warmup supports L2-only HiCache")
    pool = mr.token_to_kv_pool
    clone = clone_cache_runtime_metadata(source)
    clone.sidecar_pool_specs = []
    clone.req_to_token_pool = mr.req_to_token_pool
    params = dataclasses.replace(
        source._hicache_init_params, req_to_token_pool=mr.req_to_token_pool
    )
    started = time.monotonic()
    before = torch.cuda.memory.host_memory_stats()
    try:
        if mr.model.start_layer == mr.model.end_layer:
            build_dormant_host_stack(
                clone, source=source, projected=pool, params=params
            )
        else:
            clone.init_hicache(source._hicache_init_server_args, params)
    finally:
        if clone.host_pool_group is not source.host_pool_group:
            discard_host_cache(clone)
        # The temporary assembler registers its own layer-done counter. The
        # live empty cache keeps its original bindings throughout startup.
        getattr(pool, "full_kv_pool", pool).register_layer_transfer_counter(
            source.cache_controller.layer_done_counter
        )
        if hasattr(mr.req_to_token_pool, "register_layer_transfer_counter"):
            mr.req_to_token_pool.register_layer_transfer_counter(
                source.cache_controller.layer_done_counter
            )
    after = torch.cuda.memory.host_memory_stats()
    return {
        "interval": [mr.model.start_layer, mr.model.end_layer],
        "seconds": time.monotonic() - started,
        "pinned_allocated_bytes": after["allocated_bytes.current"],
        "pinned_active_bytes": after["active_bytes.current"],
        "new_host_allocations": after["num_host_alloc"] - before["num_host_alloc"],
        "host_allocations": after["num_host_alloc"],
    }


def adopt_prepared_host_stack(clone, prepared, *, allocator, params):
    """Move unused backing into a fresh fence-metadata clone, without allocation."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _COMPONENT_HOST_ATTR,
    )
    from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment

    if prepared.host_pool_group is None or prepared.cache_controller is None:
        raise RuntimeError("background HiCache backing has already been consumed")
    assert_host_quiescent(prepared)
    clone.host_pool_group = prepared.host_pool_group
    clone.cache_controller = prepared.cache_controller
    clone.load_cache_event = prepared.load_cache_event
    clone.token_to_kv_pool_host = clone.host_pool_group
    for ct, (cache_attr, component_attr) in _COMPONENT_HOST_ATTR.items():
        if ct in clone.components:
            host = getattr(prepared, cache_attr, None)
            setattr(clone, cache_attr, host)
            setattr(clone.components[ct], component_attr, host)
    clone.cache_controller.mem_pool_device_allocator = allocator
    clone._hicache_init_params = params
    clone._storage_attachment = StorageAttachment(clone)
    # No destructor may later retire the backing now owned by clone.
    prepared.host_pool_group = None
    prepared.cache_controller = None
    prepared.token_to_kv_pool_host = None
    for ct, (cache_attr, component_attr) in _COMPONENT_HOST_ATTR.items():
        setattr(prepared, cache_attr, None)
        if ct in prepared.components:
            setattr(prepared.components[ct], component_attr, None)


def _prepare_target_host_cache(
    cache,
    *,
    device_pool,
    target,
    generation,
    fast_copy=False,
    metrics=None,
    allocation_only=False,
    prepared=None,
    preloaded=None,
    verified_preloaded=None,
):
    """Build replacement L2 backing without resetting the live radix tree.

    Returns a private cache clone. Runtime publishes it only with the topology,
    rebinds its allocator/request pool to the real committed pools, and updates
    tree_core.components_by_type. Abort destroys only clone.host_pool_group.
    The old cache and backing must remain alive until topology publication.
    """
    import copy
    import dataclasses

    metrics = {} if metrics is None else metrics
    started = time.monotonic()
    if not allocation_only:
        assert_host_quiescent(cache)
    projected, req = project_target_device_pool(
        device_pool, target=target, generation=generation
    )
    allocator = copy.copy(cache.token_to_kv_pool_allocator)
    # Only the construction path sees this projected pool. Allocation/free
    # continues through the original shared allocator at publication.
    allocator.get_kvcache = lambda: projected
    clone = clone_cache_runtime_metadata(cache)
    clone.sidecar_pool_specs = []
    clone.token_to_kv_pool_allocator = allocator
    clone.req_to_token_pool = req or cache.req_to_token_pool
    params = dataclasses.replace(
        cache._hicache_init_params,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=clone.req_to_token_pool,
    )
    metrics["host_target_metadata_seconds"] = time.monotonic() - started
    try:
        started = time.monotonic()
        if prepared is not None:
            adopt_prepared_host_stack(
                clone, prepared, allocator=allocator, params=params
            )
        elif target.size == 0:
            build_dormant_host_stack(
                clone, source=cache, projected=projected, params=params
            )
        else:
            clone.init_hicache(cache._hicache_init_server_args, params)
        metrics["host_target_init_seconds"] = time.monotonic() - started
        if allocation_only:
            clone._elastic_pp_projected_pool = projected
            return clone
        started = time.monotonic()
        for old_entry in cache.host_pool_group.entries:
            new_pool = clone.host_pool_group.get_pool(old_entry.name)
            metadata = getattr(cache, "_elastic_host_allocator_metadata", {})
            clone_host_allocator(
                metadata.get(old_entry.name, old_entry.host_pool), new_pool
            )
        metrics["host_target_allocator_seconds"] = time.monotonic() - started
        started = time.monotonic()
        old_views = (
            {}
            if hasattr(cache, "_elastic_host_allocator_metadata")
            else cache_host_views(cache, device_pool)
        )
        new_views = cache_host_views(clone, projected)
        if fast_copy:
            old_storage = {
                t.untyped_storage().data_ptr()
                for layers in old_views.values()
                for tensors in layers.values()
                for t in tensors.values()
                if t.numel()
            }
            if any(
                t.numel() and t.untyped_storage().data_ptr() in old_storage
                for layers in new_views.values()
                for tensors in layers.values()
                for t in tensors.values()
            ):
                raise RuntimeError("private HiCache backing aliases source storage")
        nodes = host_nodes(cache)
        retained_bytes = 0
        for (_, component), node in nodes.items():
            for layer in (
                old_views.get(component, {}).keys()
                & new_views.get(component, {}).keys()
            ):
                record = (preloaded or {}).get((node.identity, component, layer))
                if record is not None:
                    from sglang.srt.elastic_pp.hicache_precopy import rows_match

                    verified = (verified_preloaded or {}).get(
                        (node.identity, component, layer)
                    )
                    if verified is record or rows_match(
                        old_views[component][layer], record, node.slots
                    ):
                        metrics["host_preloaded_retained_bytes_reused"] = metrics.get(
                            "host_preloaded_retained_bytes_reused", 0
                        ) + sum(
                            t.numel() * t.element_size()
                            for t in record.tensors.values()
                        )
                        continue
                for name, source in old_views[component][layer].items():
                    dest = new_views[component][layer][name]
                    start = _contiguous_host_rows(node.slots) if fast_copy else None
                    if start is not None:
                        dest.narrow(0, start, node.slots.numel()).copy_(
                            source.narrow(0, start, node.slots.numel())
                        )
                    else:
                        dest.index_copy_(
                            0,
                            node.slots.to(dest.device),
                            source.index_select(0, node.slots.to(source.device)),
                        )
                    retained_bytes += (
                        node.slots.numel()
                        * (source.numel() // max(1, source.shape[0]))
                        * source.element_size()
                    )
        metrics["host_retained_copy_seconds"] = time.monotonic() - started
        metrics["host_retained_copy_bytes"] = retained_bytes
    except BaseException:
        if clone.host_pool_group is not cache.host_pool_group:
            discard_host_cache(clone)
        raise
    clone._elastic_pp_projected_pool = projected
    return clone


def rebind_cache_device_pools(cache, device_pool, *, refresh_recurrent_pointers=False):
    """Replace construction-only views by the committed/rolled-back real pools."""
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

    bindings = []
    for entry in cache.host_pool_group.entries:
        host = entry.host_pool
        if not isinstance(host, (LogicalHostPool, MambaPoolHost, MHATokenToKVPoolHost)):
            raise NotImplementedError(
                f"elastic backing rebuild cannot rebind {type(host).__name__}"
            )
        if getattr(host, "mtp_draft_device_pools", ()):
            raise NotImplementedError("elastic backing rebuild does not support MTP")
        if entry.name == PoolName.KV:
            real = getattr(device_pool, "full_kv_pool", device_pool)
        elif entry.name == PoolName.MAMBA:
            real = device_pool.mamba_pool
        else:
            raise NotImplementedError(
                "elastic backing rebuild supports FULL/MAMBA only"
            )
        bindings.append((entry, host, real))
    for entry, host, real in bindings:
        entry.device_pool = real
        if hasattr(host, "device_pool"):
            host.device_pool = real
        if isinstance(host, MambaPoolHost) and refresh_recurrent_pointers:
            host.temporal_device_ptrs = torch.tensor(
                [
                    real.mamba_cache.temporal[i].data_ptr()
                    for i in range(real.num_mamba_layers)
                ],
                dtype=torch.uint64,
                device=real.device,
            )
            host.conv_device_ptrs = [
                torch.tensor(
                    [tensor[i].data_ptr() for i in range(real.num_mamba_layers)],
                    dtype=torch.uint64,
                    device=real.device,
                )
                for tensor in real.mamba_cache.conv
            ]
    cache.cache_controller.mem_pool_device = getattr(
        device_pool, "full_kv_pool", device_pool
    )
    cache.__dict__.pop("_elastic_pp_projected_pool", None)


class SchedulerHiCacheTransition:
    """Coordinator-facing transaction; source remains owned through publish."""

    def __init__(self, *, scheduler, model_runner, group, rank):
        self.scheduler, self.model_runner = scheduler, model_runner
        self.group, self.rank = group, rank
        self.adapter = None
        self.target_cache = None
        self.retired_caches = []
        self._request_state = None
        self._source_cache = None
        self._last_metrics = {}
        self._phase_metrics = {}
        self.fast_copy = bool(
            getattr(
                getattr(model_runner, "server_args", None),
                "elastic_pp_hicache_fast_copy",
                False,
            )
        )
        self._released_device_aliases_generation = None
        self.background = bool(
            getattr(
                getattr(model_runner, "server_args", None),
                "elastic_pp_hicache_background",
                False,
            )
        )
        self.batch_io = bool(
            getattr(
                getattr(model_runner, "server_args", None),
                "elastic_pp_hicache_batch_io",
                False,
            )
        )
        if self.batch_io and not self.background:
            raise ValueError("batched HiCache I/O requires background HiCache")
        if self.background and (
            not self.fast_copy
            or getattr(model_runner.server_args, "elastic_pp_transition_strategy", None)
            != "background"
            or getattr(model_runner.server_args, "elastic_pp_transition_backing", None)
            != "overlap"
        ):
            raise ValueError(
                "background HiCache requires fast-copy and background/overlap"
            )
        self.precopy = None
        self.prepared_host = None
        self._background_metrics = {}

    def begin_background(self, transition):
        """Scheduler-only CPU snapshot. Never drain or mutate active HiCache I/O."""
        if not self.background:
            return
        if self.precopy is not None or self.prepared_host is not None:
            raise RuntimeError("previous background HiCache generation still owned")
        self._background_metrics = {}
        self._last_metrics = {}
        from sglang.srt.elastic_pp.hicache_precopy import HiCachePrecopy

        cache = self.scheduler.tree_cache
        try:
            self.precopy = HiCachePrecopy.capture(
                cache,
                (
                    cache_host_views(cache)
                    if self.rank in transition.source.active_ranks
                    else {}
                ),
                transition.target.generation,
                batch_io=self.batch_io,
            )
        except Exception as exc:
            # A busy I/O epoch is not safe to snapshot. All peers select the
            # existing full-fence path instead; never read it speculatively.
            self._background_metrics["host_background_fallback"] = (
                type(exc).__name__ + ": " + str(exc)
            )

    def allocate_background(self, transition, *, enabled):
        if not enabled:
            self._reset_precopy()
            self._background_metrics["host_background_used"] = False
            return
        if (
            self.precopy is None
            or self.precopy.generation != transition.target.generation
        ):
            raise RuntimeError("missing background HiCache snapshot")
        from sglang.srt.elastic_pp.hicache_precopy import copy_records
        from sglang.srt.mem_cache.pool_host.base import hicache_capacity_group

        metrics = self._background_metrics
        metrics.update(self.precopy.metrics)
        started = time.monotonic()
        with hicache_capacity_group(self.group):
            self.prepared_host = prepare_target_host_cache(
                self.precopy.metadata_cache,
                device_pool=self.model_runner.token_to_kv_pool,
                target=transition.target.layer_range(self.rank),
                generation=transition.target.generation,
                fast_copy=True,
                metrics=metrics,
                allocation_only=True,
            )
        metrics["host_background_pool_init_seconds"] = metrics.pop(
            "host_target_init_seconds"
        )
        metrics["host_background_pool_metadata_seconds"] = metrics.pop(
            "host_target_metadata_seconds"
        )
        views = cache_host_views(
            self.prepared_host, self.prepared_host._elastic_pp_projected_pool
        )
        source_storage = {
            t.untyped_storage().data_ptr()
            for layers in cache_host_views(self.precopy.metadata_cache).values()
            for tensors in layers.values()
            for t in tensors.values()
            if t.numel()
        }
        if any(
            t.numel() and t.untyped_storage().data_ptr() in source_storage
            for layers in views.values()
            for tensors in layers.values()
            for t in tensors.values()
        ):
            raise RuntimeError("background HiCache backing aliases live source")
        self.precopy.preloaded.update(
            copy_records(views, self.precopy.records, batch_io=self.batch_io)
        )
        metrics["host_background_allocate_seconds"] = time.monotonic() - started
        metrics["host_background_used"] = True

    def copy_background(self, transition):
        if self.precopy is None:
            return
        started = time.monotonic()
        self.precopy.exchange(
            transition,
            group=self.group,
            rank=self.rank,
            target_pools=cache_host_views(
                self.prepared_host, self.prepared_host._elastic_pp_projected_pool
            ),
        )
        self._background_metrics.update(self.precopy.metrics)
        self._background_metrics["host_background_exchange_seconds"] = (
            time.monotonic() - started
        )

    def _reset_precopy(self):
        if self.precopy is not None:
            self.precopy.reset()
        self.precopy = None
        if (
            self.prepared_host is not None
            and self.prepared_host.host_pool_group is not None
        ):
            discard_host_cache(self.prepared_host)
        self.prepared_host = None

    def release_source_device_aliases(self, transition):
        """Fence-only: retire projected owners and invalidate raw GDN addresses."""
        generation = transition.target.generation
        if self._released_device_aliases_generation is not None:
            if self._released_device_aliases_generation != generation:
                raise RuntimeError("HiCache device binding generation mismatch")
            return
        cache = self._source_cache or self.scheduler.tree_cache
        self._source_cache = cache
        assert_host_quiescent(cache)
        rebind_cache_device_pools(cache, self.model_runner.token_to_kv_pool)
        # Validation/rebinding preserves serving addresses. Register before the
        # first invalidation so abort can repair a partially invalidated table.
        self._released_device_aliases_generation = generation
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        for entry in cache.host_pool_group.entries:
            if isinstance(entry.host_pool, MambaPoolHost):
                entry.host_pool.temporal_device_ptrs = None
                entry.host_pool.conv_device_ptrs = []

    def restore_source_device_aliases(self, transition):
        if self._released_device_aliases_generation is None:
            return
        if self._released_device_aliases_generation != transition.target.generation:
            raise RuntimeError("HiCache device binding generation mismatch")
        cache = self._source_cache or self.scheduler.tree_cache
        if self.scheduler.tree_cache is not cache:
            raise RuntimeError(
                "cannot restore source HiCache bindings after publication"
            )
        rebind_cache_device_pools(
            cache, self.model_runner.token_to_kv_pool, refresh_recurrent_pointers=True
        )
        self._released_device_aliases_generation = None

    def finalize_source_device_aliases(self, transition):
        if self._released_device_aliases_generation is None:
            return
        if self._released_device_aliases_generation != transition.target.generation:
            raise RuntimeError("HiCache device binding generation mismatch")
        if self.scheduler.tree_cache is self._source_cache:
            raise RuntimeError(
                "cannot finalize source HiCache bindings before publication"
            )
        self._released_device_aliases_generation = None

    def prepare_transition(self, transition):
        import copy
        from types import SimpleNamespace

        cache = self.scheduler.tree_cache
        self._phase_metrics = dict(self._background_metrics)
        started = time.monotonic()
        drain_host_io_at_fence(cache, self.group)
        self._phase_metrics["host_io_drain_seconds"] = time.monotonic() - started
        started = time.monotonic()
        self._source_cache = cache
        canonical = transition.source.first_rank
        dormant = [
            r
            for r in range(transition.source.max_pp_size)
            if r not in transition.source.active_ranks
        ]
        canonical_wire = None
        prepare_error = None
        if self.rank == canonical and dormant:
            try:
                metadata = {}
                for entry in cache.host_pool_group.entries:
                    pool = entry.host_pool
                    metadata[entry.name] = {
                        name: copy.deepcopy(getattr(pool, name))
                        for name in (
                            "logical_size",
                            "free_slots",
                            "release_slots",
                            "num_release_slots",
                            "mem_state",
                            "slot_used",
                        )
                        if hasattr(pool, name)
                    }
                canonical_wire = {
                    "tree": snapshot_tree_metadata(cache),
                    "allocators": metadata,
                    "requests": self.scheduler._elastic_pp_request_cache_state(),
                }
            except Exception as exc:
                prepare_error = exc
                canonical_wire = {"error": f"{type(exc).__name__}: {exc}"}
        for destination in dormant:
            if self.rank == canonical:
                ElasticPPLayerTransport.wait(
                    self.group.send_tensor_dict(
                        canonical_wire, dst=destination, async_send=True
                    )
                    or ()
                )
            elif self.rank == destination:
                wire = self.group.recv_tensor_dict(src=canonical)
                if "error" in wire:
                    prepare_error = RuntimeError(
                        "canonical HiCache snapshot failed: " + wire["error"]
                    )
                    continue
                local = self.scheduler._elastic_pp_request_cache_state()
                records = wire["requests"]
                if set(local) != set(records) or any(
                    local[rid]["progress"] != records[rid]["progress"] for rid in local
                ):
                    prepare_error = RuntimeError(
                        "elastic PP dormant request progress differs from canonical rank"
                    )
                    continue
                self._request_state = records
                shadow = clone_cache_runtime_metadata(cache)
                shadow.tree_core = restore_tree_metadata(wire["tree"], shadow)
                shadow.session_refs.tree_core = shadow.tree_core
                shadow._elastic_host_allocator_metadata = {
                    name: SimpleNamespace(**data)
                    for name, data in wire["allocators"].items()
                }
                cache = shadow
        if prepare_error is not None:
            raise prepare_error
        self._phase_metrics["host_canonical_metadata_seconds"] = (
            time.monotonic() - started
        )
        self.adapter = HiCacheTransitionAdapter(
            cache=cache,
            pools=(
                cache_host_views(cache)
                if self.rank in transition.source.active_ranks
                else {}
            ),
            group=self.group,
            rank=self.rank,
            fast_copy=self.fast_copy,
        )
        self.adapter.precopy = self.precopy
        self.adapter.prepare_transition(transition)
        started = time.monotonic()
        self.adapter.snapshot_at_fence(transition)
        self._phase_metrics["host_snapshot_gather_seconds"] = time.monotonic() - started

    def exchange_at_fence(self, transition):
        self.adapter.exchange_at_fence(transition)

    def apply_staged(self, transition):
        self.target_cache = prepare_target_host_cache(
            self.adapter.cache,
            device_pool=self.model_runner.token_to_kv_pool,
            target=transition.target.layer_range(self.rank),
            generation=transition.target.generation,
            fast_copy=self.fast_copy,
            metrics=self._phase_metrics,
            prepared=self.prepared_host,
            preloaded=self.precopy.preloaded if self.precopy is not None else None,
            verified_preloaded=(
                self.precopy.verified
                if self.precopy is not None and self.batch_io
                else None
            ),
        )
        self.prepared_host = None
        started = time.monotonic()
        self.adapter.apply_staged(
            transition,
            target_cache=self.target_cache,
            target_pools=cache_host_views(
                self.target_cache, self.target_cache._elastic_pp_projected_pool
            ),
            private_target=self.fast_copy,
        )
        self._phase_metrics["host_incoming_copy_seconds"] = time.monotonic() - started

    def commit(self, transition):
        import dataclasses

        clone = self.target_cache
        real_allocator = self.adapter.cache.token_to_kv_pool_allocator
        clone.token_to_kv_pool_allocator = real_allocator
        clone.req_to_token_pool = self.adapter.cache.req_to_token_pool
        clone.cache_controller.mem_pool_device_allocator = real_allocator
        rebind_cache_device_pools(clone, self.model_runner.token_to_kv_pool)
        clone._hicache_init_params = dataclasses.replace(
            clone._hicache_init_params,
            token_to_kv_pool_allocator=real_allocator,
            req_to_token_pool=clone.req_to_token_pool,
        )
        clone.tree_core.components_by_type = clone.components
        clone.tree_core.components = tuple(clone.components.values())
        for component in clone.components.values():
            component.tree_core = clone.tree_core
        self.retired_caches.append(self._source_cache)
        if self._request_state is not None:
            self.scheduler._elastic_pp_rebind_request_cache_state(self._request_state)
        self._request_state = None
        publish_scheduler_cache(self.scheduler, clone, self.model_runner)
        pool = self.model_runner.token_to_kv_pool
        getattr(pool, "full_kv_pool", pool).register_layer_transfer_counter(
            clone.cache_controller.layer_done_counter
        )
        if hasattr(clone.req_to_token_pool, "register_layer_transfer_counter"):
            clone.req_to_token_pool.register_layer_transfer_counter(
                clone.cache_controller.layer_done_counter
            )
        self._last_metrics = dict(self.adapter.metrics, **self._phase_metrics)
        self.adapter.commit(transition)
        self.target_cache = None
        self.adapter = None
        self._reset_precopy()

    def abort(self, transition):
        if self.adapter is not None:
            self.adapter.abort(transition)
        if self.target_cache is not None:
            discard_host_cache(self.target_cache)
        self.target_cache = None
        self.adapter = None
        self._reset_precopy()

    def metrics(self):
        return dict(
            self.adapter.metrics if self.adapter is not None else self._last_metrics,
            **self._phase_metrics,
            fast_copy=self.fast_copy,
        )

    def release_retired(self):
        """Only after topology publication; failures must not suggest rollback."""
        retired, self.retired_caches = self.retired_caches, []
        for cache in retired:
            discard_host_cache(cache)


def snapshot_tree_metadata(cache):
    """CPU wire snapshot of the arena, LRUs, locks and per-component sizes."""
    import copy
    import pickle

    assert_host_quiescent(cache)
    tree = cache.tree_core
    if getattr(tree, "enable_session_radix_cache", False):
        raise NotImplementedError(
            "elastic dormant session-radix frontier migration unsupported"
        )
    if getattr(tree, "_ongoing_insert_walk_state", None) is not None:
        raise RuntimeError("elastic tree snapshot requires completed insert walk")
    memo = {
        id(tree.components_by_type): {},
        id(tree.components): (),
        id(tree.kv_events): None,
    }
    for component in tree.components:
        memo[id(component)] = None
    for lru in (*tree.lru_lists.values(), *tree.host_lru_lists.values()):
        predicate = lru._is_referenced
        if predicate is not None:
            memo[id(predicate)] = None
    for node in tree._node_arena.values():
        for data in node.component_data:
            for value in vars(data).values():
                if isinstance(value, torch.Tensor):
                    memo[id(value)] = value.detach().cpu().clone()
    empty = tree._empty_match_result.device_indices
    memo[id(empty)] = empty.detach().cpu().clone()
    cloned = copy.deepcopy(tree, memo)
    cloned.device = "cpu"
    return pickle.dumps(cloned, protocol=5)


def restore_tree_metadata(payload, cache):
    import pickle

    from sglang.srt.mem_cache.unified_cache.components import tree_component
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode

    tree = pickle.loads(payload)
    tree.device = cache.tree_core.device
    tree.kv_events = cache.tree_core.kv_events
    tree._empty_match_result = tree._empty_match_result._replace(
        device_indices=tree._empty_match_result.device_indices.to(tree.device)
    )
    tree.components_by_type = cache.components
    tree.components = tuple(cache.components.values())
    for node in tree._node_arena.values():
        for data in node.component_data:
            if isinstance(data.value, torch.Tensor):
                data.value = data.value.to(tree.device)
    for component in tree.components:
        component.tree_core = tree
    # Counters are process-local. Imported IDs must never collide with a later
    # split/new node (include LRU sentinels, which are outside the arena).
    ids = [node.id for node in tree._node_arena.values()]
    for lru in (*tree.lru_lists.values(), *tree.host_lru_lists.values()):
        ids.extend(
            n.id for n in (lru.head, lru.tail, lru.mid, lru.cursor) if n is not None
        )
    newest = max(
        (
            max(node.last_access_time, node.creation_time)
            for node in tree._node_arena.values()
        ),
        default=0,
    )
    tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT = max(
        tree_component._LAST_ACCESS_TIME_COUNTER_FLOAT, newest + 1
    )
    UnifiedTreeNode.counter = max(UnifiedTreeNode.counter, max(ids, default=-1) + 1)
    return tree


def build_dormant_host_stack(clone, *, source, projected, params):
    """Keep cache allocator/consensus semantics with no layer payloads."""
    import copy
    import threading

    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _COMPONENT_HOST_ATTR,
        _logical_host_pool_size,
    )
    from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, LogicalHostPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    entries = []
    for old in source.host_pool_group.entries:
        entry = copy.copy(old)
        entry.layer_mapper = lambda _: None
        if entry.name == PoolName.KV:
            entry.device_pool = getattr(projected, "full_kv_pool", projected)
        elif entry.name == PoolName.MAMBA:
            entry.device_pool = projected.mamba_pool
        else:
            raise NotImplementedError("dormant HiCache supports FULL and MAMBA only")
        # Physical target constructors enter one PP MIN per component when
        # --hicache-size is fixed. The dormant target must join the same
        # collectives with a neutral value and adopt the new topology's
        # capacity, rather than retaining the source topology's capacity.
        entry.host_pool = LogicalHostPool(
            _logical_host_pool_size(entry.device_pool.size, old.host_pool.page_size),
            old.host_pool.page_size,
            old.host_pool.layout,
        )
        entries.append(entry)
    clone.host_pool_group = HostPoolGroup(entries)
    clone.load_cache_event = threading.Event()
    old = source.cache_controller
    clone.cache_controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        clone.host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=clone.load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=old.write_policy,
        io_backend=old.io_backend,
        storage_backend=None,
        transfer_layer_num=0,
        host_memory_mode=source.host_memory_mode,
    )
    for ct, name in (
        (ComponentType.FULL, PoolName.KV),
        (ComponentType.MAMBA, PoolName.MAMBA),
    ):
        if ct in clone.components:
            pool = clone.host_pool_group.get_pool(name)
            cache_attr, component_attr = _COMPONENT_HOST_ATTR[ct]
            setattr(clone, cache_attr, pool)
            setattr(clone.components[ct], component_attr, pool)
    clone.token_to_kv_pool_host = clone.host_pool_group
    clone._hicache_init_params = params


def drain_host_io_at_fence(cache, group, timeout_s=15.0):
    """All resident ranks poll in the same order, even locally idle ranks."""
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        controller = cache.cache_controller
        if controller.load_queue:
            controller.start_loading()
        if controller.write_queue:
            controller.start_writing()
        cache.check_hicache_events()
        cache._drain_async_work()
        pending = any(
            getattr(cache, name, None)
            for name in (
                "ongoing_backup",
                "ongoing_load_back",
                "ongoing_prefetch",
                "ongoing_write_through",
                "work_list",
            )
        )
        pending = pending or any(
            getattr(controller, name, None)
            for name in (
                "load_queue",
                "write_queue",
                "ack_load_queue",
                "ack_write_queue",
            )
        )
        statuses = group.all_gather_object(
            {"pending": bool(pending), "expired": time.monotonic() >= deadline}
        )
        if not any(s["pending"] for s in statuses):
            return
        if any(s["expired"] for s in statuses):
            raise RuntimeError("timed out retiring HiCache I/O at elastic PP fence")


def publish_scheduler_cache(scheduler, clone, model_runner):
    """Rebind retained holders before old host backing can be retired."""
    old = scheduler.tree_cache
    _rebind_scheduler_cache_contexts(scheduler, old, clone)
    holders = [scheduler]
    for name in (
        "policy",
        "schedule_policy",
        "session_controller",
        "disagg_decode_prealloc_queue",
        "disagg_decode_transfer_queue",
        "running_batch",
        "last_batch",
        "running_mbs",
        "last_mbs",
        "mbs",
        "cur_batch",
        "cur_batch_for_debug",
        "batch",
    ):
        value = getattr(scheduler, name, None)
        if isinstance(value, (list, tuple)):
            holders.extend(value)
        elif value is not None:
            holders.append(value)
    for holder in holders:
        if holder is None:
            continue
        for name in ("tree_cache", "radix_cache"):
            if getattr(holder, name, None) is old:
                setattr(holder, name, clone)
    canary = getattr(model_runner, "canary_manager", None)
    if canary is not None:
        canary.attach_radix_cache(clone)


def _rebind_scheduler_cache_contexts(scheduler, old, clone):
    """Rebuild frozen contexts and repair their shared mutable dependencies.

    These are the production Scheduler collaborators retaining a cache or a
    pool observer. Reinitializing mutable ones can reset counters or recreate
    sockets/publishers, so they keep their identity. Frozen ones are replaced
    once and all context references use the same replacement.
    """
    import dataclasses

    names = (
        "dp_attn_adapter",
        "pool_stats_observer",
        "invariant_checker",
        "kv_events_publisher",
        "load_inquirer",
        "output_streamer",
        "batch_result_processor",
    )
    roots = {
        name: value
        for name in names
        if (value := getattr(scheduler, name, None)) is not None
    }
    contexts = {id(value): value for value in roots.values()}
    replacements = {}
    mutable_updates = []

    def rebind(context):
        identity = id(context)
        if identity in replacements:
            return replacements[identity]
        # Mutable contexts may refer to each other; retain their identity.
        replacements[identity] = context
        fields = (
            (field.name for field in dataclasses.fields(context))
            if dataclasses.is_dataclass(context)
            else ("tree_cache", "radix_cache", "pool_stats_observer", "output_streamer")
        )
        changes = {}
        for name in fields:
            value = getattr(context, name, None)
            replacement = (
                clone
                if value is old
                else rebind(value) if id(value) in contexts else value
            )
            if replacement is not value:
                changes[name] = replacement
        if changes:
            params = getattr(type(context), "__dataclass_params__", None)
            if params is not None and params.frozen:
                context = dataclasses.replace(context, **changes)
            else:
                mutable_updates.append((context, changes))
        replacements[identity] = context
        return context

    target_roots = {name: rebind(value) for name, value in roots.items()}
    for context, changes in mutable_updates:
        for name, value in changes.items():
            setattr(context, name, value)
    for name, value in target_roots.items():
        setattr(scheduler, name, value)


def discard_host_cache(cache):
    """Retire host aliases explicitly, including controllers in frozen GC heaps."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _COMPONENT_HOST_ATTR,
    )

    cache.release_host_resources()
    controller = cache.cache_controller
    if controller is not None:
        controller.mem_pool_host = None
        controller.mem_pool_host_draft = None
        controller.mem_pool_device = None
        controller.mem_pool_device_draft = None
        controller.mem_pool_device_allocator = None
        controller.mtp_draft_device_pools = ()
    for ct, (cache_attr, component_attr) in _COMPONENT_HOST_ATTR.items():
        setattr(cache, cache_attr, None)
        if ct in cache.components:
            setattr(cache.components[ct], component_attr, None)
    cache.token_to_kv_pool_host = None
    cache.host_pool_group = None
    cache.cache_controller = None
    cache.__dict__.pop("_elastic_pp_projected_pool", None)
    cache.token_to_kv_pool_allocator = None
    cache.req_to_token_pool = None
    cache._hicache_init_params = None
