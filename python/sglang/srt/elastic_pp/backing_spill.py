"""CPU authority for a fenced native KV/GDN backing rebuild.

The caller drains scheduling, HiCache I/O, migration streams and CUDA graphs.
Logical token/request allocators keep their identities throughout the rebuild.
Storage witnesses detect tensor aliases; CUDA counters separately measure what
the caching allocator released. Neither implies that a GPU was returned.
"""

from __future__ import annotations

import gc
import weakref

import torch

from sglang.srt.elastic_pp.kv_residency import ElasticKVResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def _metrics():
    return dict(
        snapshot_bytes=0,
        frontier_snapshot_bytes=0,
        detached_bytes=0,
        restored_bytes=0,
        source_storage_bytes_alive=0,
        cuda_memory={},
    )


def _record_memory(metrics, device, phase, *, release_unused=False):
    device = torch.device(device)
    if device.type != "cuda":
        metrics["cuda_memory"][phase] = None
        return
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        if release_unused:
            torch.cuda.empty_cache()
        point = dict(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
        )
    baseline = metrics["cuda_memory"].get("before_snapshot", point)
    for name in ("allocated", "reserved"):
        point[f"{name}_delta_bytes"] = (
            point[f"{name}_bytes"] - baseline[f"{name}_bytes"]
        )
    metrics["cuda_memory"][phase] = point


def _storage_witnesses(tensors):
    # A detach() alias keeps storage alive without retaining the original tensor.
    storages = {t.untyped_storage()._cdata: t.untyped_storage() for t in tensors}
    return tuple((weakref.ref(s), s.nbytes()) for s in storages.values() if s.nbytes())


def _check_released(witnesses, metrics):
    gc.collect()
    alive = sum(size for ref, size in witnesses if ref() is not None)
    metrics["source_storage_bytes_alive"] = alive
    if alive:
        raise RuntimeError(
            f"source backing still has tensor/storage aliases ({alive} bytes); "
            "scheduler must remain fenced until source restoration"
        )


def _cpu_copy(tensor):
    # non_blocking=False: CPU authority is complete on return.
    return tensor.detach().to(device="cpu", copy=True)


def discard_tensor_tracebacks(error):
    """Preserve error types/messages without pinning failed tensor allocations."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend((current.__cause__, current.__context__))
    return error


def _require_cpu_space(required_bytes):
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes

    available = max(0, host_memory_budget_bytes())
    if required_bytes > available:
        raise MemoryError(
            f"elastic backing snapshot needs {required_bytes} host bytes, "
            f"but this rank's current HiCache host budget is {available}"
        )
    return available


class NativeKVSpill:
    def __init__(self, pool, *, total_num_layers):
        if type(pool) is not MHATokenToKVPool:
            raise NotImplementedError("native MHA backing required")
        pool._require_elastic_pp_kv_supported()
        self.pool = pool
        self.total_num_layers = total_num_layers
        self.source = LayerRange(pool.start_layer, pool.end_layer)
        self.state = "new"
        self.backup = {}
        self.generation = self.target = None
        self.metrics = _metrics()
        self._source_storages = ()
        self._retained = frozenset()

    def snapshot(self):
        if self.state != "new":
            raise RuntimeError("snapshot already taken")
        pool = self.pool
        residency = pool._elastic_pp_residency
        if (
            pool._elastic_pp_pending_kv
            or pool._elastic_pp_retired_kv
            or (
                residency is not None
                and (
                    residency.pending is not None
                    or residency._cleanup_generation is not None
                )
            )
        ):
            raise RuntimeError("spill requires no pending or retired generation")
        _record_memory(self.metrics, pool.device, "before_snapshot")
        current = pool._current_elastic_pp_kv_map()
        self.metrics["cpu_snapshot_budget_bytes"] = _require_cpu_space(
            sum(t.nbytes for pair in current.values() for t in pair)
        )
        witnesses = _storage_witnesses(t for pair in current.values() for t in pair)
        # Include padding and cache-resident rows, not only active requests.
        # No source mutation before EVERY blocking CPU copy has completed.
        backup = {
            layer: tuple(_cpu_copy(t) for t in pair) for layer, pair in current.items()
        }
        self.source_generation = residency.generation if residency is not None else 0
        self.backup = backup
        self._source_storages = witnesses
        self.metrics["snapshot_bytes"] = sum(
            t.nbytes for p in backup.values() for t in p
        )
        self.state = "snapshotted"
        _record_memory(self.metrics, pool.device, "after_snapshot")

    def _require_source_generation(self):
        residency = self.pool._elastic_pp_residency
        if residency is not None and (
            residency.active_range != self.source
            or residency.generation != self.source_generation
        ):
            raise RuntimeError("source rollback forbidden after topology publication")

    def release_source_backing(self):
        if self.state != "snapshotted":
            raise RuntimeError("completed CPU snapshot required before release")
        self._require_source_generation()
        pool = self.pool
        if pool._elastic_pp_residency is None:
            pool._elastic_pp_residency = ElasticKVResidency(
                num_layers=self.total_num_layers,
                initial_generation=self.source_generation,
                initial_range=self.source,
            )
        pool.k_buffer = []
        pool.v_buffer = []
        pool._kv_buffer_descs = []
        pool._kv_copy_config = None
        pool._elastic_pp_detached_layer_ids = frozenset(
            range(self.source.start, self.source.end)
        )
        for name in ("k_data_ptrs", "v_data_ptrs", "data_ptrs", "data_strides"):
            setattr(pool, name, None)
        self.metrics["detached_bytes"] = self.metrics["snapshot_bytes"]
        # A failed alias check still requires restoring the source.
        self.state = "spilled"
        _check_released(self._source_storages, self.metrics)
        _record_memory(self.metrics, pool.device, "after_release", release_unused=True)

    def prepare_target(self, *, target, generation):
        if self.state == "target_prepared" and (target, generation) == (
            self.target,
            self.generation,
        ):
            return self._retained
        if self.state != "spilled":
            raise RuntimeError("release source backing before target allocation")
        self._require_source_generation()
        _check_released(self._source_storages, self.metrics)
        self.target, self.generation = target, generation
        pool = self.pool
        pool.prepare_elastic_pp_kv_layers(
            target=target, generation=generation, total_num_layers=self.total_num_layers
        )
        retained = set()
        for layer in range(target.start, target.end):
            if layer in self.backup:
                pair = pool.get_elastic_pp_pending_kv_buffer(
                    generation=generation, layer_id=layer
                )
                for dst, src in zip(pair, self.backup[layer]):
                    dst.copy_(src)
                    self.metrics["restored_bytes"] += src.nbytes
                retained.add(layer)
        pool.mark_elastic_pp_kv_layers_loaded(generation=generation, layer_ids=retained)
        self._retained = frozenset(retained)
        self.state = "target_prepared"
        _record_memory(self.metrics, pool.device, "after_target_prepare")
        return self._retained

    def _build_source(self):
        pool = self.pool
        restored = {}
        for layer, sources in self.backup.items():
            pair = pool._allocate_elastic_pp_kv_pair(layer_id=layer)
            for dst, src in zip(pair, sources):
                dst.copy_(src)
                self.metrics["restored_bytes"] += src.nbytes
            restored[layer] = pair
        keys = [restored[i][0] for i in range(self.source.start, self.source.end)]
        values = [restored[i][1] for i in range(self.source.start, self.source.end)]
        return (
            keys,
            values,
            pool._build_elastic_pp_pointer_tables(keys, values),
            pool._build_kv_buffer_descs_for(keys, values, layer_num=self.source.size),
        )

    def restore_source(self):
        if self.state == "restored":
            self._require_source_generation()
            return
        if self.state == "snapshotted":
            self._require_source_generation()
            self.state = "restored"
            return
        if self.state not in ("spilled", "target_prepared"):
            raise RuntimeError("source restore requires unpublished spilled state")
        self._require_source_generation()
        pool = self.pool
        if pool._elastic_pp_residency.pending is not None:
            pool.abort_elastic_pp_kv_layers(
                generation=pool._elastic_pp_residency.pending.generation
            )
        self.state = "spilled"
        try:
            keys, values, tables, descs = self._build_source()
        except BaseException as exc:
            # A retained OOM traceback must not pin partially rebuilt backing.
            raise discard_tensor_tracebacks(exc)
        pool.k_buffer, pool.v_buffer = keys, values
        pool.k_data_ptrs, pool.v_data_ptrs, pool.data_ptrs, pool.data_strides = tables
        pool._kv_buffer_descs = descs
        pool.__dict__.pop("_elastic_pp_detached_layer_ids", None)
        pool.mem_usage = sum(t.nbytes for t in (*keys, *values)) / 2**30
        self.state = "restored"
        _record_memory(self.metrics, pool.device, "after_source_restore")

    def finalize(self, *, generation):
        residency = self.pool._elastic_pp_residency
        if (
            self.state != "target_prepared"
            or generation != self.generation
            or residency is None
            or residency.generation != generation
            or residency.active_range != self.target
        ):
            raise RuntimeError("cannot discard CPU authority before target publication")
        self.backup.clear()
        self.state = "published"
        _record_memory(self.metrics, self.pool.device, "after_publication")


class HybridKVSpill:
    """Native attention plus plain GDN, with unchanged request-slot identities."""

    def __init__(self, pool):
        from sglang.srt.mem_cache.elastic_hybrid_pool import ElasticHybridLinearKVPool

        if type(pool) is not ElasticHybridLinearKVPool:
            raise NotImplementedError("plain elastic GDN pool required")
        self.pool = pool
        self.attention = NativeKVSpill(
            pool.full_kv_pool, total_num_layers=len(pool._all_full_ids)
        )
        self.state = "new"
        self.source_ids = tuple(pool.req_to_token_pool.mamba_map)
        self.backup = ()
        self.frontier = {}
        self.metrics = _metrics()
        self._source_storages = ()
        self.target = self.generation = None
        self._retained = frozenset()

    def _frontier(self):
        req = self.pool.req_to_token_pool
        result = {
            "free_slots": torch.as_tensor(req.free_slots),
            "mamba_free_slots": req.mamba_allocator.free_slots,
        }
        for name in (
            "req_to_token",
            "req_generation",
            "req_index_to_mamba_index_mapping",
            "req_index_to_mamba_ping_pong_track_buffer_mapping",
        ):
            value = getattr(req, name, None)
            if value is not None:
                result[name] = torch.as_tensor(value)
        return result

    def _check_frontier(self):
        if any(
            not torch.equal(value.detach().cpu(), self.frontier[name])
            for name, value in self._frontier().items()
        ):
            raise RuntimeError(
                "request allocator/frontier changed while backing was spilled"
            )

    def snapshot(self):
        pool = self.pool
        if (
            self.state != "new"
            or pool._pending_state is not None
            or pool._retired_state is not None
            or pool._residency.pending is not None
            or pool._residency._cleanup_generation is not None
            or getattr(pool.req_to_token_pool.mamba_allocator, "_alloc_iter", None)
            is not None
        ):
            raise RuntimeError(
                "hybrid spill requires idle generation and allocator groups"
            )
        _record_memory(self.metrics, pool.device, "before_snapshot")
        tensors = pool._state_tensors(pool.mamba_pool.mamba_cache)
        frontier_tensors = self._frontier()
        self.metrics["cpu_snapshot_budget_bytes"] = _require_cpu_space(
            sum(t.nbytes for t in tensors)
            + sum(t.nbytes for t in frontier_tensors.values())
            + sum(
                t.nbytes
                for pair in pool.full_kv_pool._current_elastic_pp_kv_map().values()
                for t in pair
            )
        )
        witnesses = _storage_witnesses(tensors)
        backup = tuple(_cpu_copy(t) for t in tensors)
        frontier = {name: _cpu_copy(t) for name, t in frontier_tensors.items()}
        self.attention.snapshot()
        self.source_generation = pool._residency.generation
        self.backup, self.frontier = backup, frontier
        self._source_storages = witnesses
        self.metrics["snapshot_bytes"] = self.attention.metrics["snapshot_bytes"] + sum(
            t.nbytes for t in backup
        )
        self.metrics["frontier_snapshot_bytes"] = sum(
            t.nbytes for t in frontier.values()
        )
        self.state = "snapshotted"
        _record_memory(self.metrics, pool.device, "after_snapshot")

    def release_source_backing(self):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        if self.state != "snapshotted":
            raise RuntimeError("completed hybrid snapshot required")
        self._check_frontier()
        pool = self.pool
        old = pool.mamba_pool.mamba_cache
        empty = MambaPool.State(
            conv=[t.new_empty((0, *t.shape[1:])) for t in old.conv],
            temporal=old.temporal.new_empty((0, *old.temporal.shape[1:])),
        )
        pool.mamba_pool.mamba_cache = empty
        del old
        for name in ("_conv_slot_desc", "_conv_fuse_ok"):
            pool.mamba_pool.__dict__.pop(name, None)
        pool._elastic_pp_cpu_spill = self
        self.state = "spilled"
        self.attention.release_source_backing()
        self.metrics["detached_bytes"] = self.metrics["snapshot_bytes"]
        _check_released(self._source_storages, self.metrics)
        _record_memory(self.metrics, pool.device, "after_release", release_unused=True)

    def restore_retained_recurrent(self, target_state, target_ids):
        for dst_index, layer in enumerate(target_ids):
            if layer in self.source_ids:
                src_index = self.source_ids.index(layer)
                for dst, src in zip(
                    self.pool._state_tensors(target_state), self.backup
                ):
                    dst[dst_index].copy_(src[src_index])
                    self.metrics["restored_bytes"] += src[src_index].nbytes

    def prepare_target(self, *, target, generation):
        if self.state == "target_prepared" and (target, generation) == (
            self.target,
            self.generation,
        ):
            return self._retained
        if self.state != "spilled":
            raise RuntimeError("hybrid backing must be released first")
        self._check_frontier()
        _check_released(self._source_storages, self.metrics)
        _check_released(self.attention._source_storages, self.attention.metrics)
        self.target, self.generation = target, generation
        pool = self.pool
        pool.prepare_elastic_pp_kv_layers(
            target=target,
            generation=generation,
            total_num_layers=pool._total_num_layers,
        )
        retained = set()
        for layer in pool.elastic_pp_kv_layer_ids(target):
            ordinal = pool._full_id_to_ordinal[layer]
            if ordinal in self.attention.backup:
                pair = pool.full_kv_pool.get_elastic_pp_pending_kv_buffer(
                    generation=generation, layer_id=ordinal
                )
                for dst, src in zip(pair, self.attention.backup[ordinal]):
                    dst.copy_(src)
                    self.metrics["restored_bytes"] += src.nbytes
                retained.add(layer)
        self.restore_retained_recurrent(pool._pending_state, pool._pending_mamba_ids)
        retained.update(set(pool._pending_mamba_ids) & set(self.source_ids))
        pool.mark_elastic_pp_kv_layers_loaded(generation=generation, layer_ids=retained)
        self._retained = frozenset(retained)
        self.state = "target_prepared"
        _record_memory(self.metrics, pool.device, "after_target_prepare")
        return self._retained

    def restore_source(self):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        pool = self.pool
        if self.state == "restored":
            if pool._residency.generation != self.source_generation:
                raise RuntimeError("hybrid source rollback forbidden after publication")
            return
        if (
            self.state not in ("snapshotted", "spilled", "target_prepared")
            or tuple(pool.req_to_token_pool.mamba_map) != self.source_ids
            or pool._residency.generation != self.source_generation
        ):
            raise RuntimeError("hybrid source rollback forbidden after publication")
        self._check_frontier()
        if self.state == "snapshotted":
            self.attention.restore_source()
            self.state = "restored"
            return
        if pool._residency.pending is not None:
            pool.abort_elastic_pp_kv_layers(
                generation=pool._residency.pending.generation
            )
        self.state = "spilled"
        tensors = None
        attention_bytes_before = self.attention.metrics["restored_bytes"]
        try:
            tensors = [t.to(device=pool.device, copy=True) for t in self.backup]
            self.attention.restore_source()
        except BaseException as exc:
            tensors = None
            raise discard_tensor_tracebacks(exc)
        pool.mamba_pool.mamba_cache = MambaPool.State(
            conv=tensors[:-1], temporal=tensors[-1]
        )
        self.metrics["restored_bytes"] += (
            self.attention.metrics["restored_bytes"]
            - attention_bytes_before
            + sum(t.nbytes for t in self.backup)
        )
        pool.__dict__.pop("_elastic_pp_cpu_spill", None)
        self.state = "restored"
        _record_memory(self.metrics, pool.device, "after_source_restore")

    def finalize(self, *, generation):
        pool = self.pool
        if (
            self.state != "target_prepared"
            or generation != self.generation
            or pool._residency.generation != generation
            or pool._residency.active_range != self.target
        ):
            raise RuntimeError("cannot discard CPU authority before hybrid publication")
        # Native backing was prepared through the hybrid pool's transaction.
        self.attention.backup.clear()
        self.attention.state = "published"
        self.backup = ()
        self.frontier.clear()
        self.state = "published"
        _record_memory(self.metrics, pool.device, "after_publication")
