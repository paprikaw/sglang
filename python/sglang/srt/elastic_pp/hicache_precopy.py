"""Independent CPU snapshots and speculative, unpublished HiCache pre-copy.

Only capture() reads the live tree, on the scheduler thread with retired I/O.
The preparation worker uses owned tensors, never live allocator/node objects.
Every reused row is compared with the authoritative fence contents; a split,
eviction, slot reuse or same-slot overwrite therefore becomes a normal delta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class HostRecord:
    slots: torch.Tensor
    tensors: dict[str, torch.Tensor]


def rows_match(views, record, slots):
    from sglang.srt.elastic_pp.hicache_transition import _contiguous_host_rows

    if set(views) != set(record.tensors) or not torch.equal(slots, record.slots):
        return False
    for name, view in views.items():
        start = _contiguous_host_rows(slots)
        current = (
            view.narrow(0, start, slots.numel())
            if start is not None
            else view.index_select(0, slots)
        )
        original = record.tensors[name]
        if current.dtype != original.dtype or not torch.equal(current, original):
            return False
    return True


def copy_records(pools, records):
    """Populate a private target using speculative slot IDs; never publish it."""
    from sglang.srt.elastic_pp.hicache_transition import _copy_host_rows

    selected = {}
    operations = []
    for key, record in records.items():
        _, component, layer = key
        views = pools.get(component, {}).get(layer)
        if views is None:
            continue
        if set(views) != set(record.tensors):
            raise ValueError("background HiCache schema mismatch")
        for name, source in record.tensors.items():
            target = views[name]
            slots = record.slots
            if (
                slots.device.type != "cpu"
                or slots.dtype != torch.int64
                or slots.ndim != 1
                or len(set(slots.tolist())) != slots.numel()
                or (
                    slots.numel()
                    and (int(slots.min()) < 0 or int(slots.max()) >= target.shape[0])
                )
                or source.device.type != "cpu"
                or target.device.type != "cpu"
                or source.dtype != target.dtype
                or source.shape != (slots.numel(), *target.shape[1:])
                or (
                    source.numel()
                    and source.untyped_storage().data_ptr()
                    == target.untyped_storage().data_ptr()
                )
            ):
                raise ValueError("invalid background HiCache rows or target alias")
            operations.append((target, slots, source))
        selected[key] = record
    for target, slots, source in operations:
        _copy_host_rows(target, slots, source, fast_copy=True)
    return selected


class HiCachePrecopy:
    def __init__(self, *, generation, records, metadata_cache, captured_seconds):
        self.generation = generation
        self.records = records
        self.metadata_cache = metadata_cache
        self.preloaded = {}
        self.metrics = {
            "host_background_capture_seconds": captured_seconds,
            "host_background_snapshot_bytes": self.bytes(records),
            "host_background_bytes_sent": 0,
            "host_background_bytes_received": 0,
        }

    @staticmethod
    def bytes(records):
        return sum(
            t.numel() * t.element_size()
            for r in records.values()
            for t in r.tensors.values()
        )

    @classmethod
    def capture(cls, cache, pools, generation, *, max_bytes=1 << 30):
        from sglang.srt.elastic_pp.hicache_transition import (
            _gather_host_rows,
            assert_host_quiescent,
            clone_cache_runtime_metadata,
            host_nodes,
        )

        started = time.monotonic()
        assert_host_quiescent(cache)
        nodes = host_nodes(cache)
        estimated = sum(
            node.slots.numel() * (t.numel() // max(1, t.shape[0])) * t.element_size()
            for (_, component), node in nodes.items()
            for tensors in pools.get(component, {}).values()
            for t in tensors.values()
        )
        if estimated > max_bytes:
            raise RuntimeError("background HiCache snapshot exceeds CPU byte cap")
        records = {
            (identity, component, layer): HostRecord(
                node.slots.clone(),
                {
                    name: _gather_host_rows(t, node.slots, fast_copy=True)
                    for name, t in tensors.items()
                },
            )
            for (identity, component), node in nodes.items()
            for layer, tensors in pools.get(component, {}).items()
        }
        return cls(
            generation=generation,
            records=records,
            metadata_cache=clone_cache_runtime_metadata(cache),
            captured_seconds=time.monotonic() - started,
        )

    def exchange(self, transition, *, group, rank, target_pools):
        from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport

        if transition.target.generation != self.generation:
            raise RuntimeError("background HiCache generation mismatch")
        received = {}
        errors = []
        for edge in sorted(
            transition.kv_transfers,
            key=lambda e: (
                e.src_rank,
                e.dst_rank,
                e.layer_range.start,
                e.layer_range.end,
            ),
        ):
            if rank == edge.src_rank:
                selected = {
                    k: r
                    for k, r in self.records.items()
                    if edge.layer_range.start <= k[2] < edge.layer_range.end
                }
                wire = {"generation": self.generation, "manifest": []}
                for i, (key, record) in enumerate(sorted(selected.items())):
                    wire["manifest"].append((*key, tuple(record.tensors)))
                    wire[f"{i}:slots"] = record.slots
                    wire.update({f"{i}:{n}": t for n, t in record.tensors.items()})
                ElasticPPLayerTransport.wait(
                    group.send_tensor_dict(wire, dst=edge.dst_rank, async_send=True)
                    or ()
                )
                self.metrics["host_background_bytes_sent"] += self.bytes(selected)
            elif rank == edge.dst_rank:
                wire = group.recv_tensor_dict(src=edge.src_rank)
                if wire.get("generation") != self.generation:
                    errors.append("background HiCache payload generation mismatch")
                    continue
                selected = {}
                for i, (identity, component, layer, names) in enumerate(
                    wire["manifest"]
                ):
                    key = (identity, component, layer)
                    if (
                        not edge.layer_range.start <= layer < edge.layer_range.end
                        or key in selected
                        or key in self.preloaded
                        or key in received
                    ):
                        errors.append(
                            "duplicate or out-of-range background HiCache record"
                        )
                        continue
                    selected[key] = HostRecord(
                        wire[f"{i}:slots"], {n: wire[f"{i}:{n}"] for n in names}
                    )
                received.update(selected)
                self.metrics["host_background_bytes_received"] += self.bytes(selected)
        if errors:
            raise RuntimeError("; ".join(errors))
        self.preloaded.update(copy_records(target_pools, received))
        if set(received) - self.preloaded.keys():
            raise RuntimeError("background HiCache target lacks incoming layer")

    def reset(self):
        self.records.clear()
        self.preloaded.clear()
        # This was a non-owning metadata snapshot, not another serving owner.
        # Drop its live aliases explicitly even when GC was frozen at startup.
        if self.metadata_cache is not None:
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                _COMPONENT_HOST_ATTR,
            )

            cache = self.metadata_cache
            cache.host_pool_group = None
            cache.cache_controller = None
            cache.token_to_kv_pool_host = None
            for ct, (cache_attr, component_attr) in _COMPONENT_HOST_ATTR.items():
                setattr(cache, cache_attr, None)
                if ct in cache.components:
                    setattr(cache.components[ct], component_attr, None)
            cache.token_to_kv_pool_allocator = None
            cache.req_to_token_pool = None
            cache._hicache_init_params = None
        self.metadata_cache = None
