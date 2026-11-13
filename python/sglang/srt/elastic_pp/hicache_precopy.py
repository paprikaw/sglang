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


def layer_stack(tensors):
    """View existing CPU layer slices together; never guess unrelated storage."""
    first = tensors[0]
    if first.device.type != "cpu":
        return None
    if len(tensors) == 1:
        return first.unsqueeze(0)
    step = tensors[1].storage_offset() - first.storage_offset()
    if step <= 0 or any(
        t.device != first.device
        or t.dtype != first.dtype
        or t.shape != first.shape
        or t.stride() != first.stride()
        or t.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
        or t.storage_offset() != first.storage_offset() + i * step
        for i, t in enumerate(tensors)
    ):
        return None
    return first.as_strided(
        (len(tensors), *first.shape), (step, *first.stride()), first.storage_offset()
    )


def selected_rows(stack, slots):
    from sglang.srt.elastic_pp.hicache_transition import _contiguous_host_rows

    start = _contiguous_host_rows(slots)
    return (
        stack.narrow(1, start, slots.numel())
        if start is not None
        else stack.index_select(1, slots)
    )


def record_groups(records):
    groups = {}
    for key in sorted(records):
        groups.setdefault(key[:2], []).append(key)
    for keys in groups.values():
        first = records[keys[0]]
        if all(
            set(records[k].tensors) == set(first.tensors)
            and torch.equal(records[k].slots, first.slots)
            and all(
                records[k].tensors[name].dtype == first.tensors[name].dtype
                and records[k].tensors[name].shape == first.tensors[name].shape
                for name in first.tensors
            )
            for k in keys
        ):
            yield keys
        else:
            yield from ([key] for key in keys)


def pack_records(records):
    wire = {"blocks": []}
    for i, keys in enumerate(record_groups(records)):
        first = records[keys[0]]
        row_first = []
        wire[f"{i}:slots"] = first.slots
        for name in first.tensors:
            tensors = [records[k].tensors[name] for k in keys]
            stack = layer_stack(tensors)
            if stack is not None and stack.stride(1) > stack.stride(0):
                row_first.append(name)
                wire[f"{i}:{name}"] = stack.transpose(0, 1).contiguous()
            else:
                wire[f"{i}:{name}"] = (
                    torch.stack(tensors) if stack is None else stack.contiguous()
                )
        wire["blocks"].append(
            (*keys[0][:2], [k[2] for k in keys], tuple(first.tensors), row_first)
        )
    return wire


def unpack_records(wire):
    records = {}
    for i, (identity, component, layers, names, row_first) in enumerate(wire["blocks"]):
        slots = wire[f"{i}:slots"]
        if (
            not layers
            or len(set(layers)) != len(layers)
            or len(set(names)) != len(names)
            or not set(row_first) <= set(names)
        ):
            raise ValueError("invalid packed HiCache layer/name list")
        tensors = {
            name: (
                wire[f"{i}:{name}"].transpose(0, 1)
                if name in row_first
                else wire[f"{i}:{name}"]
            )
            for name in names
        }
        if any(
            t.ndim < 2 or t.shape[:2] != (len(layers), slots.numel())
            for t in tensors.values()
        ):
            raise ValueError("packed HiCache payload dimensions differ from metadata")
        for j, layer in enumerate(layers):
            key = (identity, component, layer)
            if key in records:
                raise ValueError("duplicate packed HiCache record")
            records[key] = HostRecord(slots, {n: t[j] for n, t in tensors.items()})
    return records


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


def copy_records(pools, records, *, batch_io=False):
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
    if not batch_io:
        for target, slots, source in operations:
            _copy_host_rows(target, slots, source, fast_copy=True)
    else:
        from sglang.srt.elastic_pp.hicache_transition import _contiguous_host_rows

        for keys in record_groups(selected):
            slots = selected[keys[0]].slots
            for name in selected[keys[0]].tensors:
                sources = [selected[k].tensors[name] for k in keys]
                targets = [pools[k[1]][k[2]][name] for k in keys]
                source, target = layer_stack(sources), layer_stack(targets)
                if source is None or target is None:
                    for dst, src in zip(targets, sources, strict=True):
                        _copy_host_rows(dst, slots, src, fast_copy=True)
                else:
                    start = _contiguous_host_rows(slots)
                    if start is not None:
                        target.narrow(1, start, slots.numel()).copy_(source)
                    else:
                        target.index_copy_(1, slots, source)
    return selected


class HiCachePrecopy:
    def __init__(
        self, *, generation, records, metadata_cache, captured_seconds, batch_io=False
    ):
        self.generation = generation
        self.records = records
        self.metadata_cache = metadata_cache
        self.preloaded = {}
        self.batch_io = batch_io
        self.verified = {}
        self.metrics = {
            "host_background_capture_seconds": captured_seconds,
            "host_background_snapshot_bytes": self.bytes(records),
            "host_background_bytes_sent": 0,
            "host_background_bytes_received": 0,
            "host_batch_io": batch_io,
        }

    @staticmethod
    def bytes(records):
        return sum(
            t.numel() * t.element_size()
            for r in records.values()
            for t in r.tensors.values()
        )

    @classmethod
    def capture(cls, cache, pools, generation, *, max_bytes=1 << 30, batch_io=False):
        from sglang.srt.elastic_pp.hicache_transition import (
            _contiguous_host_rows,
            _gather_host_rows,
            assert_host_quiescent,
            clone_cache_runtime_metadata,
            host_nodes,
        )

        started = time.monotonic()
        assert_host_quiescent(cache, allow_pp_control_sends=True)
        nodes = host_nodes(cache)
        estimated = sum(
            node.slots.numel() * (t.numel() // max(1, t.shape[0])) * t.element_size()
            for (_, component), node in nodes.items()
            for tensors in pools.get(component, {}).values()
            for t in tensors.values()
        )
        if estimated > max_bytes:
            raise RuntimeError("background HiCache snapshot exceeds CPU byte cap")
        records, copy_calls = {}, 0
        for (identity, component), node in nodes.items():
            layers = pools.get(component, {})
            keys = sorted(layers)
            if not keys:
                continue
            slots = node.slots.clone()
            for layer in keys:
                records[identity, component, layer] = HostRecord(slots, {})
            for name in layers[keys[0]]:
                stack = (
                    layer_stack([layers[layer][name] for layer in keys])
                    if batch_io
                    else None
                )
                if stack is None:
                    for layer in keys:
                        records[identity, component, layer].tensors[name] = (
                            _gather_host_rows(
                                layers[layer][name], slots, fast_copy=True
                            )
                        )
                        copy_calls += 1
                else:
                    selected = selected_rows(stack, slots)
                    # Sparse index_select already owns its result. A contiguous
                    # selection is a live view and must be independently cloned.
                    owned = (
                        selected.clone()
                        if _contiguous_host_rows(slots) is not None
                        else selected
                    )
                    copy_calls += 1
                    for i, layer in enumerate(keys):
                        records[identity, component, layer].tensors[name] = owned[i]
        metadata = clone_cache_runtime_metadata(cache)
        # This private allocation template owns no serving control messages.
        # The live scheduler keeps and retires its own send handles normally.
        metadata.work_list = []
        result = cls(
            generation=generation,
            records=records,
            metadata_cache=metadata,
            captured_seconds=time.monotonic() - started,
            batch_io=batch_io,
        )
        result.metrics["host_snapshot_copy_calls"] = copy_calls
        return result

    def verify_at_fence(self, pools, nodes):
        """Compare exact contents once while source CPU writers are fenced."""
        started = time.monotonic()
        self.verified = {}
        comparisons = 0
        for keys in record_groups(self.records):
            node = nodes.get(keys[0][:2])
            if node is None or not torch.equal(node.slots, self.records[keys[0]].slots):
                continue
            if any(k[2] not in pools.get(k[1], {}) for k in keys):
                continue
            if any(
                set(pools[k[1]][k[2]]) != set(self.records[k].tensors) for k in keys
            ):
                continue
            matched = True
            for name in self.records[keys[0]].tensors:
                current = layer_stack([pools[k[1]][k[2]][name] for k in keys])
                original = layer_stack([self.records[k].tensors[name] for k in keys])
                if current is None or original is None:
                    matched = False
                    break
                comparisons += 1
                if current.dtype != original.dtype or not torch.equal(
                    selected_rows(current, node.slots), original
                ):
                    matched = False
                    break
            for key in keys:
                if matched or rows_match(
                    pools[key[1]][key[2]], self.records[key], node.slots
                ):
                    self.verified[key] = self.records[key]
        self.metrics["host_fence_verify_seconds"] = time.monotonic() - started
        self.metrics["host_fence_bulk_comparisons"] = comparisons

    def matches(self, key, views, slots):
        if self.batch_io:
            return key in self.verified
        original = self.records.get(key)
        return original is not None and rows_match(views, original, slots)

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
                if self.batch_io:
                    wire = pack_records(selected)
                    wire["generation"] = self.generation
                else:
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
                try:
                    decoded = (
                        unpack_records(wire).items()
                        if "blocks" in wire
                        else [
                            (
                                (identity, component, layer),
                                HostRecord(
                                    wire[f"{i}:slots"],
                                    {n: wire[f"{i}:{n}"] for n in names},
                                ),
                            )
                            for i, (identity, component, layer, names) in enumerate(
                                wire["manifest"]
                            )
                        ]
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    errors.append(str(exc))
                    continue
                for key, record in decoded:
                    layer = key[2]
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
                    selected[key] = record
                received.update(selected)
                self.metrics["host_background_bytes_received"] += self.bytes(selected)
        if errors:
            raise RuntimeError("; ".join(errors))
        self.preloaded.update(
            copy_records(target_pools, received, batch_io=self.batch_io)
        )
        if set(received) - self.preloaded.keys():
            raise RuntimeError("background HiCache target lacks incoming layer")

    def reset(self):
        self.records.clear()
        self.preloaded.clear()
        self.verified.clear()
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
