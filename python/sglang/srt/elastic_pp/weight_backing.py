"""Stable weight addresses with physically evictable, layer-local CUDA pages.

Graph metadata keeps virtual addresses, never a shadow GPU weight copy. Restoring
backing allocates empty pages; the normal transactional transport must fill them
before publication. This does not evict KV, Graph workspaces or worker contexts.
"""

from __future__ import annotations

import time

import torch


def _tensor_attributes(module):
    """Include unregistered views, e.g. GDN's conv_weights, without module cycles."""
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from visit(item)

    for child in module.modules():
        for name, value in vars(child).items():
            if name != "_modules":
                yield from visit(value)


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def weight_storage_plan(layer, *, device_type="cuda"):
    """Pack parameter storages and find all their views before rebinding anything.

    Non-persistent shared position/rotary caches are not layer weights. Persistent
    buffers are included. Shared weight storage across decoder layers is rejected
    by the owner rather than evicting data still used by another layer.
    """
    roots = list(layer.parameters())
    for child in layer.modules():
        roots.extend(
            buffer
            for name, buffer in child._buffers.items()
            if buffer is not None and name not in child._non_persistent_buffers_set
        )
    groups = {}
    for tensor in roots:
        if tensor.device.type != device_type or not tensor.numel():
            continue
        if tensor.layout != torch.strided:
            raise ValueError("evictable weights require strided tensors")
        groups.setdefault(_storage_key(tensor), {"source": tensor, "views": []})
    offset = 0
    for key, group in groups.items():
        offset = (offset + 255) // 256 * 256
        group["offset"] = offset
        offset += key[2]
    for tensor in _tensor_attributes(layer):
        if tensor.device.type == device_type and tensor.numel():
            group = groups.get(_storage_key(tensor))
            if group is not None:
                group["views"].append(
                    (
                        tensor,
                        tensor.storage_offset(),
                        tuple(tensor.shape),
                        tensor.stride(),
                    )
                )
    return groups, offset


def copy_and_rebind_weight_storages(groups, storage):
    """Preserve Parameter identity and every registered/unregistered alias."""
    replacements = []
    with torch.no_grad():
        for key, group in groups.items():
            device = group["source"].device
            source = torch.empty(0, dtype=torch.uint8, device=device).set_(
                group["source"].untyped_storage(), 0, (key[2],)
            )
            target = torch.empty(0, dtype=torch.uint8, device=device).set_(
                storage, group["offset"], source.shape
            )
            target.copy_(source)
            for tensor, old_offset, shape, stride in group["views"]:
                view = torch.empty(0, dtype=tensor.dtype, device=device).set_(
                    storage,
                    group["offset"] // tensor.element_size() + old_offset,
                    shape,
                    stride,
                )
                replacements.append((tensor, view))
        if groups and device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        for tensor, view in replacements:
            tensor.data = view
    return {_storage_key(t) for t, _ in replacements}


class LayerWeightBacking:
    def __init__(self, layer, device_id, *, previous_storage_keys=()):
        from sglang.srt.cuda_vmm_utils import (
            VmmReservation,
            get_allocation_granularity,
            make_device_allocation_prop,
            tensor_from_pointer,
        )

        groups, used = weight_storage_plan(layer)
        if set(groups).intersection(previous_storage_keys):
            raise ValueError(
                "cross-layer shared weights cannot be evicted independently"
            )
        if not used:
            raise ValueError("evictable decoder layer has no CUDA weight storage")
        prop = make_device_allocation_prop(device_id, handle_types=None)
        granularity = get_allocation_granularity(prop)
        self.size = (used + granularity - 1) // granularity * granularity
        self.used_bytes = sum(key[2] for key in groups)
        self.device_id = device_id
        self.reservation = VmmReservation(self.size, prop, device_id)
        self.mapped = False
        self.allocations = self.evictions = 0
        try:
            self.restore()
            self.storage = tensor_from_pointer(
                self.reservation.base, self.size, device_id=device_id
            ).untyped_storage()
            # Only the new storage keys may be used for subsequent sharing checks.
            self.storage_keys = copy_and_rebind_weight_storages(groups, self.storage)
        except BaseException:
            self.reservation.close()
            raise

    def restore(self):
        if not self.mapped:
            self.reservation.map(0, self.size, retain_handle=False)
            self.mapped = True
            self.allocations += 1

    def evict(self):
        # Caller has drained readers. VA, Tensor objects and Graphs remain valid
        # metadata; touching tensor contents until restore + transfer is forbidden.
        if self.mapped:
            self.reservation.unmap(0, self.size)
            self.mapped = False
            self.evictions += 1


class EvictableWeightBank:
    def __init__(self):
        self.layers = {}
        self.events = []

    def prepare(self, layer_id, layer):
        started = time.monotonic()
        owner = self.layers.get(layer_id)
        if owner is None:
            device = next(layer.parameters()).device
            owner = LayerWeightBacking(
                layer,
                device.index,
                previous_storage_keys={
                    k for o in self.layers.values() for k in o.storage_keys
                },
            )
            self.layers[layer_id] = owner
        else:
            owner.restore()
        self.events.append(
            dict(
                action="map",
                layer=layer_id,
                bytes=owner.size,
                seconds=time.monotonic() - started,
            )
        )

    def evict(self, layer_ids):
        owners = [(i, self.layers[i]) for i in layer_ids if i in self.layers]
        if owners:
            torch.cuda.synchronize(owners[0][1].device_id)
        for layer_id, owner in owners:
            started = time.monotonic()
            before = torch.cuda.mem_get_info(owner.device_id)[0]
            owner.evict()
            self.events.append(
                dict(
                    action="unmap",
                    layer=layer_id,
                    bytes=owner.size,
                    seconds=time.monotonic() - started,
                    driver_free_bytes_before=before,
                    driver_free_bytes_after=torch.cuda.mem_get_info(owner.device_id)[0],
                )
            )

    def metrics(self, active_ids):
        mapped = {i: o.size for i, o in self.layers.items() if o.mapped}
        return dict(
            kind="vmm_nonresident_weights",
            mapped_physical_bytes=sum(mapped.values()),
            inactive_mapped_physical_bytes=sum(
                v for i, v in mapped.items() if i not in active_ids
            ),
            reserved_virtual_bytes=sum(o.size for o in self.layers.values()),
            mapped_layer_ids=sorted(mapped),
            unmapped_layer_ids=sorted(set(self.layers) - set(mapped)),
            allocations=sum(o.allocations for o in self.layers.values()),
            evictions=sum(o.evictions for o in self.layers.values()),
            events=list(self.events),
            scope="decoder weights only; KV, Graph workspaces and worker contexts remain",
        )
