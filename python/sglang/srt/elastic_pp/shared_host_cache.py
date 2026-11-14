"""Node-local CPU backing whose physical layer addresses outlive PP layouts.

The first integration is deliberately L2-only, page-first, kernel I/O and TP1.
Only startup creates backing. A transition borrows views; it must not copy the
historical payload or free the bank when a source cache generation retires.
"""

from __future__ import annotations

import hashlib
import logging
import math
import mmap
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

_active_bank = ContextVar("elastic_pp_shared_host_bank", default=None)
logger = logging.getLogger(__name__)


def _boot_identity():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip(), os.getuid()


def _dtype(name):
    value = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"invalid shared tensor dtype: {name}")
    return value


class SharedHostRegion:
    """An independently held memfd mapping, optionally CUDA-registered locally.

    An importer opens its own fd before the creator may exit. Never explicitly
    munmap while tensors can still reference the storage: frombuffer's storage
    owner keeps the mmap alive after this object's own references are dropped.
    """

    def __init__(self, tensor, fd, mapping, *, registered=False):
        self.tensor, self.fd, self.mapping = tensor, fd, mapping
        self.registered = registered
        self.closed = False

    @classmethod
    def create(cls, shape, dtype, *, pin_memory=False):
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        if not shape or any(int(n) <= 0 for n in shape):
            raise ValueError("shared regions require a nonempty positive shape")
        tensor, fd, mapping = alloc_shm(tuple(shape), dtype)
        result = cls(tensor, fd, mapping)
        try:
            if pin_memory:
                result.register()
        except BaseException:
            result.close()
            raise
        return result

    def descriptor(self):
        if self.closed:
            raise RuntimeError("shared region is closed")
        stat = os.fstat(self.fd)
        return dict(
            pid=os.getpid(),
            fd=self.fd,
            boot=_boot_identity(),
            inode=stat.st_ino,
            device=stat.st_dev,
            bytes=stat.st_size,
            shape=tuple(self.tensor.shape),
            dtype=str(self.tensor.dtype),
        )

    @classmethod
    def attach(cls, descriptor, *, pin_memory=False):
        if tuple(descriptor["boot"]) != _boot_identity():
            raise ValueError("shared CPU backing requires the same node and uid")
        shape = tuple(int(x) for x in descriptor["shape"])
        dtype = _dtype(descriptor["dtype"])
        if not shape or any(n <= 0 for n in shape):
            raise ValueError("invalid shared tensor shape")
        used = math.prod(shape) * dtype.itemsize
        fd = os.open(
            f"/proc/{int(descriptor['pid'])}/fd/{int(descriptor['fd'])}",
            os.O_RDWR | os.O_CLOEXEC,
        )
        try:
            stat = os.fstat(fd)
            if (
                (stat.st_dev, stat.st_ino)
                != (descriptor["device"], descriptor["inode"])
                or stat.st_uid != os.getuid()
                or stat.st_size != descriptor["bytes"]
                or used > stat.st_size
            ):
                raise ValueError("shared region identity or bounds changed")
            mapping = mmap.mmap(fd, stat.st_size, flags=mmap.MAP_SHARED)
            tensor = torch.frombuffer(mapping, dtype=dtype, count=math.prod(shape))
            result = cls(tensor.reshape(shape), fd, mapping)
        except BaseException:
            os.close(fd)
            raise
        try:
            if pin_memory:
                result.register()
        except BaseException:
            result.close()
            raise
        return result

    def register(self):
        if self.closed:
            raise RuntimeError("cannot register closed shared region")
        if not self.registered:
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

            _cuda_host_register(self.tensor)
            self.registered = True

    def close(self):
        if self.closed:
            return
        if self.registered:
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister

            _cuda_host_unregister(self.tensor)
            self.registered = False
        os.close(self.fd)
        self.fd = None
        self.tensor = None
        self.mapping = None
        self.closed = True


class SharedHostBank:
    """One physical set of full-model CPU layers, independently mapped per rank."""

    def __init__(self, schema, regions):
        self.schema, self.regions = schema, regions
        self.leases = 0
        self.closed = False

    @classmethod
    def create(cls, schema, *, pin_memory=False):
        regions = {}
        try:
            for key, spec in schema["regions"].items():
                regions[key] = SharedHostRegion.create(
                    spec["shape"], _dtype(spec["dtype"]), pin_memory=pin_memory
                )
        except BaseException:
            for region in regions.values():
                region.close()
            raise
        return cls(schema, regions)

    def descriptor(self):
        return dict(
            schema=self.schema,
            regions={key: value.descriptor() for key, value in self.regions.items()},
        )

    def evidence(self):
        """Stable physical identities, independent of each process's address."""
        return dict(
            logical_bytes=self.bytes,
            capacities=self.schema["capacities"],
            layers=self.schema["layers"],
            regions={
                key: {
                    k: v
                    for k, v in region.descriptor().items()
                    if k not in {"pid", "fd"}
                }
                for key, region in self.regions.items()
            },
            live_views=self.leases,
        )

    @classmethod
    def attach(cls, descriptor, *, pin_memory=False):
        schema, regions = descriptor["schema"], {}
        if set(schema["regions"]) != set(descriptor["regions"]):
            raise ValueError("incomplete shared bank descriptor")
        try:
            for key, spec in schema["regions"].items():
                desc = descriptor["regions"][key]
                if tuple(desc["shape"]) != tuple(spec["shape"]) or (
                    desc["dtype"] != spec["dtype"]
                ):
                    raise ValueError("shared bank shape or dtype differs")
                regions[key] = SharedHostRegion.attach(desc, pin_memory=pin_memory)
        except BaseException:
            for region in regions.values():
                region.close()
            raise
        return cls(schema, regions)

    @property
    def bytes(self):
        return sum(
            r.tensor.numel() * r.tensor.element_size() for r in self.regions.values()
        )

    def allocator(self, kind, device_pool):
        if self.closed:
            raise RuntimeError("shared host bank is closed")
        global_ids = self.schema["layers"][kind]
        if kind == "full":
            ids = global_ids[device_pool.start_layer : device_pool.end_layer]
            if len(ids) != device_pool.layer_num:
                raise ValueError("full-attention ordinal layer map differs")
        else:
            ids = tuple(device_pool.mamba_layer_ids)
        positions = [global_ids.index(layer) for layer in ids]
        if not positions or positions != list(
            range(positions[0], positions[0] + len(positions))
        ):
            raise ValueError(
                "shared host views require a nonempty contiguous layer range"
            )
        return SharedPoolAllocator(self, kind, positions[0], len(positions))

    def close(self):
        if self.leases:
            raise RuntimeError("shared host backing still has live pool views")
        if not self.closed:
            for region in self.regions.values():
                region.close()
            self.closed = True


class SharedPoolAllocator:
    manages_registration = True
    shared_host_backing = True

    def __init__(self, bank, kind, start, count):
        self.bank, self.kind, self.start, self.count = bank, kind, start, count
        self.size = bank.schema["capacities"][kind]
        self.remaining = [key for key in bank.regions if key.startswith(kind + ".")]
        self.released = False
        bank.leases += 1

    def allocate(self, dims, dtype, device):
        if self.released or self.bank.closed or device != "cpu":
            raise RuntimeError("invalid shared host allocation lifetime or device")
        axis = 2 if self.kind == "full" else 1
        for key in self.remaining:
            tensor = self.bank.regions[key].tensor.narrow(axis, self.start, self.count)
            if tuple(tensor.shape) == tuple(dims) and tensor.dtype == dtype:
                self.remaining.remove(key)
                return tensor
        raise ValueError(
            "shared host allocation shape/dtype has no registered layer view"
        )

    def release(self):
        if not self.released:
            self.bank.leases -= 1
            self.released = True

    def __del__(self):
        self.release()


@contextmanager
def use_shared_host_bank(bank):
    token = _active_bank.set(bank)
    try:
        yield
    finally:
        _active_bank.reset(token)


def shared_pool_allocator(kind, device_pool):
    bank = _active_bank.get()
    return None if bank is None else bank.allocator(kind, device_pool)


def shared_pool_capacity(kind):
    bank = _active_bank.get()
    return None if bank is None else bank.schema["capacities"][kind]


def shared_host_params(params):
    """L2-only view construction must not create new PP storage groups.

    Dormant ranks do not rebuild physical views at startup. Entering new_group
    from only active ranks deadlocks. These groups serve L3, which this mode
    explicitly excludes; normal L2 event consensus belongs to the scheduler.
    """
    import dataclasses

    return dataclasses.replace(params, pp_cache_group=None)


def validate_shared_host_args(args):
    if not (
        args.enable_elastic_pp
        and args.enable_hierarchical_cache
        and args.elastic_pp_graph_partitions
        and not args.disable_cuda_graph
        and args.nnodes == 1
        and args.tp_size == 1
        and args.ep_size == 1
        and args.pp_stage_tp_sizes is None
        and (args.elastic_pp_kv_backend == "native"
             or getattr(args, "elastic_pp_kv_capacities", None))
        and args.hicache_mem_layout == "page_first"
        and args.hicache_io_backend == "kernel"
        and args.hicache_storage_backend is None
        and args.hicache_host_memory_mode == "cache"
        and args.elastic_pp_transition_backing == "overlap"
        and args.elastic_pp_prepared_weight_policy == "offload"
    ):
        raise ValueError(
            "shared host HiCache requires single-node TP1/EP1 native elastic PP, "
            "page_first/kernel L2 cache mode, overlap backing, prepared Graphs "
            "and offloaded candidate weights"
        )


def cache_schema(cache):
    from sglang.srt.elastic_pp.hicache_transition import cache_host_views, host_nodes
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    if host_nodes(cache):
        raise RuntimeError("shared CPU bootstrap requires an empty historical cache")
    result = {}
    for component, layers in cache_host_views(cache).items():
        kind = "full" if component == int(ComponentType.FULL) else "mamba"
        result[kind] = {
            "layers": tuple(layers),
            "fields": {
                name: dict(shape=tuple(t.shape), dtype=str(t.dtype))
                for name, t in next(iter(layers.values())).items()
            },
        }
    return result


def merge_schemas(reports, total_layers):
    """Validate one fixed component schema and complete, unique layer coverage."""
    layers, capacities, fields = {}, {}, {}
    all_layers = set()
    for report in reports:
        for kind, spec in report.items():
            for layer in spec["layers"]:
                if layer in all_layers:
                    raise ValueError("duplicate global layer in shared host schema")
                all_layers.add(layer)
            layers.setdefault(kind, []).extend(spec["layers"])
            normalized = {
                name: dict(shape=tuple(x["shape"])[1:], dtype=x["dtype"])
                for name, x in spec["fields"].items()
            }
            capacity = {x["shape"][0] for x in spec["fields"].values()}
            if len(capacity) != 1:
                raise ValueError("component host capacities differ")
            capacity = next(iter(capacity))
            if kind in fields and fields[kind] != normalized:
                raise ValueError("host component shape/dtype differs across ranks")
            fields[kind] = normalized
            capacities[kind] = min(capacities.get(kind, capacity), capacity)
    if all_layers != set(range(total_layers)):
        raise ValueError("shared host schema does not cover every model layer")
    regions = {}
    for kind, ids in layers.items():
        ids.sort()
        if kind == "full":
            if (
                set(fields[kind]) != {"k", "v"}
                or fields[kind]["k"] != fields[kind]["v"]
            ):
                raise ValueError("shared host requires ordinary matching MHA K/V")
            spec = fields[kind]["k"]
            regions["full.kv"] = dict(
                shape=(2, capacities[kind], len(ids), *spec["shape"]),
                dtype=spec["dtype"],
            )
        else:
            for name, spec in fields[kind].items():
                shape = (capacities[kind], len(ids), *spec["shape"])
                if math.prod(shape):
                    regions[f"mamba.{name}"] = dict(shape=shape, dtype=spec["dtype"])
    return dict(layers=layers, capacities=capacities, regions=regions)


def metadata_signature(cache):
    """Inspect logical slot metadata only, never read historical KV/GDN bytes."""
    from sglang.srt.elastic_pp.hicache_transition import host_nodes

    nodes = sorted(
        (repr(identity), component, tuple(node.slots.tolist()))
        for (identity, component), node in host_nodes(cache).items()
    )
    allocators = []
    for entry in cache.host_pool_group.entries:
        pool = entry.host_pool
        allocators.append(
            (
                str(entry.name),
                pool.size,
                tuple(torch.as_tensor(pool.free_slots).tolist()),
                tuple(tuple(torch.as_tensor(x).tolist()) for x in pool.release_slots),
                pool.num_release_slots,
                (
                    tuple(torch.as_tensor(pool.slot_used).tolist())
                    if hasattr(pool, "slot_used")
                    else None
                ),
                (
                    tuple(torch.as_tensor(pool.mem_state).tolist())
                    if hasattr(pool, "mem_state")
                    else None
                ),
            )
        )
    return hashlib.sha256(repr((nodes, allocators)).encode()).hexdigest()


def detach_inherited_host_stack(clone):
    """A failed target constructor must never destroy the live source pools."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _COMPONENT_HOST_ATTR,
    )

    clone.host_pool_group = None
    clone.cache_controller = None
    clone.token_to_kv_pool_host = None
    for component, (cache_attr, component_attr) in _COMPONENT_HOST_ATTR.items():
        setattr(clone, cache_attr, None)
        if component in clone.components:
            setattr(clone.components[component], component_attr, None)


def initialize_scheduler_shared_host(scheduler):
    """Install empty shared CPU backing after Graph warmup, before requests."""
    mr = scheduler.tp_worker.model_runner
    if not getattr(mr.server_args, "elastic_pp_hicache_shared_host", False):
        return
    if hasattr(mr, "_elastic_pp_shared_host_bank"):
        raise RuntimeError("shared host startup was already performed")
    from sglang.srt.elastic_pp.hicache_transition import (
        assert_host_quiescent,
        clone_cache_runtime_metadata,
        discard_host_cache,
        publish_scheduler_cache,
    )

    source = scheduler.tree_cache
    topology = scheduler._pp_topology()
    group = mr.elastic_pp_migration_group
    rank = scheduler.ps.pp_rank
    active = rank in topology.active_ranks

    def consensus(stage, action):
        try:
            report = dict(ok=True, value=action(), error=None)
        except Exception as exc:
            report = dict(ok=False, value=None, error=f"{type(exc).__name__}: {exc}")
        peers = group.all_gather_object(report)
        if len(peers) != group.world_size or any(not r["ok"] for r in peers):
            raise RuntimeError(f"shared CPU startup {stage} failed: {peers}")
        logger.info("ELASTIC_PP_SHARED_HOST rank=%d startup_stage=%s", rank, stage)
        return [r["value"] for r in peers]

    def inspect_source():
        if not scheduler.is_initializing:
            raise RuntimeError("shared CPU bootstrap is startup-only")
        assert_host_quiescent(source, allow_pp_control_sends=True)
        return dict(
            boot=_boot_identity(), schema=cache_schema(source) if active else {}
        )

    sources = consensus("source", inspect_source)
    if any(tuple(x["boot"]) != _boot_identity() for x in sources):
        raise ValueError("shared CPU startup spans more than one physical node")
    schema = merge_schemas(
        [x["schema"] for x in sources], sum(x.size for x in topology.layer_ranges)
    )
    bank = None
    clone = None
    pin = torch.device(mr.device).type == "cuda"
    try:

        def create():
            nonlocal bank
            if rank != topology.first_rank:
                return None
            import psutil

            from sglang.srt.mem_cache.pool_host.base import (
                HICACHE_HOST_MEMORY_RESERVE_BYTES,
            )

            size = sum(
                math.prod(s["shape"]) * _dtype(s["dtype"]).itemsize
                for s in schema["regions"].values()
            )
            if (
                size
                > psutil.virtual_memory().available - HICACHE_HOST_MEMORY_RESERVE_BYTES
            ):
                raise MemoryError(
                    "insufficient host RAM for shared CPU startup backing"
                )
            bank = SharedHostBank.create(schema, pin_memory=pin)
            return bank.descriptor()

        descriptors = consensus("allocate", create)
        descriptor = descriptors[topology.first_rank]

        def attach():
            nonlocal bank
            if bank is None:
                bank = SharedHostBank.attach(descriptor, pin_memory=pin)
            return True

        consensus("attach", attach)

        def build():
            nonlocal clone
            if active:
                clone = clone_cache_runtime_metadata(source)
                clone.sidecar_pool_specs = []
                detach_inherited_host_stack(clone)
                with use_shared_host_bank(bank):
                    clone.init_hicache(
                        source._hicache_init_server_args,
                        shared_host_params(source._hicache_init_params),
                    )
                clone._elastic_pp_shared_host_bank = bank
            return True

        consensus("views", build)
        if clone is not None:
            publish_scheduler_cache(scheduler, clone, mr)
            pool = mr.token_to_kv_pool
            getattr(pool, "full_kv_pool", pool).register_layer_transfer_counter(
                clone.cache_controller.layer_done_counter
            )
            if hasattr(clone.req_to_token_pool, "register_layer_transfer_counter"):
                clone.req_to_token_pool.register_layer_transfer_counter(
                    clone.cache_controller.layer_done_counter
                )
            discard_host_cache(source)
        else:
            source._elastic_pp_shared_host_bank = bank
        mr._elastic_pp_shared_host_bank = bank
        mr._elastic_pp_shared_host_startup = dict(
            logical_bytes=bank.bytes,
            historical_payload_copied_bytes=0,
            layers=schema["layers"],
            capacities=schema["capacities"],
        )
    except BaseException:
        if clone is not None and scheduler.tree_cache is not clone:
            discard_host_cache(clone)
        if bank is not None and not bank.leases:
            bank.close()
        raise
