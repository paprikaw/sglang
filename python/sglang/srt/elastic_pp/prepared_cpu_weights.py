"""Bounded, startup-only native load-layout weights in CPU RAM.

No GPU tensor values are read. The ordinary Qwen loader receives fresh CPU
Parameters with the target's metadata and native weight-loader functions. This
per-worker cache is separate from the immutable canonical checkpoint bank and
must have its own admitted host-memory reserve. GPU post-load hooks still run
after the H2D copy; their optional runtime permutation is not cached here.
"""

from __future__ import annotations

import time
from copy import copy
from dataclasses import dataclass
from types import MethodType

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class CpuOnlyPacking(TorchDispatchMode):
    """Fail closed if a native loader attempts GPU work during CPU preparation."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(
            isinstance(item, torch.Tensor) and item.device.type != "cpu"
            for item in tree_leaves((args, kwargs))
        ) or (
            kwargs.get("device") is not None
            and torch.device(kwargs["device"]).type != "cpu"
        ):
            raise RuntimeError(
                f"CPU weight packing attempted a non-CPU operation: {func}"
            )
        return func(*args, **kwargs)


class NativeCopyAccounting(TorchDispatchMode):
    """Host wall time in native H2D copy calls, not isolated DMA duration.

    Pageable copy calls can block on driver staging and CUDA completion. The
    remaining native-load wall time includes Python dispatch/CPU packing and
    instrumentation. Do not present either counter as a CUDA bandwidth result.
    """

    def __init__(self):
        super().__init__()
        self.seconds = 0.0
        self.bytes = 0
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        copy = (
            func is torch.ops.aten.copy_.default
            and args[0].device.type == "cuda"
            and args[1].device.type == "cpu"
        )
        started = time.monotonic() if copy else None
        try:
            return func(*args, **(kwargs or {}))
        finally:
            if copy:
                self.seconds += time.monotonic() - started
                self.calls += 1
                self.bytes += args[0].numel() * args[0].element_size()


def layout_signature(parameters):
    return tuple(
        (name, tuple(p.shape), tuple(p.stride()), str(p.dtype), type(p).__qualname__)
        for name, p in sorted(parameters.items())
    )


class _CpuLoaderView:
    """Read-only native-loader metadata, with an independent CPU parameter map.

    Do not swap Parameters on the live model, even briefly: inference may be
    running on another host thread. Bound per-parameter loader functions receive
    only the new CPU Parameter. This adapter is qualified for the explicitly
    admitted unquantized TP1/EP1 Qwen Triton path, not arbitrary model loaders.
    """

    def __init__(self, model, parameters, layer_ids):
        self._original = model
        self._parameters = parameters
        self._elastic_pp_weight_profiler = None
        self._elastic_pp_loading_layer_ids = frozenset(layer_ids)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def named_parameters(self, *, remove_duplicate=False):
        return iter(self._parameters.items())

    def _refresh_elastic_pp_moe_weights(self):
        # CPU packing must not invalidate the live model's expert cache.
        pass


@dataclass(frozen=True)
class PreparedCpuEntry:
    signature: tuple
    parameters: dict
    buffers: dict
    loaded: frozenset
    checkpoint_bytes: int
    checkpoint_tensors: int
    prepare_seconds: float
    allocation_seconds: float
    pack_seconds: float
    nbytes: int


class PreparedCpuWeights:
    def __init__(self, *, cache_key, max_bytes, pinned):
        if max_bytes <= 0:
            raise ValueError("prepared CPU weights require an explicit positive cap")
        self.cache_key = cache_key
        self.max_bytes = max_bytes
        self.pinned = pinned
        self.entries = {}
        self.nbytes = 0
        self.prepare_seconds = 0.0
        self.allocation_seconds = 0.0
        self.pack_seconds = 0.0

    def get(self, layer_ids, parameters, buffers):
        entry = self.entries.get(frozenset(layer_ids))
        if entry is not None and entry.signature != layout_signature(
            dict(parameters, **buffers)
        ):
            raise RuntimeError("prepared CPU weight layout changed")
        return entry

    def prepare(
        self,
        *,
        model,
        layer_ids,
        parameters,
        buffers,
        weights,
        counters,
        expected_counts,
    ):
        key = frozenset(layer_ids)
        if key in self.entries:
            raise RuntimeError("prepared CPU weight entry already exists")
        unique = {id(p): p for p in parameters.values()}
        # GemmaRMSNorm's bound native loader writes both the supplied Parameter
        # and self.gemma_weight. Rebind that owner to an independent CPU module;
        # otherwise even a CPU Parameter would update the live GPU buffer.
        owners = {}
        buffer_names = {id(b): name for name, b in buffers.items()}
        derived = {}
        for parameter in unique.values():
            for attr in ("weight_loader", "_weight_loader"):
                loader = getattr(parameter, attr, None)
                if not isinstance(loader, MethodType) or not isinstance(
                    loader.__self__, torch.nn.Module
                ):
                    continue
                owner = loader.__self__
                owners[id(owner)] = owner
                for name, buf in owner.named_buffers(recurse=False):
                    if id(buf) not in buffer_names:
                        raise ValueError("CPU loader owns an unaccounted buffer")
                    if type(owner).__name__ != "GemmaRMSNorm" or name != "gemma_weight":
                        raise ValueError("CPU loader has an unqualified derived buffer")
                    derived[buffer_names[id(buf)]] = buf
        unique_buffers = {id(b): b for b in derived.values()}
        needed = sum(
            p.numel() * p.element_size()
            for p in [*unique.values(), *unique_buffers.values()]
        )
        if self.nbytes + needed > self.max_bytes:
            raise RuntimeError("prepared CPU weight memory cap exceeded")
        if any(not p.is_contiguous() for p in unique.values()):
            raise ValueError("prepared CPU weights require contiguous native storage")
        started = time.monotonic()
        copied = {}
        # Allocate directly into pinned CPU RAM: no intermediate full-size bank
        # and no GPU export. Preserve alias identity and native Parameter APIs.
        for ident, parameter in unique.items():
            data = torch.empty_strided(
                parameter.shape,
                parameter.stride(),
                dtype=parameter.dtype,
                device="cpu",
                pin_memory=self.pinned,
            )
            cpu = torch.Tensor._make_subclass(type(parameter), data, False)
            cpu.__dict__.update(parameter.__dict__)
            copied[ident] = cpu
        cpu_parameters = {name: copied[id(p)] for name, p in parameters.items()}
        cpu_buffer_ids = {
            ident: torch.empty_strided(
                b.shape, b.stride(), dtype=b.dtype, device="cpu", pin_memory=self.pinned
            )
            for ident, b in unique_buffers.items()
        }
        cpu_buffers = {name: cpu_buffer_ids[id(buf)] for name, buf in derived.items()}
        cpu_owners = {}
        for ident, owner in owners.items():
            shadow = copy(owner)
            shadow._parameters = {
                name: copied.get(id(p), p) for name, p in owner._parameters.items()
            }
            shadow._buffers = {
                name: cpu_buffer_ids.get(id(b), b) for name, b in owner._buffers.items()
            }
            cpu_owners[ident] = shadow
        for parameter in copied.values():
            for attr in ("weight_loader", "_weight_loader"):
                # BasevLLMParameter exposes weight_loader as a read-only
                # property backed by _weight_loader. Rebind stored attributes.
                loader = parameter.__dict__.get(attr)
                if isinstance(loader, MethodType) and id(loader.__self__) in cpu_owners:
                    setattr(
                        parameter,
                        attr,
                        MethodType(loader.__func__, cpu_owners[id(loader.__self__)]),
                    )
        view = _CpuLoaderView(model, cpu_parameters, layer_ids)
        allocated = time.monotonic()
        with torch.no_grad(), CpuOnlyPacking():
            loaded = frozenset(type(model).load_weights(view, weights))
        packed = time.monotonic()
        loaded_ids = {id(p) for name, p in cpu_parameters.items() if name in loaded}
        if loaded_ids != {id(p) for p in cpu_parameters.values()}:
            raise RuntimeError("CPU packing did not cover every target Parameter")
        if (
            counters["checkpoint_bytes"],
            counters["checkpoint_tensors"],
        ) != expected_counts:
            raise RuntimeError("CPU packing did not consume the full checkpoint")
        if layout_signature(cpu_parameters) != layout_signature(parameters):
            raise RuntimeError("CPU packing changed target Parameter layout")
        elapsed = time.monotonic() - started
        entry = PreparedCpuEntry(
            signature=layout_signature(dict(parameters, **buffers)),
            parameters=cpu_parameters,
            buffers=cpu_buffers,
            loaded=loaded,
            checkpoint_bytes=counters["checkpoint_bytes"],
            checkpoint_tensors=counters["checkpoint_tensors"],
            prepare_seconds=elapsed,
            allocation_seconds=allocated - started,
            pack_seconds=packed - allocated,
            nbytes=needed,
        )
        self.entries[key] = entry
        self.nbytes += needed
        self.prepare_seconds += elapsed
        self.allocation_seconds += entry.allocation_seconds
        self.pack_seconds += entry.pack_seconds
        return entry

    def copy(self, entry, parameters, stream, buffers):
        if entry.signature != layout_signature(dict(parameters, **buffers)):
            raise RuntimeError("prepared CPU weight layout changed before H2D")
        started = time.monotonic()
        begin = end = None
        if stream is not None:
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record(stream)
        seen = set()
        count = 0
        sources = dict(entry.parameters, **entry.buffers)
        targets = dict(parameters, **{name: buffers[name] for name in entry.buffers})
        for name, target in targets.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            source = sources[name]
            if source.device.type != "cpu" or (self.pinned and not source.is_pinned()):
                raise RuntimeError("prepared CPU source lost pinned host backing")
            target.copy_(source, non_blocking=True)
            count += 1
        if end is not None:
            end.record(stream)
            end.synchronize()
        elapsed = time.monotonic() - started
        return dict(
            h2d_wall_seconds=elapsed,
            h2d_stream_seconds=(begin.elapsed_time(end) / 1000 if end else None),
            h2d_copy_calls=count,
            h2d_bytes=entry.nbytes,
        )
