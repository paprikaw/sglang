"""Finite, startup-only Graph bank for experimental elastic PP execution.

This is an opt-in building block, not a replacement for CudaGraphRunner.
Capture finishes before activation. Bind every tensor read by the graph,
including weights, KV/GDN, inputs and workspaces; stable KV alone is insufficient.
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphKey:
    layer_counts: tuple[int, ...]
    batch_size: int
    profile: str = "decode"

    def __post_init__(self):
        if (
            self.batch_size <= 0
            or not self.layer_counts
            or any(value < 0 for value in self.layer_counts)
            or not any(self.layer_counts)
        ):
            raise ValueError("invalid elastic Graph layout/batch")


def binding_signature(bindings):
    """Include row strides: a packed KV view is not a dense row buffer."""
    return tuple(
        (
            name,
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
        for name, tensor in sorted(bindings.items())
    )


class TorchGraphBackend:
    def __init__(self, device):
        import torch

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Graph capture requires CUDA")

    def capture(self, forward):
        import torch

        with torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward()
            stream.synchronize()
            return graph, output

    def completion(self):
        import torch

        with torch.cuda.device(self.device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            return event

    def synchronize(self):
        import torch

        torch.cuda.synchronize(self.device)


class ElasticGraphBank:
    """Pre-capture a bounded layout set, then select without runtime capture.

    The caller owns the startup/quiescence boundary and the complete binding
    inventory. A page lease remains held until the replay completion event,
    not merely until graph.replay() returns. Runtime replay failures propagate;
    rerunning eager after a partly executed graph could duplicate writes.
    """

    def __init__(self, *, backend, max_entries=16):
        if max_entries <= 0:
            raise ValueError("Graph bank capacity must be positive")
        self.backend = backend
        self.max_entries = max_entries
        self._entries = {}
        self._inflight = []
        self._failed_leases = []
        self._phase = "building"
        self._lock = threading.RLock()
        self.counters = Counter()

    def capture(self, key, *, bindings, forward, pages_ready=True, lease=nullcontext):
        with self._lock:
            if self._phase != "building":
                raise RuntimeError("Graph capture is allowed only before activation")
            if key in self._entries or len(self._entries) >= self.max_entries:
                raise ValueError("duplicate Graph key or Graph bank capacity exceeded")
            if not pages_ready:
                raise RuntimeError("Graph capture requires mapped backing pages")
            signature = binding_signature(bindings)
            held = lease()
            held.__enter__()
            try:
                graph, output = self.backend.capture(forward)
                if signature != binding_signature(bindings):
                    raise RuntimeError("tensor bindings changed during capture")
            except BaseException:
                try:
                    self.backend.synchronize()
                except BaseException:
                    self._failed_leases.append(held)
                    self._phase = "failed"
                    raise
                held.__exit__(None, None, None)
                raise
            held.__exit__(None, None, None)
            # Strong references prevent captured allocations being recycled.
            self._entries[key] = (signature, dict(bindings), graph, output)
            self.counters["captures"] += 1

    def activate(self):
        with self._lock:
            if self._phase != "building" or not self._entries:
                raise RuntimeError("Graph bank must contain captures before activation")
            self._phase = "active"

    def poll(self):
        with self._lock:
            for entry in tuple(self._inflight):
                event, held = entry
                if event.query():
                    held.__exit__(None, None, None)
                    self._inflight.remove(entry)

    def run(self, key, *, bindings, eager, pages_ready=True, lease=nullcontext):
        with self._lock:
            if self._phase != "active":
                raise RuntimeError("Graph bank is not active")
            self.poll()
            if not pages_ready:
                self.counters["blocked_unmapped"] += 1
                raise RuntimeError("Graph backing pages are not mapped")
            entry = self._entries.get(key)
            reason = (
                "uncaptured"
                if entry is None
                else "bindings" if entry[0] != binding_signature(bindings) else None
            )
            if reason:
                self.counters["fallback_" + reason] += 1
            held = lease()
            held.__enter__()
            try:
                if reason:
                    output = eager()
                else:
                    entry[2].replay()
                    output = entry[3]
                event = self.backend.completion()
            except BaseException:
                # Keep leases if synchronization itself fails. Process cleanup
                # must then own reclamation; unsafe page release is forbidden.
                try:
                    self.backend.synchronize()
                except BaseException:
                    self._failed_leases.append(held)
                    self._phase = "failed"
                    raise
                held.__exit__(None, None, None)
                raise
            self._inflight.append((event, held))
            if not reason:
                self.counters["replays"] += 1
            return output

    def close(self):
        with self._lock:
            if self._failed_leases:
                self.backend.synchronize()
                for held in self._failed_leases:
                    held.__exit__(None, None, None)
                self._failed_leases.clear()
            while self._inflight:
                event, held = self._inflight[0]
                event.synchronize()
                held.__exit__(None, None, None)
                self._inflight.pop(0)
            self._entries.clear()
            self._phase = "closed"
