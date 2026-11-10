"""Bounded continuous copies of immutable autoregressive KV rows.

Experimental session/copy primitives. live_kv_patch.py integrates native CPU
shadows or the opt-in NIXL copier with scheduler/allocator hooks. GDN and
request metadata still require the final migration fence.
Callers must publish writes after recording their GPU completion event, and
check free/reuse through this session *before* freeing.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass


@dataclass(frozen=True)
class KVRowStamp:
    slot: int
    incarnation: int
    ready: object


class KVPatchBusy(RuntimeError):
    """A copied row must not be freed/reused until its snapshot is complete."""


class ContinuousKVPatch:
    def __init__(self, *, generation, capacity, copier, max_rows=256):
        if generation <= 0 or capacity <= 0 or max_rows <= 0:
            raise ValueError("positive generation, capacity and copy bound required")
        self.generation = generation
        self.capacity = capacity
        self.copier = copier
        self.max_rows = max_rows
        self._epochs = {}
        self._live = {}
        self._acked = {}
        self._pending = None
        self._pinned = set()
        self._phase = "copying"
        self._lock = threading.RLock()
        self.rounds = self.rows_copied = 0

    @property
    def phase(self):
        with self._lock:
            return self._phase

    def _check(self, generation, *, mutable=False):
        if generation != self.generation:
            raise ValueError("KV patch generation mismatch")
        if self._phase in {"aborted", "complete", "failed"}:
            raise RuntimeError("KV patch session is " + self._phase)
        if mutable and self._phase != "copying":
            raise RuntimeError("KV patch is fenced against source mutations")

    def allocate(self, slot, *, generation):
        with self._lock:
            self._check(generation, mutable=True)
            if not 0 <= slot < self.capacity or slot in self._live:
                raise ValueError("KV patch slot is outside capacity or already live")
            epoch = self._epochs.get(slot, 0) + 1
            self._epochs[slot] = epoch
            self._live[slot] = KVRowStamp(slot, epoch, None)
            return epoch

    def publish(self, slot, incarnation, *, ready, generation):
        with self._lock:
            self._check(generation, mutable=True)
            row = self._live.get(slot)
            if row is None or row.incarnation != incarnation:
                raise ValueError("stale KV row incarnation")
            if ready is None or row.ready is not None:
                raise ValueError(
                    "publish one completed immutable KV write per incarnation"
                )
            self._live[slot] = KVRowStamp(slot, incarnation, ready)

    def free(self, slot, incarnation, *, generation):
        with self._lock:
            self._check(generation, mutable=True)
            row = self._live.get(slot)
            if row is None or row.incarnation != incarnation:
                raise ValueError("stale KV row incarnation")
            if slot in self._pinned:
                raise KVPatchBusy("KV row copy is in flight; defer free/reuse")
            del self._live[slot]
            self._acked.pop(slot, None)

    def dirty_rows(self):
        with self._lock:
            return tuple(
                row
                for slot, row in sorted(self._live.items())
                if row.ready is not None and self._acked.get(slot) != row.incarnation
            )

    def submit(self, *, generation):
        with self._lock:
            self._check(generation)
            if self._pending is not None:
                raise KVPatchBusy("only one bounded KV copy round may be in flight")
            rows = self.dirty_rows()[: self.max_rows]
            if not rows:
                return False
            self._pinned.update(row.slot for row in rows)
            try:
                completion = self.copier.submit(rows)
            except BaseException:
                # Keep logical pins until abort drains partial GPU submissions.
                self._phase = "failed"
                raise
            self._pending = (rows, completion)
            self.rounds += 1
            return True

    def poll(self, *, generation, wait=False):
        with self._lock:
            self._check(generation)
            if self._pending is None:
                return True
            rows, completion = self._pending
            try:
                if wait:
                    completion.synchronize()
                elif not completion.query():
                    return False
            except BaseException:
                self._phase = "failed"
                raise
            for row in rows:
                current = self._live.get(row.slot)
                if current is not None and current.incarnation == row.incarnation:
                    self._acked[row.slot] = row.incarnation
            self.rows_copied += len(rows)
            self._pinned.clear()
            self._pending = None
            return True

    def finish_at_fence(self, *, generation, max_rounds=1024):
        """Caller has stopped source writers; drain before publishing topology."""
        with self._lock:
            self._check(generation)
            if max_rounds <= 0:
                raise ValueError("final copy round bound must be positive")
            self._phase = "fenced"
            if any(row.ready is None for row in self._live.values()):
                raise RuntimeError("unpublished KV writes at the migration fence")
            self.poll(generation=generation, wait=True)
            for _ in range(max_rounds):
                if not self.dirty_rows():
                    self._phase = "complete"
                    return dict(rounds=self.rounds, rows_copied=self.rows_copied)
                self.submit(generation=generation)
                self.poll(generation=generation, wait=True)
            if not self.dirty_rows():
                self._phase = "complete"
                return dict(rounds=self.rounds, rows_copied=self.rows_copied)
            raise RuntimeError("KV patch did not converge within final round bound")

    def abort(self, *, generation):
        with self._lock:
            if generation != self.generation:
                raise ValueError("KV patch generation mismatch")
            if self._phase == "complete":
                raise RuntimeError("cannot abort after KV patch completion")
            self.copier.synchronize()
            if self._pending is not None:
                completion = self._pending[1]
                getattr(completion, "drain", completion.synchronize)()
            self._pending = None
            self._pinned.clear()
            self._phase = "aborted"


class _CopyCompletion:
    def __init__(self, event, held, resources):
        self.event, self.held, self.resources = event, held, resources

    def _release(self):
        if self.held is not None:
            self.held.__exit__(None, None, None)
            self.held = None
            self.resources = None

    def query(self):
        if self.event is not None and not self.event.query():
            return False
        self._release()
        return True

    def synchronize(self):
        if self.event is not None:
            self.event.synchronize()
        self._release()


class TensorKVPatchCopier:
    """Stride-aware single-device copy on a dedicated stream.

    Multi-rank/P2P transport and production allocator hooks are separate work.
    The lease pins physical source and target pages until copy completion.
    """

    def __init__(self, source, target, *, page_lease=lambda rows: nullcontext()):
        import torch

        if not source or source.keys() != target.keys():
            raise ValueError("source/target KV layer sets must match")
        device = next(iter(source.values()))[0].device
        for layer in source:
            if len(source[layer]) != 2 or len(target[layer]) != 2:
                raise ValueError("each KV layer must contain one K/V pair")
            for src, dst in zip(source[layer], target[layer], strict=True):
                if src.shape != dst.shape or src.dtype != dst.dtype:
                    raise ValueError("KV row copy shape/dtype mismatch")
                if src.device != device or dst.device != device:
                    raise ValueError("KV patch copier requires one device")
        self.source, self.target, self.device = source, target, device
        self.page_lease = page_lease
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._failed_holds = []

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()
        for held, _resources in self._failed_holds:
            held.__exit__(None, None, None)
        self._failed_holds.clear()

    def submit(self, rows):
        import torch

        held = self.page_lease(rows)
        held.__enter__()
        context = (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )
        indices, snapshots = None, {}
        try:
            with context:
                for ready in {id(row.ready): row.ready for row in rows}.values():
                    if self.stream is not None:
                        self.stream.wait_event(ready)
                    else:
                        ready.synchronize()
                indices = torch.tensor([row.slot for row in rows], device=self.device)
                snapshots = {
                    layer: tuple(src.index_select(0, indices) for src in pair)
                    for layer, pair in self.source.items()
                }
                for layer, pair in snapshots.items():
                    for dst, snapshot in zip(self.target[layer], pair, strict=True):
                        dst.index_copy_(0, indices, snapshot)
                event = None
                if self.stream is not None:
                    event = torch.cuda.Event()
                    event.record(self.stream)
            return _CopyCompletion(event, held, (indices, snapshots, rows))
        except BaseException:
            if self.stream is not None:
                try:
                    self.stream.synchronize()
                except BaseException:
                    self._failed_holds.append((held, (indices, snapshots, rows)))
                    raise
            held.__exit__(None, None, None)
            raise
