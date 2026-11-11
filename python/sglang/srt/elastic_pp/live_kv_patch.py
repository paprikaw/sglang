"""Opt-in scheduler-owned KV snapshots while the source PP layout serves.

The default copier keeps CPU shadows. The optional NIXL copier applies rows to
unpublished target GPU layers while serving. Recurrent state and HiCache still
use the drained fence; VMM and the model Graph bank remain separate work.
"""

from __future__ import annotations

import math
import threading

import torch

from sglang.srt.elastic_pp.kv_patch import ContinuousKVPatch, KVPatchBusy


class _HostCopyCompletion:
    def __init__(self, owner, indices, snapshots, event, resources):
        self.owner = owner
        self.indices = indices
        self.snapshots = snapshots
        self.event = event
        self.resources = resources
        self.committed = False

    def _commit(self):
        if not self.committed:
            for layer, pair in self.snapshots.items():
                for dst, src in zip(self.owner.target[layer], pair, strict=True):
                    dst.index_copy_(0, self.indices, src)
            self.committed = True
            self.resources = self.snapshots = None

    def query(self):
        if self.event is not None and not self.event.query():
            return False
        self._commit()
        return True

    def synchronize(self):
        if self.event is not None:
            self.event.synchronize()
        self._commit()


class HostKVPatchCopier:
    """Gather on a private stream, then commit completed copies to CPU shadow.

    Source backing remains resident throughout preparation. Native tensors are
    held until the fence; slot reuse is protected by the allocator listener.
    """

    def __init__(self, source, *, max_rows, admit_host_bytes):
        if not source:
            raise ValueError("KV shadow requires an outgoing attention layer")
        self.source = source
        first = next(iter(source.values()))[0]
        self.device = first.device
        if any(t.device != self.device for pair in source.values() for t in pair):
            raise ValueError("KV shadow source requires a single device")
        shadow_bytes = sum(t.nbytes for pair in source.values() for t in pair)
        chunk_bytes = sum(
            min(max_rows, t.shape[0]) * math.prod(t.shape[1:]) * t.element_size()
            for pair in source.values()
            for t in pair
        )
        # Include the bounded pinned staging copy, not only the pageable shadow.
        admit_host_bytes(shadow_bytes + chunk_bytes)
        self.shadow_bytes = shadow_bytes
        self.target = {
            layer: tuple(torch.empty_like(t, device="cpu") for t in pair)
            for layer, pair in source.items()
        }
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._failed_resources = []

    def submit(self, rows):
        from contextlib import nullcontext

        host_indices = torch.tensor([r.slot for r in rows], dtype=torch.int64)
        snapshots, gathered = {}, {}
        indices = None
        try:
            context = (
                torch.cuda.stream(self.stream)
                if self.stream is not None
                else nullcontext()
            )
            with context:
                for ready in {id(r.ready): r.ready for r in rows}.values():
                    if self.stream is None:
                        ready.synchronize()
                    else:
                        self.stream.wait_event(ready)
                indices = host_indices.to(self.device)
                for layer, pair in self.source.items():
                    gathered[layer] = tuple(t.index_select(0, indices) for t in pair)
                    snapshots[layer] = tuple(
                        torch.empty(
                            t.shape,
                            dtype=t.dtype,
                            device="cpu",
                            pin_memory=self.stream is not None,
                        )
                        for t in gathered[layer]
                    )
                    for dst, src in zip(snapshots[layer], gathered[layer], strict=True):
                        dst.copy_(src, non_blocking=self.stream is not None)
                event = None
                if self.stream is not None:
                    event = torch.cuda.Event()
                    event.record(self.stream)
            return _HostCopyCompletion(
                self, host_indices, snapshots, event, (indices, gathered, rows)
            )
        except BaseException:
            # Partial async submissions retain all resources until a real drain.
            self._failed_resources.append((indices, gathered, snapshots, rows))
            self.synchronize()
            raise

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()
        self._failed_resources.clear()


class LiveKVSnapshot:
    """One transition's allocator epochs and forward-ready CPU shadow.

    Start/progress/finish run on the scheduler host thread. Abort may run on the
    preparation thread, so callbacks and retirement share a lock. Nothing reads
    an allocator's live set on the preparation thread.
    """

    def __init__(
        self, *, generation, allocator, source, max_rows, admit_host_bytes, copier=None
    ):
        self.generation = generation
        self.allocator = allocator
        self.lock = threading.RLock()
        self.error = None
        self.active = True
        self.complete = False
        self.online_submissions = self.online_completed_rows = self.free_waits = 0
        self.forward_checkpoints = 0
        self.copier = (
            copier
            if copier is not None
            else HostKVPatchCopier(
                source, max_rows=max_rows, admit_host_bytes=admit_host_bytes
            )
        )
        self.session = ContinuousKVPatch(
            generation=generation,
            capacity=next(iter(source.values()))[0].shape[0],
            copier=self.copier,
            max_rows=min(max_rows, getattr(self.copier, "max_rows", max_rows)),
        )
        self.epochs = {}
        self.published = set()
        self._allocate(allocator.elastic_pp_live_slots())
        allocator.attach_elastic_pp_live_patch(self)

    def _allocate(self, slots):
        for slot in slots.detach().cpu().tolist():
            self.epochs[slot] = self.session.allocate(slot, generation=self.generation)

    def _fail(self, exc):
        self.error = f"{type(exc).__name__}: {exc}"
        # If draining fails, propagate: allowing a physical free would be unsafe.
        self.abort()

    def allocated(self, slots):
        with self.lock:
            if self.active:
                try:
                    self._allocate(slots)
                except BaseException as exc:
                    self._fail(exc)

    def freeing(self, slots):
        with self.lock:
            if not self.active:
                return
            try:
                for slot in slots.detach().cpu().tolist():
                    epoch = self.epochs[slot]
                    try:
                        self.session.free(slot, epoch, generation=self.generation)
                    except KVPatchBusy:
                        self.free_waits += 1
                        self.session.poll(generation=self.generation, wait=True)
                        self.session.free(slot, epoch, generation=self.generation)
                    if hasattr(self.copier, "forget"):
                        self.copier.forget(slot, epoch)
                    del self.epochs[slot]
                    self.published.discard(slot)
            except BaseException as exc:
                self._fail(exc)

    def _publish(self, slots, ready):
        for slot in slots.detach().cpu().tolist():
            if slot == 0:
                continue
            if slot not in self.epochs:
                raise RuntimeError("forward references a non-live KV slot")
            if slot not in self.published:
                self.session.publish(
                    slot, self.epochs[slot], ready=ready, generation=self.generation
                )
                self.published.add(slot)

    def prepare_writes(self, slots):
        """Before forward submission, invalidate rewrites of a published row.

        Page-local appends normally touch unpublished rows. Retraction/replay
        can revisit a published row without freeing its physical page; wait for
        its old copy before allowing the new write and advance its incarnation.
        """
        with self.lock:
            if not self.active:
                return
            try:
                for slot in slots.detach().cpu().unique().tolist():
                    if slot in self.published:
                        epoch = self.epochs[slot]
                        try:
                            self.session.free(slot, epoch, generation=self.generation)
                        except KVPatchBusy:
                            self.free_waits += 1
                            self.session.poll(generation=self.generation, wait=True)
                            self.session.free(slot, epoch, generation=self.generation)
                        if hasattr(self.copier, "forget"):
                            self.copier.forget(slot, epoch)
                        self.epochs[slot] = self.session.allocate(
                            slot, generation=self.generation
                        )
                        self.published.remove(slot)
            except BaseException as exc:
                self._fail(exc)

    def progress(self, slots, ready):
        with self.lock:
            if not self.active:
                return
            try:
                self.forward_checkpoints += 1
                self._publish(slots, ready)
                if self.session.poll(generation=self.generation):
                    self.online_completed_rows = self.session.rows_copied
                    if getattr(
                        self.copier, "ready_to_submit", True
                    ) and self.session.submit(generation=self.generation):
                        self.online_submissions += 1
            except BaseException as exc:
                self._fail(exc)

    def finish_at_fence(self, slots, ready):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("online KV snapshot failed: " + self.error)
            if not self.active:
                raise RuntimeError("online KV snapshot is not active")
            current = set(slots.detach().cpu().tolist())
            if current != set(self.epochs):
                raise RuntimeError("KV allocator and patch live sets diverged")
            self._publish(slots, ready)
            self.session.finish_at_fence(
                generation=self.generation,
                max_rounds=math.ceil(self.session.capacity / self.session.max_rows) + 1,
            )
            self.active = False
            self.allocator.detach_elastic_pp_live_patch(self)
            if hasattr(self.copier, "finish"):
                self.copier.finish(self.epochs)
            self.complete = True
            self.copier.source = {}  # Do not pin retired GPU storage after commit.

    def export(self, layer_id, slots):
        with self.lock:
            if not self.complete or not set(slots.cpu().tolist()) <= set(self.epochs):
                raise RuntimeError("KV shadow export requires a complete live snapshot")
            indices = slots.cpu()
            return tuple(
                t.index_select(0, indices) for t in self.copier.target[layer_id]
            )

    def abort(self):
        with self.lock:
            if self.active:
                self.session.abort(generation=self.generation)
                self.active = False
            self.allocator.detach_elastic_pp_live_patch(self)
            self.copier.source = {}

    def metrics(self):
        with self.lock:
            return dict(
                enabled=True,
                scope=getattr(
                    self.copier,
                    "scope",
                    "native KV CPU shadow; network/GDN/HiCache at fence",
                ),
                generation=self.generation,
                complete=self.complete,
                error=self.error,
                forward_checkpoints=self.forward_checkpoints,
                online_submissions=self.online_submissions,
                online_completed_rows=self.online_completed_rows,
                total_rounds=self.session.rounds,
                total_rows=self.session.rows_copied,
                free_waits=self.free_waits,
                shadow_bytes=self.copier.shadow_bytes,
            )
