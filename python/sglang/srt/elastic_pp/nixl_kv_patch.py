"""Forward-ready KV rows sent to unpublished native target layers via NIXL."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from contextlib import nullcontext

import torch


def _signature(versions):
    return hashlib.sha256(repr(sorted(versions.items())).encode()).hexdigest()


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"invalid NIXL {name}")
    return value


class NixlKVReceiver:
    def __init__(self, endpoint, pool):
        self.endpoint, self.pool = endpoint, pool
        self.lock = threading.RLock()
        self.generation = None
        self.prepared = False
        self.expected = {}
        self.versions, self.retired = {}, {}
        self.sealed = set()
        self.fenced = False
        self.stats = {}
        self._unsafe_backing = []
        self.stream = (
            torch.cuda.Stream(device=endpoint.device)
            if endpoint.device.type == "cuda"
            else None
        )
        endpoint.handler = self.receive

    def configure(self, generation, expected):
        with self.lock:
            if self.generation is not None:
                raise RuntimeError("NIXL receiver still owns an earlier generation")
            self.generation, self.prepared = generation, False
            self.expected = {
                p: tuple(sorted(layers)) for p, layers in expected.items() if layers
            }
            self.versions = {p: {} for p in self.expected}
            self.retired = {p: {} for p in self.expected}
            self.sealed, self.fenced = set(), False
            self.stats = dict(
                generation=generation,
                online_applied_rows=0,
                applied_rows=0,
                applied_bytes=0,
                sealed_sources=0,
            )

    def ready(self, generation):
        with self.lock:
            if generation != self.generation:
                raise ValueError("NIXL receiver preparation generation mismatch")
            self.prepared = True
            sources = tuple(self.expected)
        self.endpoint.announce_ready(generation, sources)

    def _target(self, layer):
        full = getattr(self.pool, "full_kv_pool", self.pool)
        ordinal = (
            self.pool._full_id_to_ordinal[layer] if full is not self.pool else layer
        )
        return full.get_elastic_pp_pending_kv_buffer(
            generation=self.generation, layer_id=ordinal
        )

    def receive(self, peer, message, data):
        with self.lock:
            if (
                message.get("generation") != self.generation
                or not self.prepared
                or peer not in self.expected
            ):
                raise ValueError(
                    "NIXL patch has no matching prepared generation/source"
                )
            if peer in self.sealed:
                raise ValueError("NIXL source already sealed at the final fence")
            op = message.get("op")
            if op == "forget":
                rows = self._rows(message["rows"])
                for slot, incarnation in rows:
                    self.retired[peer][slot] = max(
                        self.retired[peer].get(slot, 0), incarnation
                    )
                    if self.versions[peer].get(slot, 0) <= incarnation:
                        self.versions[peer].pop(slot, None)
                return
            if op == "seal":
                versions = self.versions[peer]
                if message.get("count") != len(versions) or message.get(
                    "signature"
                ) != _signature(versions):
                    raise ValueError(
                        "NIXL target coverage differs from authoritative live rows"
                    )
                self.sealed.add(peer)
                self.stats["sealed_sources"] = len(self.sealed)
                return
            if op != "data":
                raise ValueError("unsupported NIXL KV operation")
            rows = self._rows(message["rows"])
            if not rows:
                raise ValueError("NIXL data must contain nonempty rows")
            segments = message.get("segments")
            if not isinstance(segments, list):
                raise ValueError("NIXL KV segments must be a list")
            targets, views, covered = {}, {}, set()
            end = 0
            for layer, kind, offset, shape, dtype in segments:
                if (
                    layer not in self.expected[peer]
                    or kind not in ("k", "v")
                    or (layer, kind) in covered
                ):
                    raise ValueError("NIXL KV segment layer/kind mismatch")
                pair = targets.setdefault(layer, self._target(layer))
                target = pair[0 if kind == "k" else 1]
                if shape != [len(rows), *target.shape[1:]] or dtype != str(
                    target.dtype
                ):
                    raise ValueError("NIXL KV payload shape/dtype differs from target")
                offset = _positive(offset, "offset", zero=True)
                nbytes = math.prod(shape) * target.element_size()
                if offset < end or offset % 16 or offset + nbytes > data.numel():
                    raise ValueError("NIXL KV segment exceeds/overlaps its buffer")
                if max(slot for slot, _ in rows) >= target.shape[0]:
                    raise ValueError("NIXL KV row exceeds target capacity")
                views[layer, kind] = (
                    data[offset : offset + nbytes].view(target.dtype).view(shape)
                )
                covered.add((layer, kind))
                end = offset + nbytes
            if (
                covered
                != {
                    (layer, kind)
                    for layer in self.expected[peer]
                    for kind in ("k", "v")
                }
                or end != data.numel()
            ):
                raise ValueError(
                    "NIXL data does not exactly cover the incoming KV layers"
                )
            accepted = [
                i
                for i, (slot, epoch) in enumerate(rows)
                if epoch > self.retired[peer].get(slot, 0)
                and epoch >= self.versions[peer].get(slot, 0)
            ]
            context = (
                torch.cuda.stream(self.stream)
                if self.stream is not None
                else nullcontext()
            )
            try:
                with context:
                    selected = torch.tensor(
                        accepted, dtype=torch.int64, device=data.device
                    )
                    slots = torch.tensor(
                        [rows[i][0] for i in accepted],
                        dtype=torch.int64,
                        device=data.device,
                    )
                    for layer in self.expected[peer]:
                        self.pool.import_elastic_pp_kv_slots(
                            generation=self.generation,
                            layer_id=layer,
                            slot_ids=slots,
                            cache_k=views[layer, "k"].index_select(0, selected),
                            cache_v=views[layer, "v"].index_select(0, selected),
                        )
                    if self.stream is not None:
                        applied = torch.cuda.Event()
                        applied.record(self.stream)
                # Network completion is insufficient: ACK after target scatter.
                if self.stream is not None:
                    applied.synchronize()
            except BaseException:
                if self.stream is not None:
                    try:
                        self.stream.synchronize()
                    except BaseException as drain_error:
                        # Abort may drop pending-pool ownership. Keep tensor refs
                        # until owned worker exit if CUDA cannot prove quiescence.
                        self._unsafe_backing.extend(targets.values())
                        self.endpoint._stopping.set()
                        raise RuntimeError(
                            "NIXL target stream could not drain; worker shutdown required"
                        ) from drain_error
                raise
            for i in accepted:
                slot, epoch = rows[i]
                self.versions[peer][slot] = epoch
            self.stats["applied_rows"] += len(accepted)
            self.stats["applied_bytes"] += data.numel()
            if not self.fenced:
                self.stats["online_applied_rows"] += len(accepted)

    def _rows(self, raw):
        if not isinstance(raw, list) or len(raw) > 1024:
            raise ValueError("invalid/beyond-bound NIXL row list")
        rows = []
        full = getattr(self.pool, "full_kv_pool", self.pool)
        for slot, epoch in raw:
            slot, epoch = _positive(slot, "slot"), _positive(epoch, "incarnation")
            if slot >= getattr(full, "size", 0) + getattr(full, "page_size", 1):
                raise ValueError("NIXL row is outside the native pool")
            rows.append((slot, epoch))
        if len({slot for slot, _ in rows}) != len(rows):
            raise ValueError("duplicate NIXL row")
        return rows

    def mark_fenced(self):
        with self.lock:
            self.fenced = True

    def finish(self, generation):
        with self.lock:
            if generation != self.generation or self.sealed != set(self.expected):
                raise RuntimeError(
                    "NIXL incoming live KV has not been sealed by every source"
                )
            return frozenset(
                layer for layers in self.expected.values() for layer in layers
            )

    def retire(self):
        with self.lock:
            if self._unsafe_backing:
                raise RuntimeError(
                    "NIXL target backing remains live; shutdown required"
                )
            # Serializes against target CUDA application before backing is freed.
            self.prepared, self.generation = False, None
            self.versions, self.retired = {}, {}


class NixlKVPatchCopier:
    scope = "native KV NIXL GPU staging; GDN/HiCache at fence"
    shadow_bytes = 0

    def __init__(self, source, *, endpoint, generation, targets, max_rows):
        self.source, self.endpoint, self.generation = source, endpoint, generation
        self.targets = {
            p: tuple(sorted(layers)) for p, layers in targets.items() if layers
        }
        if not self.targets or set(source) != {
            l for layers in self.targets.values() for l in layers
        }:
            raise ValueError(
                "NIXL copy destinations do not exactly cover source KV layers"
            )
        per_row = max(
            sum(
                math.prod(t.shape[1:]) * t.element_size()
                for layer in layers
                for t in source[layer]
            )
            for layers in self.targets.values()
        )
        padding = 16 * 2 * max(len(layers) for layers in self.targets.values())
        self.max_rows = min(max_rows, (endpoint.buffer_bytes - padding) // per_row)
        if self.max_rows <= 0:
            raise ValueError("one KV row exceeds the fixed NIXL GPU staging buffer")
        self.lock, self.retired = threading.Lock(), {}
        self.pending = []
        self.fence_metrics = {}

    @property
    def ready_to_submit(self):
        return self.endpoint.peers_ready(self.generation, self.targets)

    def forget(self, slot, incarnation):
        with self.lock:
            self.retired[slot] = max(incarnation, self.retired.get(slot, 0))

    def _flush_retired(self):
        with self.lock:
            retired, self.retired = self.retired, {}
        rows = sorted(retired.items())
        for start in range(0, len(rows), 256):
            for peer in self.targets:
                self.endpoint.rpc(
                    peer,
                    dict(
                        op="forget",
                        generation=self.generation,
                        rows=rows[start : start + 256],
                    ),
                )

    def _wait_ready(self):
        deadline = time.monotonic() + self.endpoint.timeout
        while not self.ready_to_submit:
            self.endpoint._poll()
            if time.monotonic() >= deadline:
                raise TimeoutError("NIXL target KV preparation timed out")
            time.sleep(0.001)

    def submit(self, rows):
        rows = tuple(rows)

        def copy():
            self._wait_ready()
            self._flush_retired()
            for ready in {id(row.ready): row.ready for row in rows}.values():
                if self.endpoint.stream is None:
                    ready.synchronize()
                else:
                    self.endpoint.stream.wait_event(ready)
            indices = torch.tensor(
                [r.slot for r in rows], dtype=torch.int64, device=self.endpoint.device
            )
            for peer, layers in self.targets.items():
                segments, end = [], 0
                for layer in layers:
                    for kind, tensor in zip(
                        ("k", "v"), self.source[layer], strict=True
                    ):
                        offset = (end + 15) // 16 * 16
                        shape = [len(rows), *tensor.shape[1:]]
                        segments.append([layer, kind, offset, shape, str(tensor.dtype)])
                        end = offset + math.prod(shape) * tensor.element_size()

                def pack(buffer):
                    for layer, kind, offset, shape, _ in segments:
                        tensor = self.source[layer][0 if kind == "k" else 1]
                        values = (
                            tensor.index_select(0, indices)
                            .contiguous()
                            .view(torch.uint8)
                            .reshape(-1)
                        )
                        buffer[offset : offset + values.numel()].copy_(values)
                    return end

                self.endpoint.rpc(
                    peer,
                    dict(
                        op="data",
                        generation=self.generation,
                        rows=[[r.slot, r.incarnation] for r in rows],
                        segments=segments,
                    ),
                    pack=pack,
                )

        completion = self.endpoint.submit(copy)
        self.pending = [p for p in self.pending if not p.future.done()]
        self.pending.append(completion)
        return completion

    def finish(self, epochs):
        versions = dict(epochs)
        queued = time.monotonic()

        def seal():
            self.fence_metrics["queue_seconds"] = time.monotonic() - queued
            started = time.monotonic()
            self._wait_ready()
            self.fence_metrics["ready_seconds"] = time.monotonic() - started
            started = time.monotonic()
            self._flush_retired()
            self.fence_metrics["retired_seconds"] = time.monotonic() - started
            self.fence_metrics["peer_seal_seconds"] = {}
            for peer in self.targets:
                started = time.monotonic()
                self.endpoint.rpc(
                    peer,
                    dict(
                        op="seal",
                        generation=self.generation,
                        count=len(versions),
                        signature=_signature(versions),
                    ),
                )
                self.fence_metrics["peer_seal_seconds"][peer] = (
                    time.monotonic() - started
                )

        completion = self.endpoint.submit(seal)
        self.pending.append(completion)
        completion.synchronize()

    def synchronize(self):
        for completion in self.pending:
            completion.drain()
        self.pending.clear()
