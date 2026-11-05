"""Sparse owner-to-owner KV transfer for elastic pipeline parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Literal

import torch

from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport
from sglang.srt.elastic_pp.topology import DynamicPPTransition, LayerTransfer

KVTransferPhase = Literal["bulk", "delta"]

_MANIFEST_KEY = "__elastic_pp_kv_manifest__"
_SLOT_IDS_KEY = "elastic_pp_kv_slot_ids"
_TENSOR_KEY_PREFIX = "elastic_pp_kv_tensor_"


def _transfer_layer_ids(transfer: LayerTransfer, layer_ids=None) -> set[int]:
    interval = set(range(transfer.layer_range.start, transfer.layer_range.end))
    if layer_ids is None:
        return interval
    expected = set(layer_ids)
    if not expected <= interval:
        raise ValueError("KV layer ids are outside the transferred model interval")
    return expected


def _slot_chunks(slot_ids: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, ...]:
    if chunk_size <= 0:
        raise ValueError("elastic PP KV slot chunk size must be positive")
    if slot_ids.ndim != 1:
        raise ValueError("elastic PP KV slot ids must be one-dimensional")
    if slot_ids.numel() == 0:
        return (slot_ids,)
    return tuple(
        slot_ids[start : start + chunk_size]
        for start in range(0, slot_ids.numel(), chunk_size)
    )


def flatten_kv_chunk(
    *,
    generation: int,
    transfer: LayerTransfer,
    phase: KVTransferPhase,
    chunk_index: int,
    total_chunks: int,
    slot_ids: torch.Tensor,
    layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    expected_layers = _transfer_layer_ids(transfer, layer_ids)
    if set(layer_kv) != expected_layers:
        raise ValueError(
            "KV payload does not exactly cover transfer layers: "
            f"expected={sorted(expected_layers)}, got={sorted(layer_kv)}"
        )
    if phase not in ("bulk", "delta"):
        raise ValueError(f"invalid elastic PP KV phase: {phase!r}")
    if not 0 <= chunk_index < total_chunks:
        raise ValueError(f"invalid elastic PP KV chunk {chunk_index}/{total_chunks}")

    payload: dict[str, Any] = {_SLOT_IDS_KEY: slot_ids.detach().contiguous()}
    entries: list[tuple[str, int, str]] = []
    tensor_index = 0
    for layer_id in sorted(layer_kv):
        cache_k, cache_v = layer_kv[layer_id]
        for kind, tensor in (("k", cache_k), ("v", cache_v)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"KV state {layer_id}:{kind} is not a tensor")
            key = f"{_TENSOR_KEY_PREFIX}{tensor_index:06d}"
            payload[key] = tensor.detach().contiguous()
            entries.append((key, int(layer_id), kind))
            tensor_index += 1
    payload[_MANIFEST_KEY] = {
        "generation": int(generation),
        "src_rank": transfer.src_rank,
        "dst_rank": transfer.dst_rank,
        "start_layer": transfer.layer_range.start,
        "end_layer": transfer.layer_range.end,
        "phase": phase,
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "slot_count": int(slot_ids.numel()),
        "entries": tuple(entries),
    }
    return payload


def unflatten_kv_chunk(
    payload: dict[str, Any],
    *,
    generation: int,
    transfer: LayerTransfer,
    phase: KVTransferPhase,
    chunk_index: int,
    total_chunks: int,
    layer_ids: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    manifest = payload.get(_MANIFEST_KEY)
    if not isinstance(manifest, dict):
        raise ValueError("elastic PP KV payload has no manifest")
    expected_header = {
        "generation": int(generation),
        "src_rank": transfer.src_rank,
        "dst_rank": transfer.dst_rank,
        "start_layer": transfer.layer_range.start,
        "end_layer": transfer.layer_range.end,
        "phase": phase,
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
    }
    for field, expected in expected_header.items():
        if manifest.get(field) != expected:
            raise ValueError(
                "elastic PP KV manifest mismatch: "
                f"{field}={manifest.get(field)!r}, expected={expected!r}"
            )
    slot_ids = payload.get(_SLOT_IDS_KEY)
    if not isinstance(slot_ids, torch.Tensor) or slot_ids.ndim != 1:
        raise ValueError("elastic PP KV payload has invalid slot ids")
    if slot_ids.dtype != torch.int64:
        raise ValueError(f"elastic PP KV slot ids must be int64, got {slot_ids.dtype}")
    if manifest.get("slot_count") != slot_ids.numel():
        raise ValueError("elastic PP KV slot count does not match its manifest")

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, (tuple, list)):
        raise ValueError("elastic PP KV manifest entries are invalid")
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    tensor_keys: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError(f"invalid elastic PP KV manifest entry: {entry!r}")
        key, layer_id, kind = entry
        if not isinstance(key, str) or kind not in ("k", "v"):
            raise ValueError(f"invalid elastic PP KV manifest entry: {entry!r}")
        layer_id = int(layer_id)
        if not (transfer.layer_range.start <= layer_id < transfer.layer_range.end):
            raise ValueError(
                f"elastic PP KV layer {layer_id} is outside {transfer.layer_range}"
            )
        if key in tensor_keys or kind in by_layer.setdefault(layer_id, {}):
            raise ValueError(f"duplicate elastic PP KV manifest entry: {entry!r}")
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"elastic PP KV tensor {key!r} is missing")
        tensor_keys.add(key)
        by_layer[layer_id][kind] = tensor

    expected_layers = _transfer_layer_ids(transfer, layer_ids)
    if set(by_layer) != expected_layers or any(
        set(item) != {"k", "v"} for item in by_layer.values()
    ):
        raise ValueError(
            "elastic PP KV payload does not contain one K/V pair per layer"
        )
    unexpected = set(payload) - tensor_keys - {_MANIFEST_KEY, _SLOT_IDS_KEY}
    if unexpected:
        raise ValueError(f"unexpected elastic PP KV payload keys: {sorted(unexpected)}")
    return slot_ids, {
        layer_id: (item["k"], item["v"]) for layer_id, item in by_layer.items()
    }


class ElasticPPKVTransport:
    """Move sparse live KV rows on the dedicated elastic PP communicator."""

    def __init__(
        self,
        *,
        group,
        model_runner,
        allocator,
        rank: int,
        slot_chunk_size: int = 1024,
    ) -> None:
        required_allocator_methods = ("elastic_pp_live_slots",)
        missing = [
            name for name in required_allocator_methods if not hasattr(allocator, name)
        ]
        if missing:
            raise NotImplementedError(
                "elastic PP sparse KV migration currently requires the standard "
                f"token allocator; missing={missing}"
            )
        self.group = group
        self.model_runner = model_runner
        self.pool = model_runner.token_to_kv_pool
        self.allocator = allocator
        self.rank = int(rank)
        self.slot_chunk_size = int(slot_chunk_size)
        self._runner_device = torch.device(getattr(model_runner, "device", "cpu"))
        self._send_stream = None
        self._receive_stream = None
        self._staged_chunks: dict[
            tuple[int, str, int, int, int, int, int],
            tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]],
        ] = {}
        if self._runner_device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self._runner_device):
                self._send_stream = torch.cuda.Stream(device=self._runner_device)
                self._receive_stream = torch.cuda.Stream(device=self._runner_device)

    @contextmanager
    def _stream_context(self, stream):
        if stream is None:
            yield
            return
        with (
            torch.cuda.device(self._runner_device),
            torch.cuda.stream(stream),
        ):
            yield

    @staticmethod
    def _record_ready_event(stream):
        if stream is None:
            return None
        ready_event = torch.cuda.Event()
        ready_event.record(stream)
        return ready_event

    @staticmethod
    def _sort_key(transfer: LayerTransfer) -> tuple[int, int, int, int]:
        return (
            transfer.src_rank,
            transfer.dst_rank,
            transfer.layer_range.start,
            transfer.layer_range.end,
        )

    def outbound_transfers(
        self, transition: DynamicPPTransition
    ) -> tuple[LayerTransfer, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in transition.kv_transfers
                    if item.src_rank == self.rank
                ),
                key=self._sort_key,
            )
        )

    def inbound_transfers(
        self, transition: DynamicPPTransition
    ) -> tuple[LayerTransfer, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in transition.kv_transfers
                    if item.dst_rank == self.rank
                ),
                key=self._sort_key,
            )
        )

    def prepare_transition(self, transition: DynamicPPTransition) -> None:
        """Prepare backing without reading or tracking the serving allocator.

        All KV payload is copied at the drained scheduler fence. A background
        snapshot is unused in that protocol and races with allocation and
        deferred frees; even a closed free-group flag is not a synchronization
        boundary. Only fence_slots may read the live set for this transition.
        """

        try:
            self.prepare_target(transition)
        except BaseException:
            # The coordinator cannot set its prepared flag until this
            # method returns. Roll back locally so a failure after backing
            # allocation cannot leak a pending generation.
            try:
                self.model_runner.abort_elastic_pp_kv_layers(
                    generation=transition.target.generation
                )
            except BaseException:
                pass
            raise

    def prepare_target(self, transition: DynamicPPTransition) -> frozenset[int]:
        with self._stream_context(self._receive_stream):
            prepared = self.model_runner.prepare_elastic_pp_kv_layers(
                target=transition.target.layer_range(self.rank),
                generation=transition.target.generation,
            )
            ready_event = self._record_ready_event(self._receive_stream)
        if ready_event is not None:
            # Allocation is on the migration stream so it may overlap serving,
            # but the bulk copy cannot publish readiness before the backing
            # tensors themselves exist.
            ready_event.synchronize()
        return prepared

    def fence_slots(self, transition: DynamicPPTransition) -> torch.Tensor:
        """Return the authoritative live rows at the drained scheduler fence.

        No allocator snapshot is taken by the background preparation thread.
        Requests can allocate, free, or reuse slots while target backing is
        prepared.  Once every PP rank has drained at the common fence, the
        current source-owner snapshot is both complete and stable for the
        synchronous transfer.
        """

        del transition
        return self.allocator.elastic_pp_live_slots()

    def send_phase(
        self,
        transition: DynamicPPTransition,
        *,
        phase: KVTransferPhase,
        slot_ids: torch.Tensor,
        source_ready_event=None,
    ) -> int:
        """Send bounded chunks; run this on the migration sender thread."""

        sent = 0
        with self._stream_context(self._send_stream):
            if source_ready_event is not None and self._send_stream is not None:
                self._send_stream.wait_event(source_ready_event)
            for transfer in self.outbound_transfers(transition):
                layer_ids = self._kv_layer_ids(transfer)
                chunks = _slot_chunks(
                    slot_ids if layer_ids else slot_ids[:0], self.slot_chunk_size
                )
                for chunk_index, chunk in enumerate(chunks):
                    layer_kv = {
                        layer_id: self.pool.export_elastic_pp_kv_slots(
                            layer_id=layer_id, slot_ids=chunk
                        )
                        for layer_id in layer_ids
                    }
                    payload = flatten_kv_chunk(
                        generation=transition.target.generation,
                        transfer=transfer,
                        phase=phase,
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                        slot_ids=chunk,
                        layer_kv=layer_kv,
                        layer_ids=layer_ids,
                    )
                    works = self.group.send_tensor_dict(
                        payload, dst=transfer.dst_rank, async_send=True
                    )
                    ElasticPPLayerTransport.wait(works or ())
                    sent += 1
            ready_event = self._record_ready_event(self._send_stream)
        if ready_event is not None:
            ready_event.synchronize()
        return sent

    def receive_phase(
        self,
        transition: DynamicPPTransition,
        *,
        phase: KVTransferPhase,
    ) -> frozenset[int]:
        loaded: set[int] = set()
        with self._stream_context(self._receive_stream):
            for transfer in self.inbound_transfers(transition):
                loaded.update(self._receive_transfer(transition, transfer, phase))
            ready_event = self._record_ready_event(self._receive_stream)
        if ready_event is not None:
            # This is a migration-thread wait, not a serving-stream sync.  The
            # control plane cannot publish buffers while NCCL/scatter is pending.
            ready_event.synchronize()
        return frozenset(loaded)

    @staticmethod
    def _manifest_total_chunks(
        payload: dict[str, Any],
        *,
        transition: DynamicPPTransition,
        transfer: LayerTransfer,
        phase: KVTransferPhase,
    ) -> int:
        manifest = payload.get(_MANIFEST_KEY)
        if not isinstance(manifest, dict):
            raise ValueError("elastic PP KV payload has no manifest")
        expected = {
            "generation": transition.target.generation,
            "src_rank": transfer.src_rank,
            "dst_rank": transfer.dst_rank,
            "start_layer": transfer.layer_range.start,
            "end_layer": transfer.layer_range.end,
            "phase": phase,
            "chunk_index": 0,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(
                    "elastic PP KV first-chunk manifest mismatch: "
                    f"{field}={manifest.get(field)!r}, expected={value!r}"
                )
        total_chunks = manifest.get("total_chunks")
        if not isinstance(total_chunks, int) or isinstance(total_chunks, bool):
            raise ValueError(f"elastic PP KV total_chunks is invalid: {total_chunks!r}")
        if total_chunks <= 0:
            raise ValueError(
                f"elastic PP KV total_chunks must be positive: {total_chunks}"
            )
        return total_chunks

    def _receive_transfer(
        self,
        transition: DynamicPPTransition,
        transfer: LayerTransfer,
        phase: KVTransferPhase,
    ) -> frozenset[int]:
        """Receive one source-authored sparse slot stream.

        PP stages can be at different points in the in-flight wave while the
        background snapshot starts.  The source rank that owns the migrating
        layers is therefore authoritative for both the slot ids and the chunk
        count; the destination allocator's instantaneous live set is not an
        equality invariant.
        """

        first_payload = self.group.recv_tensor_dict(src=transfer.src_rank)
        if not isinstance(first_payload, dict):
            raise RuntimeError(f"did not receive KV rows from rank {transfer.src_rank}")
        total_chunks = self._manifest_total_chunks(
            first_payload,
            transition=transition,
            transfer=transfer,
            phase=phase,
        )
        loaded: set[int] = set()
        for chunk_index in range(total_chunks):
            payload = (
                first_payload
                if chunk_index == 0
                else self.group.recv_tensor_dict(src=transfer.src_rank)
            )
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"did not receive KV rows from rank {transfer.src_rank}"
                )
            slot_ids, layer_kv = unflatten_kv_chunk(
                payload,
                generation=transition.target.generation,
                transfer=transfer,
                phase=phase,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                layer_ids=self._kv_layer_ids(transfer),
            )
            key = (
                transition.target.generation,
                phase,
                transfer.src_rank,
                transfer.dst_rank,
                transfer.layer_range.start,
                transfer.layer_range.end,
                chunk_index,
            )
            if key in self._staged_chunks:
                raise RuntimeError(f"duplicate staged elastic PP KV chunk: {key!r}")
            self._staged_chunks[key] = (slot_ids, layer_kv)
            loaded.update(layer_kv)
        return frozenset(loaded)

    def _kv_layer_ids(self, transfer: LayerTransfer) -> tuple[int, ...]:
        selector = getattr(self.pool, "elastic_pp_kv_layer_ids", None)
        if selector is not None:
            return selector(transfer.layer_range)
        return tuple(range(transfer.layer_range.start, transfer.layer_range.end))

    def exchange_phase(
        self,
        transition: DynamicPPTransition,
        *,
        phase: KVTransferPhase,
        slot_ids: torch.Tensor,
    ) -> frozenset[int]:
        """Execute one bounded KV phase in a globally deterministic order.

        A rank can simultaneously be a source for one moved interval and a
        destination for another.  Having every rank perform all sends before
        all receives can then form a wait cycle once chunk memory is bounded.
        Iterating the same globally sorted transfer list serializes only the
        ownership edges, keeps one chunk live at a time, and makes cycles
        impossible without rebuilding the serving communicator.
        """

        transfers = tuple(sorted(transition.kv_transfers, key=self._sort_key))
        loaded: set[int] = set()
        for transfer in transfers:
            if transfer.src_rank == self.rank:
                stream = self._send_stream
            elif transfer.dst_rank == self.rank:
                stream = self._receive_stream
            else:
                continue
            with self._stream_context(stream):
                if transfer.src_rank == self.rank:
                    # The migrating layer owner defines the sparse KV rows for
                    # this edge.  Receiver-local live slots may legitimately
                    # differ while PP microbatches are still in flight.
                    layer_ids = self._kv_layer_ids(transfer)
                    chunks = _slot_chunks(
                        slot_ids if layer_ids else slot_ids[:0], self.slot_chunk_size
                    )
                    for chunk_index, source_chunk in enumerate(chunks):
                        layer_kv = {
                            layer_id: self.pool.export_elastic_pp_kv_slots(
                                layer_id=layer_id, slot_ids=source_chunk
                            )
                            for layer_id in layer_ids
                        }
                        payload = flatten_kv_chunk(
                            generation=transition.target.generation,
                            transfer=transfer,
                            phase=phase,
                            chunk_index=chunk_index,
                            total_chunks=len(chunks),
                            slot_ids=source_chunk,
                            layer_kv=layer_kv,
                            layer_ids=layer_ids,
                        )
                        works = self.group.send_tensor_dict(
                            payload,
                            dst=transfer.dst_rank,
                            async_send=True,
                        )
                        ElasticPPLayerTransport.wait(works or ())
                else:
                    loaded.update(self._receive_transfer(transition, transfer, phase))
                ready_event = self._record_ready_event(stream)
            if ready_event is not None:
                ready_event.synchronize()
        return frozenset(loaded)

    def apply_staged(self, transition: DynamicPPTransition) -> frozenset[int]:
        """Scatter received bulk and delta rows at the drained PP fence.

        KVCacheD can need a CUDA VMM map before the first write into a newly
        owned layer. Its mapping fence synchronizes the device, so migration
        receive, ownership refresh, and the final sparse scatter all execute
        only after the scheduler has drained its microbatch ring.
        """

        generation = transition.target.generation
        inbound = self.inbound_transfers(transition)
        staged = [
            (key, value)
            for key, value in self._staged_chunks.items()
            if key[0] == generation
        ]
        if not inbound:
            if staged:
                raise RuntimeError(
                    "elastic PP rank has staged KV chunks without an inbound transfer"
                )
            return frozenset()

        expected_edges = {
            (
                transfer.src_rank,
                transfer.dst_rank,
                transfer.layer_range.start,
                transfer.layer_range.end,
            )
            for transfer in inbound
        }
        for phase in ("bulk", "delta"):
            observed_edges = {
                (key[2], key[3], key[4], key[5]) for key, _ in staged if key[1] == phase
            }
            if observed_edges != expected_edges:
                raise RuntimeError(
                    "elastic PP staged KV transfer set is incomplete: "
                    f"phase={phase}, expected={sorted(expected_edges)}, "
                    f"observed={sorted(observed_edges)}"
                )

        prepare_fence = getattr(self.pool, "prepare_elastic_pp_kv_fence", None)
        if prepare_fence is not None:
            prepare_fence(generation=generation)

        phase_order = {"bulk": 0, "delta": 1}
        loaded: set[int] = set()
        with self._stream_context(self._receive_stream):
            ordered = sorted(
                staged,
                key=lambda item: (
                    phase_order[item[0][1]],
                    item[0][2],
                    item[0][3],
                    item[0][4],
                    item[0][5],
                    item[0][6],
                ),
            )
            for _, (slot_ids, layer_kv) in ordered:
                for layer_id, (cache_k, cache_v) in layer_kv.items():
                    self.pool.import_elastic_pp_kv_slots(
                        generation=generation,
                        layer_id=layer_id,
                        slot_ids=slot_ids,
                        cache_k=cache_k,
                        cache_v=cache_v,
                    )
                    loaded.add(layer_id)
            ready_event = self._record_ready_event(self._receive_stream)
        if ready_event is not None:
            ready_event.synchronize()
        if loaded:
            self.pool.mark_elastic_pp_kv_layers_loaded(
                generation=generation,
                layer_ids=loaded,
            )
        for key, _ in staged:
            del self._staged_chunks[key]
        return frozenset(loaded)

    def abort_staged(self, transition: DynamicPPTransition) -> None:
        generation = transition.target.generation
        self._staged_chunks = {
            key: value
            for key, value in self._staged_chunks.items()
            if key[0] != generation
        }
