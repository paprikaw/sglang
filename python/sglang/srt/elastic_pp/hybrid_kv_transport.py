"""Fenced GDN request-state migration alongside sparse full-attention KV."""

from __future__ import annotations

import hashlib

import torch

from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport, _slot_chunks
from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport

_MANIFEST = "__elastic_pp_recurrent_manifest__"
_SLOTS = "__elastic_pp_recurrent_slots__"


class ElasticPPHybridKVTransport(ElasticPPKVTransport):
    """GDN state is indexed by request slots, never by token KV slots.

    Copy the complete recurrent slot pool in bounded chunks at the drained
    scheduler fence. This also preserves unused/padding slots and avoids
    depending on a stale background request snapshot. Allocator identity is
    checked collectively before any transfer; slot remapping is not supported.
    """

    def __init__(self, *, recurrent_chunk_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        if recurrent_chunk_size <= 0:
            raise ValueError("recurrent chunk size must be positive")
        self.recurrent_chunk_size = recurrent_chunk_size
        self._received_recurrent: dict[int, set[int]] = {}

    @staticmethod
    def _digest(values):
        tensor = torch.as_tensor(values, dtype=torch.int64).detach().cpu().contiguous()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

    def fence_state_signature(self):
        req_pool = self.pool.req_to_token_pool
        free = set(req_pool.free_slots)
        active = [i for i in range(1, req_pool.size + 1) if i not in free]
        active_ids = torch.tensor(active, dtype=torch.int64, device=req_pool.device)
        mapping = req_pool.req_index_to_mamba_index_mapping[active_ids]
        state = req_pool.mamba_pool.mamba_cache
        return {
            "slots": self.pool.recurrent_slot_count,
            "req_pool_size": req_pool.size,
            # Preserve free-list order too: it determines the next allocation.
            "req_free": self._digest(req_pool.free_slots),
            "mamba_free": self._digest(req_pool.mamba_allocator.free_slots),
            "active_mapping": self._digest(mapping),
            "req_generation": self._digest(req_pool.req_generation[active]),
            "layout": tuple(
                (tuple(t.shape[2:]), str(t.dtype))
                for t in (*state.conv, state.temporal)
            ),
        }

    def snapshot_at_fence(self, transition):
        size = super().snapshot_at_fence(transition)
        free = set(
            int(i) for i in self.pool.req_to_token_pool.mamba_allocator.free_slots
        )
        slots = torch.tensor(
            [i for i in range(self.pool.recurrent_slot_count) if i not in free],
            dtype=torch.int64,
        )
        layers = {
            layer
            for edge in self.outbound_transfers(transition)
            for layer in self.pool.recurrent_layer_ids(edge.layer_range)
        }
        states = {}
        for layer in sorted(layers):
            tensors = self.pool.export_recurrent_slots(
                layer_id=layer, slot_ids=slots.to(self.pool.device)
            )
            states[layer] = {
                name: t.detach().cpu().clone() for name, t in tensors.items()
            }
        req_pool = self.pool.req_to_token_pool
        self._fence_snapshot["request_state"] = {
            "free_slots": list(req_pool.free_slots),
            "recurrent_free_slots": req_pool.mamba_allocator.free_slots.detach()
            .cpu()
            .clone(),
            "mapping": req_pool.req_index_to_mamba_index_mapping.detach().cpu().clone(),
            "generation": req_pool.req_generation.detach().cpu().clone(),
        }
        self._fence_snapshot["recurrent_slots"] = slots
        self._fence_snapshot["recurrent"] = states
        return size + sum(
            t.numel() * t.element_size()
            for tensors in states.values()
            for t in tensors.values()
        )

    def send_phase(self, *args, **kwargs):
        raise NotImplementedError("hybrid PP migration requires fenced exchange_phase")

    def prepare_transition(self, transition):
        # Hybrid migration copies one authoritative snapshot at the fence.
        # Do not inspect or enable tracking on an allocator concurrently with
        # serving: its free group and live-page set may be changing here.
        try:
            self.prepare_target(transition)
        except BaseException:
            try:
                self.model_runner.abort_elastic_pp_kv_layers(
                    generation=transition.target.generation
                )
            except BaseException:
                pass
            raise
        return torch.empty(0, dtype=torch.int64, device="cpu")

    def finish_slot_tracking(self, transition):
        # No background slot tracker is installed for the fenced-only path.
        del transition

    def receive_phase(self, *args, **kwargs):
        raise NotImplementedError("hybrid PP migration requires fenced exchange_phase")

    def exchange_phase(self, transition, *, phase, slot_ids):
        loaded = super().exchange_phase(transition, phase=phase, slot_ids=slot_ids)
        if phase != "bulk":
            return loaded
        generation = transition.target.generation
        received = self._received_recurrent.setdefault(generation, set())
        # Separate address space from slot_ids (the sparse token-KV rows).
        # Use a blocking host-to-device copy: a CUDA arange on the serving
        # stream could otherwise race index_select on the migration streams.
        request_slots = torch.arange(
            self.pool.recurrent_slot_count, dtype=torch.int64
        ).to(self.pool.device)
        chunks = _slot_chunks(request_slots, self.recurrent_chunk_size)
        for transfer in sorted(transition.kv_transfers, key=self._sort_key):
            if self.rank not in (transfer.src_rank, transfer.dst_rank):
                continue
            sender = self.rank == transfer.src_rank
            stream = self._send_stream if sender else self._receive_stream
            layer_ids = self.pool.recurrent_layer_ids(transfer.layer_range)
            with self._stream_context(stream):
                for layer_id in layer_ids:
                    for chunk_index, slots in enumerate(chunks):
                        header = {
                            "generation": generation,
                            "src_rank": transfer.src_rank,
                            "dst_rank": transfer.dst_rank,
                            "start_layer": transfer.layer_range.start,
                            "end_layer": transfer.layer_range.end,
                            "layer_id": layer_id,
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunks),
                            "slot_count": self.pool.recurrent_slot_count,
                        }
                        if sender:
                            tensors = self.pool.export_recurrent_slots(
                                layer_id=layer_id, slot_ids=slots
                            )
                            payload = {_MANIFEST: header, _SLOTS: slots, **tensors}
                            works = self.group.send_tensor_dict(
                                payload, dst=transfer.dst_rank, async_send=True
                            )
                            ElasticPPLayerTransport.wait(works or ())
                        else:
                            payload = self.group.recv_tensor_dict(src=transfer.src_rank)
                            if (
                                not isinstance(payload, dict)
                                or payload.get(_MANIFEST) != header
                            ):
                                raise ValueError(
                                    "elastic PP recurrent manifest mismatch"
                                )
                            source_slots = payload.get(_SLOTS)
                            if (
                                not isinstance(source_slots, torch.Tensor)
                                or source_slots.dtype != torch.int64
                                or not torch.equal(source_slots, slots)
                            ):
                                raise ValueError(
                                    "elastic PP recurrent request slots mismatch"
                                )
                            tensors = {
                                k: v
                                for k, v in payload.items()
                                if k not in {_MANIFEST, _SLOTS}
                            }
                            self.pool.import_recurrent_slots(
                                generation=generation,
                                layer_id=layer_id,
                                slot_ids=source_slots,
                                tensors=tensors,
                            )
                    if not sender:
                        if layer_id in received:
                            raise RuntimeError(f"duplicate recurrent layer {layer_id}")
                        received.add(layer_id)
                ready_event = self._record_ready_event(stream)
            if ready_event is not None:
                ready_event.synchronize()
        return loaded | frozenset(received)

    def apply_staged(self, transition):
        generation = transition.target.generation
        expected = {
            layer_id
            for transfer in self.inbound_transfers(transition)
            for layer_id in self.pool.recurrent_layer_ids(transfer.layer_range)
        }
        received = self._received_recurrent.get(generation, set())
        if received != expected:
            raise RuntimeError(
                "elastic PP recurrent transfer set is incomplete: "
                f"expected={sorted(expected)}, received={sorted(received)}"
            )
        loaded = super().apply_staged(transition)
        if received:
            self.pool.mark_elastic_pp_kv_layers_loaded(
                generation=generation, layer_ids=received
            )
        self._received_recurrent.pop(generation, None)
        return loaded | frozenset(received)

    def abort_staged(self, transition):
        super().abort_staged(transition)
        self._received_recurrent.pop(transition.target.generation, None)
