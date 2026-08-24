"""Direct owner-to-owner model-layer transfer over a dedicated PP group."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch

from sglang.srt.elastic_pp.topology import DynamicPPTransition, LayerTransfer

_MANIFEST_KEY = "__elastic_pp_layer_manifest__"
_TENSOR_KEY_PREFIX = "elastic_pp_layer_tensor_"


def flatten_layer_states(
    *,
    generation: int,
    transfer: LayerTransfer,
    layer_states: dict[int, dict[str, torch.Tensor]],
    copy_noncontiguous: bool = True,
) -> dict[str, Any]:
    """Encode nested layer state into ``GroupCoordinator``'s flat protocol."""

    expected_layers = set(range(transfer.layer_range.start, transfer.layer_range.end))
    if set(layer_states) != expected_layers:
        raise ValueError(
            "layer-state payload does not exactly cover the transfer: "
            f"expected={sorted(expected_layers)}, got={sorted(layer_states)}"
        )

    payload: dict[str, Any] = {}
    entries: list[tuple[str, int, str]] = []
    tensor_index = 0
    for layer_id in sorted(layer_states):
        for state_name, tensor in sorted(layer_states[layer_id].items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"layer state {layer_id}:{state_name} is not a tensor")
            key = f"{_TENSOR_KEY_PREFIX}{tensor_index:06d}"
            detached = tensor.detach()
            if not detached.is_contiguous():
                if not copy_noncontiguous:
                    raise ValueError(
                        "direct elastic PP destination tensor must be contiguous: "
                        f"layer={layer_id}, state={state_name}"
                    )
                detached = detached.contiguous()
            payload[key] = detached
            entries.append((key, int(layer_id), state_name))
            tensor_index += 1
    payload[_MANIFEST_KEY] = {
        "generation": int(generation),
        "src_rank": transfer.src_rank,
        "dst_rank": transfer.dst_rank,
        "start_layer": transfer.layer_range.start,
        "end_layer": transfer.layer_range.end,
        "entries": tuple(entries),
    }
    return payload


def _validate_manifest(
    manifest: Any,
    *,
    generation: int,
    transfer: LayerTransfer,
) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(manifest, dict):
        raise ValueError("elastic PP layer payload has no manifest")
    expected_header = {
        "generation": int(generation),
        "src_rank": transfer.src_rank,
        "dst_rank": transfer.dst_rank,
        "start_layer": transfer.layer_range.start,
        "end_layer": transfer.layer_range.end,
    }
    for field, expected in expected_header.items():
        if manifest.get(field) != expected:
            raise ValueError(
                "elastic PP layer manifest mismatch: "
                f"{field}={manifest.get(field)!r}, expected={expected!r}"
            )

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, (tuple, list)):
        raise ValueError("elastic PP layer manifest entries are invalid")
    entries: list[tuple[str, int, str]] = []
    seen_keys: set[str] = set()
    seen_states: set[tuple[int, str]] = set()
    seen_layers: set[int] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, (tuple, list)) or len(raw_entry) != 3:
            raise ValueError(f"invalid elastic PP layer manifest entry: {raw_entry!r}")
        key, layer_id, state_name = raw_entry
        if not isinstance(key, str) or not isinstance(state_name, str):
            raise ValueError(f"invalid elastic PP layer manifest entry: {raw_entry!r}")
        layer_id = int(layer_id)
        if not (transfer.layer_range.start <= layer_id < transfer.layer_range.end):
            raise ValueError(
                f"elastic PP manifest layer {layer_id} is outside "
                f"{transfer.layer_range}"
            )
        if key in seen_keys:
            raise ValueError(f"duplicate elastic PP layer tensor key: {key!r}")
        state_key = (layer_id, state_name)
        if state_key in seen_states:
            raise ValueError(
                f"duplicate elastic PP layer state {layer_id}:{state_name}"
            )
        seen_keys.add(key)
        seen_states.add(state_key)
        seen_layers.add(layer_id)
        entries.append((key, layer_id, state_name))

    expected_layers = set(range(transfer.layer_range.start, transfer.layer_range.end))
    if seen_layers != expected_layers:
        raise ValueError(
            "elastic PP manifest does not exactly cover the transfer: "
            f"expected={sorted(expected_layers)}, got={sorted(seen_layers)}"
        )
    return tuple(entries)


def unflatten_layer_states(
    payload: dict[str, Any],
    *,
    generation: int,
    transfer: LayerTransfer,
) -> dict[int, dict[str, torch.Tensor]]:
    manifest = payload.get(_MANIFEST_KEY)
    entries = _validate_manifest(manifest, generation=generation, transfer=transfer)
    states: dict[int, dict[str, torch.Tensor]] = {}
    expected_tensor_keys: set[str] = set()
    for entry in entries:
        key, layer_id, state_name = entry
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"elastic PP layer tensor {key!r} is missing")
        expected_tensor_keys.add(key)
        layer_state = states.setdefault(layer_id, {})
        layer_state[state_name] = tensor

    unexpected_keys = set(payload) - expected_tensor_keys - {_MANIFEST_KEY}
    if unexpected_keys:
        raise ValueError(
            f"unexpected elastic PP layer payload keys: {sorted(unexpected_keys)}"
        )
    return states


def _validate_receive_metadata(
    metadata_list: list[tuple[str, Any]],
    *,
    generation: int,
    transfer: LayerTransfer,
) -> None:
    manifests = [value for key, value in metadata_list if key == _MANIFEST_KEY]
    if len(manifests) != 1:
        raise ValueError("elastic PP layer metadata must contain exactly one manifest")
    entries = _validate_manifest(manifests[0], generation=generation, transfer=transfer)
    expected_tensor_keys = {key for key, _, _ in entries}
    received_tensor_keys = {key for key, _ in metadata_list if key != _MANIFEST_KEY}
    if received_tensor_keys != expected_tensor_keys:
        raise ValueError(
            "elastic PP tensor metadata does not match its manifest: "
            f"missing={sorted(expected_tensor_keys - received_tensor_keys)}, "
            f"unexpected={sorted(received_tensor_keys - expected_tensor_keys)}"
        )


class ElasticPPLayerTransport:
    """Execute the weight portion of one ``DynamicPPTransition``.

    The communicator is intentionally distinct from the live PP proxy group,
    so background weight traffic cannot reorder serving send/recv operations.
    """

    def __init__(self, *, group, model_runner, rank: int) -> None:
        self.group = group
        self.model_runner = model_runner
        self.rank = int(rank)
        runner_device = torch.device(getattr(model_runner, "device", "cpu"))
        self._migration_stream = None
        if runner_device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(runner_device):
                self._migration_stream = torch.cuda.Stream(device=runner_device)

    def _stream_context(self):
        if self._migration_stream is None:
            return nullcontext()
        return torch.cuda.stream(self._migration_stream)

    @staticmethod
    def _sort_key(transfer: LayerTransfer) -> tuple[int, int, int, int]:
        return (
            transfer.src_rank,
            transfer.dst_rank,
            transfer.layer_range.start,
            transfer.layer_range.end,
        )

    def prepare_target(self, transition: DynamicPPTransition) -> frozenset[int]:
        with self._stream_context():
            return self.model_runner.prepare_elastic_pp_layers(
                target=transition.target.layer_range(self.rank),
                generation=transition.target.generation,
            )

    def outbound_transfers(
        self, transition: DynamicPPTransition
    ) -> tuple[LayerTransfer, ...]:
        return tuple(
            sorted(
                (
                    transfer
                    for transfer in transition.kv_transfers
                    if transfer.src_rank == self.rank
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
                    transfer
                    for transfer in transition.kv_transfers
                    if transfer.dst_rank == self.rank
                ),
                key=self._sort_key,
            )
        )

    def start_sends(self, transition: DynamicPPTransition) -> tuple[Any, ...]:
        works: list[Any] = []
        for transfer in self.outbound_transfers(transition):
            layer_ids = range(transfer.layer_range.start, transfer.layer_range.end)
            layer_states = self.model_runner.export_elastic_pp_layer_state(layer_ids)
            payload = flatten_layer_states(
                generation=transition.target.generation,
                transfer=transfer,
                layer_states=layer_states,
            )
            transfer_works = self.group.send_tensor_dict(
                payload, dst=transfer.dst_rank, async_send=True
            )
            if transfer_works:
                works.extend(transfer_works)
        return tuple(works)

    def receive(self, transition: DynamicPPTransition) -> frozenset[int]:
        loaded: set[int] = set()
        generation = transition.target.generation
        with self._stream_context():
            for transfer in self.inbound_transfers(transition):
                layer_ids = range(transfer.layer_range.start, transfer.layer_range.end)
                target_states = self.model_runner.export_elastic_pp_layer_state(
                    layer_ids
                )
                target_payload = flatten_layer_states(
                    generation=generation,
                    transfer=transfer,
                    layer_states=target_states,
                    copy_noncontiguous=False,
                )
                destinations = {
                    key: tensor
                    for key, tensor in target_payload.items()
                    if isinstance(tensor, torch.Tensor)
                }
                payload = self.group.recv_tensor_dict_into(
                    destinations,
                    src=transfer.src_rank,
                    metadata_validator=lambda metadata, transfer=transfer: (
                        _validate_receive_metadata(
                            metadata,
                            generation=generation,
                            transfer=transfer,
                        )
                    ),
                )
                layer_states = unflatten_layer_states(
                    payload, generation=generation, transfer=transfer
                )
                # The direct receive must preserve the target parameter
                # storage.  This assertion catches adapters that silently
                # reintroduce a full-size temporary allocation.
                for layer_id, state in layer_states.items():
                    for state_name, tensor in state.items():
                        target = target_states[layer_id][state_name]
                        if tensor.data_ptr() != target.data_ptr():
                            raise RuntimeError(
                                "elastic PP receive did not use target storage: "
                                f"layer={layer_id}, state={state_name}"
                            )
                loaded.update(layer_states)

            ready_event = None
            if self._migration_stream is not None:
                ready_event = torch.cuda.Event()
                ready_event.record(self._migration_stream)
        # This wait occurs on the background migration thread.  It does not
        # synchronize the serving compute stream, but it prevents the control
        # plane from publishing a generation before NCCL has finished writing
        # its parameter buffers.
        if ready_event is not None:
            ready_event.synchronize()
        self.model_runner.mark_elastic_pp_layers_loaded(
            generation=generation, layer_ids=loaded
        )
        return frozenset(loaded)

    @staticmethod
    def wait(works: Iterable[Any]) -> None:
        for item in works:
            work = getattr(item, "work", item)
            if work is not None:
                work.wait()
