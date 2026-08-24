"""Direct owner-to-owner model-layer transfer over a dedicated PP group."""

from __future__ import annotations

from collections.abc import Iterable
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
            payload[key] = (
                tensor.detach()
                if tensor.is_contiguous()
                else tensor.detach().contiguous()
            )
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


def unflatten_layer_states(
    payload: dict[str, Any],
    *,
    generation: int,
    transfer: LayerTransfer,
) -> dict[int, dict[str, torch.Tensor]]:
    manifest = payload.get(_MANIFEST_KEY)
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

    entries = manifest.get("entries")
    if not isinstance(entries, (tuple, list)):
        raise ValueError("elastic PP layer manifest entries are invalid")
    states: dict[int, dict[str, torch.Tensor]] = {}
    expected_tensor_keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise ValueError(f"invalid elastic PP layer manifest entry: {entry!r}")
        key, layer_id, state_name = entry
        if not isinstance(key, str) or not isinstance(state_name, str):
            raise ValueError(f"invalid elastic PP layer manifest entry: {entry!r}")
        layer_id = int(layer_id)
        tensor = payload.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"elastic PP layer tensor {key!r} is missing")
        expected_tensor_keys.add(key)
        layer_state = states.setdefault(layer_id, {})
        if state_name in layer_state:
            raise ValueError(
                f"duplicate elastic PP layer state {layer_id}:{state_name}"
            )
        layer_state[state_name] = tensor

    unexpected_keys = set(payload) - expected_tensor_keys - {_MANIFEST_KEY}
    if unexpected_keys:
        raise ValueError(
            f"unexpected elastic PP layer payload keys: {sorted(unexpected_keys)}"
        )
    expected_layers = set(range(transfer.layer_range.start, transfer.layer_range.end))
    if set(states) != expected_layers:
        raise ValueError(
            "received layer states do not exactly cover the transfer: "
            f"expected={sorted(expected_layers)}, got={sorted(states)}"
        )
    return states


class ElasticPPLayerTransport:
    """Execute the weight portion of one ``DynamicPPTransition``.

    The communicator is intentionally distinct from the live PP proxy group,
    so background weight traffic cannot reorder serving send/recv operations.
    """

    def __init__(self, *, group, model_runner, rank: int) -> None:
        self.group = group
        self.model_runner = model_runner
        self.rank = int(rank)

    @staticmethod
    def _sort_key(transfer: LayerTransfer) -> tuple[int, int, int, int]:
        return (
            transfer.src_rank,
            transfer.dst_rank,
            transfer.layer_range.start,
            transfer.layer_range.end,
        )

    def prepare_target(self, transition: DynamicPPTransition) -> frozenset[int]:
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
        for transfer in self.inbound_transfers(transition):
            payload = self.group.recv_tensor_dict(src=transfer.src_rank)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"did not receive layer state from rank {transfer.src_rank}"
                )
            layer_states = unflatten_layer_states(
                payload,
                generation=transition.target.generation,
                transfer=transfer,
            )
            loaded.update(
                self.model_runner.import_elastic_pp_layer_state(
                    generation=transition.target.generation,
                    layer_states=layer_states,
                )
            )
        return frozenset(loaded)

    @staticmethod
    def wait(works: Iterable[Any]) -> None:
        for item in works:
            work = getattr(item, "work", item)
            if work is not None:
                work.wait()
