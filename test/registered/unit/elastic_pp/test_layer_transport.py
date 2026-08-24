import pytest
import torch

from sglang.srt.elastic_pp.layer_transport import (
    ElasticPPLayerTransport,
    flatten_layer_states,
    unflatten_layer_states,
)
from sglang.srt.elastic_pp.topology import (
    DynamicPPTransition,
    LayerRange,
    LayerTransfer,
    PPTopology,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Mailbox:
    def __init__(self):
        self.payloads = {}


class _Group:
    def __init__(self, mailbox: _Mailbox, rank: int):
        self.mailbox = mailbox
        self.rank = rank
        self.received_into = None

    def send_tensor_dict(self, payload, dst, async_send):
        assert async_send
        self.mailbox.payloads[(self.rank, dst)] = payload
        return []

    def recv_tensor_dict_into(self, destination_tensors, src, metadata_validator):
        payload = self.mailbox.payloads.pop((src, self.rank))
        metadata_validator(list(payload.items()))
        self.received_into = destination_tensors
        received = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                destination_tensors[key].copy_(value)
                received[key] = destination_tensors[key]
            else:
                received[key] = value
        return received


class _Runner:
    def __init__(self, resident, templates=None):
        self.resident = dict(resident)
        self.templates = {} if templates is None else templates
        self.prepared = frozenset()
        self.loaded = frozenset()

    def prepare_elastic_pp_layers(self, target, generation):
        del generation
        target_ids = set(range(target.start, target.end))
        self.prepared = frozenset(target_ids - self.resident.keys())
        for layer_id in self.prepared:
            self.resident[layer_id] = {
                name: torch.empty_like(tensor)
                for name, tensor in self.templates[layer_id].items()
            }
        return self.prepared

    def export_elastic_pp_layer_state(self, layer_ids):
        return {layer_id: self.resident[layer_id] for layer_id in layer_ids}

    def mark_elastic_pp_layers_loaded(self, generation, layer_ids):
        del generation
        assert set(layer_ids) <= self.prepared
        self.loaded = frozenset(layer_ids)


def test_manifest_round_trip_and_generation_validation():
    transfer = LayerTransfer(0, 1, LayerRange(2, 3))
    states = {2: {"weight": torch.arange(4)}}
    payload = flatten_layer_states(generation=7, transfer=transfer, layer_states=states)

    decoded = unflatten_layer_states(payload, generation=7, transfer=transfer)
    assert torch.equal(decoded[2]["weight"], states[2]["weight"])
    with pytest.raises(ValueError, match="generation"):
        unflatten_layer_states(payload, generation=8, transfer=transfer)


def test_noncontiguous_full_storage_uses_zero_copy_transport_view():
    transfer = LayerTransfer(0, 1, LayerRange(2, 3))
    physical = torch.arange(6).reshape(2, 3)
    logical = physical.t()
    payload = flatten_layer_states(
        generation=7, transfer=transfer, layer_states={2: {"weight": logical}}
    )
    transported = next(
        value for value in payload.values() if isinstance(value, torch.Tensor)
    )

    assert transported.is_contiguous()
    assert transported.shape == (6,)
    assert transported.data_ptr() == logical.data_ptr()
    decoded = unflatten_layer_states(payload, generation=7, transfer=transfer)
    assert decoded[2]["weight"].shape == logical.shape
    assert decoded[2]["weight"].stride() == logical.stride()
    assert decoded[2]["weight"].data_ptr() == logical.data_ptr()
    assert torch.equal(decoded[2]["weight"], logical)


def test_dedicated_transport_moves_only_local_transition_segment():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 0, 0, 2], generation=0),
        PPTopology.from_layer_counts([1, 1, 1, 1], generation=1),
    )
    mailbox = _Mailbox()
    source_runner = _Runner({1: {"weight": torch.tensor([11.0])}})
    target_runner = _Runner({}, templates={1: {"weight": torch.tensor([-1.0])}})
    source = ElasticPPLayerTransport(
        group=_Group(mailbox, 0), model_runner=source_runner, rank=0
    )
    target_group = _Group(mailbox, 1)
    target = ElasticPPLayerTransport(
        group=target_group, model_runner=target_runner, rank=1
    )

    assert target.prepare_target(transition) == frozenset({1})
    destination_ptr = target_runner.resident[1]["weight"].data_ptr()
    assert source.start_sends(transition) == ()
    assert target.receive(transition) == frozenset({1})
    assert target_runner.loaded == frozenset({1})
    assert torch.equal(target_runner.resident[1]["weight"], torch.tensor([11.0]))
    assert target_runner.resident[1]["weight"].data_ptr() == destination_ptr
    assert {tensor.data_ptr() for tensor in target_group.received_into.values()} == {
        destination_ptr
    }


def test_dedicated_transport_writes_noncontiguous_state_storage_directly():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 0], generation=0),
        PPTopology.from_layer_counts([1, 1], generation=1),
    )
    mailbox = _Mailbox()
    source_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    template_weight = torch.empty(2, 3).t()
    source_runner = _Runner({1: {"weight": source_weight}})
    target_runner = _Runner({}, templates={1: {"weight": template_weight}})
    source = ElasticPPLayerTransport(
        group=_Group(mailbox, 0), model_runner=source_runner, rank=0
    )
    target = ElasticPPLayerTransport(
        group=_Group(mailbox, 1), model_runner=target_runner, rank=1
    )

    target.prepare_target(transition)
    destination = target_runner.resident[1]["weight"]
    destination_ptr = destination.data_ptr()
    source.start_sends(transition)
    target.receive(transition)

    assert destination.data_ptr() == destination_ptr
    assert destination.stride() == source_weight.stride()
    assert torch.equal(destination, source_weight)
