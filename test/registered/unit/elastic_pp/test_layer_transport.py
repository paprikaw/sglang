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

    def send_tensor_dict(self, payload, dst, async_send):
        assert async_send
        self.mailbox.payloads[(self.rank, dst)] = payload
        return []

    def recv_tensor_dict(self, src):
        return self.mailbox.payloads.pop((src, self.rank))


class _Runner:
    def __init__(self, resident):
        self.resident = dict(resident)
        self.prepared = frozenset()

    def prepare_elastic_pp_layers(self, target, generation):
        del generation
        target_ids = set(range(target.start, target.end))
        self.prepared = frozenset(target_ids - self.resident.keys())
        return self.prepared

    def export_elastic_pp_layer_state(self, layer_ids):
        return {layer_id: self.resident[layer_id] for layer_id in layer_ids}

    def import_elastic_pp_layer_state(self, generation, layer_states):
        del generation
        assert set(layer_states) <= self.prepared
        self.resident.update(layer_states)
        return frozenset(layer_states)


def test_manifest_round_trip_and_generation_validation():
    transfer = LayerTransfer(0, 1, LayerRange(2, 3))
    states = {2: {"weight": torch.arange(4)}}
    payload = flatten_layer_states(generation=7, transfer=transfer, layer_states=states)

    decoded = unflatten_layer_states(payload, generation=7, transfer=transfer)
    assert torch.equal(decoded[2]["weight"], states[2]["weight"])
    with pytest.raises(ValueError, match="generation"):
        unflatten_layer_states(payload, generation=8, transfer=transfer)


def test_dedicated_transport_moves_only_local_transition_segment():
    transition = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 0, 0, 2], generation=0),
        PPTopology.from_layer_counts([1, 1, 1, 1], generation=1),
    )
    mailbox = _Mailbox()
    source_runner = _Runner({1: {"weight": torch.tensor([11.0])}})
    target_runner = _Runner({})
    source = ElasticPPLayerTransport(
        group=_Group(mailbox, 0), model_runner=source_runner, rank=0
    )
    target = ElasticPPLayerTransport(
        group=_Group(mailbox, 1), model_runner=target_runner, rank=1
    )

    assert target.prepare_target(transition) == frozenset({1})
    assert source.start_sends(transition) == ()
    assert target.receive(transition) == frozenset({1})
    assert torch.equal(target_runner.resident[1]["weight"], torch.tensor([11.0]))
