from threading import RLock

import torch
from torch import nn

from sglang.srt.elastic_pp.layer_residency import ElasticLayerResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.llama import LlamaModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Attention(nn.Module):
    def __init__(self, start_layer: int):
        super().__init__()
        self.start_layer = start_layer


class _Layer(nn.Module):
    def __init__(self, value: float, start_layer: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), value))
        self.self_attn = _Attention(start_layer)


def _model(values: list[float | None], active: LayerRange) -> LlamaModel:
    model = LlamaModel.__new__(LlamaModel)
    nn.Module.__init__(model)
    model.quant_config = None
    model.layers = nn.ModuleList(
        [
            PPMissingLayer() if value is None else _Layer(value, active.start)
            for value in values
        ]
    )
    model.start_layer = active.start
    model.end_layer = active.end
    model._elastic_layer_lock = RLock()
    model._elastic_layer_residency = ElasticLayerResidency(
        num_layers=len(values),
        initial_generation=0,
        initial_range=active,
    )
    model._elastic_layer_factory = lambda layer_id, start_layer: _Layer(
        -1.0 - layer_id, start_layer
    )
    return model


def test_runtime_layer_state_moves_before_atomic_range_commit():
    source = _model([10.0, 20.0], LayerRange(0, 2))
    target = _model([10.0, None], LayerRange(0, 1))

    new_layers = target.prepare_elastic_pp_layers(
        target=LayerRange(0, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert new_layers == frozenset({1})
    assert target.get_elastic_pp_layer_range() == LayerRange(0, 1)
    assert torch.equal(target.layers[1].weight, torch.full((3,), -2.0))

    transferred = source.export_elastic_pp_layer_state(new_layers)
    assert target.import_elastic_pp_layer_state(
        generation=1, layer_states=transferred
    ) == frozenset({1})
    assert torch.equal(target.layers[1].weight, source.layers[1].weight)
    assert target.elastic_pp_layers_ready(1)

    assert target.commit_elastic_pp_layers(1) == LayerRange(0, 2)
    assert target.start_layer == 0
    assert target.end_layer == 2
    assert target.take_elastic_pp_cleanup_layers(1) == ()


def test_retired_layer_is_detached_only_in_post_commit_cleanup():
    model = _model([10.0, 20.0], LayerRange(0, 2))
    assert not model.prepare_elastic_pp_layers(
        target=LayerRange(1, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert model.commit_elastic_pp_layers(1) == LayerRange(1, 2)
    assert not isinstance(model.layers[0], PPMissingLayer)
    retired = model.take_elastic_pp_cleanup_layers(1)
    assert len(retired) == 1
    assert isinstance(model.layers[0], PPMissingLayer)
