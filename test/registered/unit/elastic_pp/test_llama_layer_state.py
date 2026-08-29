import gc
import weakref
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.elastic_pp.layer_residency import ElasticLayerResidency
from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.llama import LlamaModel
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForCausalLM
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


@pytest.fixture(params=[LlamaModel, Qwen3_5MoeForCausalLM])
def model_class(request):
    return request.param


class _FakeQuantMethod:
    @staticmethod
    def process_weights_after_loading(module: nn.Module) -> None:
        module.weight = nn.Parameter(module.weight.t(), requires_grad=False)
        module.weight_scale = nn.Parameter(
            module.weight_scale.max(), requires_grad=False
        )


class _QuantLinear(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 3), value))
        self.weight_scale = nn.Parameter(torch.full((2,), value))
        self.quant_method = _FakeQuantMethod()


class _QuantLayer(nn.Module):
    def __init__(self, value: float, start_layer: int, *, processed: bool):
        super().__init__()
        self.linear = _QuantLinear(value)
        self.self_attn = _Attention(start_layer)
        if processed:
            self.linear.quant_method.process_weights_after_loading(self.linear)


def _model(values: list[float | None], active: LayerRange, model_class):
    model = model_class.__new__(model_class)
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


def test_runtime_layer_state_moves_before_atomic_range_commit(model_class):
    source = _model([10.0, 20.0], LayerRange(0, 2), model_class)
    target = _model([10.0, None], LayerRange(0, 1), model_class)

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


def test_retired_layer_is_detached_only_in_post_commit_cleanup(model_class):
    model = _model([10.0, 20.0], LayerRange(0, 2), model_class)
    # Real linear/MoE parameters keep a bound loader (or a closure capturing
    # the owner). Frozen GC cannot reclaim that module -> parameter -> module
    # cycle, so cleanup must break it without relying on a collection.
    model.layers[0].weight.weight_loader = model.layers[0].forward
    model.layers[0].weight._weight_loader = model.layers[0].forward
    retired_weight = weakref.ref(model.layers[0].weight)
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
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del retired
        assert retired_weight() is None
    finally:
        if was_enabled:
            gc.enable()


def test_aborted_layer_releases_loader_reference_cycle(model_class):
    model = _model([10.0, None], LayerRange(0, 1), model_class)
    model.prepare_elastic_pp_layers(
        target=LayerRange(0, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.layers[1].weight.weight_loader = lambda layer=model.layers[1]: layer.weight
    prepared_weight = weakref.ref(model.layers[1].weight)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        model.abort_elastic_pp_layers(1)
        assert prepared_weight() is None
        assert torch.equal(model.layers[0].weight, torch.full((3,), 10.0))
    finally:
        if was_enabled:
            gc.enable()


def test_quantized_layer_is_materialized_in_runtime_layout(model_class):
    quant_name = "compressed_tensors" if model_class is LlamaModel else "moe_wna16"
    source = _model([10.0, 20.0], LayerRange(0, 2), model_class)
    source.quant_config = SimpleNamespace(get_name=lambda: quant_name)
    source.layers[1] = _QuantLayer(20.0, 0, processed=True)

    target = _model([10.0, None], LayerRange(0, 1), model_class)
    target.quant_config = SimpleNamespace(get_name=lambda: quant_name)
    target._elastic_layer_factory = lambda layer_id, start_layer: _QuantLayer(
        -1.0 - layer_id, start_layer, processed=False
    )
    new_layers = target.prepare_elastic_pp_layers(
        target=LayerRange(0, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    target_linear = target.layers[1].linear
    source_linear = source.layers[1].linear
    assert target_linear.weight.shape == source_linear.weight.shape == (3, 2)
    assert not target_linear.weight.is_contiguous()
    assert target_linear.weight_scale.shape == source_linear.weight_scale.shape == ()
    transferred = source.export_elastic_pp_layer_state(new_layers)
    target.import_elastic_pp_layer_state(generation=1, layer_states=transferred)
    assert torch.equal(target_linear.weight, source_linear.weight)
    assert torch.equal(target_linear.weight_scale, source_linear.weight_scale)
