from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.marlin_cpu_weight_reload import (
    MarlinCpuWeightReloader,
    variable_layers,
)


class Qwen3_5Fixture:
    _canonical_cpu_weight_cache_key = "admitted-initial-model"


def fixture():
    states = {
        1: {
            "packed": torch.arange(8, dtype=torch.int32).reshape(2, 4),
            "scale": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        },
        2: {"packed": torch.arange(8, 16, dtype=torch.int32).reshape(2, 4)},
    }
    ready = []
    bank = NS(building=True, partitions=((2, 0, 2), (1, 2, 1)))
    args = NS(
        quantization="fp8",
        moe_runner_backend="marlin",
        tp_size=1,
        ep_size=1,
        elastic_pp_weight_staging="cpu_prepacked",
        elastic_pp_prepared_weight_policy="offload",
        elastic_pp_cpu_weight_cache_gib=1,
    )
    runner = NS(
        model=Qwen3_5Fixture(),
        server_args=args,
        ps=NS(pp_rank=1),
        _elastic_pp_prepared_graphs=bank,
        export_elastic_pp_layer_state=lambda ids: {i: states[i] for i in ids},
        mark_elastic_pp_layers_loaded=lambda **kw: ready.append(kw),
    )
    return (
        runner,
        states,
        ready,
        MarlinCpuWeightReloader(model_runner=runner, stream=None),
    )


def test_variable_layers_only_retains_weights_that_can_become_incoming():
    p = ((24, 0, 24), (16, 16, 16))
    assert variable_layers(p, 0) == set(range(16, 24))
    assert variable_layers(p, 1) == set(range(16, 32))
    assert variable_layers(p, 2) == set(range(24, 32))


def test_processed_integer_weights_and_scales_restore_without_repacking_or_address_changes():
    runner, states, ready, cache = fixture()
    original = {i: {n: t.clone() for n, t in s.items()} for i, s in states.items()}
    pointers = {i: {n: t.data_ptr() for n, t in s.items()} for i, s in states.items()}
    cache.capture([1, 2])
    for state in states.values():
        for t in state.values():
            t.zero_()
    runner._elastic_pp_prepared_graphs.building = False
    cache.reload(generation=1, layer_ids={1, 2})
    assert ready == [dict(generation=1, layer_ids={1, 2})]
    for i, state in states.items():
        for name, t in state.items():
            assert torch.equal(t, original[i][name])
            assert t.data_ptr() == pointers[i][name]
            assert cache.entries[i][name].device.type == "cpu"
            assert cache.entries[i][name].data_ptr() != t.data_ptr()
    metrics = cache.metrics_snapshot()
    assert metrics["h2d_bytes"] > 0
    assert metrics["startup_gpu_export_bytes"] == metrics["prepared_cpu_cache_bytes"]
    assert (
        metrics["gpu_export_bytes"]
        == metrics["gloo_weight_bytes"]
        == metrics["runtime_pack_seconds"]
        == 0
    )


def test_budget_rejection_does_not_publish_partial_cpu_cache():
    _, _, _, cache = fixture()
    cache.cap = 1
    with pytest.raises(RuntimeError, match="memory cap"):
        cache.capture([1, 2])
    assert not cache.entries and cache.nbytes == 0


def test_cpu_budget_counts_strided_storage_extent_instead_of_tensor_payload():
    _, states, _, cache = fixture()
    states[1]["packed"] = torch.arange(16, dtype=torch.int32).reshape(2, 8)[:, ::2]
    logical = sum(
        t.numel() * t.element_size()
        for state in states.values()
        for t in state.values()
    )
    cache.cap = logical
    with pytest.raises(RuntimeError, match="memory cap"):
        cache.capture([1, 2])
    assert not cache.entries


@pytest.mark.parametrize("fault", ["missing", "layout", "bank"])
def test_invalid_runtime_restore_never_marks_a_layer_ready(fault):
    runner, states, ready, cache = fixture()
    cache.capture([1] if fault == "missing" else [1, 2])
    runner._elastic_pp_prepared_graphs.building = False
    if fault == "layout":
        states[1]["packed"] = states[1]["packed"].t()
    if fault == "bank":
        runner.model._canonical_cpu_weight_cache_key = "different-model"
    with pytest.raises(RuntimeError):
        cache.reload(generation=1, layer_ids={1, 2})
    assert not ready


def test_startup_export_is_not_an_available_runtime_fallback():
    runner, _, _, cache = fixture()
    runner._elastic_pp_prepared_graphs.building = False
    with pytest.raises(RuntimeError, match="GPU weight export is disabled"):
        cache.capture([1])
