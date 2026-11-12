from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.elastic_pp import cpu_weight_reload as reload_module
from sglang.srt.elastic_pp.cpu_weight_reload import CanonicalCpuWeightReloader
from sglang.srt.elastic_pp.prepared_cpu_weights import (
    CpuOnlyPacking,
    PreparedCpuWeights,
)
from sglang.srt.elastic_pp.weight_profile import WeightTensorMetadata
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Qwen3_5CpuPackingFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(2, 4, bias=False)])
        layer = self.model.layers[0]
        layer.alias = nn.Module()
        layer.alias.weight = layer.weight
        self._canonical_cpu_weight_cache_key = "admitted-bank"
        self.checkpoint = {
            "model.layers.0.q_proj.weight": torch.arange(4).float().view(2, 2),
            "model.layers.0.k_proj.weight": torch.arange(4, 8).float().view(2, 2),
        }
        self.loads = 0
        self.ready = []
        self.fault = None

    def get_elastic_pp_weight_profile(self):
        return NS(
            checkpoint_tensors=[
                WeightTensorMetadata.from_tensor(name, tensor, kind="checkpoint")
                for name, tensor in self.checkpoint.items()
            ]
        )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters(remove_duplicate=False))
        param = parameters["model.layers.0.weight"]
        for name, value in weights:
            start = 0 if name.endswith("q_proj.weight") else 2
            param.data[start : start + 2].copy_(value)
        if self.fault == "missing":
            return set()
        return {"model.layers.0.weight"}

    @contextmanager
    def elastic_pp_weight_load(self, generation, layer_ids):
        yield
        self.ready.append(generation)


@pytest.fixture
def pack_bank(tmp_path, monkeypatch):
    model = Qwen3_5CpuPackingFixture()
    save_file(model.checkpoint, str(tmp_path / "model.safetensors"))

    @contextmanager
    def view(*args):
        yield str(tmp_path), dict(
            group_key="admitted-bank", shards=[{"name": "model.safetensors"}]
        )

    monkeypatch.setattr(reload_module, "model_view", view)
    monkeypatch.setenv("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET", "/private/socket")
    monkeypatch.setenv("SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT", "source")
    runner = NS(
        model=model,
        device="cpu",
        model_config=NS(model_path=str(tmp_path), quantization=None),
        server_args=NS(
            tp_size=1,
            ep_size=1,
            quantization=None,
            moe_runner_backend="triton",
            elastic_pp_weight_staging="cpu_prepacked",
        ),
        _elastic_pp_prepared_graphs=NS(building=True),
    )
    return NS(
        model=model,
        runner=runner,
        loader=CanonicalCpuWeightReloader(model_runner=runner, stream=None),
    )


def test_prepare_on_cpu_preserves_live_model_aliases_and_addresses(pack_bank):
    b = pack_bank
    parameters = dict(b.model.named_parameters(remove_duplicate=False))
    original = b.model.model.layers[0].weight.detach().clone()
    address = b.model.model.layers[0].weight.data_ptr()
    cache = PreparedCpuWeights(cache_key="key", max_bytes=32, pinned=False)
    counters = dict(checkpoint_bytes=32, checkpoint_tensors=2)
    entry = cache.prepare(
        model=b.model,
        layer_ids={0},
        parameters=parameters,
        weights=iter(b.model.checkpoint.items()),
        counters=counters,
        expected_counts=(32, 2),
    )
    assert torch.equal(b.model.model.layers[0].weight, original)
    assert b.model.model.layers[0].weight.data_ptr() == address
    assert (
        entry.parameters["model.layers.0.weight"]
        is entry.parameters["model.layers.0.alias.weight"]
    )
    assert cache.nbytes == 32
    assert not b.model.ready
    with torch.no_grad():
        metrics = cache.copy(entry, parameters, None)
    assert metrics["h2d_copy_calls"] == 1
    assert metrics["h2d_bytes"] == 32
    assert b.model.model.layers[0].weight.data_ptr() == address
    assert torch.equal(
        b.model.model.layers[0].weight, torch.arange(8).float().view(4, 2)
    )


def test_warm_reload_uses_only_cpu_cache_and_does_not_repack(pack_bank, monkeypatch):
    b = pack_bank
    b.loader.reload(generation=1, layer_ids={0})
    assert b.loader.metrics["canonical_checkpoint_read_bytes"] == 32
    assert b.loader.metrics["cpu_prepare_seconds"] > 0
    assert not b.loader.metrics["prepared_cpu_cache_hit"]
    b.model.model.layers[0].weight.data.zero_()
    b.runner._elastic_pp_prepared_graphs.building = False
    monkeypatch.setattr(
        Qwen3_5CpuPackingFixture,
        "load_weights",
        lambda *args: pytest.fail("must not call native loader after startup"),
    )
    b.loader.reset_metrics()
    b.loader.reload(generation=2, layer_ids={0})
    assert b.loader.metrics["prepared_cpu_cache_hit"]
    assert b.loader.metrics["canonical_checkpoint_read_bytes"] == 0
    assert b.loader.metrics["cpu_prepare_seconds"] == 0
    assert (
        b.loader.metrics["gpu_export_bytes"]
        == b.loader.metrics["gloo_weight_bytes"]
        == 0
    )
    assert b.loader.metrics["storage_verified"]
    assert b.loader.metrics_snapshot()["prepared_cpu_cache_bytes"] == 32
    assert b.model.ready == [1, 2]
    assert torch.equal(
        b.model.model.layers[0].weight, torch.arange(8).float().view(4, 2)
    )


def test_runtime_cache_miss_fails_without_readiness(pack_bank):
    pack_bank.runner._elastic_pp_prepared_graphs.building = False
    with pytest.raises(RuntimeError, match="before serving"):
        pack_bank.loader.reload(generation=1, layer_ids={0})
    assert not pack_bank.model.ready
    assert pack_bank.loader.prepared.nbytes == 0


def test_cap_is_enforced_before_allocation_or_model_loading(pack_bank, monkeypatch):
    b = pack_bank
    b.loader.prepared.max_bytes = 31
    monkeypatch.setattr(
        torch,
        "empty_strided",
        lambda *args, **kwargs: pytest.fail("allocation must not happen"),
    )
    with pytest.raises(RuntimeError, match="memory cap"):
        b.loader.reload(generation=1, layer_ids={0})
    assert not b.model.ready
    assert not b.loader.prepared.entries


def test_failed_native_cpu_coverage_does_not_cache_or_publish(pack_bank):
    b = pack_bank
    b.model.fault = "missing"
    original = b.model.model.layers[0].weight.detach().clone()
    with pytest.raises(RuntimeError, match="cover every"):
        b.loader.reload(generation=1, layer_ids={0})
    assert b.loader.prepared.nbytes == 0
    assert not b.loader.prepared.entries
    assert not b.model.ready
    assert torch.equal(b.model.model.layers[0].weight, original)


def test_prepared_layout_change_rejected_before_copy(pack_bank):
    b = pack_bank
    b.loader.reload(generation=1, layer_ids={0})
    b.loader.reset_metrics()
    b.model.model.layers[0].weight.data = b.model.model.layers[0].weight.data.t()
    with pytest.raises(RuntimeError, match="layout changed"):
        b.loader.reload(generation=2, layer_ids={0})
    assert b.model.ready == [1]


def test_distinct_unloaded_parameter_is_not_hidden_by_storage_alias(pack_bank):
    b = pack_bank
    layer = b.model.model.layers[0]
    layer.alias.weight = nn.Parameter(layer.weight.detach())
    with pytest.raises(RuntimeError, match="cover every"):
        b.loader.reload(generation=1, layer_ids={0})
    assert not b.model.ready


def test_cpu_packing_rejects_gpu_allocation_without_initializing_cuda():
    with pytest.raises(RuntimeError, match="non-CPU"):
        with CpuOnlyPacking():
            torch.empty(4, device="meta")


@pytest.mark.parametrize("cap", [0, -1])
def test_positive_explicit_cap_required(cap):
    with pytest.raises(ValueError, match="positive cap"):
        PreparedCpuWeights(cache_key="key", max_bytes=cap, pinned=False)
