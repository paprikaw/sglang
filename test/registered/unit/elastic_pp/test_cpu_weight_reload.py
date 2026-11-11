from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.elastic_pp import cpu_weight_reload as reload_module
from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
from sglang.srt.elastic_pp.cpu_weight_reload import CanonicalCpuWeightReloader
from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology
from sglang.srt.elastic_pp.weight_profile import WeightTensorMetadata
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Qwen3_5ReloadFixture(nn.Module):
    def __init__(self, checkpoint, events):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(2, 4, bias=False) for _ in range(2)]
        )
        self._canonical_cpu_weight_cache_key = "initial-bank"
        self.checkpoint, self.events = checkpoint, events
        self.fault = None
        self.marked = []

    def get_elastic_pp_weight_profile(self):
        return NS(
            checkpoint_tensors=tuple(
                WeightTensorMetadata.from_tensor(name, tensor, kind="checkpoint")
                for name, tensor in self.checkpoint.items()
            )
        )

    @contextmanager
    def elastic_pp_weight_load(self, generation, layer_ids):
        yield
        self.events.append("ready")
        self.marked.append((generation, layer_ids))

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            assert tensor.device.type == "cpu"
            if self.fault == "partial":
                break
            # Target is fused, like the native Q/K/V checkpoint loaders.
            target = self.model.layers[1].weight.data
            start = 0 if name.endswith("q_proj.weight") else 2
            target[start : start + 2].copy_(tensor)
            loaded.add("model.layers.1.weight")
            if self.fault == "copy":
                raise RuntimeError("injected copy failure")
        if self.fault == "coverage":
            loaded.clear()
        if self.fault == "storage":
            self.model.layers[1].weight = nn.Parameter(
                self.model.layers[1].weight.clone()
            )
        if self.fault == "stride":
            self.model.layers[1].weight.data = self.model.layers[1].weight.data.t()
        return loaded


@pytest.fixture
def bank(tmp_path, monkeypatch):
    import json

    tensors = {
        "model.layers.1.q_proj.weight": torch.arange(4).float().reshape(2, 2),
        "model.layers.1.k_proj.weight": torch.arange(4, 8).float().reshape(2, 2),
    }
    events = []
    model = Qwen3_5ReloadFixture(tensors, events)
    model.model.layers[1].weight.data.zero_()
    names = sorted(tensors)
    shards = [{"name": f"shard-{i}.safetensors"} for i in range(2)]
    for item, name in zip(shards, names):
        save_file({name: tensors[name]}, str(tmp_path / item["name"]))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: item["name"] for item, name in zip(shards, names)}}
        )
    )
    descriptor = {"group_key": "initial-bank", "shards": shards}

    @contextmanager
    def view(*args):
        events.append("pin")
        try:
            yield str(tmp_path), descriptor
        finally:
            events.append("unpin")

    monkeypatch.setattr(reload_module, "model_view", view)
    monkeypatch.setenv("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET", "/private/bank.sock")
    monkeypatch.setenv("SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT", "source")
    args = NS(
        tp_size=1,
        ep_size=1,
        moe_runner_backend="triton",
        quantization=None,
        elastic_pp_weight_staging="cpu_cache",
    )
    runner = NS(
        model=model,
        device="cpu",
        model_config=NS(model_path=str(tmp_path), quantization=None),
        server_args=args,
    )
    reloader = CanonicalCpuWeightReloader(model_runner=runner, stream=None)
    reloader._drain = lambda: events.append("drain")
    return NS(
        model=model,
        runner=runner,
        reloader=reloader,
        events=events,
        descriptor=descriptor,
        path=tmp_path,
    )


def test_direct_bank_restores_fused_target_without_touching_active_storage(bank):
    active = bank.model.model.layers[0].weight.clone()
    address = bank.model.model.layers[1].weight.data_ptr()
    bank.reloader.reload(generation=3, layer_ids={1})
    target = bank.model.model.layers[1].weight
    assert torch.equal(
        target,
        torch.cat(
            [
                bank.model.checkpoint["model.layers.1.q_proj.weight"],
                bank.model.checkpoint["model.layers.1.k_proj.weight"],
            ]
        ),
    )
    assert torch.equal(active, bank.model.model.layers[0].weight)
    assert address == target.data_ptr()
    assert bank.events == ["pin", "drain", "ready", "unpin"]
    assert bank.model.marked == [(3, frozenset({1}))]
    assert bank.reloader.metrics["checkpoint_bytes"] == 32
    assert bank.reloader.metrics["target_weight_bytes"] == 32
    assert bank.reloader.metrics["checkpoint_tensors"] == 2
    assert bank.reloader.metrics["storage_verified"]
    assert bank.reloader.metrics["gpu_export_bytes"] == 0
    assert bank.reloader.metrics["gloo_weight_bytes"] == 0


@pytest.mark.parametrize("fault", ["partial", "coverage", "copy", "storage", "stride"])
def test_failed_copy_coverage_or_address_never_publishes_and_drains_before_unpin(
    bank, fault
):
    bank.model.fault = fault
    with pytest.raises(RuntimeError):
        bank.reloader.reload(generation=1, layer_ids={1})
    assert bank.events == ["pin", "drain", "unpin"]
    assert not bank.model.marked
    assert not bank.reloader.metrics["storage_verified"]


@pytest.mark.parametrize("fault", ["key", "index", "shard", "shape", "dtype"])
def test_bad_bank_rejected_without_ready_or_fallback(bank, fault):
    import json

    if fault == "key":
        bank.descriptor["group_key"] = "different-model"
    elif fault == "index":
        (bank.path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    elif fault == "shard":
        p = bank.path / "model.safetensors.index.json"
        index = json.loads(p.read_text())
        index["weight_map"][
            "model.layers.1.q_proj.weight"
        ] = "../disk-fallback.safetensors"
        p.write_text(json.dumps(index))
    else:
        name = "model.layers.1.q_proj.weight"
        tensor = bank.model.checkpoint[name]
        tensor = tensor.reshape(1, 4) if fault == "shape" else tensor.to(torch.float64)
        save_file({name: tensor}, str(bank.path / "shard-1.safetensors"))
    with pytest.raises(RuntimeError):
        bank.reloader.reload(generation=1, layer_ids={1})
    assert "ready" not in bank.events
    assert bank.events[-1] == "unpin"
    assert not bank.model.marked
    assert bank.reloader.metrics["fallback_count"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 2),
        ("ep_size", 2),
        ("quantization", "fp8"),
        ("moe_runner_backend", "flashinfer_trtllm"),
    ],
)
def test_unqualified_backend_rejected_before_any_bank_or_gpu_operation(
    bank, field, value
):
    setattr(bank.runner.server_args, field, value)
    with pytest.raises(ValueError, match="native Qwen3.5"):
        CanonicalCpuWeightReloader(model_runner=bank.runner, stream=None)
    assert not bank.events


def test_existing_initial_bank_contract_is_required(bank, monkeypatch):
    monkeypatch.delenv("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET")
    with pytest.raises(RuntimeError, match="existing authenticated"):
        CanonicalCpuWeightReloader(model_runner=bank.runner, stream=None)
    assert not bank.events


@pytest.mark.parametrize("replace_storage", [False, True])
def test_only_target_native_postprocessing_runs_before_readiness(bank, replace_storage):
    calls = []

    def active_postprocess(layer):
        pytest.fail("must not reprocess active source weights")

    def target_postprocess(layer):
        calls.append("target")
        assert not bank.model.marked
        if replace_storage:
            layer.weight = nn.Parameter(layer.weight.clone())
        else:
            layer.weight.copy_(layer.weight.flip(0))

    bank.model.model.layers[0].quant_method = NS(
        process_weights_after_loading=active_postprocess
    )
    bank.model.model.layers[1].quant_method = NS(
        process_weights_after_loading=target_postprocess
    )
    if replace_storage:
        with pytest.raises(RuntimeError, match="changed prepared Graph"):
            bank.reloader.reload(generation=1, layer_ids={1})
        assert not bank.model.marked
    else:
        bank.reloader.reload(generation=1, layer_ids={1})
        expected = torch.cat(
            [
                bank.model.checkpoint["model.layers.1.q_proj.weight"],
                bank.model.checkpoint["model.layers.1.k_proj.weight"],
            ]
        ).flip(0)
        assert torch.equal(bank.model.model.layers[1].weight, expected)
        assert bank.reloader.metrics["postprocess_modules"] == 1
    assert calls == ["target"]


def test_cpu_cache_coordinator_only_loads_receiver_and_never_exports_or_sends(bank):
    source = PPTopology.from_layer_counts((2, 0), generation=0)
    target = PPTopology.from_layer_counts((1, 1), generation=1)
    transition = DynamicPPTransition.build(source=source, target=target)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "CPU cache path must not export GPU weights or use a weight collective"
        )

    bank.runner.export_elastic_pp_layer_state = forbidden
    group = NS(send_tensor_dict=forbidden, recv_tensor_dict=forbidden)
    for rank in (0, 1):
        coordinator = ElasticPPMigrationCoordinator(
            rank=rank,
            current_topology=source,
            group=group,
            model_runner=bank.runner,
            layer_transport=NS(_migration_stream=None),
            kv_transport=None,
            strategy="background",
        )
        coordinator._cpu_weight_reloader = bank.reloader
        coordinator._precopy_immutable_weights(transition)
        assert coordinator._background_weight_bytes_sent == 0
        assert len(bank.model.marked) == rank
