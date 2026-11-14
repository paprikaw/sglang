"""Prepared Graph storage, finite admission and no-runtime-capture contracts."""

from threading import RLock
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from sglang.srt.elastic_pp.decode_graph import (
    ElasticPPDecodeGraphState,
    rebuild_decode_graph,
    retire_decode_graph,
)
from sglang.srt.elastic_pp.layer_residency import ElasticLayerResidency
from sglang.srt.elastic_pp.model import ElasticPPModelMixin
from sglang.srt.elastic_pp.model_wrapper import ElasticPPModelWrapperMixin
from sglang.srt.elastic_pp.prepared_decode_graph import (
    PreparedDecodeGraphBank,
    external_bindings,
)
from sglang.srt.elastic_pp.topology import DynamicPPTransition, LayerRange, PPTopology
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def args(**overrides):
    options = dict(
        model_path="unused",
        pp_size=3,
        enable_elastic_pp=True,
        elastic_pp_initial_partition="20,0,20",
        elastic_pp_graph_partitions="20,0,20;13,13,14",
        disable_cuda_graph=False,
    )
    options.update(overrides)
    result = ServerArgs(**options)
    result.get_model_config = lambda: NS(
        hf_config=NS(architectures=["LlamaForCausalLM"])
    )
    return result


@pytest.mark.parametrize(
    "change,match",
    [
        ({"enable_elastic_pp": False}, "requires --enable-elastic-pp"),
        ({"disable_cuda_graph": True}, "Graph enabled"),
        ({"elastic_pp_graph_partitions": "13,13,14;12,14,14"}, "include the initial"),
        ({"elastic_pp_graph_partitions": "20,0,20;20,20,0"}, "fixed first and last"),
        ({"elastic_pp_graph_partitions": "20,0,20;13,13,15"}, "same resident"),
        ({"enable_lora": True}, "LoRA"),
        ({"enable_pdmux": True}, "PDMux"),
    ],
)
def test_invalid_prepared_graph_config_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        args(**change)._handle_pipeline_parallelism()


def test_prepared_graph_config_projects_and_rejects_unknown_target():
    cfg = args()
    cfg._handle_pipeline_parallelism()
    assert cfg._resolved().elastic_pp_graph_partitions == "20,0,20;13,13,14"
    bank = PreparedDecodeGraphBank(NS(ps=NS(pp_rank=0)), ((20, 0, 20), (13, 13, 14)))
    bank.check_topology(PPTopology.from_layer_counts((13, 13, 14)))
    with pytest.raises(ValueError, match="not prepared"):
        bank.check_topology(PPTopology.from_layer_counts((14, 12, 14)))


def test_static_kvcached_replica_can_prepare_a_single_layout():
    cfg = args(pp_size=2, elastic_pp_initial_partition="20,20",
               elastic_pp_graph_partitions="20,20", elastic_pp_kv_backend="kvcached",
               elastic_pp_kv_capacities="8192", max_total_tokens=8192,
               disaggregation_mode="decode", elastic_pp_disagg_decode=True,
               enable_hierarchical_cache=True, pp_max_micro_batch_size=1,
               max_running_requests=1, elastic_pp_transition_strategy="background",
               elastic_pp_prepared_weight_policy="offload", elastic_pp_hicache_shared_host=True,
               elastic_pp_weight_staging="cpu_prepacked", enable_elastic_pp_kv_patch=True,
               elastic_pp_kv_patch_transport="nixl", page_size=1)
    cfg._handle_pipeline_parallelism()
    assert cfg._resolved().elastic_pp_graph_partitions == "20,20"


@pytest.mark.parametrize(
    "weight_source", ["legacy", "pinned", "cpu_cache", "cpu_prepacked"]
)
def test_private_startup_coordinator_disables_online_patch_without_changing_args(
    weight_source,
):
    from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator

    cfg = NS(
        enable_elastic_pp_kv_patch=True,
        elastic_pp_transition_strategy="background",
        elastic_pp_transition_backing="overlap",
        elastic_pp_weight_staging=weight_source,
    )
    options = dict(
        rank=0,
        current_topology=PPTopology.from_layer_counts((2, 0, 2)),
        group=None,
        model_runner=NS(server_args=cfg),
        layer_transport=None,
        kv_transport=None,
    )
    startup = ElasticPPMigrationCoordinator(
        **options, strategy="sync", enable_kv_patch=False
    )
    serving = ElasticPPMigrationCoordinator(**options)
    assert not startup.enable_kv_patch
    assert serving.enable_kv_patch and cfg.enable_elastic_pp_kv_patch


@pytest.mark.parametrize("fail", [False, True])
def test_empty_startup_keeps_endpoints_alive_and_restores_on_error(fail):
    from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport

    transport = object.__new__(ElasticPPKVTransport)
    endpoint, receiver = object(), NS(generation=None)
    transport._nixl_endpoint, transport._nixl_receiver = endpoint, receiver
    transport._live_patch = None
    try:
        with transport.empty_startup_transport():
            assert transport._nixl_endpoint is transport._nixl_receiver is None
            if fail:
                raise ValueError("startup capture failed")
    except ValueError:
        assert fail
    assert transport._nixl_endpoint is endpoint
    assert transport._nixl_receiver is receiver
    assert receiver.generation is None


def test_startup_cannot_suspend_a_serving_nixl_generation():
    from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport

    transport = object.__new__(ElasticPPKVTransport)
    transport._nixl_endpoint = object()
    transport._nixl_receiver = NS(generation=1)
    transport._live_patch = None
    with pytest.raises(RuntimeError, match="live NIXL"):
        with transport.empty_startup_transport():
            pytest.fail("serving generation suspended")
    assert transport._nixl_receiver.generation == 1


class Model(ElasticPPModelMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.start_layer, self.end_layer = 0, 2
        self._elastic_layer_lock = RLock()
        self._elastic_layer_residency = ElasticLayerResidency(
            num_layers=2, initial_generation=0, initial_range=LayerRange(0, 2)
        )
        self._elastic_layer_factory = lambda *_: pytest.fail(
            "cached layer was reallocated"
        )
        self.quant_config = None


def test_parameter_storage_survives_retirement_abort_and_return():
    model = Model()
    model.retain_elastic_pp_graph_layers()
    original = model.layers[0]
    original.weight.weight_loader = original.forward
    ptr = original.weight.data_ptr()
    model.prepare_elastic_pp_layers(
        target=LayerRange(1, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.commit_elastic_pp_layers(1)
    assert model.take_elastic_pp_cleanup_layers(1) == ()
    assert isinstance(model.layers[0], PPMissingLayer)
    for generation, abort in ((2, True), (2, False)):
        model.prepare_elastic_pp_layers(
            target=LayerRange(0, 2),
            generation=generation,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert model.layers[0].weight.data_ptr() == ptr
        if abort:
            assert model.abort_elastic_pp_layers(generation) == ()
        else:
            model.mark_elastic_pp_layers_loaded(generation, {0})
            model.commit_elastic_pp_layers(generation)
            model.take_elastic_pp_cleanup_layers(generation)
    assert hasattr(model.layers[0].weight, "weight_loader")


def test_nonresident_cleanup_and_abort_release_only_incoming_weights(monkeypatch):
    model = Model()
    model.retain_elastic_pp_graph_layers()
    mapped = {0, 1}
    calls = []

    def evict(ids):
        calls.append(("evict", set(ids)))
        mapped.difference_update(ids)

    def prepare(layer_id, layer):
        calls.append(("map", layer_id))
        mapped.add(layer_id)

    model._elastic_pp_weight_backing = NS(prepare=prepare, evict=evict)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model.prepare_elastic_pp_layers(
        target=LayerRange(1, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.commit_elastic_pp_layers(1)
    model.take_elastic_pp_cleanup_layers(1)
    assert mapped == {1}
    model.prepare_elastic_pp_layers(
        target=LayerRange(0, 2),
        generation=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mapped == {0, 1}
    model.abort_elastic_pp_layers(2)
    assert mapped == {1}
    assert isinstance(model.layers[0], PPMissingLayer)
    assert calls == [("evict", {0}), ("map", 0), ("evict", {0})]


def test_failed_cached_weight_allocation_keeps_native_loader_for_retry(monkeypatch):
    model = Model()
    model.retain_elastic_pp_graph_layers()
    original = model.layers[0]
    original.weight.weight_loader = original.forward
    loader = original.weight.weight_loader
    model.prepare_elastic_pp_layers(
        target=LayerRange(1, 2),
        generation=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.commit_elastic_pp_layers(1)
    model.take_elastic_pp_cleanup_layers(1)

    def fail(*args):
        raise RuntimeError("injected target mapping failure")

    model._elastic_pp_weight_backing = NS(prepare=fail, evict=lambda ids: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="mapping failure"):
        model.prepare_elastic_pp_layers(
            target=LayerRange(0, 2),
            generation=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    assert original.weight.weight_loader is loader
    assert isinstance(model.layers[0], PPMissingLayer)
    model._elastic_pp_weight_backing.prepare = lambda *args: None
    model.prepare_elastic_pp_layers(
        target=LayerRange(0, 2),
        generation=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert model.layers[0] is original
    assert model.layers[0].weight.weight_loader is loader
    model.abort_elastic_pp_layers(2)


def test_fixed_output_frontier_drops_spare_heads_without_touching_decoder_weights():
    class Wrapper(ElasticPPModelWrapperMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()
            self.model.norm = nn.LayerNorm(2)
            self.lm_head = nn.Linear(2, 4, bias=False)

    wrapper = Wrapper()
    layers = tuple(wrapper.model.layers.parameters())
    weight_ptrs = [t.data_ptr() for t in layers]
    assert wrapper.discard_elastic_pp_unused_output_weights() == (8 + 4) * 4
    assert isinstance(wrapper.lm_head, PPMissingLayer)
    assert isinstance(wrapper.model.norm, PPMissingLayer)
    assert [t.data_ptr() for t in wrapper.model.layers.parameters()] == weight_ptrs


def test_kv_gdn_addresses_return_but_stale_values_are_overwritten(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=3)
    pool.retain_elastic_pp_graph_buffers()
    first = pool.mamba_pool.mamba_cache
    full = pool.full_kv_pool
    ptrs = (
        first.temporal.data_ptr(),
        full.k_buffer[0].data_ptr(),
        full.data_ptrs.data_ptr(),
    )
    for generation, target in (
        (1, LayerRange(2, 4)),
        (2, LayerRange(0, 3)),
        (3, LayerRange(2, 4)),
    ):
        incoming = pool.prepare_elastic_pp_kv_layers(
            target=target, generation=generation, total_num_layers=4
        )
        # The retained recurrent layer changes after background preparation.
        current_idx = pool.req_to_token_pool.mamba_map[2]
        pool.mamba_pool.mamba_cache.temporal[current_idx].fill_(40 + generation)
        for layer in incoming:
            if layer in (1, 3):
                pool.import_elastic_pp_kv_slots(
                    generation=generation,
                    layer_id=layer,
                    slot_ids=torch.tensor([2]),
                    cache_k=torch.full((1, 1, 2), float(generation)),
                    cache_v=torch.full((1, 1, 2), float(generation)),
                )
            else:
                pool.import_recurrent_slots(
                    generation=generation,
                    layer_id=layer,
                    slot_ids=torch.tensor([1]),
                    tensors={
                        "conv_0": torch.full(
                            (1, 2, 3), generation, dtype=torch.bfloat16
                        ),
                        "temporal": torch.full((1, 2, 3, 3), float(generation)),
                    },
                )
        pool.mark_elastic_pp_kv_layers_loaded(
            generation=generation, layer_ids=set(incoming)
        )
        pool.commit_elastic_pp_kv_layers(generation=generation)
        assert (
            pool.mamba_pool.mamba_cache.temporal[
                pool.req_to_token_pool.mamba_map[2], 1, 0, 0, 0
            ]
            == 40 + generation
        )
        assert pool.cleanup_elastic_pp_kv_layers(generation=generation) == ()
        if generation == 2:
            assert (
                pool.mamba_pool.mamba_cache.temporal.data_ptr(),
                full.k_buffer[0].data_ptr(),
                full.data_ptrs.data_ptr(),
            ) == ptrs
            assert pool.mamba_pool.mamba_cache.temporal[0, 1, 0, 0, 0] == 2
            assert full.k_buffer[0][2, 0, 0] == 2


@pytest.fixture
def prepared_runner(monkeypatch):
    import sglang.srt.distributed.device_communicators.pynccl_allocator as nccl
    import sglang.srt.model_executor.runner_utils.pool as pool_module

    monkeypatch.setattr(pool_module, "get_global_graph_memory_pool", lambda: (1, 1))
    monkeypatch.setattr(pool_module, "set_global_graph_memory_pool", lambda _: None)
    monkeypatch.setattr(nccl, "set_graph_pool_id", lambda _: None)
    model = Model()
    state = ElasticPPDecodeGraphState()
    state.begin(0).update(status="ready")
    graph = NS(buffers=NS(ids=torch.zeros(2, dtype=torch.int64)))
    mr = NS(
        ps=NS(pp_rank=0),
        model=model,
        server_args=NS(disable_cuda_graph=False),
        req_to_token_pool=NS(req_to_token=torch.zeros(3, 3, dtype=torch.int32)),
        token_to_kv_pool=NS(k_buffer=[torch.zeros(4)], v_buffer=[torch.zeros(4)]),
        attn_backend=object(),
        decode_cuda_graph_runner=graph,
        prefill_cuda_graph_runner=None,
        _elastic_pp_decode_graph_state=state,
    )
    bank = mr._elastic_pp_prepared_graphs = PreparedDecodeGraphBank(
        mr, ((2, 0, 2), (2, 1, 1))
    )
    bank.remember(mr, 0.12)
    bank.seal()
    mr.init_decode_cuda_graph = lambda: pytest.fail("runtime capture")
    return mr, bank, graph


def test_runtime_selection_reuses_exact_runner_and_never_captures(prepared_runner):
    mr, bank, graph = prepared_runner
    retire_decode_graph(mr)
    assert mr.decode_cuda_graph_runner is None
    rebuild_decode_graph(mr, generation=1, active=True)
    assert mr.decode_cuda_graph_runner is graph
    mr._elastic_pp_decode_graph_state.observe(True)
    entry = mr._elastic_pp_decode_graph_state.history[-1]
    assert entry["capture_seconds"] == 0 and entry["replays"] == 1
    assert bank.metrics()["runtime_selections"] == 1


@pytest.mark.parametrize("fault", [None, "address", "version", "missing_state"])
def test_resident_weights_validate_before_marking_loaded(prepared_runner, fault):
    mr, bank, _ = prepared_runner
    model = mr.model
    # Construct a return to layers 0/1 after layer 0 was logically retired.
    source = PPTopology.from_layer_counts([0, 2], generation=1)
    target = PPTopology.from_layer_counts([2, 0], generation=2)
    transition = DynamicPPTransition.build(source, target)
    bank.partitions = ((0, 2), (2, 0))
    marked = []
    mr.export_elastic_pp_layer_state = model.export_elastic_pp_layer_state
    mr.mark_elastic_pp_layers_loaded = lambda **kw: marked.append(kw)
    if fault == "address":
        model.layers[0].weight = nn.Parameter(model.layers[0].weight.clone())
    elif fault == "version":
        with torch.no_grad():
            model.layers[1].weight.add_(1)
    elif fault == "missing_state":
        model.layers[0].register_parameter("bias", None)
    if fault:
        with pytest.raises(RuntimeError, match="prepared weight"):
            bank.reuse_weights(mr, transition)
        assert marked == []
    else:
        assert bank.reuse_weights(mr, transition) == 48
        assert marked == [{"generation": 2, "layer_ids": {0, 1}}]


def test_prepared_candidates_reject_online_weight_mutation_before_loading():
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )

    manager = NS(
        tp_worker=NS(
            model_runner=NS(server_args=NS(elastic_pp_graph_partitions="2,0;1,1"))
        )
    )
    error = SchedulerWeightUpdaterManager._cpu_weight_cache_online_update_error(manager)
    assert "immutable weight version" in error


@pytest.mark.parametrize("fault", ["weight", "kv", "request", "input", "interval"])
def test_changed_address_or_uncaptured_interval_cannot_replay(prepared_runner, fault):
    mr, bank, graph = prepared_runner
    retire_decode_graph(mr)
    if fault == "weight":
        mr.model.layers[0].weight = nn.Parameter(mr.model.layers[0].weight.clone())
    elif fault == "kv":
        mr.token_to_kv_pool.k_buffer[0] = mr.token_to_kv_pool.k_buffer[0].clone()
    elif fault == "request":
        mr.req_to_token_pool.req_to_token = mr.req_to_token_pool.req_to_token.clone()
    elif fault == "input":
        graph.buffers.ids = graph.buffers.ids.clone()
    else:
        mr.model.end_layer = 1
    with pytest.raises(RuntimeError, match="bindings changed|not prepared"):
        rebuild_decode_graph(mr, generation=1, active=True)
    assert mr.decode_cuda_graph_runner is None
    assert mr._elastic_pp_decode_graph_state.history[-1]["status"] == "failed"


def test_retired_registered_layers_are_not_captured_bindings():
    mr = NS(model=Model(), req_to_token_pool=NS(), token_to_kv_pool=NS())
    mr.model.start_layer = 1
    bindings = external_bindings(mr, NS())
    assert "weight.layers.0.weight" not in bindings
    assert "weight.layers.1.weight" in bindings
