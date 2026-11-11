"""CPU checks of the production native transition peak-admission method."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def capacity_report(runner, transition, **kwargs):
    # Load the production method without importing CUDA model backends.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "elastic_pp_transition_capacity_report"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name](runner, transition, **kwargs)


def transition():
    return DynamicPPTransition.build(
        PPTopology.from_layer_counts([4, 0], generation=0),
        PPTopology.from_layer_counts([2, 2], generation=1),
    )


def runner(pool, device="cpu"):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        ps=SimpleNamespace(pp_rank=1),
        elastic_pp_weight_profile=SimpleNamespace(layer_weight_sizes=(10, 20, 30, 40)),
        device=device,
    )


def test_native_admission_counts_full_new_backing_and_reserve():
    pool = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=0,
        device="cpu",
        enable_memory_saver=False,
    )
    result = capacity_report(runner(pool), transition())
    assert result["weight_bytes"] == 70
    assert result["kv_bytes"] == 2 * 2 * 9 * 2 * 4
    assert result["required_free_bytes"] == 70 + result["kv_bytes"] + 256 * 2**20


def test_prepared_graph_admission_counts_unmapped_weight_pages():
    pool = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=0,
        device="cpu",
        enable_memory_saver=False,
    )
    instance = runner(pool)
    instance._elastic_pp_prepared_graphs = SimpleNamespace(
        building=False,
        check_topology=lambda target: None,
        weight_backing=SimpleNamespace(
            layers={
                2: SimpleNamespace(size=4096, mapped=False),
                3: SimpleNamespace(size=8192, mapped=True),
            }
        ),
    )
    report = capacity_report(instance, transition())
    assert report["weight_bytes"] == 4096
    assert report["kv_bytes"] == report["recurrent_bytes"] == 0
    assert report["required_free_bytes"] == 4096 + 256 * 2**20


def test_dormant_hybrid_counts_complete_replacement_gdn(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=4, end_layer=4)
    result = capacity_report(runner(pool), transition())
    state = pool.mamba_pool.mamba_cache
    expected = sum(
        t.shape[1:].numel() * t.element_size() for t in (*state.conv, state.temporal)
    )
    assert result["recurrent_bytes"] == expected
    assert result["kv_bytes"] > 0


def test_native_peak_rejection_occurs_before_allocation(monkeypatch):
    pool = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=0,
        device="cpu",
        enable_memory_saver=False,
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 80 * 2**30))
    with pytest.raises(RuntimeError, match="peak capacity rejected"):
        capacity_report(runner(pool, device="cuda:0"), transition())
    assert not pool._elastic_pp_pending_kv


def test_spilled_admission_counts_retained_target_backing():
    pool = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=4,
        device="cpu",
        enable_memory_saver=False,
    )
    change = transition()
    instance = runner(pool)
    instance.ps.pp_rank = 0
    overlap = capacity_report(instance, change)
    spilled = capacity_report(instance, change, source_backing_released=True)
    assert overlap["kv_bytes"] == 0
    assert spilled["kv_bytes"] == 2 * 2 * 9 * 2 * 4
    assert spilled["source_backing_released"]
    assert not spilled["native_overlap_checked"]


def test_spilled_unchanged_gdn_layer_map_still_needs_replacement(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    topology = PPTopology.from_layer_counts([4, 0], generation=0)
    change = DynamicPPTransition.build(
        topology, PPTopology.from_layer_counts([4, 0], generation=1)
    )
    instance = runner(pool)
    instance.ps.pp_rank = 0
    overlap = capacity_report(instance, change)
    spilled = capacity_report(instance, change, source_backing_released=True)
    assert overlap["recurrent_bytes"] == 0
    assert spilled["recurrent_bytes"] > 0
