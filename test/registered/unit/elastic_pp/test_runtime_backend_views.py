"""CPU regression for real fused GDN tracking views across backing changes."""

import ast
import __future__
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.backing_spill import HybridKVSpill
from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
from sglang.srt.elastic_pp.topology import LayerRange


ENGINE = Path(__file__).resolve().parents[4]


def production_methods(relative_path, class_name, names):
    # Execute the production methods verbatim without loading CUDA model kernels.
    path = ENGINE / "python/sglang/srt" / relative_path
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            namespace = {"LayerRange": LayerRange, "torch": torch}
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
            methods[node.name] = namespace[node.name]
    return methods


def runtime(pool):
    backend_class = type(
        "ProductionTrackPools",
        (),
        production_methods(
            "layers/attention/hybrid_linear_attn_backend.py",
            "MambaAttnBackendBase",
            {"_track_pools"},
        ),
    )
    backends = [backend_class() for _ in range(3)]
    for backend in backends:
        backend.req_to_token_pool = pool.req_to_token_pool
        backend.forward_metadata = object()
        assert backend._track_pools() is not None
    runner_class = type(
        "ProductionRunnerMethods",
        (),
        production_methods(
            "model_executor/model_runner.py",
            "ModelRunner",
            {
                "invalidate_elastic_pp_attention_state_views",
                "commit_elastic_pp_kv_layers",
            },
        ),
    )
    runner = runner_class()
    runner.device = "cpu"
    runner.token_to_kv_pool = pool
    runner.attn_backend = SimpleNamespace(linear_attn_backend=backends[0])
    runner.decode_attn_backend = SimpleNamespace(
        prefill_backend=backends[1], decode_backend=backends[2]
    )
    # Repeated roots must be safe; each backend owns its own full-pool views.
    runner.decode_attn_backend_group = [backends[0], runner.decode_attn_backend]
    return runner, backends


def source_copies(pool):
    return tuple(t.clone() for t in pool._state_tensors(pool.mamba_pool.mamba_cache))


def test_real_fused_track_cache_reproduces_storage_alias_rejection(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    runner, _ = runtime(pool)
    state_bytes = sum(
        t.nbytes for t in pool._state_tensors(pool.mamba_pool.mamba_cache)
    )
    spill = HybridKVSpill(pool)
    spill.snapshot()
    with pytest.raises(
        RuntimeError, match=f"storage aliases \\({state_bytes} bytes\\)"
    ):
        spill.release_source_backing()
    assert spill.metrics["source_storage_bytes_alive"] == state_bytes
    # Retaining the runtime object is exactly what keeps these source views alive.
    assert runner.attn_backend.linear_attn_backend._track_pools_cache is not None


def test_fenced_runtime_release_drops_views_and_rollback_rebinds_source(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    pool.mamba_pool.mamba_cache.temporal.fill_(17)
    pool.mamba_pool.mamba_cache.conv[0].fill_(5)
    expected = source_copies(pool)
    original = weakref.ref(pool.mamba_pool.mamba_cache.temporal)
    req_pool, allocator = pool.req_to_token_pool, pool.req_to_token_pool.mamba_allocator
    runner, backends = runtime(pool)
    metadata = [backend.forward_metadata for backend in backends]
    holder = SimpleNamespace(spill=None)

    def release(_transition):
        holder.spill = HybridKVSpill(pool)
        holder.spill.snapshot()
        holder.spill.release_source_backing()
        return holder.spill.metrics.copy()

    coordinator = ElasticPPMigrationCoordinator.__new__(ElasticPPMigrationCoordinator)
    coordinator.model_runner = runner
    coordinator.cache_transition = None
    coordinator.before_backing_release = None
    coordinator.after_backing_restore = None
    coordinator.after_topology_commit = None
    coordinator._source_backing_attempted = False
    coordinator._source_backing_metrics = {}
    coordinator.kv_transport = SimpleNamespace(
        release_source_backing=release,
        restore_source_backing=lambda transition: holder.spill.restore_source(),
        restore_target_from_snapshot=lambda transition: None,
        finalize_source_backing=lambda transition: None,
    )
    coordinator._release_source_backing_at_fence(None)
    gc.collect()
    assert original() is None
    assert holder.spill.metrics["source_storage_bytes_alive"] == 0
    assert all("_track_pools_cache" not in backend.__dict__ for backend in backends)
    assert [backend.forward_metadata for backend in backends] == metadata
    assert pool.req_to_token_pool is req_pool and req_pool.mamba_allocator is allocator
    coordinator.kv_transport.restore_source_backing(None)
    assert all(
        torch.equal(actual, saved)
        for actual, saved in zip(
            pool._state_tensors(pool.mamba_pool.mamba_cache), expected
        )
    )
    for backend in backends:
        assert backend._track_pools()[0] is pool.mamba_pool.mamba_cache.conv[0]
        assert backend._track_pools()[1] is pool.mamba_pool.mamba_cache.temporal


def test_overlap_publication_rebuilds_tracking_views_for_new_layer_map(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    runner, backends = runtime(pool)
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(0, 2), generation=1, total_num_layers=4
    )
    assert pool.elastic_pp_kv_ready(generation=1)
    assert runner.commit_elastic_pp_kv_layers(generation=1) == LayerRange(0, 2)
    assert all("_track_pools_cache" not in backend.__dict__ for backend in backends)
    for backend in backends:
        current = backend._track_pools()
        assert current is not None
        assert current[0] is pool.mamba_pool.mamba_cache.conv[0]
        assert current[1] is pool.mamba_pool.mamba_cache.temporal
        assert current[2] == 0
        assert current[0].shape[0] == current[1].shape[0] == 1


def test_rejected_kv_commit_keeps_source_tracking_views(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    runner, backends = runtime(pool)
    with pytest.raises((RuntimeError, ValueError)):
        runner.commit_elastic_pp_kv_layers(generation=1)
    for backend in backends:
        assert backend._track_pools()[1] is pool.mamba_pool.mamba_cache.temporal
        assert backend._track_pools()[2] == 1


def test_spilled_target_publication_tracks_restored_current_state(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    pool.mamba_pool.mamba_cache.temporal.fill_(29)
    pool.mamba_pool.mamba_cache.conv[0].fill_(7)
    runner, backends = runtime(pool)
    runner.invalidate_elastic_pp_attention_state_views()
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    spill.prepare_target(target=LayerRange(0, 2), generation=1)
    assert runner.commit_elastic_pp_kv_layers(generation=1) == LayerRange(0, 2)
    for backend in backends:
        current = backend._track_pools()
        assert current[0] is pool.mamba_pool.mamba_cache.conv[0]
        assert current[1] is pool.mamba_pool.mamba_cache.temporal
        assert current[0].eq(7).all() and current[1].eq(29).all()
        assert current[2] == 0
    spill.finalize(generation=1)
    with pytest.raises(RuntimeError):
        spill.restore_source()


def test_real_external_detach_alias_still_prevents_release(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    alias = pool.mamba_pool.mamba_cache.temporal.detach()
    runner, _ = runtime(pool)
    runner.invalidate_elastic_pp_attention_state_views()
    spill = HybridKVSpill(pool)
    spill.snapshot()
    with pytest.raises(
        RuntimeError, match=f"storage aliases \\({alias.nbytes} bytes\\)"
    ):
        spill.release_source_backing()
    assert spill.metrics["source_storage_bytes_alive"] == alias.nbytes
    spill.restore_source()
    assert runner.attn_backend.linear_attn_backend._track_pools()[1] is not alias


def test_backend_lists_duplicate_roots_and_dormant_cache_are_invalidated(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    runner, backends = runtime(pool)
    full_backend = type(backends[0])()
    list_backend = type(backends[0])()
    for backend in (full_backend, list_backend):
        backend.req_to_token_pool = pool.req_to_token_pool
        assert backend._track_pools() is not None
    dormant = type(backends[0])()
    dormant.req_to_token_pool = SimpleNamespace(mamba_map={})
    dormant._track_pools_cache = None
    runner.attn_backend.full_attn_backend = full_backend
    runner.attn_backend.attn_backend_list = [
        full_backend,
        list_backend,
        dormant,
        runner.attn_backend,
    ]
    runner.invalidate_elastic_pp_attention_state_views()
    assert all(
        "_track_pools_cache" not in backend.__dict__
        for backend in (*backends, full_backend, list_backend, dormant)
    )
    # Invalidating a dormant cache must not execute max(empty layer map).
    assert dormant.req_to_token_pool.mamba_map == {}


def test_real_tbo_constructor_primary_and_children_views_are_invalidated(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    runner, backends = runtime(pool)
    for backend in backends:
        backend.token_to_kv_pool = pool
    path = ENGINE / "python/sglang/srt/layers/attention/tbo_backend.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "TboAttnBackend"
    )
    # Keep the real constructor, using object for the kernel-facing base class.
    cls.bases = [ast.Name(id="object", ctx=ast.Load())]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {}
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    wrapper = namespace["TboAttnBackend"](backends[0], backends[1:])
    runner.attn_backend = wrapper
    runner.decode_attn_backend = None
    runner.decode_attn_backend_group = []
    runner.invalidate_elastic_pp_attention_state_views()
    assert wrapper.req_to_token_pool is pool.req_to_token_pool
    assert all("_track_pools_cache" not in backend.__dict__ for backend in backends)
