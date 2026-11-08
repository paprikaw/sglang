import gc
import weakref

import pytest
import torch

from sglang.srt.elastic_pp.backing_spill import NativeKVSpill
from sglang.srt.elastic_pp.topology import LayerRange
from test_kv_transport import _make_pool


def test_detach_releases_owner_and_restore_is_exact():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.k_buffer[0].fill_(17)
    original = weakref.ref(pool.k_buffer[0])
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    spill.release_source_backing()
    gc.collect()
    assert original() is None
    assert pool.k_buffer == []
    assert spill.metrics["detached_bytes"] > 0
    spill.restore_source()
    assert torch.all(pool.k_buffer[0] == 17)
    assert pool.layer_num == 2


def test_target_only_allocates_after_release_and_rollback_restores_source():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.k_buffer[1].fill_(21)
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    spill.release_source_backing()
    spill.prepare_target(target=LayerRange(1, 3), generation=1)
    assert len(pool._elastic_pp_pending_kv) == 2
    k, _ = pool.get_elastic_pp_pending_kv_buffer(generation=1, layer_id=1)
    assert torch.all(k == 21)
    spill.restore_source()
    assert pool._elastic_pp_residency.pending is None
    assert torch.all(pool.k_buffer[1] == 21)


def test_failed_restore_preserves_cpu_authority_for_retry(monkeypatch):
    pool = _make_pool(start_layer=0, layer_num=2)
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    spill.release_source_backing()
    allocate = pool._allocate_elastic_pp_kv_pair
    monkeypatch.setattr(
        pool,
        "_allocate_elastic_pp_kv_pair",
        lambda **kw: (_ for _ in ()).throw(MemoryError("oom")),
    )
    with pytest.raises(MemoryError):
        spill.restore_source()
    assert spill.state == "spilled"
    assert pool.k_buffer == []
    monkeypatch.setattr(pool, "_allocate_elastic_pp_kv_pair", allocate)
    spill.restore_source()
    assert spill.state == "restored"


def test_release_requires_completed_snapshot():
    spill = NativeKVSpill(_make_pool(start_layer=0, layer_num=1), total_num_layers=4)
    with pytest.raises(RuntimeError):
        spill.release_source_backing()


def test_hybrid_spill_releases_gdn_and_preserves_frontier(hybrid_pool_factory):
    from sglang.srt.elastic_pp.backing_spill import HybridKVSpill

    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    req = pool.req_to_token_pool
    req.req_generation[1] = 7
    req.req_index_to_mamba_index_mapping[1] = 2
    pool.mamba_pool.mamba_cache.temporal.fill_(19)
    original = weakref.ref(pool.mamba_pool.mamba_cache.temporal)
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    gc.collect()
    assert original() is None
    assert req.req_generation[1] == 7
    assert req.req_index_to_mamba_index_mapping[1] == 2
    spill.prepare_target(target=LayerRange(0, 4), generation=1)
    assert pool._pending_state.temporal[0].eq(19).all()
    spill.restore_source()
    assert pool.mamba_pool.mamba_cache.temporal.eq(19).all()
    assert req.req_generation[1] == 7


def test_hybrid_commit_restores_retained_from_cpu_not_empty_source(hybrid_pool_factory):
    from sglang.srt.elastic_pp.backing_spill import HybridKVSpill

    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    pool.mamba_pool.mamba_cache.temporal.fill_(23)
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    spill.prepare_target(target=LayerRange(0, 4), generation=1)
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2, 3})
    pool.commit_elastic_pp_kv_layers(generation=1)
    assert pool.mamba_pool.mamba_cache.temporal[0].eq(23).all()
    with pytest.raises(RuntimeError):
        spill.restore_source()
