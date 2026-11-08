"""CPU lifetime/failure boundaries using real native pools and transports."""

import gc
import weakref
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.backing_spill import HybridKVSpill, NativeKVSpill
from sglang.srt.elastic_pp.hicache_transition import (
    SchedulerHiCacheTransition,
    discard_host_cache,
    project_target_device_pool,
    rebind_cache_device_pools,
)
from sglang.srt.elastic_pp.hybrid_kv_transport import ElasticPPHybridKVTransport
from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport
from sglang.srt.elastic_pp.topology import DynamicPPTransition, LayerRange, PPTopology
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from test_kv_transport import _Allocator, _Group, _Mailbox, _make_pool, _Runner

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_host_budget_rejection_precedes_any_source_detachment(monkeypatch):
    from sglang.srt.mem_cache.pool_host import base

    pool = _make_pool(start_layer=0, layer_num=1)
    original = pool.k_buffer[0]
    monkeypatch.setattr(base, "host_memory_budget_bytes", lambda: 1)
    spill = NativeKVSpill(pool, total_num_layers=4)
    with pytest.raises(MemoryError, match="current HiCache host budget"):
        spill.snapshot()
    assert pool.k_buffer[0] is original
    assert spill.state == "new"


def test_detached_tensor_alias_blocks_capacity_relief_and_remains_rollbackable():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.k_buffer[0].fill_(31)
    original = weakref.ref(pool.k_buffer[0])
    alias = pool.k_buffer[0].detach()
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    with pytest.raises(RuntimeError, match="storage aliases"):
        spill.release_source_backing()
    gc.collect()
    assert original() is None  # Tensor weakrefs alone would incorrectly pass.
    assert spill.metrics["source_storage_bytes_alive"] == alias.nbytes
    del alias
    spill.restore_source()
    assert pool.k_buffer[0].eq(31).all()


def test_failed_source_rebuild_drops_partial_owners_even_with_retained_exception(
    monkeypatch,
):
    pool = _make_pool(start_layer=0, layer_num=2)
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    spill.release_source_backing()
    allocate = pool._allocate_elastic_pp_kv_pair
    witnesses = []

    def fail_second(*, layer_id):
        if layer_id == 1:
            raise MemoryError("second layer OOM")
        pair = allocate(layer_id=layer_id)
        witnesses.extend(weakref.ref(t.untyped_storage()) for t in pair)
        return pair

    monkeypatch.setattr(pool, "_allocate_elastic_pp_kv_pair", fail_second)
    with pytest.raises(MemoryError) as retained_error:
        spill.restore_source()
    gc.collect()
    assert retained_error.value is not None
    assert all(w() is None for w in witnesses)
    assert spill.backup and spill.state == "spilled"
    monkeypatch.setattr(pool, "_allocate_elastic_pp_kv_pair", allocate)
    spill.restore_source()
    assert len(pool.k_buffer) == 2


def test_target_oom_releases_partial_owners_with_retained_exception(monkeypatch):
    source = PPTopology.from_layer_counts([2, 0, 2], generation=0)
    target = PPTopology.from_layer_counts([1, 2, 1], generation=1)
    tx = DynamicPPTransition.build(source, target)
    pool = _make_pool(start_layer=2, layer_num=0)
    transport = ElasticPPKVTransport(
        group=_Group(_Mailbox(), 1),
        model_runner=_Runner(pool),
        allocator=_Allocator(torch.tensor([1, 3])),
        rank=1,
    )
    transport.release_source_backing(tx)
    allocate = pool._allocate_elastic_pp_kv_pair
    witnesses = []

    def fail_second(*, layer_id):
        if layer_id == 2:
            raise MemoryError("second target layer OOM")
        pair = allocate(layer_id=layer_id)
        witnesses.extend(weakref.ref(t.untyped_storage()) for t in pair)
        return pair

    monkeypatch.setattr(pool, "_allocate_elastic_pp_kv_pair", fail_second)
    with pytest.raises(MemoryError) as retained_error:
        transport.prepare_transition(tx)
    gc.collect()
    assert retained_error.value is not None
    assert all(w() is None for w in witnesses)
    assert pool._elastic_pp_residency.pending is None
    assert pool.abort_elastic_pp_kv_layers(generation=1) == ()
    transport.restore_source_backing(tx)
    assert transport._spill is None
    assert pool.k_buffer == []


def test_host_prepare_failure_tracebacks_do_not_pin_device_owners(monkeypatch):
    from sglang.srt.elastic_pp import hicache_transition

    witnesses = []

    def fail_with_tensor(*args, **kwargs):
        tensor = torch.ones(16)
        witnesses.append(weakref.ref(tensor.untyped_storage()))
        raise MemoryError("target host OOM") from ValueError("allocation failed")

    monkeypatch.setattr(
        hicache_transition, "_prepare_target_host_cache", fail_with_tensor
    )
    with pytest.raises(MemoryError) as retained_error:
        hicache_transition.prepare_target_host_cache(
            None, device_pool=None, target=None, generation=1
        )
    gc.collect()
    assert retained_error.value.__cause__ is not None
    assert all(w() is None for w in witnesses)


def test_detached_shrink_cleanup_and_publication_discard_cpu_authority():
    pool = _make_pool(start_layer=0, layer_num=2)
    spill = NativeKVSpill(pool, total_num_layers=4)
    spill.snapshot()
    spill.release_source_backing()
    assert spill.prepare_target(target=LayerRange(0, 1), generation=1) == {0}
    copied = spill.metrics["restored_bytes"]
    spill.prepare_target(target=LayerRange(0, 1), generation=1)
    assert spill.metrics["restored_bytes"] == copied
    with pytest.raises(RuntimeError, match="before target publication"):
        spill.finalize(generation=1)
    pool.commit_elastic_pp_kv_layers(generation=1)
    assert pool.cleanup_elastic_pp_kv_layers(generation=1) == ()
    assert pool._elastic_pp_residency._cleanup_generation is None
    with pytest.raises(RuntimeError, match="after topology publication"):
        spill.restore_source()
    spill.finalize(generation=1)
    assert not spill.backup
    assert all(v is None for v in spill.metrics["cuda_memory"].values())


def test_hybrid_frontier_authority_rejects_scheduler_progress(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    assert spill.metrics["frontier_snapshot_bytes"] > 0
    pool.req_to_token_pool.req_generation[1] += 1
    with pytest.raises(RuntimeError, match="frontier changed"):
        spill.restore_source()
    assert spill.backup and spill.state == "spilled"
    pool.req_to_token_pool.req_generation[1] -= 1
    spill.restore_source()


@pytest.mark.parametrize("fail_attention", [False, True])
def test_hybrid_failed_restore_counts_completed_copies_and_releases_owners(
    hybrid_pool_factory, monkeypatch, fail_attention
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=4)
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    witnesses = []
    original_to = torch.Tensor.to
    original_allocate = pool.full_kv_pool._allocate_elastic_pp_kv_pair
    completed_gdn_bytes = 0

    def fail_copy(tensor, *args, **kwargs):
        nonlocal completed_gdn_bytes
        is_source = any(tensor is source for source in spill.backup)
        if is_source and completed_gdn_bytes and not fail_attention:
            raise MemoryError("GDN rebuild OOM")
        result = original_to(tensor, *args, **kwargs)
        if is_source:
            completed_gdn_bytes += tensor.nbytes
            witnesses.append(weakref.ref(result.untyped_storage()))
        return result

    def fail_second_attention(*, layer_id):
        if layer_id == 1:
            raise MemoryError("attention rebuild OOM")
        pair = original_allocate(layer_id=layer_id)
        witnesses.extend(weakref.ref(t.untyped_storage()) for t in pair)
        return pair

    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    if fail_attention:
        monkeypatch.setattr(
            pool.full_kv_pool, "_allocate_elastic_pp_kv_pair", fail_second_attention
        )
    with pytest.raises(MemoryError) as retained_error:
        spill.restore_source()
    gc.collect()
    assert retained_error.value is not None and witnesses
    assert all(w() is None for w in witnesses)
    failed_bytes = completed_gdn_bytes + spill.attention.metrics["restored_bytes"]
    assert spill.metrics["restored_bytes"] == failed_bytes > 0
    monkeypatch.setattr(torch.Tensor, "to", original_to)
    monkeypatch.setattr(
        pool.full_kv_pool, "_allocate_elastic_pp_kv_pair", original_allocate
    )
    spill.restore_source()
    assert (
        spill.metrics["restored_bytes"]
        == failed_bytes + spill.metrics["snapshot_bytes"]
    )


def _host_cache(pool):
    full = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    full.device_pool = pool.full_kv_pool
    mamba = MambaPoolHost.__new__(MambaPoolHost)
    mamba.device_pool = pool.mamba_pool
    mamba.temporal_device_ptrs = torch.tensor([0], dtype=torch.uint64)
    mamba.conv_device_ptrs = []
    return NS(
        host_pool_group=NS(
            entries=[
                NS(name=PoolName.KV, host_pool=full, device_pool=pool.full_kv_pool),
                NS(name=PoolName.MAMBA, host_pool=mamba, device_pool=pool.mamba_pool),
            ]
        ),
        cache_controller=NS(),
    )


def test_host_binding_abort_refreshes_gdn_addresses_and_preserves_pool_identity(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    cache = _host_cache(pool)
    scheduler = NS(tree_cache=cache)
    host = SchedulerHiCacheTransition(
        scheduler=scheduler, model_runner=NS(token_to_kv_pool=pool), group=None, rank=0
    )
    tx = NS(target=NS(generation=1))
    host.restore_source_device_aliases(tx)  # Unattempted release is a no-op.
    host.release_source_device_aliases(tx)
    mamba_host = cache.host_pool_group.entries[1].host_pool
    assert mamba_host.temporal_device_ptrs is None
    spill = HybridKVSpill(pool)
    spill.snapshot()
    spill.release_source_backing()
    spill.restore_source()
    host.restore_source_device_aliases(tx)
    assert mamba_host.device_pool is pool.mamba_pool
    assert mamba_host.temporal_device_ptrs.tolist() == [
        pool.mamba_pool.mamba_cache.temporal[0].data_ptr()
    ]
    assert cache.cache_controller.mem_pool_device is pool.full_kv_pool


def test_projected_view_does_not_retain_source_descriptors(hybrid_pool_factory):
    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    old = weakref.ref(pool.full_kv_pool.k_buffer[0].untyped_storage())
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    projected, req = project_target_device_pool(
        pool, target=LayerRange(2, 4), generation=1
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2, 3})
    pool.commit_elastic_pp_kv_layers(generation=1)
    pool.cleanup_elastic_pp_kv_layers(generation=1)
    gc.collect()
    assert old() is None
    cache = _host_cache(projected)
    cache._elastic_pp_projected_pool = projected
    rebind_cache_device_pools(cache, pool)
    assert "_elastic_pp_projected_pool" not in vars(cache)
    assert cache.host_pool_group.entries[0].device_pool is pool.full_kv_pool
    assert cache.host_pool_group.entries[1].host_pool.device_pool is pool.mamba_pool
    del projected, req


def test_discarded_frozen_cache_drops_projected_device_owners():
    tensor = torch.ones(32)
    witness = weakref.ref(tensor.untyped_storage())
    projected = NS(k_buffer=[tensor])
    controller = NS(
        mem_pool_device=projected, mem_pool_device_allocator=NS(owner=projected)
    )
    cache = NS(
        release_host_resources=lambda: None,
        cache_controller=controller,
        components={},
        _elastic_pp_projected_pool=projected,
        token_to_kv_pool_allocator=NS(owner=projected),
        req_to_token_pool=projected,
        _hicache_init_params=NS(owner=projected),
    )
    del tensor, projected
    discard_host_cache(cache)
    gc.collect()
    assert controller.mem_pool_device is None
    assert witness() is None


def test_three_rank_hybrid_spill_grow_shrink_preserves_live_kv_and_all_gdn(
    hybrid_pool_factory,
):
    source = PPTopology.from_layer_counts([2, 0, 2], generation=0)
    expanded = PPTopology.from_layer_counts([1, 2, 1], generation=1)
    shrunk = PPTopology.from_layer_counts([2, 0, 2], generation=2)
    mailbox = _Mailbox()
    slots = torch.tensor([1, 3], dtype=torch.int64)
    pools = [
        hybrid_pool_factory(start_layer=r.start, end_layer=r.end)
        for r in (source.layer_range(i) for i in range(3))
    ]
    transports = [
        ElasticPPHybridKVTransport(
            group=_Group(mailbox, rank),
            model_runner=_Runner(pool),
            allocator=_Allocator(slots),
            rank=rank,
            slot_chunk_size=1,
            recurrent_chunk_size=2,
        )
        for rank, pool in enumerate(pools)
    ]
    for pool in pools:
        for layer in pool.elastic_pp_kv_layer_ids(
            LayerRange(pool.start_layer, pool.end_layer)
        ):
            ordinal = pool._full_id_to_ordinal[layer]
            pool.full_kv_pool.get_key_buffer(ordinal).fill_(layer + 10)
            pool.full_kv_pool.get_value_buffer(ordinal).fill_(layer + 20)
        for layer, index in pool.req_to_token_pool.mamba_map.items():
            for tensor in pool._state_tensors(pool.mamba_pool.mamba_cache):
                tensor[index].fill_(layer + 30)
    # Initialization must not carry a full backing alias across the fence.
    del tensor

    for old_topology, target in ((source, expanded), (expanded, shrunk)):
        tx = DynamicPPTransition.build(old_topology, target)
        for transport in transports:
            transport.release_source_backing(tx)
            transport.prepare_transition(tx)
            before = transport.backing_spill_metrics()["restored_bytes"]
            transport.restore_target_from_snapshot(tx)
            assert transport.backing_spill_metrics()["restored_bytes"] == before
        senders = sorted({edge.src_rank for edge in tx.kv_transfers})
        order = senders + [r for r in range(3) if r not in senders]
        for phase in ("bulk", "delta"):
            for rank in order:
                transports[rank].exchange_phase(
                    tx, phase=phase, slot_ids=slots if phase == "bulk" else slots[:0]
                )
        for transport, pool in zip(transports, pools):
            transport.apply_staged(tx)
            assert pool.elastic_pp_kv_ready(generation=target.generation)
            pool.commit_elastic_pp_kv_layers(generation=target.generation)
            pool.cleanup_elastic_pp_kv_layers(generation=target.generation)
            transport.finalize_source_backing(tx)
            assert transport._spill is None
            assert transport.backing_spill_metrics()["source_storage_bytes_alive"] == 0
            for layer in pool.elastic_pp_kv_layer_ids(
                target.layer_range(transport.rank)
            ):
                ordinal = pool._full_id_to_ordinal[layer]
                assert (
                    pool.full_kv_pool.get_key_buffer(ordinal)[slots]
                    .eq(layer + 10)
                    .all()
                )
                assert (
                    pool.full_kv_pool.get_value_buffer(ordinal)[slots]
                    .eq(layer + 20)
                    .all()
                )
            for layer, index in pool.req_to_token_pool.mamba_map.items():
                assert all(
                    t[index].eq(layer + 30).all()
                    for t in pool._state_tensors(pool.mamba_pool.mamba_cache)
                )
    assert all(not payloads for payloads in mailbox.payloads.values())
