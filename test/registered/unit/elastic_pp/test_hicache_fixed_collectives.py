"""Resident zero-layer ranks must join fixed-capacity HiCache PP collectives."""

from functools import partial
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_fixed_full_and_mamba_capacity_collectives_include_dormant_rank(
    hybrid_pool_factory, monkeypatch
):
    memory = NS(
        hicache_size=1,
        hicache_ratio=2,
        hicache_mem_layout="page_first",
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
    )
    group = object()
    current_rank = 0
    events = {rank: [] for rank in range(3)}

    def all_reduce(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is pp_group.cpu_group
        index = len(events[current_rank])
        events[current_rank].append((index, tensor.item()))
        tensor.fill_((32, 7)[index])

    pp_group = NS(world_size=3, cpu_group=group)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(assembler, "get_memory", lambda: memory)
    monkeypatch.setattr(assembler, "get_parallel", lambda: NS(dcp_enabled=False))
    monkeypatch.setattr(assembler, "_get_allocator_type", lambda _: "default")
    monkeypatch.setattr(
        assembler,
        "get_mha_host_pool_cls",
        lambda _: partial(MHATokenToKVPoolHost, pin_memory=False),
    )
    monkeypatch.setattr(
        assembler, "MambaPoolHost", partial(MambaPoolHost, pin_memory=False)
    )
    monkeypatch.setattr(assembler, "HybridCacheController", lambda *a, **kw: NS())
    capacities = []
    logical = []
    for current_rank, (start, end) in enumerate(((0, 2), (2, 2), (2, 4))):
        pool = hybrid_pool_factory(start_layer=start, end_layer=end)
        full = pool.full_kv_pool
        req = pool.req_to_token_pool
        params = NS(
            req_to_token_pool=req,
            token_to_kv_pool_allocator=NS(),
            mtp_draft_device_pools=(),
            page_size=2,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        host_group, _ = assembler.build_hybrid_mamba_stack(
            params=params,
            server_args=NS(hicache_host_memory_mode="cache"),
            kv_pool=full,
            mamba_pool=pool.mamba_pool,
            full_layer_mapping={i: i for i in range(full.layer_num)},
            mamba_layer_mapping={
                full.layer_num + i: i for i in range(pool.mamba_pool.num_mamba_layers)
            },
            load_cache_event=None,
            storage_backend=None,
            use_mla=False,
        )
        kv = host_group.get_pool(PoolName.KV)
        mamba = host_group.get_pool(PoolName.MAMBA)
        capacities.append((kv.size, mamba.size))
        logical.append(
            (isinstance(kv, LogicalHostPool), isinstance(mamba, LogicalHostPool))
        )
        host_group.destroy()
    assert capacities == [(34, 8)] * 3
    assert logical == [(False, False), (True, True), (False, False)]
    assert all([index for index, _ in calls] == [0, 1] for calls in events.values())
    assert [value for _, value in events[1]] == [2**63 - 1] * 2
    assert all(value < 2**63 - 1 for rank in (0, 2) for _, value in events[rank])


def test_all_logical_component_uses_ratio_after_neutral_fixed_min(monkeypatch):
    calls = []
    monkeypatch.setattr(
        assembler, "get_memory", lambda: NS(hicache_size=1, hicache_ratio=2)
    )

    def neutral(size, host_size):
        calls.append((size, host_size))
        return size

    monkeypatch.setattr(assembler, "sync_fixed_hicache_size", neutral)
    assert assembler._logical_host_pool_size(8, 2) == 18
    assert calls == [(2**63 - 1, 1)]


def test_ratio_logical_capacity_has_no_fixed_collective(monkeypatch):
    monkeypatch.setattr(
        assembler, "get_memory", lambda: NS(hicache_size=0, hicache_ratio=2)
    )

    def unexpected(*args):
        raise AssertionError("ratio mode must not enter a fixed-size collective")

    monkeypatch.setattr(assembler, "sync_fixed_hicache_size", unexpected)
    assert assembler._logical_host_pool_size(8, 2) == 18


@pytest.mark.parametrize(
    ("hicache_size", "live_high_slot"), [(1, False), (1, True), (0, False)]
)
@pytest.mark.parametrize("preallocate", [False, True])
def test_shrinking_host_constructor_keeps_dormant_rank_in_capacity_protocol(
    hybrid_pool_factory, monkeypatch, hicache_size, live_high_slot, preallocate
):
    from sglang.srt.elastic_pp.hicache_transition import prepare_target_host_cache
    from sglang.srt.elastic_pp.topology import LayerRange
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.session.streaming_session import StreamingSession

    memory = NS(
        hicache_size=hicache_size,
        hicache_ratio=2,
        hicache_mem_layout="page_first",
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
    )
    pp_group = NS(world_size=3, cpu_group=object())
    events = {rank: [] for rank in range(3)}
    current_rank = 1
    source_phase = True

    def all_reduce(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is pp_group.cpu_group
        index = len(events[current_rank])
        events[current_rank].append(tensor.item())
        tensor.fill_(((32, 7) if source_phase else (16, 3))[index])

    def controller(*args, **kwargs):
        return NS(write_policy=kwargs["write_policy"], io_backend=kwargs["io_backend"])

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(assembler, "get_memory", lambda: memory)
    monkeypatch.setattr(assembler, "get_parallel", lambda: NS(dcp_enabled=False))
    monkeypatch.setattr(assembler, "_get_allocator_type", lambda _: "default")
    monkeypatch.setattr(
        assembler,
        "get_mha_host_pool_cls",
        lambda _: partial(MHATokenToKVPoolHost, pin_memory=False),
    )
    monkeypatch.setattr(
        assembler, "MambaPoolHost", partial(MambaPoolHost, pin_memory=False)
    )
    # CPU tensors and the production host constructors are used; the controller
    # is the I/O-only boundary, so no CUDA streams or background threads start.
    monkeypatch.setattr(assembler, "HybridCacheController", controller)
    monkeypatch.setattr(hybrid_cache_controller, "HybridCacheController", controller)

    def params(pool):
        return CacheInitParams(
            disable=False,
            req_to_token_pool=pool.req_to_token_pool,
            token_to_kv_pool_allocator=NS(device="cpu", get_kvcache=lambda: pool),
            page_size=2,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        )

    def physical_stack(pool, cache_params):
        full = pool.full_kv_pool
        return assembler.build_hybrid_mamba_stack(
            params=cache_params,
            server_args=NS(hicache_host_memory_mode="cache"),
            kv_pool=full,
            mamba_pool=pool.mamba_pool,
            full_layer_mapping={i: i for i in range(full.layer_num)},
            mamba_layer_mapping={
                full.layer_num + i: i for i in range(pool.mamba_pool.num_mamba_layers)
            },
            load_cache_event=None,
            storage_backend=None,
            use_mla=False,
        )

    class Cache:
        def release_host_resources(self):
            self.host_pool_group.destroy()

    source_pool = hybrid_pool_factory(start_layer=1, end_layer=3)
    cache_params = params(source_pool)
    source = Cache()
    source.components = {ct: NS() for ct in cache_params.tree_components}
    source.tree_core = UnifiedTreeCore(cache_params, source.components)
    source.session_refs = NS(
        tree_core=source.tree_core, components=tuple(source.components.values())
    )
    source.session = StreamingSession(inner=source)
    source.req_to_token_pool = cache_params.req_to_token_pool
    source.token_to_kv_pool_allocator = cache_params.token_to_kv_pool_allocator
    source._hicache_init_params = cache_params
    source._hicache_init_server_args = NS(hicache_host_memory_mode="cache")
    source.host_memory_mode = "cache"
    source.host_pool_group, source.cache_controller = physical_stack(
        source_pool, cache_params
    )
    source.full_kv_pool_host = source.host_pool_group.get_pool(PoolName.KV)
    source.mamba_pool_host = source.host_pool_group.get_pool(PoolName.MAMBA)
    full, mamba = source.full_kv_pool_host, source.mamba_pool_host
    if live_high_slot:
        slots = full.alloc(full.size)
        full.free(slots[:-2])
    else:
        assert full.alloc(2).tolist() == [0, 1]
    assert mamba.alloc(1).tolist() == [0]
    free_before = full.free_slots.clone()
    source_capacity = (full.size, mamba.size)
    source_phase = False
    events = {rank: [] for rank in range(3)}
    target_groups = []
    target = None
    source_pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 2), generation=1, total_num_layers=4
    )
    try:
        for current_rank, (start, end) in enumerate(((0, 2), (2, 2), (2, 4))):
            if current_rank != 1:
                pool = hybrid_pool_factory(start_layer=start, end_layer=end)
                group, _ = physical_stack(pool, params(pool))
                target_groups.append(group)
            elif live_high_slot:
                with pytest.raises(RuntimeError, match="preserve live high host slots"):
                    prepare_target_host_cache(
                        source,
                        device_pool=source_pool,
                        target=LayerRange(2, 2),
                        generation=1,
                    )
                continue
            else:
                prepared = None
                if preallocate:
                    prepared = prepare_target_host_cache(
                        source,
                        device_pool=source_pool,
                        target=LayerRange(2, 2),
                        generation=1,
                        allocation_only=True,
                    )
                target = prepare_target_host_cache(
                    source,
                    device_pool=source_pool,
                    target=LayerRange(2, 2),
                    generation=1,
                    prepared=prepared,
                )
                if prepared is not None:
                    assert prepared.host_pool_group is None
                    assert prepared.cache_controller is None
                group = target.host_pool_group
                target_groups.append(group)
                assert all(
                    isinstance(entry.host_pool, LogicalHostPool)
                    for entry in group.entries
                )
                assert group.get_pool(PoolName.KV).free_slots.tolist() == list(
                    range(2, group.size)
                )
                assert group.get_pool(PoolName.MAMBA).free_slots.tolist() == list(
                    range(1, group.get_pool(PoolName.MAMBA).size)
                )
        expected_capacity = (18, 4) if hicache_size else (18, 9)
        assert all(
            (group.get_pool(PoolName.KV).size, group.get_pool(PoolName.MAMBA).size)
            == expected_capacity
            for group in target_groups
        )
        if hicache_size:
            assert [len(events[rank]) for rank in range(3)] == [2, 2, 2]
            assert events[1] == [2**63 - 1] * 2
            assert source_capacity == (34, 8)
        else:
            assert events == {0: [], 1: [], 2: []}
        assert (full.size, mamba.size) == source_capacity
        assert torch.equal(full.free_slots, free_before)
        assert full.k_buffer is not None and mamba.temporal_buffer is not None
        assert source.req_to_token_pool is cache_params.req_to_token_pool
        assert source_pool.end_layer == 3
    finally:
        for group in target_groups:
            group.destroy()
        source.host_pool_group.destroy()
        source_pool.abort_elastic_pp_kv_layers(generation=1)
