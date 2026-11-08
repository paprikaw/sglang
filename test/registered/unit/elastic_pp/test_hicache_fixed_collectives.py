"""Resident zero-layer ranks must join fixed-capacity HiCache PP collectives."""

from functools import partial
from types import SimpleNamespace as NS

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host import base
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
