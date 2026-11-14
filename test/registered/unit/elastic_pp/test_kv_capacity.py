from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.kv_capacity import capacity_frontier, topology_capacity
from sglang.srt.elastic_pp.topology import PPTopology
from sglang.srt.mem_cache.kvcached_pool import KVCacheDTokenToKVPoolAllocator


def args(**changes):
    return NS(**(dict(elastic_pp_kv_backend="kvcached",
                      elastic_pp_kv_capacities="4;8",
                      elastic_pp_graph_partitions="24,0,24;16,16,16",
                      max_total_tokens=8, page_size=1, disaggregation_mode="decode") | changes))


def test_capacity_frontier_fixes_virtual_maximum_and_layouts():
    assert capacity_frontier(args()) == {(24, 0, 24): 4, (16, 16, 16): 8}
    assert topology_capacity(args(), PPTopology.from_layer_counts((16, 16, 16))) == 8


@pytest.mark.parametrize("changes", [dict(max_total_tokens=7), dict(page_size=64),
    dict(elastic_pp_kv_capacities="4"), dict(elastic_pp_kv_capacities="0;8"),
    dict(elastic_pp_kv_backend="native"), dict(disaggregation_mode="prefill")])
def test_capacity_frontier_rejects_unqualified_modes(changes):
    with pytest.raises(ValueError):
        capacity_frontier(args(**changes))


class Manager:
    def clear(self):
        pass


def allocator():
    pool = NS(kvcached_allocator=Manager(), attach_logical_allocator=lambda a: None)
    result = KVCacheDTokenToKVPoolAllocator(8, torch.float32, "cpu", pool, False)
    result.set_serving_capacity(4, initialize=True)
    return result


def test_grow_preserves_live_ids_and_adds_real_free_slots():
    pool = allocator()
    live = pool.alloc(3)
    assert pool.alloc(2) is None
    pool.set_serving_capacity(8)
    extra = pool.alloc(5)
    assert live.tolist() == [1, 2, 3]
    assert extra.tolist() == [4, 5, 6, 7, 8]
    pool.free(extra)
    pool.set_serving_capacity(4)
    assert pool.elastic_pp_live_slots().tolist() == [1, 2, 3]
    assert pool.available_size() == 1
    assert pool.size == 4
    assert pool.virtual_capacity == 8  # Tensor/Graph address space stays stable.


def test_shrink_refuses_a_live_tail_even_when_live_count_fits():
    pool = allocator()
    pool.set_serving_capacity(8)
    slots = pool.alloc(8)
    pool.free(slots[:7])
    with pytest.raises(RuntimeError, match="drain or evict"):
        pool.set_serving_capacity(4)
    assert pool.serving_capacity == 8
    assert pool.elastic_pp_live_slots().tolist() == [8]


def test_clear_preserves_capacity_and_does_not_rebuild_registered_backing():
    pool = allocator()
    pool.alloc(3)
    pool.kvcached_allocator.clear = lambda: pytest.fail("registered pages must survive cache flush")
    pool.clear()
    assert pool.available_size() == 4
    assert not pool.elastic_pp_live_slots().numel()


def test_free_group_keeps_slots_unavailable_until_it_closes():
    pool = allocator()
    slots = pool.alloc(4)
    pool.free_group_begin()
    pool.free(slots)
    assert pool.available_size() == 0
    with pytest.raises(RuntimeError, match="deferred frees"):
        pool.set_serving_capacity(8)
    pool.free_group_end()
    assert pool.available_size() == 4


def test_native_prefill_copies_into_strided_kvcached_rows_without_overwrite():
    from unittest.mock import Mock
    import numpy as np
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.pp_size = 2
    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
    manager.enable_custom_mem_pool = False
    manager._transfer_data = Mock(return_value=0)
    assert manager._send_kvcache_generic(
        "peer", [1000, 2000], [10000, 10004], [4, 4],
        np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32),
        None, src_layer_ids=[3, 3], dst_layer_ids=[3, 3], dst_row_strides=[8, 8],
    ) == 0
    blocks = manager._transfer_data.call_args.args[1]
    assert blocks == [(1004, 10024, 4), (1008, 10032, 4),
                      (2004, 10028, 4), (2008, 10036, 4)]


def test_hybrid_idle_accounting_uses_committed_capacity_and_keeps_virtual_bound():
    from sglang.srt.managers.scheduler_components.invariant_checker import SchedulerInvariantChecker
    from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats

    alloc = allocator()
    checker = NS(is_hybrid_swa=False, is_hybrid_ssm=True,
                 tree_cache=NS(supports_mamba=lambda: True, full_protected_size=lambda: 0),
                 pool_stats_observer=NS(session_held_tokens=lambda: 0),
                 token_to_kv_pool_allocator=alloc,
                 _check_pool_invariant=SchedulerInvariantChecker._check_pool_invariant)
    for capacity in (4, 8, 4):
        alloc.set_serving_capacity(capacity)
        stats = PoolStats(full_available_size=capacity, full_evictable_size=0, full_num_used=0, full_token_usage=0)
        leaked, _ = SchedulerInvariantChecker._check_full_pool(checker, stats)
        assert not leaked
        assert alloc.virtual_capacity == 8


def test_capacity_commit_refreshes_immutable_statistics_owner():
    from dataclasses import dataclass
    from sglang.srt.elastic_pp.kv_capacity import publish_scheduler_capacity

    @dataclass(frozen=True)
    class Stats:
        max_total_num_tokens: int

    old = Stats(4)
    runner = NS(token_to_kv_pool=NS(capacity_managed=True, serving_capacity=8))
    checker = NS(max_total_num_tokens=4, pool_stats_observer=old)
    scheduler = NS(tp_worker=NS(model_runner=runner), pool_stats_observer=old,
                   invariant_checker=checker, disagg_decode_prealloc_queue=NS(max_total_num_tokens=4))
    publish_scheduler_capacity(scheduler)
    assert old.max_total_num_tokens == 4
    assert scheduler.pool_stats_observer.max_total_num_tokens == 8
    assert checker.pool_stats_observer is scheduler.pool_stats_observer
    assert checker.max_total_num_tokens == scheduler.max_total_num_tokens == 8
