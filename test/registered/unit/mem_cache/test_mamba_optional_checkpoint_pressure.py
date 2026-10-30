"""Unfinished cache insertion is optional when all recurrent states are pinned.

Exercise native cache/component/pool methods on CPU tensors, including ownership
when a deferred checkpoint is later donated at request completion.
"""
from types import MethodType, SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class StateAllocator:
    def __init__(self):
        self.free_slots=set()
        self.freed=[]
        self.allocations=0

    def alloc(self,n):
        self.allocations+=1
        if len(self.free_slots)<n:
            return None
        selected=sorted(self.free_slots)[:n]
        self.free_slots.difference_update(selected)
        return torch.tensor(selected,dtype=torch.int64)

    def free(self,slots):
        values=slots.reshape(-1).tolist()
        assert not self.free_slots.intersection(values), 'Double free'
        self.free_slots.update(values)
        self.freed.extend(values)


def fixture(*,extra_buffer=True,reclaim=False):
    states=torch.arange(12,dtype=torch.float32).reshape(4,3)
    allocator=StateAllocator()
    pool=NS(mamba_allocator=allocator,enable_mamba_extra_buffer=extra_buffer,
        enable_mamba_extra_buffer_lazy=False,mamba_ping_pong_track_buffer_size=2,
        req_to_token=torch.tensor([[10,11,12]],dtype=torch.int64),
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor([[1,2]]),
        translate_mamba_indices=lambda value:value,
        mamba_pool=NS(replayssm_write_pos=None,
            copy_from=lambda src,dst:states.__setitem__(dst,states[src].clone())))
    for name in ('get_mamba_ping_pong_keep_idx','set_mamba_ping_pong_slot',
                 'donate_mamba_ping_pong_slot','free_mamba_cache'):
        setattr(pool,name,MethodType(getattr(HybridReqToTokenPool,name),pool))
    req=NS(rid='cpu-pressure',req_pool_idx=0,mamba_pool_idx=torch.tensor(0),
        mamba_ping_pong_track_buffer=torch.tensor([1,2]),
        mamba_last_track_idx=1,mamba_next_track_idx=0,mamba_last_track_seqlen=2,
        cache_protected_len=2,last_node=object(),extra_key=None,cache_salt=None,
        prefix_indices=torch.tensor([10,11]),get_fill_ids=lambda:[10,11,12])
    cache=NS(enable_mamba_extra_buffer=extra_buffer,req_to_token_pool=pool,
        session=NS(try_cache_unfinished_req=MagicMock(return_value=False)),
        disable=False,tree_core=NS(is_eagle=False),page_size=1,
        token_to_kv_pool_allocator=MagicMock(),_dec_req_lock=MagicMock(),
        insert=MagicMock(),evict=MagicMock())
    if reclaim:
        cache.evict.side_effect=lambda params:allocator.free(torch.tensor([3]))
    component=object.__new__(MambaComponent)
    component.cache=cache
    full=object.__new__(FullComponent)
    full.cache=cache
    cache._components_tuple=(full,component)
    return component,cache,req,states


def prepare(component,req,*,finished=False):
    params=InsertParams(prev_prefix_len=req.cache_protected_len)
    length=component.prepare_for_caching_req(req,params,3,is_finished=finished)
    return length,params


def test_required_state_allocation_remains_strict():
    component,cache,_,_=fixture()
    with pytest.raises(AssertionError,match='Can not alloc mamba cache'):
        component._alloc_mamba_slot()
    cache.evict.assert_called_once()


def test_optional_snapshot_does_not_overwrite_or_free_live_state_when_full():
    component,cache,req,states=fixture()
    before=states.clone()
    length,params=prepare(component,req)
    assert length==0 and params.mamba_value is None
    assert torch.equal(states,before)
    assert req.mamba_pool_idx.item()==0
    assert req.mamba_ping_pong_track_buffer.tolist()==[1,2]
    assert req.mamba_last_track_seqlen==2
    assert cache.req_to_token_pool.mamba_allocator.freed==[]
    cache.evict.assert_called_once()


def test_native_unified_skip_keeps_kv_indices_prefix_lock_and_checkpoint_depth():
    component,cache,req,states=fixture()
    node=req.last_node
    before=states.clone()
    UnifiedRadixCache.cache_unfinished_req(cache,req)
    assert req.prefix_indices.tolist()==[10,11,12]
    assert req.last_node is node and req.cache_protected_len==2
    assert req.mamba_last_track_seqlen==2
    assert torch.equal(states,before)
    cache._dec_req_lock.assert_not_called()
    cache.insert.assert_not_called()
    cache.token_to_kv_pool_allocator.free_segments.assert_not_called()


def test_completion_can_donate_deferred_checkpoint_without_extra_slot():
    component,cache,req,states=fixture()
    UnifiedRadixCache.cache_unfinished_req(cache,req)
    before=states.clone()
    length,params=prepare(component,req,finished=True)
    assert length==2 and params.mamba_value.tolist()==[2]
    component.cleanup_after_caching_req(req,is_finished=True,
        insert_result=NS(mamba_exist=False),insert_params=params)
    # The donated checkpoint survives; only the live and other tracking slot
    # are freed by the native pool method. The pre-existing cache slot 3 stays.
    assert set(cache.req_to_token_pool.mamba_allocator.freed)=={0,1}
    assert req.mamba_pool_idx is None and req.mamba_ping_pong_track_buffer is None
    assert torch.equal(states[2],before[2])


def test_retry_after_pressure_reuses_the_still_valid_tracked_state():
    component,cache,req,states=fixture()
    UnifiedRadixCache.cache_unfinished_req(cache,req)
    cache.req_to_token_pool.mamba_allocator.free(torch.tensor([3]))
    length,params=prepare(component,req)
    assert length==2 and params.mamba_value.tolist()==[2]
    assert req.mamba_ping_pong_track_buffer.tolist()==[1,3]
    assert cache.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping.tolist()==[[1,3]]
    component.cleanup_after_caching_req(req,is_finished=False,
        insert_result=NS(mamba_exist=False),insert_params=params)
    assert req.mamba_last_track_seqlen is None
    assert 2 not in cache.req_to_token_pool.mamba_allocator.freed


def test_evictable_snapshot_is_reclaimed_and_cached_normally():
    component,cache,req,_=fixture(reclaim=True)
    length,params=prepare(component,req)
    assert length==2 and params.mamba_value.tolist()==[2]
    assert req.mamba_ping_pong_track_buffer.tolist()==[1,3]
    cache.evict.assert_called_once()


def test_no_buffer_mode_does_not_copy_when_optional_snapshot_is_full():
    component,cache,req,states=fixture(extra_buffer=False)
    before=states.clone()
    length,params=prepare(component,req)
    assert length==0 and params.mamba_value is None
    assert req.mamba_pool_idx.item()==0 and torch.equal(states,before)
    assert cache.req_to_token_pool.mamba_allocator.freed==[]


def test_no_valid_checkpoint_does_not_attempt_an_allocation():
    component,cache,req,_=fixture()
    req.mamba_last_track_seqlen=None
    length,params=prepare(component,req)
    assert length==0 and params.mamba_value is None
    assert cache.req_to_token_pool.mamba_allocator.allocations==0


def test_no_buffer_completion_keeps_donated_live_state_after_defer():
    component,cache,req,states=fixture(extra_buffer=False)
    UnifiedRadixCache.cache_unfinished_req(cache,req)
    before=states.clone()
    length,params=prepare(component,req,finished=True)
    assert length==3 and params.mamba_value.tolist()==[0]
    component.cleanup_after_caching_req(req,is_finished=True,
        insert_result=NS(mamba_exist=False),insert_params=params)
    assert cache.req_to_token_pool.mamba_allocator.freed==[]
    assert torch.equal(states[0],before[0])
