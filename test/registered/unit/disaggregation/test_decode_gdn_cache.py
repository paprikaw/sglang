"""Regression coverage for transferred GDN state, prefix reuse and host restore."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.mamba_cache import (
    prepare_transferred_mamba_state,
    reserve_decode_mamba_slots,
)
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def fixture(depth=512):
    states = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    # Distinct old-prefix, transferred-current and tracking-slot contents.
    states[1] = -10
    states[2] = 97
    pool = SimpleNamespace(
        enable_mamba_extra_buffer=True,
        enable_mamba_extra_buffer_lazy=False,
        mamba_ping_pong_track_buffer_size=2,
        translate_mamba_indices=lambda x: x,
        get_mamba_ping_pong_other_idx=lambda i: 1-i,
        mamba_pool=SimpleNamespace(copy_from=lambda src,dst: states.__setitem__(dst, states[src].clone())),
        mamba_allocator=MagicMock(),
    )
    req = SimpleNamespace(
        mamba_pool_idx=torch.tensor(2),
        mamba_ping_pong_track_buffer=torch.tensor([3, 4]),
        mamba_next_track_idx=0,mamba_last_track_idx=1,
        mamba_cow_src_index=torch.tensor([1]),mamba_needs_clear=True,
        mamba_branching_seqlen=256,mamba_last_track_seqlen=None,
        kv_committed_len=depth,
    )
    cache=SimpleNamespace(
        supports_mamba=lambda: True,req_to_token_pool=pool,
        disable=False,enable_mamba_extra_buffer=True,page_size=1,
        evict=MagicMock(),
    )
    return cache,req,states


def test_transferred_state_is_snapshotted_without_reusing_stale_prefix():
    cache,req,states=fixture()
    original=states.clone()
    with patch('sglang.srt.disaggregation.mamba_cache.mamba_checkpoint_grid',return_value=64):
        prepare_transferred_mamba_state(req,cache)
    assert torch.equal(states[3],original[2])
    assert torch.equal(states[2],original[2])
    assert torch.equal(states[1],original[1])
    assert req.mamba_last_track_seqlen==512
    assert req.mamba_last_track_idx==0 and req.mamba_next_track_idx==1
    assert req.mamba_cow_src_index is None and not req.mamba_needs_clear
    # Mutating the live Decode state must not mutate its cache checkpoint.
    states[2].add_(1)
    assert torch.equal(states[3],original[2])


def test_unaligned_transferred_state_is_not_mislabeled_as_earlier_prefix():
    cache,req,states=fixture(513)
    original=states.clone()
    with patch('sglang.srt.disaggregation.mamba_cache.mamba_checkpoint_grid',return_value=64):
        prepare_transferred_mamba_state(req,cache)
    assert torch.equal(states,original)
    assert req.mamba_last_track_seqlen is None
    assert req.mamba_cow_src_index is None and not req.mamba_needs_clear


def test_state_admission_reclaims_cache_and_waits_if_states_are_protected():
    cache,req,_=fixture()
    req.mamba_pool_idx=None;req.mamba_ping_pong_track_buffer=None
    cache.req_to_token_pool.mamba_allocator.available_size.side_effect=[1,4]
    assert reserve_decode_mamba_slots(cache,req)
    params=cache.evict.call_args.args[0]
    assert params.num_tokens==0 and params.mamba_num==3
    cache.req_to_token_pool.mamba_allocator.available_size.side_effect=[1,1]
    assert not reserve_decode_mamba_slots(cache,req)


def test_hicache_restores_checkpoint_without_touching_live_transfer_slot():
    cache,req,states=fixture()
    original=states.clone()
    req.last_node=1;req.swa_host_hit_length=0;req.mamba_host_hit_length=1
    tree=SimpleNamespace(
        buffer_pipeline=None,
        tree_core=SimpleNamespace(
            is_full_device_evicted=lambda n: True,
            collect_full_device_indices=lambda a,b: torch.arange(4),
        ),
    )
    calls=[]
    def load_back(node,quota,req):
        calls.append(req)
        # Model the recurrent-state destination exposed to component transfers.
        if req is not None:states[req.mamba_pool_idx]=states[1]
        return True
    tree.load_back=load_back
    values,node=UnifiedRadixCache.init_load_back(tree,InitLoadBackParams(
        best_match_node=2,host_hit_length=4,req=req,copy_mamba_to_request=False))
    assert node==2 and len(values)==4 and calls==[None]
    assert torch.equal(states[2],original[2])
    # Ordinary Prefill keeps the existing per-request restore semantics.
    UnifiedRadixCache.init_load_back(tree,InitLoadBackParams(best_match_node=2,host_hit_length=4,req=req))
    assert calls[-1] is req
    assert torch.equal(states[2],states[1])
