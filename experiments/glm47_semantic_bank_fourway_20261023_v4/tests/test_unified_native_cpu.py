"""Actual pinned URC/TreeCore CPU interface probe; no model and no GPU.

Tree, FULL component, key hashing, insertion, matches and cache-action dispatch
are the real engine classes. Only the slot allocator is a recording CPU double;
this does not qualify CUDA transfers or the physical HostPoolGroup allocation.
"""
from array import array
import importlib
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parents[1] / '.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023'


@pytest.fixture(scope='module')
def native():
    # Run on the login CPU. Any accidental CUDA initialization is an error.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ENGINE / 'python'))
    sys.path.insert(0, str(ROOT / 'runtime'))
    import torch
    assert not torch.cuda.is_initialized()
    old_lazy = torch.cuda._lazy_init
    def reject_cuda(*args, **kwargs):
        # FlashInfer's import-time property probe catches RuntimeError, exactly
        # like a CPU-only Torch build. It must not reach the real CUDA initializer.
        raise RuntimeError('CPU-only URC probe blocked CUDA initialization')
    torch.cuda._lazy_init = reject_cuda
    try:
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
        import semantic_bank
        def reject_test_cuda(*args, **kwargs):
            raise AssertionError('Actual CPU tree operation attempted CUDA initialization')
        torch.cuda._lazy_init=reject_test_cuda
        source = Path(importlib.import_module(UnifiedRadixCache.__module__).__file__).resolve()
        assert source.is_relative_to(ENGINE / 'python')
        yield dict(torch=torch,params=CacheInitParams,evict=EvictParams,match=MatchPrefixParams,
                   key=RadixKey,full=ComponentType.FULL,cache=UnifiedRadixCache,
                   free_device=FreeDeviceKV,bank=semantic_bank)
        assert not torch.cuda.is_initialized()
    finally:
        torch.cuda._lazy_init = old_lazy


def make_cache(native, eagle):
    cache = native['cache'](native['params'](
        disable=False,req_to_token_pool=None,token_to_kv_pool_allocator=None,
        page_size=64,tree_components=(native['full'],),is_eagle=eagle))
    cache.tree_core.enable_hicache = True
    return cache


@pytest.mark.parametrize('eagle', [False, True])
def test_actual_urc_mode_host_insert_nodeid_chain_and_eviction(native, eagle):
    torch=native['torch'];bank=native['bank'];cache=make_cache(native,eagle)
    assert type(cache).__name__=='UnifiedRadixCache'
    assert type(cache.tree_core).__name__=='UnifiedTreeCore'
    assert str(cache.tree_core.device)=='cpu'
    assert bank._cache_mode(cache) is eagle
    assert not hasattr(cache,'is_eagle')
    root=bank._root(cache)
    assert type(cache.root_node_handle()) is int
    assert cache.resolve_node_handle(cache.root_node_handle()) is root
    assert bank._node_full(cache,root.id)[1] is root.component_data[native['full']]

    tokens=list(range(4097))
    key=bank._native_key(tokens,eagle)
    meta=bank.native_key_metadata(tokens,eagle)
    assert len(key)==4032 and len(meta['page_hashes'])==63
    assert meta['is_bigram'] is eagle
    slots=torch.arange(64,4096,dtype=torch.int64,device='cpu')
    bank._insert_host(cache,key,slots,meta['page_hashes'])
    match=bank._match(cache,tokens)
    assert type(match.last_host_node) is int and type(match.last_device_node) is int
    assert type(match.best_match_node) is int
    assert match.last_device_node==root.id
    assert match.host_hit_length==4032 and match.device_indices.numel()==0
    assert torch.equal(bank._host_indices(cache,match),slots)
    leaf=cache.resolve_node_handle(match.last_host_node)
    assert leaf.component_data[native['full']].value is None
    assert torch.equal(leaf.component_data[native['full']].host_value,slots)

    # A real partial match splits the host-only FULL path. Reassembling through
    # NodeIds must retain the exact allocator index order across both nodes.
    short=tokens[:2048+int(eagle)]
    partial=cache.match_prefix(native['match'](key=native['key'](array('q',short))))
    assert partial.host_hit_length==2048 and partial.device_indices.numel()==0
    match=bank._match(cache,tokens)
    leaf=cache.resolve_node_handle(match.last_host_node)
    assert leaf.parent is not root
    assert torch.equal(bank._host_indices(cache,match),slots)
    assert not match.cache_actions  # Actual URC consumes actions before return.

    # Device eviction leaves the CPU prefix intact. Host eviction uses the real
    # FULL component's free path with a recording CPU allocator.
    result=cache.evict(native['evict'](num_tokens=1<<60))
    assert result.num_tokens_evicted==0
    assert bank._match(cache,tokens).host_hit_length==4032
    freed=[]
    class HostSlots:
        def free(self,indices):freed.append(indices.clone())
    cache.components[native['full']]._full_kv_pool_host=HostSlots()
    assert cache.evict_host(1<<60)==4032
    assert torch.equal(torch.sort(torch.cat(freed)).values,slots)
    assert bank._match(cache,tokens).host_hit_length==0
    cache.reset()
    assert not bank._root(cache).children
    assert bank._match(cache,tokens).device_indices.numel()==0
    assert not cache.tree_core.has_ongoing_insert()


def test_actual_urc_cache_actions_consume_owned_cpu_indices_once(native):
    torch=native['torch'];cache=make_cache(native,False);freed=[]
    class Slots:
        def free_segment(self,indices,start_pos):
            assert start_pos==0
            freed.append(indices.clone())
    cache.token_to_kv_pool_allocator=Slots()
    indices=torch.arange(64,dtype=torch.int64,device='cpu')
    actions=[native['free_device']([indices])]
    cache._apply_cache_actions(actions)
    assert not actions and len(freed)==1 and torch.equal(freed[0],indices)
    cache._apply_cache_actions(actions)
    assert len(freed)==1
