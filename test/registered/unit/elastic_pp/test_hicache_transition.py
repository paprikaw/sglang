from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.hicache_transition import (
    HiCacheTransitionAdapter,
    host_layer_views,
    host_nodes,
)


def cache(slot=2, tokens=(1, 2), namespace=None):
    def key(tokens):
        return NS(
            extra_key=namespace,
            cache_salt=None,
            is_bigram=False,
            raw_token_ids=lambda: tokens,
        )

    root = NS(key=key(()), parent=None, component_data=[], children={})
    child = NS(
        key=key(tokens),
        parent=root,
        component_data=[NS(host_value=torch.tensor([slot]))],
        children={},
    )
    root.children[1] = child
    return NS(tree_core=NS(root_node=root))


def transition():
    edge = NS(src_rank=0, dst_rank=1, layer_range=NS(start=3, end=4))
    return NS(target=NS(generation=1), kv_transfers=[edge])


class Wire:
    def send_tensor_dict(self, payload, **kwargs):
        self.payload = payload
        return []

    def recv_tensor_dict(self, **kwargs):
        return self.payload


def test_cpu_only_prefix_moves_with_global_layer_and_local_slot_remap():
    wire = Wire()
    source = torch.arange(20).reshape(5, 4).float()
    target = torch.zeros(5, 4)
    src = HiCacheTransitionAdapter(
        cache=cache(2), pools={0: {3: {"k": source}}}, group=wire, rank=0
    )
    dst = HiCacheTransitionAdapter(cache=cache(4), pools={}, group=wire, rank=1)
    tx = transition()
    src.prepare_transition(tx)
    dst.prepare_transition(tx)
    src.exchange_at_fence(tx)
    dst.exchange_at_fence(tx)
    dst.apply_staged(tx, target_cache=cache(4), target_pools={0: {3: {"k": target}}})
    assert torch.equal(target[4], source[2])
    dst.abort(tx)
    assert not target.any()


def test_identity_ignores_local_slot_and_separates_namespace():
    assert set(host_nodes(cache(2))) == set(host_nodes(cache(4)))
    assert set(host_nodes(cache(namespace="a"))) != set(
        host_nodes(cache(namespace="b"))
    )


def test_missing_prefix_fails_before_mutation():
    adapter = HiCacheTransitionAdapter(cache=cache(), pools={}, group=Wire(), rank=1)
    tx = transition()
    adapter.prepare_transition(tx)
    identity, component = next(iter(host_nodes(cache())))
    adapter._staged[(identity, component, 3)] = {"k": torch.ones(1, 4)}
    target = torch.zeros(5, 4)
    with pytest.raises(RuntimeError, match="lacks source CPU prefix"):
        adapter.apply_staged(
            tx, target_cache=cache(tokens=(9,)), target_pools={0: {3: {"k": target}}}
        )
    assert not target.any()


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
def test_mha_views_map_sparse_global_layers(layout):
    data = torch.zeros(2, 5, 1, 3)
    if layout == "page_first":
        data = data.transpose(0, 1)
    pool = NS(k_buffer=data, v_buffer=data.clone(), layer_num=2, layout=layout)
    views = host_layer_views(pool, [7, 12])
    views[12]["k"][3].fill_(9)
    assert (data[1, 3] if layout == "layer_first" else data[3, 1]).eq(9).all()


def test_mamba_views_keep_distinct_global_layer_ids():
    pool = NS(
        conv_buffer=[torch.zeros(5, 2, 1, 3)],
        temporal_buffer=torch.zeros(5, 2, 1, 3, 3),
        num_mamba_layers=2,
        layout="page_first",
    )
    views = host_layer_views(pool, [0, 4])
    views[4]["temporal"][2].fill_(7)
    assert pool.temporal_buffer[2, 1].eq(7).all()


def test_pending_io_rejected():
    source = cache()
    source.ongoing_backup = {42: object()}
    adapter = HiCacheTransitionAdapter(cache=source, pools={}, group=Wire(), rank=0)
    with pytest.raises(RuntimeError, match="ongoing_backup"):
        adapter.prepare_transition(transition())


def test_snapshot_cannot_alias_source_mutation():
    wire = Wire()
    source = torch.ones(5, 4)
    src = HiCacheTransitionAdapter(
        cache=cache(2), pools={0: {3: {"k": source}}}, group=wire, rank=0
    )
    tx = transition()
    src.prepare_transition(tx)
    assert src.snapshot_at_fence(tx) == 16
    source.zero_()
    src.exchange_at_fence(tx)
    assert wire.payload["0:k"].eq(1).all()


def test_allocator_clone_retains_order_and_appends_capacity():
    from sglang.srt.elastic_pp.hicache_transition import clone_host_allocator

    src = NS(
        logical_size=3,
        free_slots=torch.tensor([2, 0]),
        release_slots=[],
        num_release_slots=0,
        mem_state=torch.tensor([0, 1, 0], dtype=torch.uint8),
        slot_used=torch.tensor([False, True, False]),
    )
    dst = NS(
        logical_size=5,
        mem_state=torch.zeros(5, dtype=torch.uint8),
        slot_used=torch.zeros(5, dtype=torch.bool),
    )
    clone_host_allocator(src, dst)
    assert dst.free_slots.tolist() == [2, 0, 3, 4]
    assert dst.slot_used.tolist() == [False, True, False, False, False]
    src.free_slots.zero_()
    assert dst.free_slots.tolist() == [2, 0, 3, 4]


def test_real_tree_snapshot_preserves_arena_lru_and_private_components():
    from array import array
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.elastic_pp.hicache_transition import (
        snapshot_tree_metadata,
        restore_tree_metadata,
    )

    ct = ComponentType.FULL
    component = NS()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_components=(ct,),
    )
    tree = UnifiedTreeCore(params, {ct: component})
    node = tree._new_node()
    node.parent = tree.root_node
    node.key = RadixKey(array("q", [1, 2]))
    node.component_data[ct].value = torch.tensor([3, 4])
    node.component_data[ct].host_value = torch.tensor([5, 6])
    tree.root_node.children[1] = node
    tree.lru_lists[ct].insert_mru(node)
    tree.component_evictable_size_[ct] = 2
    source = NS(tree_core=tree)
    payload = snapshot_tree_metadata(source)
    target_component = NS()
    target = NS(tree_core=tree, components={ct: target_component})
    restored = restore_tree_metadata(payload, target)
    assert restored is not tree
    assert restored._node_arena[node.id] is restored.root_node.children[1]
    assert restored.lru_lists[ct].cache[node.id] is restored._node_arena[node.id]
    assert restored.component_evictable_size_[ct] == 2
    assert restored.components == (target_component,)
    assert target_component.tree_core is restored
    assert restored._empty_match_result.device_indices.device == torch.device("cpu")
    restored._node_arena[node.id].component_data[ct].host_value.zero_()
    assert tree._node_arena[node.id].component_data[ct].host_value.tolist() == [5, 6]


def test_project_pending_hybrid_does_not_publish(hybrid_pool_factory):
    from sglang.srt.elastic_pp.hicache_transition import project_target_device_pool
    from sglang.srt.elastic_pp.topology import LayerRange

    pool = hybrid_pool_factory(start_layer=0, end_layer=2)
    original_state = pool.mamba_pool.mamba_cache
    original_map = dict(pool.req_to_token_pool.mamba_map)
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(0, 4), generation=1, total_num_layers=4
    )
    view, req = project_target_device_pool(pool, target=LayerRange(0, 4), generation=1)
    assert pool.end_layer == 2
    assert pool.mamba_pool.mamba_cache is original_state
    assert pool.req_to_token_pool.mamba_map == original_map
    assert view.full_attention_layer_id_mapping == {1: 0, 3: 1}
    assert req.mamba_map == {0: 0, 2: 1}
    assert req.mamba_pool.mamba_cache is pool._pending_state
    pool.abort_elastic_pp_kv_layers(generation=1)


def test_logical_host_allocator_preserves_slot_order_into_physical():
    from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
    from sglang.srt.elastic_pp.hicache_transition import clone_host_allocator

    logical = LogicalHostPool(8, 1)
    assert logical.alloc(3).tolist() == [0, 1, 2]
    logical.free(torch.tensor([1]))
    physical = NS(
        logical_size=8,
        mem_state=torch.zeros(8, dtype=torch.uint8),
        slot_used=torch.zeros(8, dtype=torch.bool),
    )
    clone_host_allocator(logical, physical)
    assert physical.free_slots.tolist() == [3, 4, 5, 6, 7]
    assert physical.slot_used.tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_zero_layer_loading_event_has_completion(monkeypatch):
    import sglang.srt.managers.cache_controller as module

    class Event:
        def query(self):
            return True

    monkeypatch.setattr(module, "device_module", NS(Event=Event))
    counter = module.LayerDoneCounter(0)
    assert counter.update_producer() == 0
    counter.set_consumer(0)
    counter.wait_until(0)
    assert counter.events[0].finish_event is counter.events[0].start_event


def test_publish_rebinds_batches_policy_and_sessions():
    from sglang.srt.elastic_pp.hicache_transition import publish_scheduler_cache

    old, new = object(), object()
    batch, policy, session = NS(tree_cache=old), NS(tree_cache=old), NS(tree_cache=old)
    scheduler = NS(
        tree_cache=old, running_mbs=[batch], policy=policy, session_controller=session
    )
    publish_scheduler_cache(scheduler, new, NS(canary_manager=None))
    assert all(
        holder.tree_cache is new for holder in (scheduler, batch, policy, session)
    )
