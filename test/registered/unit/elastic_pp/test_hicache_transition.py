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
    node.last_access_time = 1e12
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
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        get_and_increase_time_counter,
    )

    assert get_and_increase_time_counter() > node.last_access_time
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


def test_publish_updates_pd_queues_before_retired_cache_controller_is_destroyed():
    from sglang.srt.disaggregation.decode_hicache_mixin import (
        DecodeHiCacheTransferMixin,
    )
    from sglang.srt.elastic_pp.hicache_transition import publish_scheduler_cache

    counter = NS(producer_index=0, num_counters=3)
    polled = []
    old = NS(
        cache_controller=NS(layer_done_counter=counter),
        is_load_back_event_done=lambda _: True,
    )
    new = NS(
        cache_controller=NS(layer_done_counter=counter),
        is_load_back_event_done=lambda index: polled.append(index) or True,
    )
    metadata_queue, socket = [], object()
    prealloc = NS(tree_cache=old, queue=metadata_queue, socket=socket)
    transfer = DecodeHiCacheTransferMixin()
    transfer.tree_cache, transfer.queue = old, []
    scheduler = NS(
        tree_cache=old,
        disagg_decode_prealloc_queue=prealloc,
        disagg_decode_transfer_queue=transfer,
    )
    publish_scheduler_cache(scheduler, new, NS(canary_manager=None))
    old.cache_controller = None
    # Invoke the real native Decode restore poll which failed on GPU after
    # commit, even with no pending restore requests.
    transfer._process_hicache_local_restores(transfer.queue)
    assert polled == [1]
    assert prealloc.tree_cache is transfer.tree_cache is new
    assert scheduler.disagg_decode_prealloc_queue is prealloc
    assert scheduler.disagg_decode_transfer_queue is transfer
    assert prealloc.queue is metadata_queue and prealloc.socket is socket


def test_allocator_shrink_clips_only_free_high_capacity():
    from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
    from sglang.srt.elastic_pp.hicache_transition import clone_host_allocator

    source = LogicalHostPool(8, 1)
    source.alloc(2)
    target = LogicalHostPool(4, 1)
    clone_host_allocator(source, target)
    assert target.free_slots.tolist() == [2, 3]
    source.alloc(4)
    with pytest.raises(RuntimeError, match="live high host slots"):
        clone_host_allocator(source, target)


def test_native_work_wrapper_is_waited():
    class Handle:
        waited = False

        def wait(self):
            self.waited = True

    handle = Handle()

    class NativeWire(Wire):
        def send_tensor_dict(self, payload, **kwargs):
            self.payload = payload
            return [NS(work=handle, payload=payload)]

    src = HiCacheTransitionAdapter(
        cache=cache(),
        pools={0: {3: {"k": torch.ones(5, 4)}}},
        group=NativeWire(),
        rank=0,
    )
    tx = transition()
    src.prepare_transition(tx)
    src.exchange_at_fence(tx)
    assert handle.waited


def test_failed_hybrid_host_construction_releases_new_full_pool(monkeypatch):
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as module

    released = []
    pool = NS(destroy=lambda: released.append(True))
    monkeypatch.setattr(
        module,
        "get_memory",
        lambda: NS(hicache_size=0, hicache_ratio=2, hicache_mem_layout="page_first"),
    )
    monkeypatch.setattr(module, "build_kv_host_pool", lambda **kwargs: pool)
    monkeypatch.setattr(module, "_get_allocator_type", lambda args: "default")

    def fail(*args, **kwargs):
        raise RuntimeError("injected target mamba allocation failure")

    monkeypatch.setattr(module, "MambaPoolHost", fail)
    params = NS(
        req_to_token_pool=NS(mamba_allocator=NS()),
        mtp_draft_device_pools=(),
        page_size=1,
    )
    with pytest.raises(RuntimeError, match="target mamba allocation"):
        module.build_hybrid_mamba_stack(
            params=params,
            server_args=None,
            kv_pool=NS(layer_num=1),
            mamba_pool=NS(num_mamba_layers=1),
            full_layer_mapping={0: 0},
            mamba_layer_mapping={1: 0},
            load_cache_event=None,
            storage_backend=None,
            use_mla=False,
        )
    assert released == [True]


def test_initial_later_elastic_stage_uses_local_host_layer_ids(
    monkeypatch, hybrid_pool_factory
):
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as module

    pool = hybrid_pool_factory(start_layer=2, end_layer=4)
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return NS(get_pool=lambda name: NS()), NS()

    monkeypatch.setattr(module, "build_hybrid_mamba_stack", build)
    module._MambaStrategy().build(
        cache=NS(),
        kvcache=pool,
        params=NS(req_to_token_pool=pool.req_to_token_pool),
        server_args=None,
        load_cache_event=None,
    )
    assert pool.full_attention_layer_id_mapping == {3: 1}
    assert captured["full_layer_mapping"] == {1: 0}
    assert captured["mamba_layer_mapping"] == {0: 0}


def test_discard_drops_frozen_controller_host_aliases():
    from sglang.srt.elastic_pp.hicache_transition import discard_host_cache
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    group = object()
    controller = NS(mem_pool_host=group, mem_pool_host_draft=group)
    component = NS(_full_kv_pool_host=group)
    released = []
    cache = NS(
        release_host_resources=lambda: released.append(True),
        cache_controller=controller,
        host_pool_group=group,
        full_kv_pool_host=group,
        token_to_kv_pool_host=group,
        components={ComponentType.FULL: component},
    )
    discard_host_cache(cache)
    assert released == [True]
    assert (
        cache.host_pool_group
        is cache.cache_controller
        is cache.full_kv_pool_host
        is None
    )
    assert component._full_kv_pool_host is controller.mem_pool_host is None


def test_fence_host_drain_keeps_idle_rank_in_all_rounds():
    from sglang.srt.elastic_pp.hicache_transition import drain_host_io_at_fence

    calls = []
    cache = NS(
        cache_controller=NS(load_queue=[], write_queue=[]),
        check_hicache_events=lambda: calls.append("check"),
        _drain_async_work=lambda: calls.append("retire"),
    )

    class Group:
        rounds = 0

        def all_gather_object(self, local):
            self.rounds += 1
            return [local, {"pending": self.rounds == 1, "expired": False}]

    group = Group()
    metrics = drain_host_io_at_fence(cache, group)
    assert group.rounds == 2
    assert calls == ["retire", "check", "retire"]
    assert metrics["host_io_event_rounds"] == 1


def test_idle_fence_retires_existing_sends_without_creating_another_round():
    from sglang.srt.elastic_pp.hicache_transition import drain_host_io_at_fence

    cache = NS(
        cache_controller=NS(load_queue=[], write_queue=[]),
        work_list=[object()],
        check_hicache_events=lambda: pytest.fail("idle fence created a PP packet"),
    )
    cache._drain_async_work = cache.work_list.clear
    flags = []

    def reduce(status):
        assert status.device.type == "cpu" and status.dtype == torch.int64
        flags.append(status.tolist())

    metrics = drain_host_io_at_fence(cache, NS(all_reduce_cpu=reduce))
    assert not cache.work_list
    assert flags == [[0, 0]]
    assert metrics["host_io_status_rounds"] == 1
    assert metrics["host_io_event_rounds"] == 0


def test_tensor_fence_keeps_idle_rank_polling_for_remote_io():
    from sglang.srt.elastic_pp.hicache_transition import drain_host_io_at_fence

    events = []
    cache = NS(
        cache_controller=NS(load_queue=[], write_queue=[]),
        check_hicache_events=lambda: events.append("poll"),
        _drain_async_work=lambda: events.append("retire"),
    )
    rounds = []

    def reduce(status):
        rounds.append(status.tolist())
        if len(rounds) == 1:
            status[0] += 1  # Another rank owns an unfinished transfer.

    metrics = drain_host_io_at_fence(cache, NS(all_reduce_cpu=reduce))
    assert rounds == [[0, 0], [0, 0]]
    assert events == ["retire", "poll", "retire"]
    assert metrics["host_io_event_rounds"] == 1


def test_fence_submits_and_retires_both_queued_io_directions():
    from sglang.srt.elastic_pp.hicache_transition import drain_host_io_at_fence

    events = []
    cc = NS(load_queue=[1], write_queue=[2], ack_load_queue=[], ack_write_queue=[])
    cache = NS(cache_controller=cc, _drain_async_work=lambda: None)

    def submit(direction):
        events.append(direction)
        getattr(cc, direction + "_queue").clear()
        getattr(cc, "ack_" + direction + "_queue").append(object())

    def poll():
        assert cc.ack_load_queue and cc.ack_write_queue
        events.append("finish")
        cc.ack_load_queue.clear()
        cc.ack_write_queue.clear()

    cc.start_loading = lambda: submit("load")
    cc.start_writing = lambda: submit("write")
    cache.check_hicache_events = poll
    metrics = drain_host_io_at_fence(cache, NS(all_reduce_cpu=lambda status: None))
    assert events == ["load", "write", "finish"]
    assert metrics["host_io_status_rounds"] == 2
    assert metrics["host_io_event_rounds"] == 1


def test_fence_does_not_ignore_a_remote_timeout():
    from sglang.srt.elastic_pp.hicache_transition import drain_host_io_at_fence

    cache = NS(
        cache_controller=NS(load_queue=[], write_queue=[]),
        _drain_async_work=lambda: None,
        check_hicache_events=lambda: pytest.fail("poll after timeout"),
    )

    def reduce(status):
        status += 1  # A peer has both pending I/O and an expired deadline.

    with pytest.raises(RuntimeError, match="timed out retiring HiCache"):
        drain_host_io_at_fence(cache, NS(all_reduce_cpu=reduce))


def test_canonical_snapshot_failure_is_delivered_to_dormant_peer(monkeypatch):
    from sglang.srt.elastic_pp import hicache_transition as module
    from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology

    tx = DynamicPPTransition.build(
        PPTopology.from_layer_counts([2, 2, 0], generation=0),
        PPTopology.from_layer_counts([1, 2, 1], generation=1),
    )
    cache = NS(
        cache_controller=NS(load_queue=[], write_queue=[]),
        host_pool_group=NS(entries=[]),
        check_hicache_events=lambda: None,
        _drain_async_work=lambda: None,
    )

    class Group(Wire):
        def all_gather_object(self, local):
            return [local]

    wire = Group()

    def fail(cache):
        raise RuntimeError("injected snapshot failure")

    monkeypatch.setattr(module, "snapshot_tree_metadata", fail)
    source = module.SchedulerHiCacheTransition(
        scheduler=NS(tree_cache=cache), model_runner=None, group=wire, rank=0
    )
    target = module.SchedulerHiCacheTransition(
        scheduler=NS(tree_cache=cache), model_runner=None, group=wire, rank=2
    )
    with pytest.raises(RuntimeError, match="injected snapshot failure"):
        source.prepare_transition(tx)
    with pytest.raises(RuntimeError, match="canonical HiCache snapshot failed"):
        target.prepare_transition(tx)


def _canonical_allocator_cache():
    # Deliberately noncontiguous free slots and an empty release batch.
    pool = NS(
        logical_size=8,
        free_slots=torch.tensor([6, 99, 2, 99, 5, 99, 1, 99])[::2],
        release_slots=[
            torch.tensor([4]),
            torch.empty(0, dtype=torch.int64),
            torch.tensor([7]),
        ],
        num_release_slots=2,
        slot_used=torch.tensor([True, False, False, True, False, False, False, False]),
        mem_state=torch.arange(8, dtype=torch.uint8),
    )
    return NS(host_pool_group=NS(entries=[NS(name="kv", host_pool=pool)]))


def test_canonical_allocator_native_wire_is_private_and_bypasses_tensor_pickle():
    import pickle

    from sglang.srt.distributed.parallel_state import _split_tensor_dict
    from sglang.srt.elastic_pp.hicache_transition import (
        canonical_host_allocator_wire,
        restore_canonical_host_allocators,
    )

    source = _canonical_allocator_cache()
    pool = source.host_pool_group.entries[0].host_pool
    wire = canonical_host_allocator_wire(source, 9)
    metadata, tensors = _split_tensor_dict(wire)
    assert len(tensors) == 6
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in tensors)

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_tensor(v) for v in value)
        return False

    # The real GroupCoordinator splitter leaves nested tensors in metadata.
    # This verifies that its actual pickle lane carries descriptors only.
    assert not contains_tensor(pickle.loads(pickle.dumps(metadata)))
    expected = restore_canonical_host_allocators(wire, 9)["kv"]
    assert expected["free_slots"].tolist() == [6, 2, 5, 1]
    assert [v.tolist() for v in expected["release_slots"]] == [[4], [], [7]]
    assert expected["slot_used"].dtype == torch.bool
    assert expected["mem_state"].dtype == torch.uint8
    for name in ["free_slots", "slot_used", "mem_state"]:
        assert (
            expected[name].untyped_storage().data_ptr()
            != getattr(pool, name).untyped_storage().data_ptr()
        )
    pool.free_slots.zero_()
    pool.release_slots[0].zero_()
    assert expected["free_slots"].tolist() == [6, 2, 5, 1]
    assert expected["release_slots"][0].tolist() == [4]
    expected["mem_state"].zero_()
    assert pool.mem_state.tolist() == list(range(8))


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "generation",
        "missing",
        "extra",
        "repeated",
        "pool",
        "encoding",
        "not_tensor",
    ],
)
def test_canonical_allocator_native_wire_rejects_incomplete_or_wrong_generation(fault):
    from sglang.srt.elastic_pp.hicache_transition import (
        _HOST_ALLOCATOR_MANIFEST,
        _HOST_ALLOCATOR_TENSOR_PREFIX,
        canonical_host_allocator_wire,
        restore_canonical_host_allocators,
    )

    wire = canonical_host_allocator_wire(_canonical_allocator_cache(), 9)
    manifest = wire[_HOST_ALLOCATOR_MANIFEST]
    fields = manifest["pools"][0][1]
    if fault in {"version", "generation"}:
        manifest[fault] += 1
    elif fault == "missing":
        del wire[fields["free_slots"][1]]
    elif fault == "extra":
        wire[_HOST_ALLOCATOR_TENSOR_PREFIX + "999"] = torch.tensor([99])
    elif fault == "repeated":
        fields["slot_used"] = fields["free_slots"]
    elif fault == "pool":
        manifest["pools"].append(manifest["pools"][0])
    elif fault == "encoding":
        fields["mem_state"] = ("unknown", None)
    else:
        wire[fields["free_slots"][1]] = [6, 2, 5, 1]
    with pytest.raises(RuntimeError, match="canonical host allocator"):
        restore_canonical_host_allocators(wire, 9)


@pytest.mark.parametrize("target_capacity", [4, 8, 10])
def test_allocator_resize_preserves_full_state_and_private_free_release_buffers(
    target_capacity,
):
    from sglang.srt.elastic_pp.hicache_transition import clone_host_allocator

    source = _canonical_allocator_cache().host_pool_group.entries[0].host_pool
    target = NS(
        logical_size=target_capacity,
        slot_used=torch.zeros(target_capacity, dtype=torch.bool),
        mem_state=torch.zeros(target_capacity, dtype=torch.uint8),
    )
    clone_host_allocator(source, target)
    assert (
        target.free_slots.tolist()
        == {4: [2, 1], 8: [6, 2, 5, 1], 10: [6, 2, 5, 1, 8, 9]}[target_capacity]
    )
    assert [v.tolist() for v in target.release_slots] == (
        [] if target_capacity == 4 else [[4], [7]]
    )
    assert target.num_release_slots == (0 if target_capacity == 4 else 2)
    assert (
        target.slot_used[: min(8, target_capacity)].tolist()
        == source.slot_used[:target_capacity].tolist()
    )
    target.free_slots.zero_()
    for released in target.release_slots:
        released.zero_()
    target.mem_state.zero_()
    assert source.free_slots.tolist() == [6, 2, 5, 1]
    assert [v.tolist() for v in source.release_slots] == [[4], [], [7]]
    assert source.mem_state.tolist() == list(range(8))
    source.slot_used[0] = False
    with pytest.raises(RuntimeError, match="free lists and used slots disagree"):
        clone_host_allocator(source, target)
