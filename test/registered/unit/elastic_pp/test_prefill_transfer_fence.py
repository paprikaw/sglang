import pytest
from sglang.srt.elastic_pp.prefill_transfer import PrefillTransferFence


def test_queued_and_deferred_chunks_pin_the_source():
    fence = PrefillTransferFence()
    first, second = object(), object()
    fence.admit(first)
    fence.admit(second)
    fence.begin_io(23)
    assert fence.sending(23) and not fence.sending(24)
    with pytest.raises(RuntimeError):
        fence.close()
    fence.end_io(23)
    fence.finish(first)
    assert fence.pending() == 1
    with pytest.raises(RuntimeError):
        fence.close()
    fence.finish(second)
    fence.close()
    with pytest.raises(RuntimeError):
        fence.admit(first)
    fence.reopen()
    fence.admit(first)
    fence.finish(first)


def test_shrink_demotes_only_safe_backed_high_rows():
    from types import SimpleNamespace as NS
    from sglang.srt.elastic_pp.pd_registration import PrefillRegistrationTransaction
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    nodes = [NS(id=i, backuped=backed, locked=locked,
                component_data={ComponentType.FULL: NS(value=[row])})
             for i, (row, backed, locked) in enumerate(
                 [(3, True, False), (9, True, False), (10, False, False), (11, True, True)])]
    allocator = NS(serving_capacity=12, _as_int_list=list, _live_page_ids={3, 9, 10, 11})
    tree = NS(evictable_device_leaves=nodes.copy(), _is_device_leaf=lambda n: not n.locked)
    def demote(i, tracker):
        node = nodes[i]
        allocator._live_page_ids.remove(node.component_data[ComponentType.FULL].value[0])
        tree.evictable_device_leaves.remove(node)
        tracker[ComponentType.FULL] += 1
    scheduler = NS(tree_cache=NS(tree_core=tree, _demote=demote), ps=NS(pp_rank=0),
                   disagg_prefill_bootstrap_queue=NS(kv_manager=NS(prefill_transfer_fence=PrefillTransferFence())))
    PrefillRegistrationTransaction(scheduler).demote_cached_rows_for_shrink(allocator, 8)
    assert allocator._live_page_ids == {3, 10, 11}


def test_shrink_can_offload_low_cache_to_make_room_for_high_active_rows():
    from types import SimpleNamespace as NS

    from sglang.srt.elastic_pp.pd_registration import PrefillRegistrationTransaction
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    gdn = NS(value=[4], host_value=None)
    leaf = NS(id=7, backuped=True, component_data={
        ComponentType.FULL: NS(value=[1]), ComponentType.MAMBA: gdn,
    })
    allocator = NS(serving_capacity=8, _as_int_list=list, _live_page_ids={1, 2, 7})
    leaves = [leaf]
    tree = NS(
        evictable_device_leaves=leaves, _is_device_leaf=lambda node: node is leaf,
        _build_backup_kv_action=lambda node: node.id,
    )
    events = []

    def write(node_id, *, write_back):
        assert node_id == leaf.id and write_back
        events.append("backup")

    def ack(*, write_back):
        assert write_back
        events.append("ack")
        gdn.host_value = [0]

    def demote(node_id, tracker):
        assert node_id == leaf.id and gdn.host_value is not None
        events.append("demote")
        leaves.clear()
        allocator._live_page_ids.remove(1)
        tracker[ComponentType.FULL] += 1

    cache = NS(
        tree_core=tree, _execute_and_commit_kv_backup=write,
        writing_check=ack, _demote=demote,
    )
    scheduler = NS(
        tree_cache=cache, ps=NS(pp_rank=0),
        disagg_prefill_bootstrap_queue=NS(
            kv_manager=NS(prefill_transfer_fence=PrefillTransferFence())
        ),
    )
    group = NS(all_gather_object=lambda value: [value, value])
    registration = PrefillRegistrationTransaction(scheduler)
    assert registration.demote_cached_rows_for_shrink(
        allocator, 2, write_missing=True, group=group
    ) == 1
    assert events == ["backup", "ack", "demote"]
    assert allocator._live_page_ids == {2, 7}


def test_source_trim_unmaps_only_after_rdma_retirement():
    from types import SimpleNamespace as NS

    from sglang.srt.elastic_pp.pd_registration import PrefillRegistrationTransaction

    fence = PrefillTransferFence()
    events = []
    resident = {(100, 4), (104, 4), (200, 8)}
    pool = NS(serving_capacity=7)

    def kv_regions(capacity=None):
        capacity = pool.serving_capacity if capacity is None else capacity
        return [(100, 4), (104, 4)] if capacity == 7 else [(100, 4)]

    def resize(capacity):
        events.append(("resize", capacity))
        pool.serving_capacity = capacity

    def reclaim():
        assert resident == set(kv_regions()) | {(200, 8)}
        events.append("unmap")

    def unregister(addresses):
        events.append(("deregister", addresses))
        resident.difference_update({row for row in resident if row[0] in addresses})
        return 0

    def register(addresses, sizes):
        events.append(("register", addresses))
        resident.update(zip(addresses, sizes, strict=True))
        return 0

    pool.pd_registration_regions = kv_regions
    pool.resize_serving_capacity_at_fence = resize
    pool.reclaim_capacity_at_fence = reclaim
    manager = NS(
        kv_args={},
        prefill_transfer_fence=fence,
        engine=NS(batch_register=register, batch_deregister=unregister),
    )
    manager._registerable_regions = lambda: manager.kv_args["regions"]
    queue = NS(
        kv_manager=manager,
        _build_kv_args=lambda: {"regions": [*kv_regions(), (200, 8)]},
    )
    manager.kv_args = queue._build_kv_args()
    registration = PrefillRegistrationTransaction(
        NS(disagg_prefill_bootstrap_queue=queue)
    )
    registration.resize_source_capacity_at_fence(pool, 3, generation=2)
    assert events == [("deregister", [104]), ("resize", 3), "unmap"]
    registration.resize_source_capacity_at_fence(pool, 7, generation=2)
    assert events[3:] == [("resize", 7), ("register", [104]), "unmap"]
