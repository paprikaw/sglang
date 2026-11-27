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
