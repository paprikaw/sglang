import pytest
import torch

from sglang.srt.elastic_pp.kv_slots import ElasticKVSlotTracker
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_delta_includes_new_and_reused_slots_but_excludes_freed_slots():
    tracker = ElasticKVSlotTracker(
        generation=1,
        size=8,
        initial_live_slots=torch.tensor([1, 2, 3]),
    )
    tracker.record_alloc(torch.tensor([4, 5]))
    # Slot 2 belonged to the bulk snapshot, was freed, and was then reused.
    tracker.record_alloc(torch.tensor([2]))
    current_live = torch.tensor([1, 2, 3, 4])
    assert torch.equal(tracker.changed_live_slots(current_live), torch.tensor([2, 4]))


def test_token_allocator_tracks_only_during_transition():
    allocator = TokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    assert torch.equal(allocator.alloc(2), torch.tensor([1, 2]))
    assert torch.equal(
        allocator.begin_elastic_pp_slot_tracking(generation=3),
        torch.tensor([1, 2]),
    )
    allocated = allocator.alloc(2)
    assert torch.equal(allocated, torch.tensor([3, 4]))
    allocator.free(allocated[1:])
    assert torch.equal(
        allocator.elastic_pp_delta_slots(generation=3), torch.tensor([3])
    )
    allocator.finish_elastic_pp_slot_tracking(generation=3)
    assert allocator._elastic_pp_slot_tracker is None


def test_token_allocator_rejects_unstable_snapshot_and_clear():
    allocator = TokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    allocator.free_group_begin()
    with pytest.raises(RuntimeError, match="closed allocator free group"):
        allocator.begin_elastic_pp_slot_tracking(generation=1)
    allocator.free_group_end()

    allocator.begin_elastic_pp_slot_tracking(generation=1)
    with pytest.raises(RuntimeError, match="cannot clear"):
        allocator.clear()
    allocator.finish_elastic_pp_slot_tracking(generation=1)
    allocator.clear()


def test_token_allocator_snapshot_after_committed_free_group():
    allocator = TokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    allocated = allocator.alloc(2)
    allocator.free_group_begin()
    allocator.free(allocated[:1])

    with pytest.raises(RuntimeError, match="closed allocator free group"):
        allocator.begin_elastic_pp_slot_tracking(generation=1)

    allocator.free_group_end()
    assert allocator.is_not_in_free_group
    assert allocator.free_group == []
    assert torch.equal(
        allocator.begin_elastic_pp_slot_tracking(generation=1),
        allocated[1:],
    )
    allocator.finish_elastic_pp_slot_tracking(generation=1)
