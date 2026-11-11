from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport
from sglang.srt.elastic_pp.live_kv_patch import LiveKVSnapshot
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def fixture(*, size=4, max_rows=2, need_sort=False):
    allocator = TokenToKVPoolAllocator(size, torch.float32, "cpu", object(), need_sort)
    source = {
        7: (
            torch.arange((size + 1) * 2).view(size + 1, 2).float(),
            torch.arange(size + 1).view(size + 1, 1).float(),
        )
    }
    slots = allocator.alloc(2)
    patch = LiveKVSnapshot(
        generation=1,
        allocator=allocator,
        source=source,
        max_rows=max_rows,
        admit_host_bytes=lambda n: None,
    )
    return allocator, source, slots, patch


def ready():
    return SimpleNamespace(synchronize=lambda: None)


def test_real_allocator_online_snapshot_and_fence_include_unpublished_loadback():
    allocator, source, slots, patch = fixture()
    patch.progress(slots[:1], ready())
    # An asynchronously loaded cache row is allocated but has not been used
    # by any forward yet. It must not be speculatively marked ready.
    loaded = allocator.alloc(1)
    source[7][0][loaded] = 99
    patch.progress(slots, ready())
    assert loaded.item() not in patch.published
    assert patch.session.rows_copied == 1
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    final = allocator.elastic_pp_live_slots()
    for actual, expected in zip(patch.export(7, final), source[7], strict=True):
        assert torch.equal(actual, expected[final])
    assert patch.online_submissions == 2
    assert patch.complete and allocator._elastic_pp_live_patch is None
    assert patch.copier.source == {}


@pytest.mark.parametrize("need_sort", [False, True])
def test_free_waits_for_snapshot_then_reused_slot_gets_a_new_epoch(need_sort):
    allocator, source, slots, patch = fixture(size=2, need_sort=need_sort)
    patch.progress(slots, ready())
    allocator.free(slots[:1])
    assert patch.free_waits == 1
    again = allocator.alloc(1)
    assert again.item() == slots[0].item()
    source[7][0][again] = 123
    assert patch.epochs[again.item()] == 2
    patch.progress(again, ready())
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.export(7, again)[0].tolist() == [[123, 123]]


def test_grouped_frees_pin_until_the_actual_allocator_release():
    allocator, source, slots, patch = fixture(size=2)
    patch.progress(slots, ready())
    allocator.free_group_begin()
    allocator.free(slots[:1])
    assert slots[0].item() in patch.epochs and patch.free_waits == 0
    allocator.free_group_end()
    assert slots[0].item() not in patch.epochs and patch.free_waits == 1
    patch.abort()
    assert allocator._elastic_pp_live_patch is None


def test_abort_drains_pending_copy_and_preserves_serving_allocator():
    allocator, source, slots, patch = fixture()
    patch.progress(slots, ready())
    patch.abort()
    assert patch.session.phase == "aborted"
    assert allocator._elastic_pp_live_patch is None
    allocator.free(slots)
    allocator.clear()
    assert allocator.available_size() == allocator.size


def test_copy_failure_aborts_only_patch_and_rejects_transition(monkeypatch):
    allocator, source, slots, patch = fixture()

    def fail(_):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(patch.copier, "submit", fail)
    patch.progress(slots, ready())
    assert "injected copy failure" in patch.error
    assert allocator._elastic_pp_live_patch is None
    allocator.free(slots)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())


def test_host_admission_failure_never_installs_allocator_hook():
    allocator = TokenToKVPoolAllocator(2, torch.float32, "cpu", object(), False)

    def reject(_):
        raise MemoryError("budget")

    with pytest.raises(MemoryError, match="budget"):
        LiveKVSnapshot(
            generation=1,
            allocator=allocator,
            source={0: (torch.empty(3, 2), torch.empty(3, 1))},
            max_rows=1,
            admit_host_bytes=reject,
        )
    assert allocator._elastic_pp_live_patch is None


def test_allocator_clear_and_second_tracker_are_blocked_until_fence():
    allocator, source, slots, patch = fixture()
    with pytest.raises(RuntimeError, match="online KV patch"):
        allocator.clear()
    with pytest.raises(RuntimeError, match="cannot resize"):
        allocator.resize(SimpleNamespace(max_total_num_tokens=1))
    assert allocator.size == 4
    with pytest.raises(RuntimeError, match="already attached"):
        allocator.begin_elastic_pp_slot_tracking(generation=2)
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    allocator.clear()


def test_forward_mapping_publishes_used_prefix_not_all_allocated_rows():
    allocator, source, slots, patch = fixture()
    transport = object.__new__(ElasticPPKVTransport)
    transport._live_patch = patch
    transport._live_patch_error = None
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1]), seq_lens_cpu=torch.tensor([1])
    )
    table = SimpleNamespace(req_to_token=torch.tensor([[0, 0], [1, 2]]))
    transport.progress_live_patch(batch, ready(), table)
    assert patch.published == {1}
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())


def test_fence_detects_untracked_allocator_mutation():
    allocator, source, slots, patch = fixture()
    with pytest.raises(RuntimeError, match="live sets diverged"):
        patch.finish_at_fence(slots[:1], ready())
    patch.abort()


def test_rewrite_without_physical_free_advances_epoch_before_gpu_write():
    allocator, source, slots, patch = fixture()
    patch.progress(slots, ready())
    patch.prepare_writes(slots[:1])
    assert patch.epochs[slots[0].item()] == 2 and patch.free_waits == 1
    source[7][0][slots[0]] = 55
    patch.progress(slots[:1], ready())
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.export(7, slots[:1])[0].tolist() == [[55, 55]]


def page_fixture():
    allocator = PagedTokenToKVPoolAllocator(
        12, 4, torch.float32, "cpu", object(), False
    )
    slots = allocator.alloc(4)
    source = {3: (torch.zeros(16, 2), torch.zeros(16, 1))}
    patch = LiveKVSnapshot(
        generation=1,
        allocator=allocator,
        source=source,
        max_rows=2,
        admit_host_bytes=lambda n: None,
    )
    return allocator, source, slots, patch


def test_page_tail_append_is_not_published_as_completed_prefix():
    allocator, source, slots, patch = page_fixture()
    patch.progress(slots[:2], ready())
    assert patch.published == {4, 5}
    patch.prepare_writes(slots[2:3])
    source[3][0][slots[2:3]] = 29
    patch.progress(slots[2:3], ready())
    assert patch.epochs[6] == 1
    assert 7 not in patch.published
    patch.finish_at_fence(allocator.elastic_pp_live_slots(), ready())
    assert patch.export(3, slots)[0].tolist() == [[0, 0], [0, 0], [29, 29], [0, 0]]


@pytest.mark.parametrize("grouped", [False, True])
def test_page_segment_free_protects_every_row_and_page_reuse(grouped):
    allocator, source, slots, patch = page_fixture()
    patch.progress(slots[:2], ready())
    if grouped:
        allocator.free_group_begin()
    allocator.free_segment(slots[1:], start_pos=1)
    if grouped:
        assert patch.free_waits == 0
        allocator.free_group_end()
    assert patch.free_waits == 1 and patch.epochs == {}
    again = allocator.alloc(4)
    assert again.tolist() == slots.tolist()
    assert all(patch.epochs[i] == 2 for i in again.tolist())
    patch.abort()


def test_page_extend_and_decode_notify_full_new_pages_but_not_tail_reuse(monkeypatch):
    import sglang.srt.mem_cache.allocator.paged as module

    allocator, source, slots, patch = page_fixture()

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return self.fn

    def extend(prefix, seq, last, pages, out, bs, page_size):
        module.alloc_extend_naive(prefix, seq, last, pages, out, page_size, "cpu")

    def decode(seq, last, pages, out, bs, page_size):
        out.copy_(
            torch.where(seq % page_size == 1, pages[: len(seq)] * page_size, last + 1)
        )

    monkeypatch.setattr(module, "alloc_extend_kernel", Kernel(extend))
    monkeypatch.setattr(module, "alloc_decode_kernel", Kernel(decode))
    new = allocator.alloc_extend(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([-1]),
        3,
    )
    assert new.tolist() == [8, 9, 10]
    assert set(patch.epochs) == set(range(4, 12))
    tail = allocator.alloc_decode(torch.tensor([4]), torch.tensor([4]), new[-1:])
    assert tail.tolist() == [11] and patch.epochs[11] == 1
    next_page = allocator.alloc_decode(torch.tensor([5]), torch.tensor([5]), tail)
    assert next_page.tolist() == [12]
    assert set(patch.epochs) == set(range(4, 16))
    with pytest.raises(RuntimeError, match="cannot resize"):
        allocator.resize(SimpleNamespace(max_total_num_tokens=4))
    assert allocator.size == 12
    patch.abort()
