import sys
from types import ModuleType, SimpleNamespace

import torch

from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.mem_cache.kvcached_pool import (
    KVCacheDMHATokenToKVPool,
    KVCacheDPagedTokenToKVPoolAllocator,
    KVCacheDTokenToKVPoolAllocator,
    _explicit_kvcached_device,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_indexless_cuda_device_resolves_to_process_active_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert _explicit_kvcached_device("cuda") == "cuda:3"
    assert _explicit_kvcached_device(torch.device("cuda:1")) == "cuda:1"
    assert _explicit_kvcached_device("cpu") == "cpu"


class _FakeKVCacheDManager:
    def __init__(self, size: int):
        self.size = int(size)
        self.num_blocks = self.size + 1
        self.layer_group_granularity = 2
        self.physical_group_page_size = 128
        self.blocks_per_physical_page = 2
        self.block_mem_size = 1
        self.block_level_layer_groups = True
        self.null_block = [0]
        self.page_allocator = SimpleNamespace(
            get_page_id=lambda block_id, block_mem_size: int(block_id) // 2
        )
        self.clear()

    def clear(self):
        self.free_ids = list(range(1, self.size + 1))
        self.live_ids: set[int] = set()

    def alloc(self, count: int):
        count = int(count)
        if count > len(self.free_ids):
            return None
        result = self.free_ids[:count]
        self.free_ids = self.free_ids[count:]
        self.live_ids.update(result)
        return result

    def free(self, page_ids):
        page_ids = [int(page_id) for page_id in page_ids]
        self.live_ids.difference_update(page_ids)
        self.free_ids = sorted(set(self.free_ids).union(page_ids))

    def available_size(self):
        return len(self.free_ids)

    def stats(self):
        return {
            "fake_live_pages": len(self.live_ids),
            "mapped_memory_bytes": 4096,
        }

    def get_layer_group_total_span(self):
        return ((self.num_blocks + 1) // 2) * self.physical_group_page_size


class _FakeKVCacheDPool(KVCacheDMHATokenToKVPool):
    @staticmethod
    def _detect_physical_layer_partitioning():
        return False

    def _get_kvcached_interfaces(self):
        return SimpleNamespace()

    def _allocate_kvcached_global_buffers(self):
        k_shape, v_shape = self._kv_buffer_shapes()
        return (
            [
                torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
                for _ in range(self._kvcached_total_num_layers)
            ],
            [
                torch.zeros(v_shape, dtype=self.store_dtype, device=self.device)
                for _ in range(self._kvcached_total_num_layers)
            ],
        )

    def _create_kvcached_manager(self):
        return _FakeKVCacheDManager(size=self.size // self.page_size)


class _RecordingPhysicalKVCacheDPool(_FakeKVCacheDPool):
    def __init__(self, *args, **kwargs):
        self.mapped: list[tuple[frozenset[int], torch.Tensor]] = []
        self.unmapped: list[frozenset[int]] = []
        self.fail_next_map = False
        self.fail_next_unmap = False
        super().__init__(*args, **kwargs)
        self.physical_layer_partitioned = True

    def _map_layer_slots(self, *, layer_ids, slot_ids):
        self.mapped.append(
            (frozenset(int(layer_id) for layer_id in layer_ids), slot_ids.clone())
        )
        if self.fail_next_map:
            self.fail_next_map = False
            raise RuntimeError("injected physical map failure")

    def _unmap_layer_groups(self, layer_ids):
        self.unmapped.append(frozenset(int(layer_id) for layer_id in layer_ids))
        if self.fail_next_unmap:
            self.fail_next_unmap = False
            raise RuntimeError("injected physical unmap failure")


def _make_pool(
    *,
    start_layer: int = 0,
    layer_num: int = 2,
    page_size: int = 1,
    pool_cls=_FakeKVCacheDPool,
) -> _FakeKVCacheDPool:
    pool = pool_cls(
        size=8,
        page_size=page_size,
        dtype=torch.float32,
        head_num=1,
        head_dim=2,
        layer_num=layer_num,
        total_num_layers=4,
        device="cpu",
        enable_memory_saver=False,
        start_layer=start_layer,
        end_layer=start_layer + layer_num,
        enable_alt_stream=False,
    )
    pool._elastic_pp_enabled = True
    return pool


def test_global_backing_is_rebound_only_at_transaction_commit():
    pool = _make_pool()
    global_layer_one = pool._kvcached_global_k_buffer[1]
    global_layer_two = pool._kvcached_global_k_buffer[2]
    global_layer_one[3].fill_(7)

    assert pool.k_buffer[1].data_ptr() == global_layer_one.data_ptr()
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(1, 3), generation=1, total_num_layers=4
    ) == frozenset({2})
    assert pool.start_layer == 0
    assert [item.data_ptr() for item in pool.k_buffer] == [
        pool._kvcached_global_k_buffer[0].data_ptr(),
        global_layer_one.data_ptr(),
    ]

    pool.import_elastic_pp_kv_slots(
        generation=1,
        layer_id=2,
        slot_ids=torch.tensor([3]),
        cache_k=torch.tensor([[[5.0, 5.0]]]),
        cache_v=torch.tensor([[[6.0, 6.0]]]),
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2})
    assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(1, 3)
    assert pool.k_buffer[0].data_ptr() == global_layer_one.data_ptr()
    assert pool.k_buffer[1].data_ptr() == global_layer_two.data_ptr()
    assert torch.equal(pool.get_key_buffer(2)[3], torch.tensor([[5.0, 5.0]]))

    released = pool.cleanup_elastic_pp_kv_layers(generation=1)
    assert len(released) == 2
    # Cleanup drops the old active view but never destroys global KVCacheD VA.
    assert pool._kvcached_global_k_buffer[0].data_ptr() == released[0].data_ptr()


def test_kvcached_token_allocator_tracks_bulk_and_delta_live_slots():
    pool = _make_pool()
    allocator = KVCacheDTokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    assert torch.equal(allocator.alloc(2), torch.tensor([1, 2]))
    assert torch.equal(
        allocator.begin_elastic_pp_slot_tracking(generation=4),
        torch.tensor([1, 2]),
    )
    allocated = allocator.alloc(2)
    assert torch.equal(allocated, torch.tensor([3, 4]))
    allocator.free(allocated[1:])
    assert torch.equal(
        allocator.elastic_pp_delta_slots(generation=4), torch.tensor([3])
    )
    allocator.finish_elastic_pp_slot_tracking(generation=4)
    assert allocator.available_size() == 5


def test_kvcached_paged_allocator_tracks_complete_page_rows():
    pool = _make_pool(page_size=2)
    allocator = KVCacheDPagedTokenToKVPoolAllocator(
        size=8,
        page_size=2,
        dtype=torch.float32,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    first = allocator.alloc(2)
    assert torch.equal(first, torch.tensor([2, 3]))
    assert torch.equal(
        allocator.begin_elastic_pp_slot_tracking(generation=2),
        torch.tensor([2, 3]),
    )
    second = allocator.alloc(2)
    assert torch.equal(second, torch.tensor([4, 5]))
    assert torch.equal(
        allocator.elastic_pp_delta_slots(generation=2), torch.tensor([4, 5])
    )
    allocator.free(second)
    assert allocator.elastic_pp_delta_slots(generation=2).numel() == 0
    allocator.finish_elastic_pp_slot_tracking(generation=2)


def test_kvcached_stats_do_not_claim_physical_layer_partitioning():
    pool = _make_pool(start_layer=1, layer_num=2)
    stats = pool.kvcached_physical_stats()
    assert stats["backing_num_layers"] == 4
    assert stats["active_start_layer"] == 1
    assert stats["active_end_layer"] == 3
    assert stats["physical_layer_partitioned"] is False
    assert stats["mapped_memory_scope"] == "allocator_estimate"


def test_kvcached_physical_stats_use_local_worker_mapping_ledger(monkeypatch):
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.physical_layer_partitioned = True
    fake_tp_ipc = ModuleType("kvcached.tp_ipc_util")
    fake_tp_ipc.snapshot_worker_mapped_offsets = lambda group_id=0: (0, 128, 640)
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", fake_tp_ipc)

    stats = pool.kvcached_physical_stats()

    assert stats["allocator_estimated_mapped_memory_bytes"] == 4096
    assert stats["mapped_memory_bytes"] == 384
    assert stats["mapped_memory_scope"] == "local_worker_offset_ledger"
    assert stats["local_worker_mapped_offset_count"] == 3
    assert stats["local_worker_mapped_layer_groups"] == [0, 1]


def test_layer_group_offsets_cover_null_and_sparse_live_pages():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.physical_layer_partitioned = True
    offsets = pool._offsets_for_layer_groups_and_slots(
        layer_ids={2, 3}, slot_ids=torch.tensor([1, 7])
    )
    group_span = ((pool.kvcached_allocator.num_blocks + 1) // 2) * 128
    assert offsets == [group_span, group_span + 3 * 128]


def test_layer_stacking_rejects_misaligned_topology_boundary():
    pool = _make_pool(start_layer=0, layer_num=2)
    pool.physical_layer_partitioned = True
    try:
        pool._active_layer_groups(LayerRange(1, 3))
    except ValueError as exc:
        assert "aligned" in str(exc)
        assert "granularity 2" in str(exc)
    else:
        raise AssertionError("misaligned KVCacheD layer ownership was accepted")


def _prepare_physical_move():
    pool = _make_pool(pool_cls=_RecordingPhysicalKVCacheDPool)
    allocator = KVCacheDTokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    assert torch.equal(allocator.alloc(2), torch.tensor([1, 2]))
    return pool, allocator


def test_physical_prepare_failure_rolls_back_mapping_and_transaction():
    pool, _ = _prepare_physical_move()
    pool.fail_next_map = True
    try:
        pool.prepare_elastic_pp_kv_layers(
            target=LayerRange(2, 4), generation=1, total_num_layers=4
        )
    except RuntimeError as exc:
        assert "injected physical map failure" in str(exc)
    else:
        raise AssertionError("injected KVCacheD map failure was ignored")

    assert pool.mapped[0][0] == frozenset({2, 3})
    assert torch.equal(pool.mapped[0][1], torch.tensor([1, 2]))
    assert pool.unmapped == [frozenset({2, 3})]
    assert pool._elastic_pp_pending_kv == {}
    assert pool._elastic_pp_residency.pending is None


def test_physical_abort_keeps_transaction_retryable_until_unmap_succeeds():
    pool, _ = _prepare_physical_move()
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    pool.fail_next_unmap = True
    try:
        pool.abort_elastic_pp_kv_layers(generation=1)
    except RuntimeError as exc:
        assert "injected physical unmap failure" in str(exc)
    else:
        raise AssertionError("injected KVCacheD unmap failure was ignored")

    assert set(pool._elastic_pp_pending_kv) == {2, 3}
    assert pool._elastic_pp_residency.pending.generation == 1
    pool.abort_elastic_pp_kv_layers(generation=1)
    assert pool._elastic_pp_pending_kv == {}
    assert pool._elastic_pp_residency.pending is None


def test_physical_cleanup_keeps_retired_mapping_retryable():
    pool, _ = _prepare_physical_move()
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={2, 3})
    assert pool.commit_elastic_pp_kv_layers(generation=1) == LayerRange(2, 4)
    pool.fail_next_unmap = True
    try:
        pool.cleanup_elastic_pp_kv_layers(generation=1)
    except RuntimeError as exc:
        assert "injected physical unmap failure" in str(exc)
    else:
        raise AssertionError("injected KVCacheD cleanup failure was ignored")

    assert set(pool._elastic_pp_retired_kv) == {0, 1}
    released = pool.cleanup_elastic_pp_kv_layers(generation=1)
    assert len(released) == 4
    assert pool._elastic_pp_retired_kv == {}
    assert pool.unmapped[-1] == frozenset({0, 1})
