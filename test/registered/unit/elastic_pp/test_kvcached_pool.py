import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.elastic_pp.coordinator import (
    ElasticPPMigrationCoordinator,
    ElasticPPMigrationPhase,
)
from sglang.srt.elastic_pp.kv_transport import ElasticPPKVTransport
from sglang.srt.elastic_pp.topology import DynamicPPTransition, LayerRange, PPTopology
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
        self.inuse_page_ids = [0, 2]
        self.trim_calls = 0
        self.reserved_pages = 3
        self.page_allocator = SimpleNamespace(
            get_page_id=lambda block_id, block_mem_size: int(block_id) // 2,
            get_num_reserved_pages=lambda: self.reserved_pages,
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

    def trim(self):
        self.trim_calls += 1
        self.reserved_pages = 0

    def snapshot_inuse_page_ids(self):
        return list(self.inuse_page_ids)


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


def test_hybrid_kvcached_uses_attention_ordinals_and_shared_logical_allocator(
    hybrid_pool_factory,
):
    pool = hybrid_pool_factory(
        start_layer=0, end_layer=3, full_pool_class=_FakeKVCacheDPool
    )
    allocator = KVCacheDTokenToKVPoolAllocator(
        size=8, dtype=torch.float32, device="cpu", kvcache=pool, need_sort=False
    )
    slots = allocator.alloc(2)
    assert pool.kvcached_allocator is pool.full_kv_pool.kvcached_allocator
    assert pool.full_kv_pool._kvcached_total_num_layers == 2
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    pool.prepare_elastic_pp_kv_fence(generation=1)
    pool.import_elastic_pp_kv_slots(
        generation=1,
        layer_id=3,
        slot_ids=slots,
        cache_k=torch.full((2, 1, 2), 31.0),
        cache_v=torch.full((2, 1, 2), 37.0),
    )
    pool.mark_elastic_pp_kv_layers_loaded(generation=1, layer_ids={3})
    pool.commit_elastic_pp_kv_layers(generation=1)
    assert pool.full_kv_pool.start_layer == 1
    assert torch.all(pool.get_key_buffer(3)[slots] == 31)
    pool.cleanup_elastic_pp_kv_layers(generation=1)
    allocator.free(slots)
    assert allocator.available_size() == 8


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


def test_kvcached_token_allocator_snapshot_after_committed_free_group():
    pool = _make_pool()
    allocator = KVCacheDTokenToKVPoolAllocator(
        size=8,
        dtype=torch.float32,
        device="cpu",
        kvcache=pool,
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


def test_physical_mapping_is_deferred_until_import_at_drained_fence():
    pool, _ = _prepare_physical_move()
    assert pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    ) == frozenset({2, 3})
    assert pool.mapped == []

    pool.fail_next_map = True
    try:
        pool.import_elastic_pp_kv_slots(
            generation=1,
            layer_id=2,
            slot_ids=torch.tensor([1, 2]),
            cache_k=torch.zeros((2, 1, 2)),
            cache_v=torch.zeros((2, 1, 2)),
        )
    except RuntimeError as exc:
        assert "injected physical map failure" in str(exc)
    else:
        raise AssertionError("injected KVCacheD map failure was ignored")

    assert pool.mapped[0][0] == frozenset({2, 3})
    assert torch.equal(pool.mapped[0][1], torch.tensor([1, 2]))
    assert set(pool._elastic_pp_pending_kv) == {2, 3}
    pool.abort_elastic_pp_kv_layers(generation=1)
    assert pool.unmapped == [frozenset({2, 3})]
    assert pool._elastic_pp_pending_kv == {}
    assert pool._elastic_pp_residency.pending is None


def test_physical_fence_refreshes_gained_groups_for_reusable_pages(monkeypatch):
    pool, _ = _prepare_physical_move()
    pool.prepare_elastic_pp_kv_layers(
        target=LayerRange(2, 4), generation=1, total_num_layers=4
    )
    mapped_offsets = []
    fake_tp_ipc = ModuleType("kvcached.tp_ipc_util")
    fake_tp_ipc.map_worker_offsets_for_migration = (
        lambda offsets, group_id=0: mapped_offsets.extend(offsets) or True
    )
    monkeypatch.setitem(sys.modules, "kvcached.tp_ipc_util", fake_tp_ipc)

    pool.prepare_elastic_pp_kv_fence(generation=1)

    group_span = pool.kvcached_allocator.get_layer_group_total_span()
    assert pool.kvcached_allocator.trim_calls == 1
    assert mapped_offsets == [group_span, group_span + 2 * 128]


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


def _make_snapshot_coordinator(*, rank=0, failure=None):
    """Use real coordinator/transport/allocator logic with CPU-only backing."""
    source = PPTopology.from_layer_counts([2, 2, 0, 0], generation=0)
    target = PPTopology.from_layer_counts([1, 1, 1, 1], generation=1)
    transition = DynamicPPTransition.build(source, target)
    pool = _make_pool(start_layer=source.layer_range(rank).start, layer_num=2)
    allocator = KVCacheDTokenToKVPoolAllocator(
        size=8, dtype=torch.float32, device="cpu", kvcache=pool, need_sort=False
    )

    def prepare_kv(*, target, generation):
        prepared = pool.prepare_elastic_pp_kv_layers(
            target=target, generation=generation, total_num_layers=4
        )
        if failure == "local_prepare":
            raise RuntimeError("injected failure after KV preparation")
        return prepared

    def gather(report):
        reports = [dict(report, rank=peer) for peer in range(4)]
        if failure == "peer_prepare" and report["stage"] == "prepare_allocations":
            reports[3].update(success=False, error="injected peer failure")
        return reports

    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        prepare_elastic_pp_kv_layers=prepare_kv,
        abort_elastic_pp_kv_layers=pool.abort_elastic_pp_kv_layers,
        abort_elastic_pp_layers=Mock(),
        elastic_pp_generation_ready=pool.elastic_pp_kv_ready,
        commit_elastic_pp_generation=pool.commit_elastic_pp_kv_layers,
        cleanup_elastic_pp_generation=pool.cleanup_elastic_pp_kv_layers,
    )
    group = SimpleNamespace(
        world_size=4,
        all_gather_object=gather,
        send_tensor_dict=Mock(return_value=[]),
    )
    transport = ElasticPPKVTransport(
        group=group, model_runner=runner, allocator=allocator, rank=rank
    )
    layers = Mock()
    layers.start_sends.return_value = ()
    coordinator = ElasticPPMigrationCoordinator(
        rank=rank,
        current_topology=source,
        group=group,
        model_runner=runner,
        layer_transport=layers,
        kv_transport=transport,
    )
    return coordinator, transition, pool, allocator


def _wait_for_snapshot_prepare(coordinator, transition):
    coordinator.begin(transition)
    snapshot = coordinator.wait_for_phase(
        {ElasticPPMigrationPhase.BULK_READY, ElasticPPMigrationPhase.FAILED}, timeout=5
    )
    assert snapshot.phase == ElasticPPMigrationPhase.BULK_READY, snapshot.error


def test_coordinator_prepare_during_free_group_copies_only_fence_live_slots(
    monkeypatch,
):
    coordinator, transition, pool, allocator = _make_snapshot_coordinator()
    snapshot = Mock(wraps=allocator.elastic_pp_live_slots)
    monkeypatch.setattr(allocator, "elastic_pp_live_slots", snapshot)
    allocated = allocator.alloc(3)
    allocator.free_group_begin()
    allocator.free(allocated[:1])

    _wait_for_snapshot_prepare(coordinator, transition)
    snapshot.assert_not_called()
    assert allocator._elastic_pp_slot_tracker is None
    coordinator.group.send_tensor_dict.assert_not_called()
    allocator.free_group_end()

    # Reuse one slot, free two others, then allocate both reused and new slots.
    # The payload must reflect the final state, not any pre-prepare snapshot.
    assert allocator.alloc(1).tolist() == [1]
    allocator.free(allocated[1:])
    assert allocator.alloc(3).tolist() == [2, 3, 4]
    allocator.free(torch.tensor([3]))
    expected = torch.tensor([1, 2, 4])
    pool.k_buffer[1][expected] = torch.tensor(
        [[[11.0, 12.0]], [[21.0, 22.0]], [[41.0, 42.0]]]
    )
    pool.v_buffer[1][expected] = pool.k_buffer[1][expected] + 100
    assert coordinator.finalize_at_fence(lambda topology: None) == transition.target

    snapshot.assert_called_once_with()
    bulk, delta = [
        call.args[0] for call in coordinator.group.send_tensor_dict.call_args_list
    ]
    assert torch.equal(bulk["elastic_pp_kv_slot_ids"], expected)
    assert torch.equal(
        bulk["elastic_pp_kv_tensor_000000"], pool._kvcached_global_k_buffer[1][expected]
    )
    assert torch.equal(
        bulk["elastic_pp_kv_tensor_000001"], pool._kvcached_global_v_buffer[1][expected]
    )
    assert delta["elastic_pp_kv_slot_ids"].numel() == 0
    assert allocator._elastic_pp_slot_tracker is None
    coordinator.start_cleanup()
    coordinator.wait_for_phase(ElasticPPMigrationPhase.DONE, timeout=5)
    assert pool._elastic_pp_residency.pending is None
    assert pool._elastic_pp_pending_kv == {}


def test_coordinator_prepare_does_not_trust_early_closed_free_group_flag(monkeypatch):
    coordinator, transition, _, allocator = _make_snapshot_coordinator()
    snapshot = Mock(wraps=allocator.elastic_pp_live_slots)
    monkeypatch.setattr(allocator, "elastic_pp_live_slots", snapshot)
    allocated = allocator.alloc(2)
    allocator.free_group_begin()
    allocator.free(allocated[:1])
    free_entered, release_free = threading.Event(), threading.Event()
    original_free = allocator.free
    errors = []

    def held_free(indices):
        free_entered.set()
        if not release_free.wait(timeout=10):
            raise TimeoutError("deferred-free test barrier timed out")
        original_free(indices)

    def finish_group():
        try:
            allocator.free_group_end()
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(allocator, "free", held_free)
    scheduler = threading.Thread(target=finish_group)
    scheduler.start()
    try:
        assert free_entered.wait(timeout=5)
        assert allocator.is_not_in_free_group and allocator.free_group == []
        assert allocator._live_page_ids == {1, 2}
        _wait_for_snapshot_prepare(coordinator, transition)
        snapshot.assert_not_called()
        assert allocator._elastic_pp_slot_tracker is None
    finally:
        release_free.set()
        scheduler.join(timeout=5)
    assert not scheduler.is_alive() and not errors

    coordinator.finalize_at_fence(lambda topology: None)
    snapshot.assert_called_once_with()
    bulk = coordinator.group.send_tensor_dict.call_args_list[0].args[0]
    assert bulk["elastic_pp_kv_slot_ids"].tolist() == [2]
    coordinator.start_cleanup()
    coordinator.wait_for_phase(ElasticPPMigrationPhase.DONE, timeout=5)


@pytest.mark.parametrize("failure", ["local_prepare", "peer_prepare", "fence_snapshot"])
def test_coordinator_snapshot_failure_aborts_backing_without_tracker(failure):
    coordinator, transition, pool, allocator = _make_snapshot_coordinator(
        rank=1, failure=failure
    )
    allocated = allocator.alloc(2)
    allocator.free_group_begin()
    allocator.free(allocated[:1])
    coordinator.begin(transition)
    snapshot = coordinator.wait_for_phase(
        {ElasticPPMigrationPhase.BULK_READY, ElasticPPMigrationPhase.FAILED}, timeout=5
    )
    if failure == "fence_snapshot":
        assert snapshot.phase == ElasticPPMigrationPhase.BULK_READY
        # The guard must remain active if a caller violates the drained-fence
        # contract; removing the unsafe background read must not weaken it.
        with pytest.raises(RuntimeError, match="closed allocator free group"):
            coordinator.finalize_at_fence(lambda topology: None)
    else:
        assert snapshot.phase == ElasticPPMigrationPhase.FAILED
        assert "injected" in snapshot.error
    assert coordinator.phase == ElasticPPMigrationPhase.FAILED
    assert coordinator.snapshot().source_usable
    assert coordinator.current_topology == transition.source
    assert (pool.start_layer, pool.end_layer) == (2, 4)
    assert pool._elastic_pp_pending_kv == {}
    assert pool._elastic_pp_residency.pending is None
    assert allocator._elastic_pp_slot_tracker is None
    allocator.free_group_end()
    assert allocator.elastic_pp_live_slots().tolist() == [2]
