from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.hicache_precopy import (
    HiCachePrecopy,
    HostRecord,
    copy_records,
    rows_match,
)
from sglang.srt.elastic_pp.hicache_transition import (
    HiCacheTransitionAdapter,
    SchedulerHiCacheTransition,
    adopt_prepared_host_stack,
    host_nodes,
)


def cache(slots=(1, 2), tokens=(7, 8)):
    def key(value):
        return NS(
            extra_key=None,
            cache_salt=None,
            is_bigram=False,
            raw_token_ids=lambda: value,
        )

    root = NS(key=key(()), parent=None, component_data=[], children={})
    child = NS(
        key=key(tokens),
        parent=root,
        component_data=[NS(host_value=torch.tensor(slots))],
        children={},
    )
    root.children[7] = child
    return NS(tree_core=NS(root_node=root))


def transition():
    return NS(
        target=NS(generation=1),
        kv_transfers=[NS(src_rank=0, dst_rank=1, layer_range=NS(start=3, end=4))],
    )


class Wire:
    def send_tensor_dict(self, payload, **kwargs):
        self.payload = payload
        return []

    def recv_tensor_dict(self, **kwargs):
        return self.payload


@pytest.mark.parametrize(
    "change", ["unchanged", "overwrite", "remap", "split", "new_prefix"]
)
def test_background_then_fence_sends_only_unchanged_proven_rows(change):
    source_cache, target_cache = cache(), cache()
    identity, component = next(iter(host_nodes(source_cache)))
    values = torch.arange(24).reshape(6, 4).float()
    original = HostRecord(torch.tensor([1, 2]), {"k": values[1:3].clone()})
    key = (identity, component, 3)
    src = HiCachePrecopy(
        generation=1, records={key: original}, metadata_cache=None, captured_seconds=0
    )
    dst = HiCachePrecopy(
        generation=1, records={}, metadata_cache=None, captured_seconds=0
    )
    target = torch.full_like(values, -1)
    pools = {0: {3: {"k": target}}}
    wire = Wire()
    tx = transition()
    src.exchange(tx, group=wire, rank=0, target_pools={})
    dst.exchange(tx, group=wire, rank=1, target_pools=pools)
    assert torch.equal(target[1:3], original.tensors["k"])
    if change == "overwrite":
        values[1].add_(100)
    elif change == "remap":
        source_cache, target_cache = cache((3, 4)), cache((4, 5))
    elif change in {"split", "new_prefix"}:
        source_cache, target_cache = cache(tokens=(7,)), cache(tokens=(7,))
    sender = HiCacheTransitionAdapter(
        cache=source_cache,
        pools={0: {3: {"k": values}}},
        group=wire,
        rank=0,
        fast_copy=True,
    )
    receiver = HiCacheTransitionAdapter(
        cache=target_cache, pools={}, group=wire, rank=1, fast_copy=True
    )
    sender.precopy, receiver.precopy = src, dst
    sender.prepare_transition(tx)
    receiver.prepare_transition(tx)
    sender.snapshot_at_fence(tx)
    sender.exchange_at_fence(tx)
    assert bool(wire.payload["manifest"]) is (change != "unchanged")
    receiver.exchange_at_fence(tx)
    receiver.apply_staged(
        tx,
        target_cache=NS(**vars(target_cache)),
        target_pools=pools,
        private_target=True,
    )
    src_slots = next(iter(host_nodes(source_cache).values())).slots
    dst_slots = next(iter(host_nodes(target_cache).values())).slots
    assert torch.equal(
        target.index_select(0, dst_slots), values.index_select(0, src_slots)
    )
    assert source_cache is sender.cache
    if change == "unchanged":
        assert sender.metrics["host_bytes_sent"] == 0
        assert receiver.metrics["host_preloaded_incoming_bytes_reused"] == 32
    else:
        assert sender.metrics["host_bytes_sent"] == 32


def test_receiver_slot_remap_copies_base_without_resending_unchanged_source():
    source = torch.arange(24).reshape(6, 4).float()
    identity, component = next(iter(host_nodes(cache())))
    key = (identity, component, 3)
    record = HostRecord(torch.tensor([1, 2]), {"k": source[1:3].clone()})
    precopy = HiCachePrecopy(
        generation=1, records={}, metadata_cache=None, captured_seconds=0
    )
    precopy.preloaded[key] = record
    target = torch.full_like(source, -1)
    copy_records({0: {3: {"k": target}}}, precopy.preloaded)
    adapter = HiCacheTransitionAdapter(
        cache=cache((4, 5)), pools={}, group=None, rank=1, fast_copy=True
    )
    adapter.prepare_transition(transition())
    adapter.precopy = precopy
    adapter._staged[key] = record.tensors
    adapter.apply_staged(
        transition(), target_cache=adapter.cache, target_pools={0: {3: {"k": target}}}
    )
    assert torch.equal(target[4:6], source[1:3])
    assert adapter.metrics.get("host_preloaded_incoming_bytes_reused", 0) == 0


@pytest.mark.parametrize("slots", [[-1], [8], [1, 1]])
def test_preload_rejects_invalid_rows_before_any_write(slots):
    target = torch.zeros(6, 4)
    record = HostRecord(torch.tensor(slots), {"k": torch.ones(len(slots), 4)})
    with pytest.raises(ValueError, match="invalid background"):
        copy_records({0: {3: {"k": target}}}, {("id", 0, 3): record})
    assert not target.any()


def test_preload_rejects_storage_alias_and_same_slot_overwrite_is_not_reused():
    target = torch.zeros(6, 4)
    record = HostRecord(torch.tensor([1, 2]), {"k": target[1:3]})
    with pytest.raises(ValueError, match="invalid background"):
        copy_records({0: {3: {"k": target}}}, {("id", 0, 3): record})
    independent = HostRecord(torch.tensor([1, 2]), {"k": target[1:3].clone()})
    assert rows_match({"k": target}, independent, independent.slots)
    target[2].fill_(9)
    assert not rows_match({"k": target}, independent, independent.slots)


def test_background_generation_mismatch_does_not_touch_target():
    precopy = HiCachePrecopy(
        generation=2, records={}, metadata_cache=None, captured_seconds=0
    )
    with pytest.raises(RuntimeError, match="generation mismatch"):
        precopy.exchange(transition(), group=Wire(), rank=1, target_pools={})


def test_capture_budget_precedes_copy_and_owned_snapshot_survives_slot_reuse(
    monkeypatch,
):
    from sglang.srt.elastic_pp import hicache_transition as module

    live = cache()
    values = torch.arange(24).reshape(6, 4).float()
    pools = {0: {3: {"k": values}}}
    with pytest.raises(RuntimeError, match="byte cap"):
        HiCachePrecopy.capture(live, pools, 1, max_bytes=1)
    monkeypatch.setattr(
        module, "clone_cache_runtime_metadata", lambda _: NS(components={})
    )
    captured = HiCachePrecopy.capture(live, pools, 1)
    record = next(iter(captured.records.values()))
    before = record.tensors["k"].clone()
    values.fill_(-1)
    live.tree_core.root_node.children[7].component_data[0].host_value.fill_(5)
    assert torch.equal(record.slots, torch.tensor([1, 2]))
    assert torch.equal(record.tensors["k"], before)
    captured.reset()
    assert not captured.records and captured.metadata_cache is None


def test_prepared_backing_is_moved_without_reusing_stale_tree_or_live_source():
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    ct = ComponentType.FULL
    pool = object()
    controller = NS(mem_pool_device_allocator=object())
    group = object()
    prepared = NS(
        host_pool_group=group,
        cache_controller=controller,
        load_cache_event=object(),
        token_to_kv_pool_host=group,
        full_kv_pool_host=pool,
        components={ct: NS(_kv_pool_host=pool)},
    )
    fresh_tree = object()
    clone = NS(tree_core=fresh_tree, components={ct: NS()})
    allocator, params = object(), object()
    adopt_prepared_host_stack(clone, prepared, allocator=allocator, params=params)
    assert clone.tree_core is fresh_tree
    assert clone.host_pool_group is group and clone.cache_controller is controller
    assert clone.full_kv_pool_host is pool
    assert clone._storage_attachment._cache is clone
    assert controller.mem_pool_device_allocator is allocator
    assert prepared.host_pool_group is None and prepared.cache_controller is None
    with pytest.raises(RuntimeError, match="already been consumed"):
        adopt_prepared_host_stack(clone, prepared, allocator=allocator, params=params)


def test_abort_discards_private_background_backing_and_resets_usage(monkeypatch):
    from sglang.srt.elastic_pp import hicache_transition as module

    source = object()
    scheduler = NS(tree_cache=source)
    worker = NS(
        server_args=NS(
            elastic_pp_hicache_fast_copy=True,
            elastic_pp_hicache_background=True,
            elastic_pp_transition_strategy="background",
            elastic_pp_transition_backing="overlap",
        )
    )
    manager = SchedulerHiCacheTransition(
        scheduler=scheduler, model_runner=worker, group=None, rank=0
    )
    manager.precopy = HiCachePrecopy(
        generation=1,
        records={"owned": HostRecord(torch.tensor([0]), {"k": torch.ones(1, 4)})},
        metadata_cache=None,
        captured_seconds=0,
    )
    precopy = manager.precopy
    manager.prepared_host = NS(host_pool_group=object())
    discarded = []
    monkeypatch.setattr(module, "discard_host_cache", discarded.append)
    manager.abort(transition())
    assert len(discarded) == 1
    assert manager.precopy is None and manager.prepared_host is None
    assert not precopy.records and not precopy.preloaded
    assert scheduler.tree_cache is source


def test_capacity_collective_override_is_thread_local_and_restored(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from sglang.srt.distributed import parallel_state
    from sglang.srt.mem_cache.pool_host.base import (
        hicache_capacity_group,
        sync_fixed_hicache_size,
    )

    serving, migration = NS(world_size=2, cpu_group="serving"), NS(
        world_size=2, cpu_group="migration"
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: serving)
    groups = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, **kw: groups.append(kw["group"])
    )
    with hicache_capacity_group(migration):
        assert sync_fixed_hicache_size(8, 1) == 8
        with ThreadPoolExecutor(1) as executor:
            assert executor.submit(sync_fixed_hicache_size, 8, 1).result() == 8
    assert sync_fixed_hicache_size(8, 1) == 8
    assert groups == ["migration", "serving", "serving"]
