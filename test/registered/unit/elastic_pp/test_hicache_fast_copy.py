from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.hicache_transition import (
    HiCacheTransitionAdapter,
    SchedulerHiCacheTransition,
    _copy_host_rows,
    _gather_host_rows,
    host_nodes,
)


def cache(slots):
    def key(tokens):
        return NS(
            extra_key=None,
            cache_salt=None,
            is_bigram=False,
            raw_token_ids=lambda: tokens,
        )

    root = NS(key=key(()), parent=None, component_data=[], children={})
    child = NS(
        key=key((7, 8)),
        parent=root,
        component_data=[NS(host_value=torch.tensor(slots, dtype=torch.int64))],
        children={},
    )
    root.children[7] = child
    return NS(tree_core=NS(root_node=root))


def transition():
    return NS(
        target=NS(generation=1, layer_range=lambda rank: NS(start=3, end=5, size=2))
    )


@pytest.mark.parametrize("slots", [[2], [1, 2, 3], [4, 1, 3], []])
@pytest.mark.parametrize("strided", [False, True])
def test_row_copy_preserves_order_strides_and_snapshot_independence(slots, strided):
    storage = torch.arange(6 * 3 * 4).reshape(6, 3, 4).float()
    source = storage[:, 1] if strided else storage[:, 1].clone()
    original = source.clone()
    ids = torch.tensor(slots, dtype=torch.int64)
    rows = _gather_host_rows(source, ids, fast_copy=True)
    expected = original.index_select(0, ids)
    assert torch.equal(rows, expected)
    source.fill_(-1)
    assert torch.equal(rows, expected)
    target_storage = torch.full_like(storage, -9)
    target = target_storage[:, 2] if strided else target_storage[:, 2].clone()
    before = target.clone()
    _copy_host_rows(target, ids, rows, fast_copy=True)
    before.index_copy_(0, ids, expected)
    assert torch.equal(target, before)
    if strided:
        assert target_storage[:, :2].eq(-9).all()


def staged_adapter(source, slots=(2,)):
    src = cache(slots)
    adapter = HiCacheTransitionAdapter(
        cache=src, pools={0: {3: {"k": source}}}, group=None, rank=0, fast_copy=True
    )
    adapter.prepare_transition(transition())
    identity, component = next(iter(host_nodes(src)))
    adapter._staged[(identity, component, 4)] = {"k": torch.ones(len(slots), 4)}
    return adapter


def test_private_target_abort_keeps_live_source_and_needs_no_undo():
    source = torch.arange(24).reshape(6, 4).float()
    before = source.clone()
    adapter = staged_adapter(source)
    target = torch.full_like(source, -1)
    adapter.apply_staged(
        transition(),
        target_cache=cache([4]),
        target_pools={0: {4: {"k": target}}},
        private_target=True,
    )
    assert target[4].eq(1).all()
    assert adapter.metrics["private_target_undo_bytes_avoided"] == 16
    assert not adapter._undo
    adapter.abort(transition())
    assert torch.equal(source, before)


def test_private_target_rejects_live_storage_alias_before_any_write():
    source = torch.arange(24).reshape(6, 4).float()
    before = source.clone()
    adapter = staged_adapter(source)
    with pytest.raises(RuntimeError, match="aliases live source"):
        adapter.apply_staged(
            transition(),
            target_cache=cache([4]),
            target_pools={0: {4: {"k": source.view_as(source)}}},
            private_target=True,
        )
    assert torch.equal(source, before)


def test_fast_copy_without_private_target_still_rolls_back_shared_rows():
    source = torch.zeros(6, 4)
    adapter = staged_adapter(source)
    adapter.apply_staged(
        transition(), target_cache=cache([2]), target_pools={0: {4: {"k": source}}}
    )
    assert source[2].eq(1).all()
    adapter.abort(transition())
    assert not source.any()


def test_failed_private_import_is_discarded_without_publishing(monkeypatch):
    from sglang.srt.elastic_pp import hicache_transition as module

    source_tensor = torch.zeros(6, 4)
    adapter = staged_adapter(source_tensor)
    source_cache = adapter.cache
    scheduler = NS(tree_cache=source_cache)
    worker = NS(
        server_args=NS(elastic_pp_hicache_fast_copy=True), token_to_kv_pool=object()
    )
    transaction = SchedulerHiCacheTransition(
        scheduler=scheduler, model_runner=worker, group=None, rank=0
    )
    transaction.adapter = adapter
    private = cache([4])
    private._elastic_pp_projected_pool = object()
    target = torch.zeros_like(source_tensor)
    discarded = []
    monkeypatch.setattr(module, "prepare_target_host_cache", lambda *a, **kw: private)
    monkeypatch.setattr(module, "cache_host_views", lambda *a: {0: {4: {"k": target}}})
    monkeypatch.setattr(module, "discard_host_cache", discarded.append)
    copy_rows = module._copy_host_rows

    def fail_after_write(*args, **kwargs):
        copy_rows(*args, **kwargs)
        raise RuntimeError("injected import failure")

    monkeypatch.setattr(module, "_copy_host_rows", fail_after_write)
    with pytest.raises(RuntimeError, match="injected import failure"):
        transaction.apply_staged(transition())
    transaction.abort(transition())
    assert scheduler.tree_cache is source_cache
    assert not source_tensor.any()
    assert discarded == [private]
    assert transaction.target_cache is None


def test_private_target_cannot_be_source_cache():
    adapter = staged_adapter(torch.zeros(6, 4))
    with pytest.raises(RuntimeError, match="must not be the source"):
        adapter.apply_staged(
            transition(),
            target_cache=adapter.cache,
            target_pools={},
            private_target=True,
        )
