from types import SimpleNamespace as NS

import pytest

from sglang.srt.elastic_pp import pd_capacity as m
from sglang.srt.elastic_pp.pd_registration import (
    DecodeRegistrationTransaction,
    pending_pd_work,
)


def waiter(rid="wait"):
    return NS(
        req=NS(rid=rid, req_pool_idx=None),
        metadata_buffer_index=-1,
        prefix_match=None,
        hicache_restored_kv_indices=None,
    )


def scheduler():
    pre = NS(
        queue=[waiter()],
        pending_reqs=[],
        retracted_queue=[],
        has_published_destinations=False,
    )
    transfer = NS(queue=[], has_pending_deferred_releases=lambda: False)
    args = NS(
        elastic_pp_disagg_decode=True,
        elastic_pp_kv_backend="kvcached",
        elastic_pp_kv_capacities="16;24",
        max_running_requests=4,
    )
    return NS(
        server_args=args,
        disagg_decode_prealloc_queue=pre,
        disagg_decode_transfer_queue=transfer,
    )


def test_capacity_wait_counts_only_current_fresh_native_failures(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    q = NS(queue=[waiter()], max_total_num_tokens=16)
    assert not m.capacity_wait_snapshot(q)["blocked"]
    m.record_capacity_waiter(q, "wait", required_tokens=9, available_tokens=8)
    clock[0] = 12
    m.record_capacity_waiter(q, "wait", required_tokens=9, available_tokens=7)
    assert m.capacity_wait_snapshot(q)["oldest_wait_s"] == 2
    assert m.capacity_wait_snapshot(q)["requests"][0]["available_tokens"] == 7
    clock[0] = 16
    assert not m.capacity_wait_snapshot(q)["blocked"]
    q.queue = []
    m.prune_capacity_waiters(q)
    assert q._elastic_kv_waits == {}


def test_other_admission_cause_clears_kv_pressure():
    q = NS(queue=[waiter()], max_total_num_tokens=16)
    m.record_capacity_waiter(q, "wait", required_tokens=9, available_tokens=8)
    m.clear_capacity_waiter(q, "wait")
    assert not m.capacity_wait_snapshot(q)["blocked"]


def test_capacity_reservation_spans_distinct_live_pp_slots():
    a = NS(rid="a", finished=lambda: False)
    b = NS(rid="b", finished=lambda: False)
    ended = NS(rid="old", finished=lambda: True)
    batch = NS(reqs=[a])
    s = NS(
        server_args=NS(elastic_pp_disagg_decode=True, max_running_requests=4),
        running_batch=batch,
        running_mbs=[batch, NS(reqs=[b, ended])],
        mbs=[None, batch],
    )
    assert [r.rid for r in m.capacity_running_reqs(s)] == ["a", "b"]
    s.server_args.max_running_requests = 1
    assert m.capacity_running_reqs(s) == [a]


def test_unallocated_receiver_republishes_new_addresses_after_registration():
    s = scheduler()
    events = []
    receiver = NS(
        prefill_dp_rank=0,
        conclude_state=None,
        _setup_bootstrap_infos=lambda: events.append("republish"),
    )
    s.disagg_decode_prealloc_queue.queue[0].kv_receiver = receiver
    manager = NS(
        deregister_buffer_to_engine=lambda: events.append("deregister"),
        register_buffer_to_engine=lambda: events.append("register"),
    )
    s.disagg_decode_prealloc_queue.kv_manager = manager
    s.disagg_decode_prealloc_queue._build_kv_args = lambda: "fresh-pointers"
    transaction = DecodeRegistrationTransaction(s)
    transaction.suspend()
    transaction.refresh()
    assert manager.kv_args == "fresh-pointers"
    assert events == ["deregister", "register", "republish"]


def test_unpublished_waiter_does_not_deadlock_capacity_grow():
    s = scheduler()
    assert m.unpublished_waiters_are_safe(s)
    assert pending_pd_work(s) == 0
    calls = []
    s.disagg_decode_prealloc_queue.kv_manager = NS(
        deregister_buffer_to_engine=lambda: calls.append("deregister")
    )
    DecodeRegistrationTransaction(s).suspend()
    assert calls == ["deregister"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("metadata_buffer_index", 0),
        ("prefix_match", object()),
        ("hicache_restored_kv_indices", object()),
    ],
)
def test_partially_allocated_waiter_still_blocks_registration(field, value):
    s = scheduler()
    setattr(s.disagg_decode_prealloc_queue.queue[0], field, value)
    assert pending_pd_work(s) > 0
    with pytest.raises(RuntimeError, match="transfers must finish"):
        DecodeRegistrationTransaction(s).suspend()


@pytest.mark.parametrize("cause", ["slot", "published", "transfer", "ack", "legacy"])
def test_each_possible_writer_or_unqualified_profile_still_blocks(cause):
    s = scheduler()
    pre = s.disagg_decode_prealloc_queue
    if cause == "slot":
        pre.queue[0].req.req_pool_idx = 1
    elif cause == "published":
        pre.has_published_destinations = True
    elif cause == "transfer":
        s.disagg_decode_transfer_queue.queue = [object()]
    elif cause == "ack":
        s.disagg_decode_transfer_queue.has_pending_deferred_releases = lambda: True
    elif cause == "legacy":
        s.server_args.max_running_requests = 1
    assert pending_pd_work(s) > 0
