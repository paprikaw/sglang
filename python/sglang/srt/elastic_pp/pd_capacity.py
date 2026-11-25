"""Observe actual Decode KV admission failures without touching CUDA.

Failures are recorded by the native preallocator after prefix matching,
evictable-cache accounting and decode reservations. Request-count, handshake,
transfer and recurrent-state waits are separate causes.
"""

import time


def capacity_running_reqs(scheduler):
    args = getattr(scheduler, "server_args", None)
    if not (
        getattr(args, "elastic_pp_disagg_decode", False)
        and getattr(args, "max_running_requests", 1) > 1
    ):
        return scheduler.running_batch.reqs
    # Reservations belong to the whole PP pipeline, not only the current slot.
    batches = [
        scheduler.running_batch,
        *getattr(scheduler, "running_mbs", ()),
        *getattr(scheduler, "mbs", ()),
    ]
    return list(
        {
            req.rid: req
            for batch in batches
            if batch is not None
            for req in batch.reqs
            if not req.finished()
        }.values()
    )


def prune_capacity_waiters(queue):
    live = {entry.req.rid for entry in queue.queue}
    queue._elastic_kv_waits = {
        rid: row
        for rid, row in getattr(queue, "_elastic_kv_waits", {}).items()
        if rid in live
    }


def clear_capacity_waiter(queue, rid):
    getattr(queue, "_elastic_kv_waits", {}).pop(rid, None)


def record_capacity_waiter(queue, rid, *, required_tokens, available_tokens):
    now = time.monotonic()
    waits = getattr(queue, "_elastic_kv_waits", None)
    if waits is None:
        waits = queue._elastic_kv_waits = {}
    old = waits.get(rid)
    waits[rid] = dict(
        rid=rid,
        since=old["since"] if old else now,
        observed_at=now,
        required_tokens=int(required_tokens),
        available_tokens=int(available_tokens),
        serving_capacity=int(queue.max_total_num_tokens),
        reason="kv_tokens",
    )


def capacity_wait_snapshot(queue, now=None):
    now = time.monotonic() if now is None else now
    live = {entry.req.rid for entry in queue.queue}
    rows = [
        dict(row, wait_s=max(0.0, now - row["since"]))
        for rid, row in getattr(queue, "_elastic_kv_waits", {}).items()
        if rid in live and 0 <= now - row["observed_at"] <= 3
    ]
    return dict(
        blocked=bool(rows),
        waiting=len(rows),
        oldest_wait_s=max((r["wait_s"] for r in rows), default=0.0),
        requests=rows,
    )


def unpublished_waiters_are_safe(scheduler):
    """Only unallocated waiters may survive registration refresh.

    Preserve the legacy strict boundary outside the managed-capacity concurrent
    path. A waiter must never have published a writable destination or retained
    request/KV storage. Transfers and deferred-release acknowledgements remain
    independently protected by pending_pd_work.
    """
    args = getattr(scheduler, "server_args", None)
    if not (
        getattr(args, "elastic_pp_disagg_decode", False)
        and getattr(args, "elastic_pp_kv_backend", None) == "kvcached"
        and getattr(args, "elastic_pp_kv_capacities", None)
        and getattr(args, "max_running_requests", 1) > 1
    ):
        return False
    pre = scheduler.disagg_decode_prealloc_queue
    if pre.has_published_destinations or getattr(pre, "retracted_queue", ()):
        return False
    entries = [*pre.queue, *getattr(pre, "pending_reqs", ())]
    return all(
        entry.req.req_pool_idx is None
        and entry.metadata_buffer_index == -1
        and entry.prefix_match is None
        and entry.hicache_restored_kv_indices is None
        for entry in entries
    )
