"""Bounded KVCacheD capacity frontier; tensor addresses keep the VA upper bound.

Physical backing is committed per serving-capacity budget, rather than mapping
on each request allocation. This keeps RDMA registrations valid between fences.
"""

from __future__ import annotations


def capacity_frontier(args):
    raw = getattr(args, "elastic_pp_kv_capacities", None)
    if raw is None:
        return {}
    from sglang.srt.elastic_pp.autoscaling import parse_elastic_pp_partitions

    if getattr(args, "elastic_pp_kv_backend", None) != "kvcached":
        raise ValueError("elastic KV capacities require the KVCacheD backend")
    if not getattr(args, "elastic_pp_graph_partitions", None):
        raise ValueError("elastic KV capacities require prepared PP partitions")
    # A static replica uses the same backing/Graph path with one admitted layout.
    parts = parse_elastic_pp_partitions(
        args.elastic_pp_graph_partitions, allow_single=True
    )
    if len(set(parts)) != len(parts):
        raise ValueError("capacity frontier cannot contain duplicate PP partitions")
    if getattr(args, "page_size", None) != 1:
        raise ValueError("capacity-managed KVCacheD currently requires page-size 1")
    role = getattr(args, "disaggregation_mode", None)
    prefill = role == "prefill" and getattr(args, "elastic_pp_disagg_prefill", False)
    if raw == "auto":
        if (
            not prefill
            or len(parts) != 2
            or getattr(args, "max_total_tokens", None) is not None
            or (args.tp_size, args.ep_size, args.dp_size) != (1, 1, 1)
        ):
            raise ValueError(
                "automatic KV capacity requires two TP1/EP1/DP1 elastic Prefill layouts and "
                "no --max-total-tokens"
            )
        from sglang.srt.elastic_pp.prefill_kv_capacity import prefill_capacity_layouts

        prefill_capacity_layouts(args)
        return {}
    if role != "decode" and not prefill:
        raise ValueError("capacity-managed P/D requires Decode or the elastic Prefill sender")
    try:
        caps = tuple(int(x.strip()) for x in raw.split(";"))
    except (TypeError, ValueError) as exc:
        raise ValueError("KV capacities must be semicolon-separated integers") from exc
    if len(caps) != len(parts) or any(c <= 0 for c in caps):
        raise ValueError("one positive KV capacity is required per PP partition")
    if max(caps) != getattr(args, "max_total_tokens", None):
        raise ValueError("max-total-tokens must equal the maximum virtual KV capacity")
    return dict(zip(parts, caps))


def automatic_prefill_capacity(args):
    return getattr(args, "elastic_pp_kv_capacities", None) == "auto"


def topology_capacity(args, topology, *, pool=None):
    frontier = (
        pool.auto_capacity_by_layout
        if automatic_prefill_capacity(args)
        else capacity_frontier(args)
    )
    if not frontier:
        return None
    counts = tuple(r.size for r in topology.layer_ranges)
    if counts not in frontier:
        raise ValueError("PP target has no admitted KV capacity")
    return frontier[counts]


def prepared_topology_capacity(args, topology, *, pool):
    """Temporary capacity used while both PP weight layouts are resident."""
    capacity = topology_capacity(args, topology, pool=pool)
    staging = getattr(pool, "prefill_staging_capacity", None)
    return capacity if staging is None else min(capacity, staging)


def full_pool(runner):
    pool = runner.token_to_kv_pool
    return getattr(pool, "full_kv_pool", pool)


def configure_capacity(runner):
    if automatic_prefill_capacity(runner.server_args):
        from sglang.srt.elastic_pp.prefill_kv_capacity import (
            prefill_capacity_layouts,
        )

        pool = full_pool(runner)
        # The private Graph walk is request-free. Give it just the physical page
        # containing the null slot. Profile serving capacity after that walk.
        capture_capacity = pool.kvcached_allocator.blocks_per_physical_page - 1
        pool.configure_serving_capacity(capture_capacity)
        pool.auto_capacity_by_layout = dict.fromkeys(
            prefill_capacity_layouts(runner.server_args), capture_capacity
        )
        runner.max_total_num_tokens = capture_capacity
        return
    frontier = capacity_frontier(runner.server_args)
    if not frontier:
        return
    counts = tuple(int(x) for x in runner.server_args.elastic_pp_initial_partition.split(","))
    pool = full_pool(runner)
    pool.configure_serving_capacity(frontier[counts])
    runner.max_total_num_tokens = frontier[counts]


def publish_scheduler_capacity(scheduler):
    """Publish admission limits after model/KV commit, before router admission."""
    worker = getattr(scheduler, "tp_worker", None)
    if worker is None:
        return
    runner = worker.model_runner
    pool = full_pool(runner)
    if not getattr(pool, "capacity_managed", False):
        return
    capacity = pool.serving_capacity
    runner.max_total_num_tokens = capacity
    scheduler.max_total_num_tokens = capacity
    queue = (scheduler.disagg_prefill_bootstrap_queue
             if getattr(getattr(scheduler, "server_args", None), "elastic_pp_disagg_prefill", False)
             else scheduler.disagg_decode_prealloc_queue)
    queue.max_total_num_tokens = capacity
    observer = getattr(scheduler, "pool_stats_observer", None)
    if observer is not None:
        from dataclasses import replace

        scheduler.pool_stats_observer = replace(observer, max_total_num_tokens=capacity)
    checker = getattr(scheduler, "invariant_checker", None)
    if checker is not None:
        checker.max_total_num_tokens = capacity
        checker.pool_stats_observer = scheduler.pool_stats_observer
    # Shapes, Graph bindings and request slot tables retain the virtual maximum.
    # HTTP status exposes the actual admission limit through these live owners.
