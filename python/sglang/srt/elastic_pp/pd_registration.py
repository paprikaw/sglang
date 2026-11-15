"""Quiescent P/D registration transaction for the initial Decode qualification.

No transport restart, in-flight writes, or stale registered CUDA addresses are
permitted.  Admission at the external router must be held during migration.
Existing decode requests retain their slots and continue after the PP fence.
"""

from __future__ import annotations


def validate_decode_args(args):
    required = dict(
        elastic_pp_disagg_decode=True,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="mooncake",
        enable_hierarchical_cache=True,
        pp_async_batch_depth=0,
    )
    mismatches = [
        f"{name}={getattr(args, name, None)!r} (requires {value!r})"
        for name, value in required.items()
        if getattr(args, name, None) != value
    ]
    sync = (
        args.elastic_pp_transition_strategy == "sync"
        and args.elastic_pp_transition_backing == "cpu_spill"
        and args.disable_cuda_graph
    )
    optimized = (
        args.elastic_pp_transition_strategy == "background"
        and args.elastic_pp_transition_backing == "overlap"
        and not args.disable_cuda_graph
        and getattr(args, "elastic_pp_graph_partitions", None)
        and getattr(args, "elastic_pp_prepared_weight_policy", None) == "offload"
        and getattr(args, "elastic_pp_hicache_shared_host", False)
        and getattr(args, "elastic_pp_weight_staging", None) == "cpu_prepacked"
        and getattr(args, "enable_elastic_pp_kv_patch", False)
        and getattr(args, "elastic_pp_kv_patch_transport", None) == "nixl"
        and (
            getattr(args, "elastic_pp_kv_backend", None) == "native"
            or getattr(args, "elastic_pp_kv_capacities", None)
        )
    )
    count = getattr(args, "max_running_requests", None)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= (4 if optimized else 1)
    ):
        mismatches.append(
            f"max_running_requests={count!r}: requires 1 (sync) or 1..4 (bounded optimized pilot)"
        )
    micro = getattr(args, "pp_max_micro_batch_size", None)
    if (
        not isinstance(micro, int)
        or isinstance(micro, bool)
        or not 1 <= micro <= (4 if optimized else 1)
    ):
        mismatches.append(
            f"pp_max_micro_batch_size={micro!r}: requires 1 (sync) or 1..4 (bounded optimized pilot)"
        )
    if not (sync or optimized):
        mismatches.append(
            f"elastic_pp_transition_strategy={args.elastic_pp_transition_strategy!r}, "
            f"elastic_pp_transition_backing={args.elastic_pp_transition_backing!r}, "
            f"disable_cuda_graph={args.disable_cuda_graph!r}: requires sync/cpu_spill/Eager "
            "or background/overlap/prepared Graphs/shared HiCache/CPU-prepacked/NIXL"
        )
    if mismatches:
        raise ValueError("elastic P/D Decode qualification: " + "; ".join(mismatches))


def pending_pd_work(scheduler):
    """Do not confuse running decode requests with outstanding RDMA writers."""
    prealloc = scheduler.disagg_decode_prealloc_queue
    transfer = scheduler.disagg_decode_transfer_queue
    from sglang.srt.elastic_pp.pd_capacity import unpublished_waiters_are_safe

    unallocated = (
        0
        if unpublished_waiters_are_safe(scheduler)
        else (len(prealloc.queue) + len(getattr(prealloc, "pending_reqs", ())))
    )
    return (
        unallocated
        + len(getattr(prealloc, "retracted_queue", ()))
        + len(transfer.queue)
        + int(transfer.has_pending_deferred_releases())
    )


def align_decode_hicache_polls_at_fence(scheduler, group):
    """Consume each pre-fence HiCache round before blocking on its sends.

    Native PD ranks may advance different numbers of idle slots while the
    request/output rings drain. Cache packets follow physical PP ranks and
    outlive those rings. Catch up lagging receivers without creating another
    round at the leading rank; only then can ordinary host-I/O drain wait.
    """
    before = scheduler._pp_pd_hicache_polls
    reports = group.all_gather_object(
        {
            "polls": before,
            "microsteps": scheduler._pp_pd_microsteps,
        }
    )
    if not reports or any(
        not isinstance(row.get(name), int) or row[name] < 0
        for row in reports
        for name in ("polls", "microsteps")
    ):
        raise RuntimeError("invalid PD HiCache fence poll counts")
    target = max(row["polls"] for row in reports)
    poll = getattr(
        scheduler.tree_cache,
        "check_hicache_events_pp_decode",
        scheduler.tree_cache.check_hicache_events,
    )
    while scheduler._pp_pd_hicache_polls < target:
        poll()
        scheduler._pp_pd_hicache_polls += 1
    # Preserve the same polling cadence after resuming, including intervals >1.
    scheduler._pp_pd_microsteps = max(row["microsteps"] for row in reports)
    return {"before": before, "target": target, "catchup_rounds": target - before}


class DecodeRegistrationTransaction:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.suspended = False
        self.prepared_regions = None
        self.prepared_generation = None
        self.last_metrics = {}

    def prepare_target(self, generation):
        runner = self.scheduler.tp_worker.model_runner
        if (
            getattr(runner.server_args, "elastic_pp_transition_strategy", None)
            != "background"
        ):
            return
        pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
        if not getattr(pool, "capacity_managed", False):
            return
        if self.prepared_regions is not None:
            raise RuntimeError("previous P/D region preparation still owned")
        from sglang.srt.elastic_pp.pd_registered_regions import PreparedDecodeRegions

        manager = self.scheduler.disagg_decode_prealloc_queue.kv_manager
        target = pool.prepared_pd_registration_regions(generation)
        lease = PreparedDecodeRegions(
            manager.engine, manager._registerable_regions(), target
        )
        self.prepared_regions, self.prepared_generation = lease, generation
        lease.prepare()

    def abort_prepared(self, generation):
        if self.prepared_regions is None:
            return
        if self.prepared_generation != generation:
            raise RuntimeError("P/D registration abort generation mismatch")
        self.prepared_regions.abort()
        if not self.suspended:
            self.last_metrics = self.metrics()
            self.prepared_regions = self.prepared_generation = None

    def metrics(self):
        lease = self.prepared_regions
        if lease is None:
            return dict(self.last_metrics)
        return dict(
            lease.metrics, generation=self.prepared_generation, phase=lease.phase
        )

    def suspend(self):
        if self.suspended:
            raise RuntimeError("P/D registration already suspended")
        if pending_pd_work(self.scheduler):
            raise RuntimeError("P/D transfers must finish before backing release")
        manager = self.scheduler.disagg_decode_prealloc_queue.kv_manager
        self.suspended = True
        if self.prepared_regions is None:
            manager.deregister_buffer_to_engine()
        else:
            self.prepared_regions.suspend()
            # Retained MR addresses do not make old peer layout information valid.
            if hasattr(manager, "connection_pool"):
                with manager.connection_lock:
                    manager.connection_pool.clear()

    def refresh(self):
        if not self.suspended:
            return
        queue = self.scheduler.disagg_decode_prealloc_queue
        manager = queue.kv_manager
        from sglang.srt.elastic_pp.kv_capacity import publish_scheduler_capacity

        publish_scheduler_capacity(self.scheduler)
        # The pool object survives migration; its layer storage and addresses
        # do not.  Rebuild all KV/GDN/aux registrations from the committed pool.
        manager.kv_args = queue._build_kv_args()
        if self.prepared_regions is None:
            manager.register_buffer_to_engine()
        else:
            from sglang.srt.elastic_pp.pd_registered_regions import regions

            actual = manager._registerable_regions()
            if self.prepared_regions.phase == "aborted":
                if regions(actual) != self.prepared_regions.source:
                    raise RuntimeError("restored P/D source regions differ")
                self.prepared_regions.restore_source()
            else:
                self.prepared_regions.commit(actual)
        # These rooms bootstrapped before migration, but have not published any
        # token destinations. Republish pool addresses before their first send.
        # Future rooms use the same refreshed connection cache.
        for entry in [*queue.queue, *getattr(queue, "pending_reqs", ())]:
            receiver = entry.kv_receiver
            if hasattr(receiver, "prefill_dp_rank"):
                receiver._setup_bootstrap_infos()
                if receiver.conclude_state is not None:
                    raise RuntimeError(
                        "unallocated P/D receiver re-registration failed"
                    )
        self.suspended = False
        self.last_metrics = self.metrics()
        self.prepared_regions = self.prepared_generation = None
