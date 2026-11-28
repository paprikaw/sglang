"""Quiescent P/D registration transaction for the initial Decode qualification.

No transport restart, in-flight writes, or stale registered CUDA addresses are
permitted.  Admission at the external router must be held during migration.
Existing decode requests retain their slots and continue after the PP fence.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


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


def validate_prefill_args(args):
    if not (args.disaggregation_transfer_backend == "mooncake"
            and args.tp_size == 1 and args.pp_async_batch_depth == 0
            and args.enable_hierarchical_cache
            and args.elastic_pp_transition_strategy == "background"
            and args.elastic_pp_transition_backing == "overlap"
            and args.elastic_pp_prepared_weight_policy == "offload"
            and args.elastic_pp_hicache_shared_host
            and args.enable_elastic_pp_kv_patch):
        raise ValueError("elastic Prefill requires TP1 Mooncake, HiCache, overlapping CPU weight recovery and KV patch")


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

    def _queue(self):
        return self.scheduler.disagg_decode_prealloc_queue

    def prepare_target(self, generation):
        runner = self.scheduler.tp_worker.model_runner
        if (
            getattr(runner.server_args, "elastic_pp_transition_strategy", None)
            != "background"
        ):
            return
        device_pool = runner.token_to_kv_pool
        pool = getattr(device_pool, "full_kv_pool", device_pool)
        if not getattr(pool, "capacity_managed", False):
            return
        if self.prepared_regions is not None:
            raise RuntimeError("previous P/D region preparation still owned")
        from sglang.srt.elastic_pp.pd_registered_regions import (
            PreparedDecodeRegions,
            regions,
        )

        manager = self._queue().kv_manager
        target = pool.prepared_pd_registration_regions(generation)
        # Metadata buffers belong to the persistent P/D queue, not to a PP
        # layer generation. Keep their registrations throughout the handoff.
        args = manager.kv_args
        auxiliary = regions(
            zip(
                getattr(args, "aux_data_ptrs", ()) or (),
                getattr(args, "aux_data_lens", ()) or (),
                strict=True,
            )
        )
        prepare_state = getattr(
            device_pool, "prepared_pd_state_registration_regions", None
        )
        if prepare_state is not None:
            auxiliary.update(regions(prepare_state(generation)))
        lease = PreparedDecodeRegions(
            manager.engine,
            manager._registerable_regions(),
            target,
            target_aux=auxiliary,
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
        if self.pending_work():
            raise RuntimeError("P/D transfers must finish before backing release")
        manager = self._queue().kv_manager
        self.suspended = True
        if self.prepared_regions is None:
            manager.deregister_buffer_to_engine()
        else:
            self.prepared_regions.suspend()
            # Retained MR addresses do not make old peer layout information valid.
            if hasattr(manager, "connection_pool"):
                with manager.connection_lock:
                    manager.connection_pool.clear()

    def pending_work(self):
        return pending_pd_work(self.scheduler)

    def refresh(self):
        if not self.suspended:
            return
        queue = self._queue()
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
        self._refresh_peers(queue)
        self.suspended = False
        self.last_metrics = self.metrics()
        self.prepared_regions = self.prepared_generation = None

    def _refresh_peers(self, queue):
        for entry in [*queue.queue, *getattr(queue, "pending_reqs", ())]:
            receiver = entry.kv_receiver
            if hasattr(receiver, "prefill_dp_rank"):
                receiver._setup_bootstrap_infos()
                if receiver.conclude_state is not None:
                    raise RuntimeError(
                        "unallocated P/D receiver re-registration failed"
                    )


class PrefillRegistrationTransaction(DecodeRegistrationTransaction):
    """Keep sender resources registered until all old chunks have retired."""

    def _queue(self):
        return self.scheduler.disagg_prefill_bootstrap_queue

    def pending_work(self):
        return self._queue().kv_manager.prefill_transfer_fence.pending()

    def demote_cached_rows_for_shrink(
        self, allocator, capacity, *, write_missing=False, group=None
    ):
        """Keep host-backed prefixes while retiring high-numbered device rows."""
        if capacity >= allocator.serving_capacity:
            return 0
        high = {row for row in allocator._live_page_ids if row > capacity}
        if not high and group is None:
            return 0
        from collections import defaultdict
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        cache = self.scheduler.tree_cache
        tree = cache.tree_core
        tracker = defaultdict(int)
        before = len(high)
        consider_low = False

        def host_backed(node):
            if not node.backuped:
                return False
            if not write_missing:
                return True
            state = node.component_data[ComponentType.MAMBA]
            return state.value is None or state.host_value is not None

        # Demoting a leaf can expose its parent. No row is freed until the
        # existing host copy is valid or an explicit write-back completes.
        while True:
            candidates = []
            ready = {}
            error = None
            needed = max(0, len(allocator._live_page_ids) - capacity)
            try:
                if self.pending_work():
                    raise RuntimeError("Cannot demote Prefill KV referenced by an active transfer")
                selecting = needed > 0 if consider_low else bool(high)
                leaves = tree.evictable_device_leaves if selecting else ()
                for node in sorted(leaves, key=lambda n: n.id):
                    if not tree._is_device_leaf(node):
                        continue
                    rows = node.component_data[ComponentType.FULL].value
                    if rows is None:
                        continue
                    if not consider_low and not high.intersection(allocator._as_int_list(rows)):
                        continue
                    if write_missing and not host_backed(node):
                        action = tree._build_backup_kv_action(node)
                        cache._execute_and_commit_kv_backup(action, write_back=True)
                        cache.writing_check(write_back=True)
                    candidates.append((node, len(rows)))
                ready = {
                    node.id: count for node, count in candidates if host_backed(node)
                }
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            reports = (
                group.all_gather_object((ready, error, needed))
                if group is not None
                else [(ready, error, needed)]
            )
            if any(failure for _, failure, _ in reports):
                raise RuntimeError(f"Prefill HiCache backup failed: {reports}")
            if any(count != needed for _, _, count in reports):
                raise RuntimeError("Prefill shrink live row counts differ across ranks")
            common = set.intersection(*(set(ids) for ids, _, _ in reports))
            if not common and needed and write_missing and not consider_low:
                # Locked high rows may still fit after releasing a few
                # unlocked low rows with durable CPU copies.
                consider_low = True
                continue
            if not common:
                break
            failure = None
            try:
                rows_to_free = needed
                for node_id in sorted(common):
                    if consider_low and rows_to_free <= 0:
                        break
                    cache._demote(node_id, tracker)
                    rows_to_free -= min(ids[node_id] for ids, _, _ in reports)
            except BaseException as exc:
                failure = f"{type(exc).__name__}: {exc}"
            failures = group.all_gather_object(failure) if group is not None else [failure]
            if any(failures):
                raise RuntimeError(f"Prefill HiCache demotion failed: {failures}")
            high.intersection_update(allocator._live_page_ids)
        if high:
            cached = set()
            locked = set()
            for node in getattr(tree, "_node_arena", {}).values():
                rows = node.component_data[ComponentType.FULL].value
                if rows is None:
                    continue
                affected = high.intersection(allocator._as_int_list(rows))
                cached.update(affected)
                if any(component.lock_ref > 0 for component in node.component_data):
                    locked.update(affected)
            logger.warning(
                "Prefill shrink kept high KV rows: cached=%d locked_cached=%d "
                "outside_tree=%d", len(cached), len(locked), len(high - cached),
            )
        logger.info(
            "Elastic Prefill shrink rank=%d cap=%d high_before=%d high_after=%d "
            "live=%d demoted_KV=%d",
            self.scheduler.ps.pp_rank, capacity, before, len(high),
            len(allocator._live_page_ids), tracker[ComponentType.FULL],
        )
        return tracker[ComponentType.FULL]

    def resize_source_capacity_at_fence(self, pool, capacity, generation):
        """Update the current layout's MR set before relinquishing its pages."""
        from sglang.srt.elastic_pp.pd_registered_regions import (
            PreparedDecodeRegions,
            regions,
        )

        if self.suspended or self.prepared_regions is not None:
            raise RuntimeError("source resize requires active source registrations")
        manager = self._queue().kv_manager
        old = regions(manager._registerable_regions())
        old_kv = regions(pool.pd_registration_regions())
        shrinking = capacity < pool.serving_capacity
        if not shrinking:
            # Registration additions need mapped addresses. The scheduler is
            # fenced throughout, so no sender can observe the new capacity.
            pool.resize_serving_capacity_at_fence(capacity)
        lease = PreparedDecodeRegions(
            manager.engine, old, pool.pd_registration_regions(capacity),
            target_aux=old - old_kv,
        )
        self.prepared_regions, self.prepared_generation = lease, generation
        lease.prepare()
        rank = self.scheduler.ps.pp_rank
        logger.info(
            "Elastic Prefill resize rank=%d generation=%d retiring_MRs=%d",
            rank, generation, len(lease.source - lease.target_prepared),
        )
        self.suspend()
        logger.info(
            "Elastic Prefill resize rank=%d generation=%d retired_MRs_seconds=%.4f",
            rank, generation, lease.metrics["suspend_seconds"],
        )
        started = time.monotonic()
        if shrinking:
            pool.resize_serving_capacity_at_fence(capacity)
        pool.reclaim_capacity_at_fence()
        logger.info(
            "Elastic Prefill resize rank=%d generation=%d reclaim_seconds=%.4f",
            rank, generation, time.monotonic() - started,
        )
        self.refresh()

    def suspend(self):
        fence = self._queue().kv_manager.prefill_transfer_fence
        fence.close()
        try:
            super().suspend()
        except BaseException:
            if not self.suspended:
                fence.reopen()
            raise

    def _refresh_peers(self, queue):
        # The static D registered each physical P rank once. Destination
        # addresses and room IDs are unchanged, even on zero-layer stages.
        queue.kv_manager.prefill_transfer_fence.reopen()
