"""Rank-local orchestration for transactional elastic PP migration."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition, RLock, Thread
from typing import Any

import torch

from sglang.srt.elastic_pp.layer_transport import (
    flatten_layer_states,
    unflatten_layer_states,
)
from sglang.srt.elastic_pp.topology import (
    DynamicPPTransition,
    LayerRange,
    LayerTransfer,
    PPTopology,
)

logger = logging.getLogger(__name__)


class ElasticPPMigrationPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    BULK_READY = "bulk_ready"
    FENCING = "fencing"
    COMMITTED = "committed"
    CLEANING = "cleaning"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class ElasticPPMigrationSnapshot:
    phase: ElasticPPMigrationPhase
    generation: int | None
    error: str | None
    source_usable: bool
    fence_microstep: int | None
    prepare_seconds: float | None
    fence_seconds: float | None
    total_seconds: float | None


class ElasticPPMigrationCoordinator:
    """Prepare backing off-path, copy/commit at a fence, then clean up later.

    One coordinator exists in every resident PP scheduler process.  Long
    allocation work runs on a daemon migration thread.  The scheduler thread
    calls :meth:`finalize_at_fence` only after its microbatch ring is empty;
    that method owns every migration NCCL launch, copies the current live KV
    rows, and publishes the new model/KV/route generation.  Keeping serving
    and migration communicator launches on one host thread avoids undefined
    multi-thread NCCL ordering on the same device.

    The opt-in KV patch integration gathers forward-ready native rows while
    serving. The default CPU shadow crosses ranks at the drained fence; the
    NIXL variant applies them online through independent registered buffers.
    Recurrent/cache handoff and final coverage checks remain at the fence;
    no background allocator snapshot is permitted.
    """

    def __init__(
        self,
        *,
        rank: int,
        current_topology: PPTopology,
        group: Any,
        model_runner: Any,
        layer_transport: Any,
        kv_transport: Any,
        fence_step_provider: Callable[[], int] | None = None,
        fence_lead_steps: int = 0,
        strategy: str | None = None,
        cache_transition: Any = None,
        backing: str | None = None,
        enable_kv_patch: bool | None = None,
    ) -> None:
        # Test doubles without server_args retain the pre-existing allocation-only
        # protocol. Real runners select one of the three explicit policies.
        args = getattr(model_runner, "server_args", None)
        self.strategy = strategy or getattr(
            args, "elastic_pp_transition_strategy", "legacy"
        )
        if self.strategy not in {"sync", "background", "batched", "legacy"}:
            raise ValueError(f"unknown elastic PP transition strategy: {self.strategy}")
        self.backing = backing or getattr(
            args, "elastic_pp_transition_backing", "overlap"
        )
        if self.backing not in {"overlap", "cpu_spill"}:
            raise ValueError(f"unknown elastic PP transition backing: {self.backing}")
        if self.backing == "cpu_spill" and self.strategy not in {"sync", "batched"}:
            raise ValueError("CPU-spill backing requires sync or batched transition")
        self.cache_transition = cache_transition
        self.enable_kv_patch = (
            bool(getattr(args, "enable_elastic_pp_kv_patch", False))
            if enable_kv_patch is None
            else enable_kv_patch
        )
        if self.enable_kv_patch and (
            self.strategy != "background" or self.backing != "overlap"
        ):
            raise ValueError("online KV patch requires background/overlap transition")
        self._source_backing_attempted = False
        self._source_backing_metrics: dict[str, Any] = {}
        self._layers_precopied = False
        self._background_weight_bytes_sent = 0
        self._weight_stager = None
        # Startup Graph preparation and the serving coordinator share the CPU
        # bank. It contains host tensors only, never retained candidate weights.
        self._cpu_weight_reloader = getattr(
            model_runner, "_elastic_pp_cpu_weight_reloader", None
        )
        self.weight_staging = getattr(args, "elastic_pp_weight_staging", "legacy")
        if self.weight_staging not in {
            "legacy",
            "pinned",
            "cpu_cache",
            "cpu_prepacked",
        }:
            raise ValueError("unknown elastic PP weight staging policy")
        if self.weight_staging in {"cpu_cache", "cpu_prepacked"} and (
            self.strategy not in {"background", "sync"} or self.backing != "overlap"
        ):
            raise ValueError(
                "CPU cache reload requires background or sync with overlap backing"
            )
        self._observed_cuda_allocated_peak = 0
        self.rank = int(rank)
        self.current_topology = current_topology
        self.group = group
        self.model_runner = model_runner
        self.layer_transport = layer_transport
        self.kv_transport = kv_transport
        self.fence_step_provider = fence_step_provider or (lambda: 0)
        self.fence_lead_steps = int(fence_lead_steps)
        if self.fence_lead_steps < 0:
            raise ValueError("elastic PP fence lead steps must be non-negative")

        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._phase = ElasticPPMigrationPhase.IDLE
        self._transition: DynamicPPTransition | None = None
        self._error: str | None = None
        self._source_usable = True
        self._thread: Thread | None = None
        self._started_at: float | None = None
        self._bulk_ready_at: float | None = None
        self._fence_started_at: float | None = None
        self._committed_at: float | None = None
        self._done_at: float | None = None
        self._fence_microstep: int | None = None
        self._layer_prepared = False
        self._kv_prepared = False
        self._drain_consensus_tensor: torch.Tensor | None = None
        self._drain_consensus_work: Any = None
        self._stage_seconds: dict[str, float] = {}
        self._prepared_weight_bytes_reused = 0
        self._weights_reused = False

    def _timed(self, name, action, *args, **kwargs):
        started = time.monotonic()
        try:
            return action(*args, **kwargs)
        finally:
            with self._condition:
                self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + (
                    time.monotonic() - started
                )

    @property
    def phase(self) -> ElasticPPMigrationPhase:
        with self._lock:
            return self._phase

    @property
    def transition(self) -> DynamicPPTransition | None:
        with self._lock:
            return self._transition

    def snapshot(self) -> ElasticPPMigrationSnapshot:
        with self._lock:
            generation = (
                None if self._transition is None else self._transition.target.generation
            )
            prepare_seconds = (
                None
                if self._started_at is None or self._bulk_ready_at is None
                else self._bulk_ready_at - self._started_at
            )
            fence_seconds = (
                None
                if self._fence_started_at is None or self._committed_at is None
                else self._committed_at - self._fence_started_at
            )
            end = self._done_at or self._committed_at
            total_seconds = (
                None
                if self._started_at is None or end is None
                else end - self._started_at
            )
            return ElasticPPMigrationSnapshot(
                phase=self._phase,
                generation=generation,
                error=self._error,
                source_usable=self._source_usable,
                fence_microstep=self._fence_microstep,
                prepare_seconds=prepare_seconds,
                fence_seconds=fence_seconds,
                total_seconds=total_seconds,
            )

    def begin(self, transition: DynamicPPTransition) -> None:
        bank = getattr(self.model_runner, "_elastic_pp_prepared_graphs", None)
        if bank is not None:
            bank.check_topology(transition.target)
        with self._condition:
            if self._phase not in {
                ElasticPPMigrationPhase.IDLE,
                ElasticPPMigrationPhase.DONE,
                ElasticPPMigrationPhase.FAILED,
            }:
                raise RuntimeError(
                    "an elastic PP transition is already active: "
                    f"phase={self._phase.value}"
                )
            if not self._source_usable:
                raise RuntimeError(
                    "cannot start another transition after a partial commit failure"
                )
            if transition.source != self.current_topology:
                raise ValueError(
                    "elastic PP transition source does not match the committed "
                    f"topology: source={transition.source}, "
                    f"committed={self.current_topology}"
                )
            if transition.source.max_pp_size != self.group.world_size:
                raise ValueError(
                    "elastic PP transition does not cover the migration group: "
                    f"topology={transition.source.max_pp_size}, "
                    f"group={self.group.world_size}"
                )

            self._transition = transition
            self._phase = ElasticPPMigrationPhase.PREPARING
            self._error = None
            self._source_usable = True
            self._started_at = time.monotonic()
            self._bulk_ready_at = None
            self._fence_started_at = None
            self._committed_at = None
            self._done_at = None
            self._fence_microstep = None
            self._layer_prepared = False
            self._kv_prepared = False
            self._layers_precopied = False
            self._background_weight_bytes_sent = 0
            if self._cpu_weight_reloader is not None:
                self._cpu_weight_reloader.reset_metrics()
            self._observed_cuda_allocated_peak = 0
            self._source_backing_attempted = False
            self._source_backing_metrics = {}
            self._stage_seconds = {}
            self._prepared_weight_bytes_reused = 0
            self._weights_reused = False
            self._drain_consensus_tensor = None
            self._drain_consensus_work = None
            if self.cache_transition is not None:
                capture = getattr(self.cache_transition, "begin_background", None)
                if capture is not None:
                    capture(transition)
            if self.enable_kv_patch:
                # This caller is the scheduler; allocator snapshots must never
                # be taken from the preparation worker.
                self.kv_transport.begin_live_patch(transition)
            self._thread = Thread(
                target=self._prepare_worker,
                name=f"elastic-pp-prepare-g{transition.target.generation}-r{self.rank}",
                daemon=True,
            )
            self._thread.start()

    def progress_live_patch(self, batch, ready_event, req_to_token_pool):
        if self.enable_kv_patch:
            self.kv_transport.progress_live_patch(batch, ready_event, req_to_token_pool)

    def prepare_live_patch_writes(self, batch):
        if self.enable_kv_patch:
            self.kv_transport.prepare_live_patch_writes(batch)

    @staticmethod
    def _slot_signature(slot_ids: torch.Tensor) -> tuple[int, str]:
        cpu_slots = slot_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest = hashlib.sha256(cpu_slots.numpy().tobytes()).hexdigest()
        return int(cpu_slots.numel()), digest

    def _consensus(
        self,
        stage: str,
        *,
        success: bool,
        error: str | None = None,
        details: Any = None,
    ) -> list[dict[str, Any]]:
        transition = self._require_transition()
        report = {
            "stage": stage,
            "generation": transition.target.generation,
            "rank": self.rank,
            "success": bool(success),
            "error": error,
            "details": details,
        }
        reports = self._timed(
            f"consensus.{stage}", self.group.all_gather_object, report
        )
        if len(reports) != self.group.world_size:
            raise RuntimeError(
                "elastic PP consensus returned the wrong rank count: "
                f"got={len(reports)}, expected={self.group.world_size}"
            )
        if any(not isinstance(peer, dict) for peer in reports) or {
            peer.get("rank") for peer in reports
        } != set(range(self.group.world_size)):
            raise RuntimeError("elastic PP consensus requires every resident rank once")
        for peer in reports:
            if not isinstance(peer, dict):
                raise RuntimeError(f"invalid elastic PP consensus report: {peer!r}")
            if peer.get("stage") != stage:
                raise RuntimeError(
                    "elastic PP consensus stage mismatch: "
                    f"local={stage}, peer={peer.get('stage')!r}"
                )
            if peer.get("generation") != transition.target.generation:
                raise RuntimeError(
                    "elastic PP consensus generation mismatch: "
                    f"local={transition.target.generation}, "
                    f"peer={peer.get('generation')!r}"
                )
        return reports

    @staticmethod
    def _raise_peer_failures(stage: str, reports: list[dict[str, Any]]) -> None:
        failures = [report for report in reports if not report.get("success")]
        if failures:
            messages = [
                f"rank {report.get('rank')}: {report.get('error') or 'unknown error'}"
                for report in failures
            ]
            raise RuntimeError(
                f"elastic PP {stage} failed on resident ranks: " + " | ".join(messages)
            )

    def _set_worker_device(self) -> None:
        # CUDA's current device is thread-local. New preparation/cleanup
        # threads otherwise default to GPU 0 even on another resident rank.
        device = torch.device(getattr(self.model_runner, "device", "cpu"))
        if device.type == "cuda":
            index = device.index
            if index is None:
                index = self.model_runner.gpu_id
            torch.cuda.set_device(index)

    def poll_all_ranks_drained_at_fence(self, local_drained: bool) -> bool:
        """Poll a non-blocking consensus that every scheduler ring is empty.

        A rank that drains early must keep advancing the resident data plane;
        blocking it in a control collective can prevent its peers from draining.
        Keep at most one CPU all-reduce in flight and poll it between PP
        microsteps instead.
        """

        transition = self._require_transition()
        if self.group.world_size == 1:
            return bool(local_drained)

        if self._drain_consensus_work is None:
            self._drain_consensus_tensor = torch.tensor(
                [int(bool(local_drained)), transition.target.generation],
                dtype=torch.int64,
                device="cpu",
            )
            self._drain_consensus_work = self.group.all_reduce_cpu(
                self._drain_consensus_tensor,
                async_op=True,
            )
            if self._drain_consensus_work is None:
                raise RuntimeError(
                    "elastic PP drain consensus did not return asynchronous work"
                )
            return False

        if not self._drain_consensus_work.is_completed():
            return False

        self._drain_consensus_work.wait()
        assert self._drain_consensus_tensor is not None
        ready_count, generation_sum = self._drain_consensus_tensor.tolist()
        self._drain_consensus_tensor = None
        self._drain_consensus_work = None

        expected_generation_sum = transition.target.generation * self.group.world_size
        if generation_sum != expected_generation_sum:
            raise RuntimeError(
                "elastic PP drain generation mismatch: "
                f"expected_sum={expected_generation_sum}, got={generation_sum}"
            )
        if ready_count < 0 or ready_count > self.group.world_size:
            raise RuntimeError(
                "elastic PP drain consensus returned an invalid ready count: "
                f"{ready_count}"
            )
        return ready_count == self.group.world_size

    def aligned_resume_microstep(self, microstep: int, ring_size: int) -> int:
        """Agree on one post-commit ring boundary across scheduler ranks."""

        microstep = int(microstep)
        ring_size = int(ring_size)
        if microstep < 0:
            raise ValueError(f"elastic PP microstep must be non-negative: {microstep}")
        if ring_size <= 0:
            raise ValueError(f"elastic PP ring size must be positive: {ring_size}")

        reports = self._consensus(
            "scheduler_alignment",
            success=True,
            details=(microstep, ring_size),
        )
        peer_microsteps = []
        for report in reports:
            details = report.get("details")
            if (
                not isinstance(details, (tuple, list))
                or len(details) != 2
                or type(details[0]) is not int
                or type(details[1]) is not int
            ):
                raise RuntimeError(
                    "elastic PP scheduler alignment returned invalid details: "
                    f"{details!r}"
                )
            peer_microstep, peer_ring_size = details
            if peer_microstep < 0 or peer_ring_size != ring_size:
                raise RuntimeError(
                    "elastic PP scheduler alignment mismatch: "
                    f"local_ring_size={ring_size}, peer={details!r}"
                )
            peer_microsteps.append(peer_microstep)

        max_microstep = max(peer_microsteps)
        return (max_microstep // ring_size + 1) * ring_size

    def _observe_memory(self):
        device = torch.device(getattr(self.model_runner, "device", "cpu"))
        if device.type == "cuda":
            self._observed_cuda_allocated_peak = max(
                self._observed_cuda_allocated_peak,
                torch.cuda.memory_allocated(device),
            )

    def transition_metrics(self):
        snapshot = self.snapshot()
        metrics = {
            "generation": snapshot.generation,
            "rank": self.rank,
            "strategy": self.strategy,
            "backing_policy": self.backing,
            "source_backing": dict(self._source_backing_metrics),
            "phase": snapshot.phase.value,
            "prepare_seconds": snapshot.prepare_seconds,
            "fence_seconds": snapshot.fence_seconds,
            "total_seconds": snapshot.total_seconds,
            "background_weight_bytes_sent": self._background_weight_bytes_sent,
            "weight_staging": self.weight_staging,
            "weight_staging_metrics": (
                self._cpu_weight_reloader.metrics_snapshot()
                if self._cpu_weight_reloader is not None
                else (
                    dict(self._weight_stager.metrics)
                    if self._weight_stager is not None
                    else {}
                )
            ),
            "prepared_weight_bytes_reused": self._prepared_weight_bytes_reused,
            "stage_seconds": dict(self._stage_seconds),
            "stage_timing_scope": "rank-local wall time; nested stages and parallel ranks must not be summed",
            "observed_cuda_allocated_peak_bytes": self._observed_cuda_allocated_peak,
            "peak_scope": "sampled at prepare/copy/commit; not allocator peak proof",
            "gpu_released": False,
        }
        graph_metrics = getattr(
            self.model_runner, "elastic_pp_decode_graph_metrics", None
        )
        if graph_metrics is not None:
            metrics["decode_graph"] = graph_metrics()
        prepared = getattr(self.model_runner, "elastic_pp_prepared_graph_metrics", None)
        if prepared is not None:
            metrics["prepared_decode_graphs"] = prepared()
        for name, component in (
            ("kv", self.kv_transport),
            ("host", self.cache_transition),
        ):
            provider = getattr(component, "metrics", None)
            if provider is not None:
                metrics[name] = provider() if callable(provider) else provider
        return metrics

    def _preflight_capacity(self, transition):
        local_error, details = None, None
        try:
            method = getattr(
                self.model_runner, "elastic_pp_transition_capacity_report", None
            )
            if method is not None:
                details = (
                    method(transition, source_backing_released=True)
                    if self._source_backing_attempted
                    else method(transition)
                )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        reports = self._consensus(
            "capacity", success=local_error is None, error=local_error, details=details
        )
        self._raise_peer_failures("capacity", reports)
        logger.info(
            "Elastic PP strategy=%s capacity reports=%s", self.strategy, reports
        )

    def _prepare_backing(self, transition: DynamicPPTransition) -> None:
        """Shared target allocation; sync policies call only after draining."""
        self._layer_prepared = True
        self._timed("prepare_layers", self.layer_transport.prepare_target, transition)
        bank = getattr(self.model_runner, "_elastic_pp_prepared_graphs", None)
        policy = getattr(
            getattr(self.model_runner, "server_args", None),
            "elastic_pp_prepared_weight_policy",
            "reuse",
        )
        if bank is not None and not bank.building and policy == "reuse":
            self._prepared_weight_bytes_reused = self._timed(
                "reuse_prepared_weights",
                bank.reuse_weights,
                self.model_runner,
                transition,
            )
            self._weights_reused = True
            self._layers_precopied = True
        self._kv_prepared = True
        self._timed("prepare_kv", self.kv_transport.prepare_transition, transition)
        if self._source_backing_attempted:
            self._source_backing_metrics["target_restore"] = (
                self.kv_transport.restore_target_from_snapshot(transition)
            )
        self._observe_memory()

    def _release_source_backing_at_fence(self, transition):
        if getattr(self.model_runner, "_unified_memory_pool", None) is not None:
            raise NotImplementedError(
                "CPU spill requires independently owned KV backing"
            )
        for name in (
            "release_source_backing",
            "restore_source_backing",
            "restore_target_from_snapshot",
            "finalize_source_backing",
        ):
            if not callable(getattr(self.kv_transport, name, None)):
                raise NotImplementedError(f"CPU spill transport is missing {name}")
        if self.cache_transition is not None:
            for name in (
                "release_source_device_aliases",
                "restore_source_device_aliases",
                "finalize_source_device_aliases",
            ):
                if not callable(getattr(self.cache_transition, name, None)):
                    raise NotImplementedError(
                        f"CPU spill host adapter is missing {name}"
                    )
        device = torch.device(getattr(self.model_runner, "device", "cpu"))

        def sample():
            return {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "free_bytes": torch.cuda.mem_get_info(device)[0],
            }

        before = sample() if device.type == "cuda" else None
        # Partial release failures also require source reconstruction.
        self._source_backing_attempted = True
        invalidate = getattr(
            self.model_runner, "invalidate_elastic_pp_attention_state_views", None
        )
        if invalidate is not None:
            invalidate()
        if self.cache_transition is not None:
            self.cache_transition.release_source_device_aliases(transition)
        self._source_backing_metrics["release"] = (
            self.kv_transport.release_source_backing(transition)
        )
        if device.type == "cuda":
            self._timed("fence_device_drain", torch.cuda.synchronize, device)
            with torch.cuda.device(device):
                # Admission uses physical free bytes. Return unused allocator
                # blocks before that measurement, after tensor owners detach.
                torch.cuda.empty_cache()
            after = sample()
            self._source_backing_metrics["cuda_release"] = {
                "before": before,
                "after": after,
                "allocated_drop_bytes": before["allocated_bytes"]
                - after["allocated_bytes"],
            }
        self._observe_memory()

    def _precopy_immutable_weights(self, transition: DynamicPPTransition) -> None:
        """Restore immutable weights while the old pipeline executes.

        CPU payloads deliberately avoid launching NCCL from a second host thread.
        One layer is live in host staging at a time. KV/GDN remain authoritative
        at the final fence, since their rows can change during serving.
        """
        if self.weight_staging in {"cpu_cache", "cpu_prepacked"}:
            return self._precopy_cpu_cached_weights(transition)
        if self.weight_staging == "pinned":
            return self._precopy_pinned_weights(transition)
        for edge in sorted(transition.kv_transfers, key=self.layer_transport._sort_key):
            for layer_id in range(edge.layer_range.start, edge.layer_range.end):
                transfer = LayerTransfer(
                    edge.src_rank, edge.dst_rank, LayerRange(layer_id, layer_id + 1)
                )
                if self.rank == edge.src_rank:
                    state = self.model_runner.export_elastic_pp_layer_state([layer_id])
                    started = time.monotonic()
                    cpu_state = {
                        layer_id: {
                            name: tensor.detach().to("cpu", copy=True)
                            for name, tensor in state[layer_id].items()
                        }
                    }
                    self._stage_seconds["weight_legacy_d2h"] = (
                        self._stage_seconds.get("weight_legacy_d2h", 0.0)
                        + time.monotonic()
                        - started
                    )
                    payload = flatten_layer_states(
                        generation=transition.target.generation,
                        transfer=transfer,
                        layer_states=cpu_state,
                    )
                    self._background_weight_bytes_sent += sum(
                        t.numel() * t.element_size()
                        for values in cpu_state.values()
                        for t in values.values()
                    )
                    started = time.monotonic()
                    works = self.group.send_tensor_dict(
                        payload, dst=edge.dst_rank, async_send=True
                    )
                    self.layer_transport.wait(works or ())
                    self._stage_seconds["weight_legacy_send"] = (
                        self._stage_seconds.get("weight_legacy_send", 0.0)
                        + time.monotonic()
                        - started
                    )
                elif self.rank == edge.dst_rank:
                    payload = self._timed(
                        "weight_legacy_recv",
                        self.group.recv_tensor_dict,
                        src=edge.src_rank,
                    )
                    state = unflatten_layer_states(
                        payload,
                        generation=transition.target.generation,
                        transfer=transfer,
                    )
                    if any(
                        t.device.type != "cpu"
                        for values in state.values()
                        for t in values.values()
                    ):
                        raise RuntimeError(
                            "background weight transport requires CPU payloads"
                        )
                    started = time.monotonic()
                    with self.layer_transport._stream_context():
                        self.model_runner.import_elastic_pp_layer_state(
                            generation=transition.target.generation, layer_states=state
                        )
                        if self.layer_transport._migration_stream is not None:
                            self.layer_transport._migration_stream.synchronize()
                    self._stage_seconds["weight_legacy_h2d"] = (
                        self._stage_seconds.get("weight_legacy_h2d", 0.0)
                        + time.monotonic()
                        - started
                    )

    def _precopy_cpu_cached_weights(self, transition):
        from sglang.srt.elastic_pp.cpu_weight_reload import CanonicalCpuWeightReloader

        if self._cpu_weight_reloader is None:
            self._cpu_weight_reloader = CanonicalCpuWeightReloader(
                model_runner=self.model_runner,
                stream=self.layer_transport._migration_stream,
            )
            self.model_runner._elastic_pp_cpu_weight_reloader = (
                self._cpu_weight_reloader
            )
        incoming = {
            layer_id
            for edge in transition.kv_transfers
            if edge.dst_rank == self.rank
            for layer_id in range(edge.layer_range.start, edge.layer_range.end)
        }
        self._cpu_weight_reloader.reload(
            generation=transition.target.generation, layer_ids=incoming
        )

    def _precopy_pinned_weights(self, transition):
        from sglang.srt.elastic_pp.weight_staging import (
            PinnedWeightStager,
            WeightPacketPlan,
        )

        if self._weight_stager is None:
            self._weight_stager = PinnedWeightStager(
                device=self.model_runner.device,
                stream=self.layer_transport._migration_stream,
            )
        self._weight_stager.reset_metrics()
        for edge in sorted(transition.kv_transfers, key=self.layer_transport._sort_key):
            if self.rank not in {edge.src_rank, edge.dst_rank}:
                continue
            for layer_id in range(edge.layer_range.start, edge.layer_range.end):
                transfer = LayerTransfer(
                    edge.src_rank, edge.dst_rank, LayerRange(layer_id, layer_id + 1)
                )
                plan = WeightPacketPlan(
                    generation=transition.target.generation,
                    transfer=transfer,
                    layer_states=self.model_runner.export_elastic_pp_layer_state(
                        [layer_id]
                    ),
                )
                if self.rank == edge.src_rank:
                    self._weight_stager.send(
                        plan=plan,
                        group=self.group,
                        destination=edge.dst_rank,
                        wait=self.layer_transport.wait,
                    )
                    self._background_weight_bytes_sent += plan.total_bytes
                else:
                    self._weight_stager.receive(
                        plan=plan, group=self.group, source=edge.src_rank
                    )
                    self.model_runner.mark_elastic_pp_layers_loaded(
                        generation=transition.target.generation, layer_ids={layer_id}
                    )

    def _prepare_worker(self) -> None:
        transition = self._require_transition()
        generation = transition.target.generation
        try:
            local_error = None
            try:
                self._set_worker_device()
                if self.enable_kv_patch:
                    self.kv_transport.check_live_patch()
                if self.strategy in {"background", "legacy"}:
                    if self.strategy == "background":
                        self._timed("capacity", self._preflight_capacity, transition)
                    self._prepare_backing(transition)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"

            logger.info(
                "Elastic PP generation=%d rank=%d allocation consensus begin",
                generation,
                self.rank,
            )
            prepare_policy = {
                "reuse_weights": self._weights_reused,
                "background_host": bool(
                    getattr(self.cache_transition, "background", False)
                ),
            }
            reports = self._consensus(
                "prepare_allocations",
                success=local_error is None,
                error=local_error,
                details=prepare_policy,
            )
            self._raise_peer_failures("prepare allocations", reports)
            if any(r.get("details") != prepare_policy for r in reports):
                raise RuntimeError("prepared weight policy differs across ranks")
            # Never enter P2P pre-copy before all peers have allocated backing:
            # a peer allocation failure must not strand a sender/receiver.
            if self.strategy == "background":
                local_error = None
                try:
                    if not self._weights_reused:
                        self._timed(
                            "precopy_weights",
                            self._precopy_immutable_weights,
                            transition,
                        )
                    self._layers_precopied = True
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                reports = self._consensus(
                    "background_copy", success=local_error is None, error=local_error
                )
                self._raise_peer_failures("background copy", reports)
            if self.cache_transition is not None and getattr(
                self.cache_transition, "background", False
            ):
                reports = self._consensus(
                    "host_background_capture",
                    success=True,
                    details=self.cache_transition.precopy is not None,
                )
                enabled = all(r.get("details") is True for r in reports)
                local_error = None
                try:
                    self._timed(
                        "background_host_allocate",
                        self.cache_transition.allocate_background,
                        transition,
                        enabled=enabled,
                    )
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                reports = self._consensus(
                    "host_background_allocate",
                    success=local_error is None,
                    error=local_error,
                )
                self._raise_peer_failures("background host allocation", reports)
                local_error = None
                try:
                    self._timed(
                        "background_host_copy",
                        self.cache_transition.copy_background,
                        transition,
                    )
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                reports = self._consensus(
                    "host_background_copy",
                    success=local_error is None,
                    error=local_error,
                )
                self._raise_peer_failures("background host copy", reports)
            logger.info(
                "Elastic PP generation=%d rank=%d allocation consensus complete",
                generation,
                self.rank,
            )
            # Scheduler loops on different ranks can observe background
            # allocation completion at slightly different wall-clock times.
            # Stopping as soon as each local thread becomes ready can therefore
            # leave one stage launching a microbatch whose neighbour has already
            # begun draining. Agree on a sufficiently-future *logical* PP step
            # so every stage stops scheduling at exactly the same boundary.
            local_error = None
            local_step = None
            try:
                local_step = int(self.fence_step_provider())
                if local_step < 0:
                    raise ValueError(
                        f"elastic PP microstep must be non-negative: {local_step}"
                    )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "arm_fence",
                success=local_error is None,
                error=local_error,
                details=local_step,
            )
            self._raise_peer_failures("arm fence", reports)
            peer_steps = [report.get("details") for report in reports]
            if any(not isinstance(step, int) or step < 0 for step in peer_steps):
                raise RuntimeError(
                    f"elastic PP arm fence returned invalid microsteps: {peer_steps!r}"
                )
            fence_microstep = max(peer_steps) + self.fence_lead_steps
            logger.info(
                "Elastic PP generation=%d rank=%d armed fence at microstep=%d",
                generation,
                self.rank,
                fence_microstep,
            )
            with self._condition:
                self._fence_microstep = fence_microstep
                self._bulk_ready_at = time.monotonic()
                self._phase = ElasticPPMigrationPhase.BULK_READY
                self._condition.notify_all()
        except BaseException as exc:
            self._fail_before_publication(exc)

    def finalize_at_fence(
        self, commit_topology: Callable[[PPTopology], None]
    ) -> PPTopology:
        """Copy weights/current KV and publish a fully drained generation.

        This method is called by the scheduler thread on every resident rank.
        No migration communicator operation is launched by the background
        prepare thread, so serving and migration NCCL have a deterministic
        single-host-thread order on each device.
        """

        transition = self._require_transition()
        with self._condition:
            if self._phase != ElasticPPMigrationPhase.BULK_READY:
                raise RuntimeError(
                    "elastic PP fence requires globally bulk-ready state: "
                    f"phase={self._phase.value}"
                )
            self._phase = ElasticPPMigrationPhase.FENCING
            self._fence_started_at = time.monotonic()
            self._condition.notify_all()

        commit_started = False
        try:
            # Host I/O and live allocator reads occur only at the drained fence.
            # Keep the source generation intact until all ranks can publish.
            local_error = None
            try:
                device = torch.device(getattr(self.model_runner, "device", "cpu"))
                if device.type == "cuda":
                    # Dedicated transfer streams must observe all serving producers.
                    self._timed("fence_device_drain", torch.cuda.synchronize, device)
                if self.cache_transition is not None:
                    self._timed(
                        "fence_host_snapshot",
                        self.cache_transition.prepare_transition,
                        transition,
                    )
                if self.enable_kv_patch:
                    self._timed(
                        "fence_patch_finish",
                        self.kv_transport.finish_live_patch,
                        transition,
                    )
                if self.strategy in {"sync", "batched"}:
                    snapshot = getattr(self.kv_transport, "snapshot_at_fence", None)
                    if snapshot is not None:
                        snapshot(transition)
                    # Snapshot completion and peak admission precede allocation.
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "fence_snapshot", success=local_error is None, error=local_error
            )
            self._raise_peer_failures("fence snapshot", reports)
            if self.backing == "cpu_spill":
                local_error = None
                try:
                    self._release_source_backing_at_fence(transition)
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                reports = self._consensus(
                    "source_backing_release",
                    success=local_error is None,
                    error=local_error,
                )
                self._raise_peer_failures("source backing release", reports)
            if self.strategy in {"sync", "batched"}:
                self._preflight_capacity(transition)
            local_error = None
            try:
                if self.strategy in {"sync", "batched"}:
                    self._prepare_backing(transition)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "fence_prepare", success=local_error is None, error=local_error
            )
            self._raise_peer_failures("fence prepare", reports)
            local_error = None
            fence_slots = None
            slot_signature = None
            try:
                fence_slots = self.kv_transport.fence_slots(transition)
                slot_signature = self._slot_signature(fence_slots)
                state_signature = getattr(
                    self.kv_transport, "fence_state_signature", None
                )
                if state_signature is not None:
                    slot_signature = {
                        "kv": slot_signature,
                        "recurrent": state_signature(),
                    }
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "fence_slots",
                success=local_error is None,
                error=local_error,
                details=slot_signature,
            )
            self._raise_peer_failures("fence slots", reports)
            if isinstance(slot_signature, dict) and "recurrent" in slot_signature:
                # All ranks retain scheduler state, including inactive ranks.
                # A recurrent slot is meaningful only if every owner agrees on
                # its request mapping. Reject skew before entering P2P copies.
                for report in reports:
                    details = report.get("details")
                    if (
                        not isinstance(details, dict)
                        or details.get("recurrent") != slot_signature["recurrent"]
                    ):
                        raise RuntimeError(
                            "elastic PP recurrent slot mapping differs across ranks; "
                            "cannot migrate GDN state without request-slot remapping"
                        )
            assert fence_slots is not None

            # Startup's private Graph walk uses synchronous transitions. It
            # must use the same CPU bank, and reach agreement before any KV
            # P2P operation if a rank cannot restore its target checkpoint.
            if (
                self.weight_staging in {"cpu_cache", "cpu_prepacked"}
                and not self._layers_precopied
            ):
                local_error = None
                try:
                    self._timed(
                        "fence_cpu_cached_weights",
                        self._precopy_cpu_cached_weights,
                        transition,
                    )
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                reports = self._consensus(
                    "fence_cpu_cached_weights",
                    success=local_error is None,
                    error=local_error,
                )
                self._raise_peer_failures("fence CPU cached weights", reports)
                self._layers_precopied = True

            local_error = None
            works = ()
            try:
                stage_started = time.monotonic()
                logger.info(
                    "Elastic PP generation=%d rank=%d fenced layer copy begin",
                    transition.target.generation,
                    self.rank,
                )
                if not self._layers_precopied:
                    works = self.layer_transport.start_sends(transition)
                    try:
                        self.layer_transport.receive(transition)
                    finally:
                        self.layer_transport.wait(works)
                logger.info(
                    "Elastic PP generation=%d rank=%d fenced layer copy complete "
                    "in %.3f s",
                    transition.target.generation,
                    self.rank,
                    time.monotonic() - stage_started,
                )

                stage_started = time.monotonic()
                logger.info(
                    "Elastic PP generation=%d rank=%d fenced KV copy begin "
                    "with %d live slots",
                    transition.target.generation,
                    self.rank,
                    int(fence_slots.numel()),
                )
                previous_chunk_size = getattr(
                    self.kv_transport, "slot_chunk_size", None
                )
                if (
                    self.strategy == "sync"
                    and self.backing == "overlap"
                    and previous_chunk_size is not None
                ):
                    self.kv_transport.slot_chunk_size = max(1, int(fence_slots.numel()))
                try:
                    logger.info(
                        "Elastic PP generation=%d rank=%d KV/GDN bulk exchange begin",
                        transition.target.generation,
                        self.rank,
                    )
                    self._timed(
                        "fence_kv_bulk",
                        self.kv_transport.exchange_phase,
                        transition,
                        phase="bulk",
                        slot_ids=fence_slots,
                    )
                    logger.info(
                        "Elastic PP generation=%d rank=%d KV/GDN bulk exchange complete",
                        transition.target.generation,
                        self.rank,
                    )
                finally:
                    if previous_chunk_size is not None:
                        self.kv_transport.slot_chunk_size = previous_chunk_size
                # Keep the existing two-phase staging invariant while the
                # authoritative snapshot is now taken only once at the fence.
                # An empty delta still carries one manifest per ownership edge
                # and proves the receive set is complete before publication.
                logger.info(
                    "Elastic PP generation=%d rank=%d KV/GDN delta exchange begin",
                    transition.target.generation,
                    self.rank,
                )
                self._timed(
                    "fence_kv_delta",
                    self.kv_transport.exchange_phase,
                    transition,
                    phase="delta",
                    slot_ids=fence_slots[:0],
                )
                logger.info(
                    "Elastic PP generation=%d rank=%d KV/GDN delta exchange complete; import begin",
                    transition.target.generation,
                    self.rank,
                )
                self._timed(
                    "fence_kv_apply", self.kv_transport.apply_staged, transition
                )
                logger.info(
                    "Elastic PP generation=%d rank=%d KV/GDN import complete",
                    transition.target.generation,
                    self.rank,
                )
                if self.cache_transition is not None:
                    logger.info(
                        "Elastic PP generation=%d rank=%d HiCache host exchange begin",
                        transition.target.generation,
                        self.rank,
                    )
                    self._timed(
                        "fence_host_exchange",
                        self.cache_transition.exchange_at_fence,
                        transition,
                    )
                    logger.info(
                        "Elastic PP generation=%d rank=%d HiCache host exchange complete; host preparation/import begin",
                        transition.target.generation,
                        self.rank,
                    )
                    self._timed(
                        "fence_host_apply",
                        self.cache_transition.apply_staged,
                        transition,
                    )
                    logger.info(
                        "Elastic PP generation=%d rank=%d HiCache host preparation/import complete",
                        transition.target.generation,
                        self.rank,
                    )
                logger.info(
                    "Elastic PP generation=%d rank=%d fenced KV copy complete "
                    "in %.3f s",
                    transition.target.generation,
                    self.rank,
                    time.monotonic() - stage_started,
                )
                if not self.model_runner.elastic_pp_generation_ready(
                    generation=transition.target.generation
                ):
                    raise RuntimeError(
                        "local model/KV generation lost readiness at the fence"
                    )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "fence_copy",
                success=local_error is None,
                error=local_error,
            )
            self._raise_peer_failures("fence copy", reports)

            local_error = None
            try:
                commit_started = True
                self._timed(
                    "fence_model_commit",
                    self.model_runner.commit_elastic_pp_generation,
                    generation=transition.target.generation,
                )
                if self.cache_transition is not None:
                    self._timed(
                        "fence_host_commit", self.cache_transition.commit, transition
                    )
                # From this point onward the local source generation cannot be
                # restored, even if route publication or peer commit fails.
                self._timed("fence_route_publish", commit_topology, transition.target)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "commit",
                success=local_error is None,
                error=local_error,
            )
            self._raise_peer_failures("commit", reports)

            # Every rank has installed its target weights, pool and PP route.
            # Select the startup-prepared runner when enabled; otherwise native
            # pools changed addresses and require drained recapture.
            local_error = None
            try:
                rebuild = getattr(
                    self.model_runner, "rebuild_elastic_pp_decode_graph", None
                )
                if rebuild is not None:
                    self._timed(
                        "fence_graph_select",
                        rebuild,
                        generation=transition.target.generation,
                        active=not transition.target.layer_range(self.rank).empty,
                    )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "decode_graph", success=local_error is None, error=local_error
            )
            self._raise_peer_failures("Decode Graph rebuild", reports)

            with self._condition:
                self.current_topology = transition.target
                self._committed_at = time.monotonic()
                self._phase = ElasticPPMigrationPhase.COMMITTED
                self._condition.notify_all()
            self._observe_memory()
            logger.info(
                "ELASTIC_PP_TRANSITION_METRICS %s",
                json.dumps(self.transition_metrics()),
            )
            return transition.target
        except BaseException as exc:
            if not commit_started:
                self._fail_before_publication(exc)
            else:
                self._fail(exc, source_usable=False)
            raise

    def start_cleanup(self, completion_event: Any = None) -> None:
        transition = self._require_transition()
        with self._condition:
            if self._phase != ElasticPPMigrationPhase.COMMITTED:
                raise RuntimeError(
                    "elastic PP cleanup requires a committed generation: "
                    f"phase={self._phase.value}"
                )
            self._phase = ElasticPPMigrationPhase.CLEANING
            self._thread = Thread(
                target=self._cleanup_worker,
                args=(transition, completion_event),
                name=f"elastic-pp-cleanup-g{transition.target.generation}-r{self.rank}",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()

    def _cleanup_worker(
        self, transition: DynamicPPTransition, completion_event: Any
    ) -> None:
        try:
            local_error = None
            try:
                self._set_worker_device()
                if completion_event is not None:
                    completion_event.synchronize()
                cuda_cleanup = (
                    torch.device(getattr(self.model_runner, "device", "cpu")).type
                    == "cuda"
                )
                allocated_before = (
                    torch.cuda.memory_allocated() if cuda_cleanup else None
                )
                self.model_runner.cleanup_elastic_pp_generation(
                    generation=transition.target.generation
                )
                if self._source_backing_attempted:
                    # Every rank already acknowledged publication. CPU source
                    # authority can now be retired, never used for rollback.
                    self.kv_transport.finalize_source_backing(transition)
                    if self.cache_transition is not None:
                        self.cache_transition.finalize_source_device_aliases(transition)
                if self.enable_kv_patch:
                    self.kv_transport.release_live_patch()
                if self.cache_transition is not None:
                    self.cache_transition.release_retired()
                release_snapshot = getattr(
                    self.kv_transport, "release_fence_snapshot", None
                )
                if release_snapshot is not None:
                    release_snapshot(transition)
                if cuda_cleanup:
                    logger.info(
                        "Elastic PP generation=%d rank=%d cleanup CUDA allocated "
                        "%.3f -> %.3f GiB",
                        transition.target.generation,
                        self.rank,
                        allocated_before / 2**30,
                        torch.cuda.memory_allocated() / 2**30,
                    )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "cleanup",
                success=local_error is None,
                error=local_error,
            )
            self._raise_peer_failures("cleanup", reports)
            with self._condition:
                self._done_at = time.monotonic()
                self._phase = ElasticPPMigrationPhase.DONE
                self._condition.notify_all()
        except BaseException as exc:
            # The target generation remains the committed serving generation;
            # cleanup failure cannot make the retired source generation usable.
            self._fail(exc, source_usable=False)

    def _abort_prepared_generation(self) -> None:
        transition = self._transition
        if transition is None:
            return
        generation = transition.target.generation
        errors = []

        def attempt(name, action):
            try:
                action()
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

        if self.cache_transition is not None:
            attempt("host abort", lambda: self.cache_transition.abort(transition))
        attempt("KV staging abort", lambda: self.kv_transport.abort_staged(transition))
        if self._kv_prepared:
            attempt(
                "KV backing abort",
                lambda: self.model_runner.abort_elastic_pp_kv_layers(
                    generation=generation
                ),
            )
            self._kv_prepared = False
        if self._layer_prepared:
            attempt(
                "weight backing abort",
                lambda: self.model_runner.abort_elastic_pp_layers(
                    generation=generation
                ),
            )
            self._layer_prepared = False
        if self._source_backing_attempted:
            attempt(
                "source backing restore",
                lambda: self.kv_transport.restore_source_backing(transition),
            )
            if self.cache_transition is not None:
                attempt(
                    "source host device aliases restore",
                    lambda: self.cache_transition.restore_source_device_aliases(
                        transition
                    ),
                )
        if errors:
            raise RuntimeError("; ".join(errors))

    def _fail_before_publication(self, exc: BaseException) -> None:
        """Resume only after every resident rank proves source rollback."""
        try:
            local_error = None
            try:
                self._abort_prepared_generation()
            except BaseException as rollback_exc:
                local_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
            reports = self._consensus(
                "abort_restore", success=local_error is None, error=local_error
            )
            self._raise_peer_failures("source rollback", reports)
        except BaseException as rollback_exc:
            self._fail(
                RuntimeError(f"{exc}; source rollback unverified: {rollback_exc}"),
                source_usable=False,
            )
        else:
            self._fail(exc, source_usable=True)

    def _fail(self, exc: BaseException, *, source_usable: bool) -> None:
        with self._condition:
            self._error = f"{type(exc).__name__}: {exc}"
            self._source_usable = source_usable
            self._phase = ElasticPPMigrationPhase.FAILED
            self._condition.notify_all()
        logger.error(
            "Elastic PP rank=%d migration failed (source_usable=%s): %s",
            self.rank,
            source_usable,
            self._error,
        )

    def _require_transition(self) -> DynamicPPTransition:
        with self._lock:
            if self._transition is None:
                raise RuntimeError("no elastic PP transition is active")
            return self._transition

    def wait_for_phase(
        self,
        phases: ElasticPPMigrationPhase | set[ElasticPPMigrationPhase],
        *,
        timeout: float,
    ) -> ElasticPPMigrationSnapshot:
        expected = {phases} if isinstance(phases, ElasticPPMigrationPhase) else phases
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._phase not in expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for elastic PP phase "
                        f"{sorted(item.value for item in expected)}; "
                        f"current={self._phase.value}"
                    )
                self._condition.wait(remaining)
        return self.snapshot()
