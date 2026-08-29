"""Rank-local orchestration for transactional elastic PP migration."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Condition, RLock, Thread
from typing import Any

import torch

from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology

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
    ) -> None:
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
        self._slot_tracking = False
        self._bulk_slots: torch.Tensor | None = None

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
            self._slot_tracking = False
            self._bulk_slots = None
            self._thread = Thread(
                target=self._prepare_worker,
                name=f"elastic-pp-prepare-g{transition.target.generation}-r{self.rank}",
                daemon=True,
            )
            self._thread.start()

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
        reports = self.group.all_gather_object(report)
        if len(reports) != self.group.world_size:
            raise RuntimeError(
                "elastic PP consensus returned the wrong rank count: "
                f"got={len(reports)}, expected={self.group.world_size}"
            )
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

    def _prepare_worker(self) -> None:
        transition = self._require_transition()
        generation = transition.target.generation
        try:
            local_error = None
            try:
                self._set_worker_device()
                stage_started = time.monotonic()
                logger.info(
                    "Elastic PP generation=%d rank=%d layer prepare begin",
                    generation,
                    self.rank,
                )
                self.layer_transport.prepare_target(transition)
                self._layer_prepared = True
                logger.info(
                    "Elastic PP generation=%d rank=%d layer prepare complete "
                    "in %.3f s",
                    generation,
                    self.rank,
                    time.monotonic() - stage_started,
                )
                stage_started = time.monotonic()
                logger.info(
                    "Elastic PP generation=%d rank=%d KV prepare begin",
                    generation,
                    self.rank,
                )
                self._bulk_slots = self.kv_transport.prepare_transition(transition)
                self._kv_prepared = True
                self._slot_tracking = True
                slot_signature = self._slot_signature(self._bulk_slots)
                logger.info(
                    "Elastic PP generation=%d rank=%d KV prepare complete "
                    "in %.3f s with %d live slots",
                    generation,
                    self.rank,
                    time.monotonic() - stage_started,
                    slot_signature[0],
                )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
                slot_signature = None

            logger.info(
                "Elastic PP generation=%d rank=%d allocation consensus begin",
                generation,
                self.rank,
            )
            reports = self._consensus(
                "prepare_allocations",
                success=local_error is None,
                error=local_error,
                details=slot_signature,
            )
            self._raise_peer_failures("prepare allocations", reports)
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
            self._abort_prepared_generation()
            self._fail(exc, source_usable=True)

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

            local_error = None
            works = ()
            try:
                stage_started = time.monotonic()
                logger.info(
                    "Elastic PP generation=%d rank=%d fenced layer copy begin",
                    transition.target.generation,
                    self.rank,
                )
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
                self.kv_transport.exchange_phase(
                    transition,
                    phase="bulk",
                    slot_ids=fence_slots,
                )
                # Keep the existing two-phase staging invariant while the
                # authoritative snapshot is now taken only once at the fence.
                # An empty delta still carries one manifest per ownership edge
                # and proves the receive set is complete before publication.
                self.kv_transport.exchange_phase(
                    transition,
                    phase="delta",
                    slot_ids=fence_slots[:0],
                )
                self.kv_transport.apply_staged(transition)
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
                self.model_runner.commit_elastic_pp_generation(
                    generation=transition.target.generation
                )
                # From this point onward the local source generation cannot be
                # restored, even if route publication or peer commit fails.
                commit_topology(transition.target)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            reports = self._consensus(
                "commit",
                success=local_error is None,
                error=local_error,
            )
            self._raise_peer_failures("commit", reports)

            self.kv_transport.finish_slot_tracking(transition)
            self._slot_tracking = False
            with self._condition:
                self.current_topology = transition.target
                self._committed_at = time.monotonic()
                self._phase = ElasticPPMigrationPhase.COMMITTED
                self._condition.notify_all()
            return transition.target
        except BaseException as exc:
            if not commit_started:
                self._abort_prepared_generation()
            self._fail(exc, source_usable=not commit_started)
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
            # cleanup failure is a leak/retry condition, not a rollback.
            self._fail(exc, source_usable=True)

    def _abort_prepared_generation(self) -> None:
        transition = self._transition
        if transition is None:
            return
        generation = transition.target.generation
        try:
            self.kv_transport.abort_staged(transition)
        except BaseException:
            pass
        if self._slot_tracking:
            try:
                self.kv_transport.finish_slot_tracking(transition)
            except BaseException:
                pass
            self._slot_tracking = False
        if self._kv_prepared:
            try:
                self.model_runner.abort_elastic_pp_kv_layers(generation=generation)
            except BaseException:
                pass
            self._kv_prepared = False
        if self._layer_prepared:
            try:
                self.model_runner.abort_elastic_pp_layers(generation=generation)
            except BaseException:
                pass
            self._layer_prepared = False

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
