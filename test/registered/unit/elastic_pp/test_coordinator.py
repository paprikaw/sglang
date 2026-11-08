from __future__ import annotations

import threading

import pytest
import torch

from sglang.srt.elastic_pp.coordinator import (
    ElasticPPMigrationCoordinator,
    ElasticPPMigrationPhase,
)
from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CompletedWork:
    @staticmethod
    def is_completed():
        return True

    @staticmethod
    def wait():
        return True


class _Group:
    world_size = 2

    def __init__(
        self,
        *,
        fail_stage=None,
        mismatch_stage=None,
        peer_fence_step=None,
        peer_drain_state=None,
        peer_drain_microstep=None,
    ):
        self.fail_stage = fail_stage
        self.mismatch_stage = mismatch_stage
        self.peer_fence_step = peer_fence_step
        self.peer_drain_state = peer_drain_state
        self.peer_drain_microstep = peer_drain_microstep
        self.stages = []
        self.drain_reductions = []

    def all_gather_object(self, report):
        self.stages.append(report["stage"])
        peer = dict(report)
        peer["rank"] = 1
        if report["stage"] == self.fail_stage:
            peer["success"] = False
            peer["error"] = "injected peer failure"
        if report["stage"] == self.mismatch_stage:
            peer["details"] = (999, "different")
        if report["stage"] == "arm_fence" and self.peer_fence_step is not None:
            peer["details"] = self.peer_fence_step
        if (
            report["stage"] == "scheduler_alignment"
            and self.peer_drain_microstep is not None
        ):
            peer["details"] = (
                self.peer_drain_microstep,
                report["details"][1],
            )
        return [report, peer]

    def all_reduce_cpu(self, tensor, *, async_op=False):
        assert async_op
        local_ready, generation = tensor.tolist()
        peer_ready = (
            local_ready if self.peer_drain_state is None else self.peer_drain_state
        )
        self.drain_reductions.append((local_ready, generation))
        tensor.add_(
            torch.tensor(
                [int(peer_ready), generation],
                dtype=tensor.dtype,
            )
        )
        return _CompletedWork()


class _LayerTransport:
    def __init__(self):
        self.prepared = False
        self.received = False
        self.copy_threads = []

    def prepare_target(self, transition):
        del transition
        self.prepared = True
        return frozenset({1})

    def start_sends(self, transition):
        del transition
        self.copy_threads.append(threading.current_thread().name)
        return ()

    def receive(self, transition):
        del transition
        self.received = True
        self.copy_threads.append(threading.current_thread().name)
        return frozenset({1})

    @staticmethod
    def wait(works):
        assert works == ()


class _KVTransport:
    def __init__(self):
        self.phases = []
        self.finished = False
        self.applied = False
        self.aborted_staged = False
        self.snapshot_released = False
        self.exchange_threads = []

    def release_fence_snapshot(self, transition):
        assert transition.target.generation > 0
        self.snapshot_released = True

    def prepare_transition(self, transition):
        del transition

    def exchange_phase(self, transition, phase, slot_ids):
        del transition
        self.phases.append((phase, tuple(slot_ids.tolist())))
        self.exchange_threads.append(threading.current_thread().name)
        return frozenset({1})

    def fence_slots(self, transition):
        del transition
        return torch.tensor([1, 3, 5], dtype=torch.long)

    def delta_slots(self, transition):
        del transition
        return torch.tensor([5], dtype=torch.long)

    def finish_slot_tracking(self, transition):
        del transition
        self.finished = True

    def apply_staged(self, transition):
        del transition
        self.applied = True
        return frozenset({1})

    def abort_staged(self, transition):
        del transition
        self.aborted_staged = True


class _Runner:
    def __init__(self):
        self.committed = False
        self.cleaned = False
        self.aborted_layers = False
        self.aborted_kv = False

    def elastic_pp_generation_ready(self, generation):
        del generation
        return True

    def commit_elastic_pp_generation(self, generation):
        del generation
        self.committed = True

    def cleanup_elastic_pp_generation(self, generation):
        del generation
        self.cleaned = True

    def abort_elastic_pp_layers(self, generation):
        del generation
        self.aborted_layers = True

    def abort_elastic_pp_kv_layers(self, generation):
        del generation
        self.aborted_kv = True


def _transition():
    source = PPTopology.from_layer_counts([2, 2], generation=0)
    target = PPTopology.from_layer_counts([1, 3], generation=1)
    return source, DynamicPPTransition.build(source, target)


def _coordinator(group=None, *, fence_step_provider=None, fence_lead_steps=0):
    source, transition = _transition()
    runner = _Runner()
    kv_transport = _KVTransport()
    coordinator = ElasticPPMigrationCoordinator(
        rank=0,
        current_topology=source,
        group=group or _Group(),
        model_runner=runner,
        layer_transport=_LayerTransport(),
        kv_transport=kv_transport,
        fence_step_provider=fence_step_provider,
        fence_lead_steps=fence_lead_steps,
    )
    return coordinator, transition, runner, kv_transport


@pytest.mark.parametrize(
    "device, expected_gpu", [("cpu", None), ("cuda", 3), ("cuda:2", 2)]
)
def test_prepare_fence_commit_and_deferred_cleanup(monkeypatch, device, expected_gpu):
    coordinator, transition, runner, kv_transport = _coordinator()
    installed = []
    runner.device = device
    runner.gpu_id = 3
    selected_devices = []
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda index: selected_devices.append((index, threading.current_thread().name)),
    )

    coordinator.begin(transition)
    ready = coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    assert ready.source_usable
    assert kv_transport.phases == []
    assert not coordinator.layer_transport.received

    assert coordinator.finalize_at_fence(installed.append) == transition.target
    assert coordinator.phase == ElasticPPMigrationPhase.COMMITTED
    assert runner.committed
    assert installed == [transition.target]
    assert kv_transport.phases == [("bulk", (1, 3, 5)), ("delta", ())]
    assert kv_transport.exchange_threads == [
        threading.current_thread().name,
        threading.current_thread().name,
    ]
    assert coordinator.layer_transport.copy_threads == [
        threading.current_thread().name,
        threading.current_thread().name,
    ]
    assert kv_transport.applied
    assert not kv_transport.finished

    coordinator.start_cleanup()
    done = coordinator.wait_for_phase(ElasticPPMigrationPhase.DONE, timeout=2)
    assert runner.cleaned
    assert kv_transport.snapshot_released
    assert done.prepare_seconds is not None
    assert done.fence_seconds is not None
    assert done.total_seconds is not None
    if expected_gpu is None:
        assert selected_devices == []
    else:
        assert selected_devices == [
            (expected_gpu, "elastic-pp-prepare-g1-r0"),
            (expected_gpu, "elastic-pp-cleanup-g1-r0"),
        ]


def test_prepare_arms_one_future_fence_from_the_slowest_scheduler_rank():
    coordinator, transition, _, _ = _coordinator(
        _Group(peer_fence_step=17),
        fence_step_provider=lambda: 11,
        fence_lead_steps=8,
    )

    coordinator.begin(transition)
    ready = coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)

    assert ready.fence_microstep == 25


def test_scheduler_fence_waits_for_every_resident_rank_to_drain():
    group = _Group(peer_drain_state=False, peer_drain_microstep=11)
    coordinator, transition, _, _ = _coordinator(group)
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)

    assert not coordinator.poll_all_ranks_drained_at_fence(True)
    assert not coordinator.poll_all_ranks_drained_at_fence(True)
    group.peer_drain_state = True
    assert not coordinator.poll_all_ranks_drained_at_fence(True)
    assert coordinator.poll_all_ranks_drained_at_fence(True)
    assert group.drain_reductions == [(1, 1), (1, 1)]
    assert coordinator.aligned_resume_microstep(10, 4) == 12
    assert group.stages[-1] == "scheduler_alignment"


def test_prepare_peer_failure_aborts_without_changing_source_generation():
    group = _Group(fail_stage="prepare_allocations")
    coordinator, transition, runner, kv_transport = _coordinator(group)

    coordinator.begin(transition)
    failed = coordinator.wait_for_phase(ElasticPPMigrationPhase.FAILED, timeout=2)
    assert failed.source_usable
    assert coordinator.current_topology == transition.source
    assert runner.aborted_layers
    assert runner.aborted_kv
    assert not kv_transport.finished


def test_fence_slot_failure_reaches_consensus_before_abort():
    group = _Group()
    coordinator, transition, runner, kv_transport = _coordinator(group)
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)

    def fail_fence_slots(transition):
        del transition
        raise RuntimeError("injected live-slot failure")

    kv_transport.fence_slots = fail_fence_slots
    try:
        coordinator.finalize_at_fence(lambda topology: None)
    except RuntimeError as exc:
        assert "fence slots failed" in str(exc)
        assert "injected live-slot failure" in str(exc)
    else:
        raise AssertionError("fence slot failure unexpectedly committed")

    assert group.stages[-1] == "fence_slots"
    assert coordinator.phase == ElasticPPMigrationPhase.FAILED
    assert coordinator.snapshot().source_usable
    assert runner.aborted_layers
    assert runner.aborted_kv
    assert not kv_transport.finished


def test_rank_skewed_bulk_and_delta_snapshots_are_allowed():
    group = _Group(mismatch_stage="prepare_allocations")
    coordinator, transition, runner, _ = _coordinator(group)
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    group.mismatch_stage = "fence_slots"

    assert coordinator.finalize_at_fence(lambda topology: None) == transition.target
    assert coordinator.phase == ElasticPPMigrationPhase.COMMITTED
    assert runner.committed


def test_recurrent_slot_skew_aborts_collectively_before_copy():
    group = _Group(mismatch_stage="fence_slots")
    coordinator, transition, runner, kv_transport = _coordinator(group)
    kv_transport.fence_state_signature = lambda: {"slots": 5, "mapping": "abc"}
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    with pytest.raises(RuntimeError, match="recurrent slot mapping"):
        coordinator.finalize_at_fence(lambda topology: None)
    assert coordinator.phase == ElasticPPMigrationPhase.FAILED
    assert coordinator.snapshot().source_usable
    assert not runner.committed
    assert not coordinator.layer_transport.copy_threads
    assert runner.aborted_layers and runner.aborted_kv


def test_peer_commit_failure_is_reported_as_nonrecoverable_partial_commit():
    group = _Group(fail_stage="commit")
    coordinator, transition, runner, _ = _coordinator(group)
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)

    try:
        coordinator.finalize_at_fence(lambda topology: None)
    except RuntimeError as exc:
        assert "commit failed" in str(exc)
    else:
        raise AssertionError("peer commit failure unexpectedly succeeded")
    assert runner.committed
    assert not coordinator.snapshot().source_usable


def test_pp2_pp4_pp2_reuses_coordinator_without_slot_tracker_lifecycle():
    group = _Group()
    group.world_size = 4
    group.all_gather_object = lambda report: [
        dict(report, rank=rank) for rank in range(4)
    ]
    coordinator, _, _, kv_transport = _coordinator(group)
    coordinator.current_topology = PPTopology.from_layer_counts(
        [2, 2, 0, 0], generation=0
    )
    for generation, counts in enumerate(([1, 1, 1, 1], [2, 2, 0, 0]), start=1):
        transition = DynamicPPTransition.build(
            coordinator.current_topology,
            PPTopology.from_layer_counts(counts, generation=generation),
        )
        coordinator.begin(transition)
        coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
        coordinator.finalize_at_fence(lambda topology: None)
        assert not kv_transport.finished
        coordinator.start_cleanup()
        coordinator.wait_for_phase(ElasticPPMigrationPhase.DONE, timeout=2)
        assert coordinator.current_topology == transition.target
    assert kv_transport.phases == [("bulk", (1, 3, 5)), ("delta", ())] * 2


@pytest.mark.parametrize("strategy", ["sync", "batched"])
def test_synchronous_policies_snapshot_before_target_allocation(strategy):
    coordinator, transition, runner, kv = _coordinator()
    coordinator.strategy = strategy
    events = []
    kv.snapshot_at_fence = lambda transition: events.append("snapshot")
    old_prepare = coordinator.layer_transport.prepare_target
    coordinator.layer_transport.prepare_target = lambda transition: (
        events.append("allocate"),
        old_prepare(transition),
    )[-1]
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    assert events == []
    assert not coordinator.layer_transport.prepared
    coordinator.finalize_at_fence(lambda target: events.append("publish"))
    assert events == ["snapshot", "allocate", "publish"]
    assert runner.committed


@pytest.mark.parametrize("failure_stage", ["prepare", "apply", "commit"])
def test_host_cache_transaction_failure_obeys_publication_boundary(failure_stage):
    coordinator, transition, runner, kv = _coordinator()
    events = []

    class Cache:
        def run(self, stage):
            events.append(stage)
            if stage == failure_stage:
                raise RuntimeError("injected cache failure")

        def prepare_transition(self, transition):
            self.run("prepare")

        def exchange_at_fence(self, transition):
            self.run("exchange")

        def apply_staged(self, transition):
            self.run("apply")

        def commit(self, transition):
            self.run("commit")

        def abort(self, transition):
            events.append("abort")

    coordinator.cache_transition = Cache()
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    with pytest.raises(RuntimeError, match="injected cache failure"):
        coordinator.finalize_at_fence(lambda target: None)
    assert coordinator.snapshot().source_usable is (failure_stage != "commit")
    assert ("abort" in events) is (failure_stage != "commit")
    assert runner.committed is (failure_stage == "commit")


def test_background_precopy_uses_cpu_payload_and_leaves_live_model_unpublished():
    import contextlib
    import queue
    from sglang.srt.elastic_pp.layer_transport import ElasticPPLayerTransport

    payloads = queue.Queue()
    source, transition = _transition()
    tensors = {"weight": torch.arange(12).reshape(3, 4).float()}
    imported = {}

    class Group:
        world_size = 2

        def send_tensor_dict(self, payload, *, dst, async_send):
            assert dst == 1 and async_send
            assert all(
                t.device.type == "cpu"
                for t in payload.values()
                if isinstance(t, torch.Tensor)
            )
            payloads.put(payload)
            return ()

        def recv_tensor_dict(self, *, src):
            assert src == 0
            return payloads.get(timeout=2)

    class Runner(_Runner):
        def export_elastic_pp_layer_state(self, layer_ids):
            assert list(layer_ids) == [1]
            return {1: tensors}

        def import_elastic_pp_layer_state(self, *, generation, layer_states):
            assert generation == 1
            imported.update(layer_states)

    class Layers(_LayerTransport):
        _migration_stream = None
        _sort_key = staticmethod(ElasticPPLayerTransport._sort_key)
        _stream_context = staticmethod(contextlib.nullcontext)

    def make(rank):
        return ElasticPPMigrationCoordinator(
            rank=rank,
            current_topology=source,
            group=Group(),
            model_runner=Runner(),
            layer_transport=Layers(),
            kv_transport=_KVTransport(),
            strategy="background",
        )

    sender, receiver = make(0), make(1)
    sender._precopy_immutable_weights(transition)
    receiver._precopy_immutable_weights(transition)
    assert torch.equal(imported[1]["weight"], tensors["weight"])
    assert imported[1]["weight"].data_ptr() != tensors["weight"].data_ptr()
    assert not sender.model_runner.committed and not receiver.model_runner.committed


def test_cleanup_failure_never_promises_retired_source_rollback():
    coordinator, transition, runner, _ = _coordinator()
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    coordinator.finalize_at_fence(lambda topology: None)

    def fail_cleanup(**kwargs):
        raise RuntimeError("injected retired backing cleanup failure")

    runner.cleanup_elastic_pp_generation = fail_cleanup
    coordinator.start_cleanup()
    failed = coordinator.wait_for_phase(ElasticPPMigrationPhase.FAILED, timeout=2)
    assert not failed.source_usable
    assert coordinator.current_topology == transition.target
    assert runner.committed
