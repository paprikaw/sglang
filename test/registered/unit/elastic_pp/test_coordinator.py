from __future__ import annotations

import torch

from sglang.srt.elastic_pp.coordinator import (
    ElasticPPMigrationCoordinator,
    ElasticPPMigrationPhase,
)
from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Group:
    world_size = 2

    def __init__(self, *, fail_stage=None, mismatch_stage=None, peer_fence_step=None):
        self.fail_stage = fail_stage
        self.mismatch_stage = mismatch_stage
        self.peer_fence_step = peer_fence_step
        self.stages = []

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
        return [report, peer]


class _LayerTransport:
    def __init__(self):
        self.prepared = False
        self.received = False

    def prepare_target(self, transition):
        del transition
        self.prepared = True
        return frozenset({1})

    def start_sends(self, transition):
        del transition
        return ()

    def receive(self, transition):
        del transition
        self.received = True
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

    def prepare_transition(self, transition):
        del transition
        return torch.tensor([1, 3], dtype=torch.long)

    def exchange_phase(self, transition, phase, slot_ids):
        del transition
        self.phases.append((phase, tuple(slot_ids.tolist())))
        return frozenset({1})

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


def test_prepare_fence_commit_and_deferred_cleanup():
    coordinator, transition, runner, kv_transport = _coordinator()
    installed = []

    coordinator.begin(transition)
    ready = coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    assert ready.source_usable
    assert kv_transport.phases == [("bulk", (1, 3))]

    assert coordinator.finalize_at_fence(installed.append) == transition.target
    assert coordinator.phase == ElasticPPMigrationPhase.COMMITTED
    assert runner.committed
    assert installed == [transition.target]
    assert kv_transport.phases[-1] == ("delta", (5,))
    assert kv_transport.applied
    assert kv_transport.finished

    coordinator.start_cleanup()
    done = coordinator.wait_for_phase(ElasticPPMigrationPhase.DONE, timeout=2)
    assert runner.cleaned
    assert done.prepare_seconds is not None
    assert done.fence_seconds is not None
    assert done.total_seconds is not None


def test_prepare_arms_one_future_fence_from_the_slowest_scheduler_rank():
    coordinator, transition, _, _ = _coordinator(
        _Group(peer_fence_step=17),
        fence_step_provider=lambda: 11,
        fence_lead_steps=8,
    )

    coordinator.begin(transition)
    ready = coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)

    assert ready.fence_microstep == 25


def test_prepare_peer_failure_aborts_without_changing_source_generation():
    group = _Group(fail_stage="prepare_allocations")
    coordinator, transition, runner, kv_transport = _coordinator(group)

    coordinator.begin(transition)
    failed = coordinator.wait_for_phase(ElasticPPMigrationPhase.FAILED, timeout=2)
    assert failed.source_usable
    assert coordinator.current_topology == transition.source
    assert runner.aborted_layers
    assert runner.aborted_kv
    assert kv_transport.finished


def test_rank_skewed_bulk_and_delta_snapshots_are_allowed():
    group = _Group(mismatch_stage="prepare_allocations")
    coordinator, transition, runner, _ = _coordinator(group)
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    group.mismatch_stage = "fence_slots"

    assert coordinator.finalize_at_fence(lambda topology: None) == transition.target
    assert coordinator.phase == ElasticPPMigrationPhase.COMMITTED
    assert runner.committed


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
