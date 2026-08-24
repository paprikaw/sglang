from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.distributed.parallel_state import GroupCoordinator, TensorMetadata
from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationPhase
from sglang.srt.elastic_pp.topology import DynamicPPTransition, PPTopology
from sglang.srt.managers.io_struct import (
    GetElasticPPStateReqInput,
    ReconfigureElasticPPReqInput,
)
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    SchedulerPPMixin,
)


class _FakePPGroup:
    def __init__(self, rank: int, size: int = 4):
        self.rank = rank
        self.size = size
        self.pynccl_comm = SimpleNamespace(available=True)
        self.sent = []
        self.received = deque()

    @property
    def is_first_rank(self):
        return self.rank == 0

    @property
    def is_last_rank(self):
        return self.rank == self.size - 1

    def send_tensor_dict(self, **kwargs):
        self.sent.append(kwargs)
        return []

    def recv_tensor_dict(self, **kwargs):
        self.last_recv_kwargs = kwargs
        return self.received.popleft()


class _FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


class _FakeDeviceModule:
    @staticmethod
    def Event():
        return _FakeEvent()

    @staticmethod
    def current_stream():
        return "schedule-stream"


class _FakeScheduler(SchedulerPPMixin):
    def __init__(self, rank: int, size: int = 4):
        self.ps = SimpleNamespace(pp_rank=rank, pp_size=size)
        self.pp_group = _FakePPGroup(rank, size)
        self.require_attn_tp_allgather = False
        self._pp_tensor_dict_inbox = defaultdict(deque)
        self.device_module = _FakeDeviceModule()
        self.forward_ct = 0
        self._sched_idled = True

    def _record_batch_launch(self, batch):
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = 1.0
        batch.after_idle_gap = self._sched_idled
        self._sched_idled = False


class _FakeSender:
    def __init__(self):
        self.outputs = []

    def send_output(self, output, recv_req):
        self.outputs.append((output, recv_req))


class _FakeMigrationCoordinator:
    def __init__(
        self,
        transition,
        *,
        fence_microstep=10,
        phase=ElasticPPMigrationPhase.BULK_READY,
    ):
        self.transition = transition
        self.current_topology = transition.source
        self.phase = phase
        self.fence_microstep = fence_microstep
        self.cleanup_event = None
        self.begun = False

    def snapshot(self):
        return SimpleNamespace(
            phase=self.phase,
            generation=self.transition.target.generation,
            error=None,
            source_usable=True,
            fence_microstep=self.fence_microstep,
            prepare_seconds=1.0,
            fence_seconds=(
                0.1 if self.phase != ElasticPPMigrationPhase.BULK_READY else None
            ),
            total_seconds=(
                1.1 if self.phase != ElasticPPMigrationPhase.BULK_READY else None
            ),
        )

    def finalize_at_fence(self, commit_topology):
        self.phase = ElasticPPMigrationPhase.FENCING
        commit_topology(self.transition.target)
        self.current_topology = self.transition.target
        self.phase = ElasticPPMigrationPhase.COMMITTED
        return self.transition.target

    def begin(self, transition):
        self.transition = transition
        self.begun = True
        self.phase = ElasticPPMigrationPhase.PREPARING

    def start_cleanup(self, completion_event=None):
        self.cleanup_event = completion_event
        self.phase = ElasticPPMigrationPhase.CLEANING


def _pp2_topology(generation: int = 0):
    return PPTopology.from_layer_counts([4, 0, 0, 4], generation=generation)


def test_proxy_send_bypasses_inactive_middle_ranks_and_carries_generation():
    scheduler = _FakeScheduler(rank=0)
    scheduler._pp_install_topology(_pp2_topology(generation=5))

    tensors = {"hidden_states": "payload"}
    scheduler._pp_send_dict_to_next_stage(tensors, msg_type="proxy")

    assert len(scheduler.pp_group.sent) == 1
    sent = scheduler.pp_group.sent[0]
    assert sent["dst"] == 3
    assert sent["use_pynccl"]
    assert sent["tensor_dict"]["__pp_generation__"] == 5
    assert sent["tensor_dict"]["__msg_type__"] == "proxy"


def test_static_pp_send_path_is_unchanged_without_topology():
    scheduler = _FakeScheduler(rank=0)

    tensors = {"hidden_states": "payload"}
    scheduler._pp_send_dict_to_next_stage(tensors, msg_type="proxy")

    sent = scheduler.pp_group.sent[0]
    assert sent["dst"] is None
    assert not sent["use_pynccl"]
    assert "__pp_generation__" not in sent["tensor_dict"]


def test_proxy_receive_uses_previous_active_rank_and_rejects_stale_generation():
    scheduler = _FakeScheduler(rank=3)
    scheduler._pp_install_topology(_pp2_topology(generation=5))
    scheduler.pp_group.received.append(
        {
            "__msg_type__": "proxy",
            "__pp_generation__": 5,
            "hidden_states": "payload",
        }
    )

    proxy = scheduler._pp_recv_proxy_tensors()

    assert proxy["hidden_states"] == "payload"
    assert scheduler.pp_group.last_recv_kwargs["src"] == 0
    assert scheduler.pp_group.last_recv_kwargs["pynccl_message_types"] == ("proxy",)

    scheduler.pp_group.received.append(
        {
            "__msg_type__": "proxy",
            "__pp_generation__": 4,
            "hidden_states": "stale",
        }
    )
    with pytest.raises(RuntimeError, match="generation mismatch"):
        scheduler._pp_recv_proxy_tensors()


def test_elastic_output_ring_stays_on_process_group_nccl():
    scheduler = _FakeScheduler(rank=3)
    scheduler._pp_install_topology(_pp2_topology(generation=5))

    scheduler._pp_send_dict_to_next_stage(
        {"next_token_ids": "payload"}, msg_type="output"
    )

    sent = scheduler.pp_group.sent[0]
    assert sent["dst"] is None
    assert not sent["use_pynccl"]


def test_elastic_proxy_route_rejects_missing_pynccl():
    scheduler = _FakeScheduler(rank=0)
    scheduler._pp_install_topology(_pp2_topology())
    scheduler.pp_group.pynccl_comm = None

    with pytest.raises(RuntimeError, match="requires the full-group PyNCCL"):
        scheduler._pp_send_dict_to_next_stage(
            {"hidden_states": "payload"}, msg_type="proxy"
        )


@pytest.mark.parametrize(
    ("message_type", "expect_pynccl"),
    [("proxy", True), ("output", False)],
)
def test_tensor_dict_receive_selects_transport_from_actual_metadata(
    monkeypatch, message_type, expect_pynccl
):
    group = object.__new__(GroupCoordinator)
    group.world_size = 2
    group.rank_in_group = 1
    group.ranks = [0, 1]
    group.device_group = object()
    group.cpu_group = object()
    group.unique_name = "test-pp"
    group.recv_object = lambda src: [
        ("__msg_type__", message_type),
        ("payload", TensorMetadata("meta", torch.float32, (2,))),
    ]

    pynccl_recvs = []
    group.pynccl_comm = SimpleNamespace(
        available=True,
        change_state=lambda enable: nullcontext(),
        recv=lambda tensor, src: pynccl_recvs.append((tensor, src)),
    )
    process_group_recvs = []

    class _Work:
        @staticmethod
        def wait():
            return None

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "irecv",
        lambda tensor, src, group: (
            process_group_recvs.append((tensor, src, group)) or _Work()
        ),
    )

    received = group.recv_tensor_dict(
        src=0,
        pynccl_message_types=("proxy",),
    )

    assert received["__msg_type__"] == message_type
    assert tuple(received["payload"].shape) == (2,)
    assert bool(pynccl_recvs) is expect_pynccl
    assert bool(process_group_recvs) is not expect_pynccl


def test_inactive_rank_neither_receives_proxy_nor_launches_forward():
    scheduler = _FakeScheduler(rank=1)
    scheduler._pp_install_topology(_pp2_topology())

    assert scheduler._pp_recv_proxy_tensors() is None
    metadata = [None]
    batch = SimpleNamespace()
    result, event = scheduler._pp_launch_or_skip_batch(
        0,
        batch,
        None,
        metadata,
        deque(),
    )

    assert result is None
    assert metadata == [PPBatchMetadata(can_run_cuda_graph=False)]
    assert event.recorded_stream == "schedule-stream"
    assert batch.forward_iter == 1
    assert batch.launch_ts == 1.0
    assert batch.after_idle_gap
    assert scheduler.forward_ct == 1
    assert not scheduler._sched_idled


def test_topology_install_requires_static_endpoints_new_generation_and_fence():
    scheduler = _FakeScheduler(rank=0)

    with pytest.raises(NotImplementedError, match="physical first/last"):
        scheduler._pp_install_topology(
            PPTopology.from_layer_counts([4, 4, 0, 0], generation=0)
        )

    scheduler._pp_install_topology(_pp2_topology(generation=1))
    with pytest.raises(ValueError, match="generation must increase"):
        scheduler._pp_install_topology(_pp2_topology(generation=1))

    scheduler.mbs = [object(), None]
    with pytest.raises(RuntimeError, match="scheduler fence"):
        scheduler._pp_install_topology(
            PPTopology.from_layer_counts([2, 2, 2, 2], generation=2)
        )


def test_common_future_microstep_arms_drain_on_the_exact_boundary():
    scheduler = _FakeScheduler(rank=0)
    source = _pp2_topology(generation=0)
    target = PPTopology.from_layer_counts([2, 2, 0, 4], generation=1)
    coordinator = _FakeMigrationCoordinator(
        DynamicPPTransition.build(source, target), fence_microstep=10
    )
    scheduler.elastic_pp_migration_coordinator = coordinator
    scheduler._elastic_pp_pending_req = None
    scheduler._elastic_pp_draining = False

    scheduler._pp_microstep = 9
    assert not scheduler._elastic_pp_drain_due()
    scheduler._pp_microstep = 10
    assert scheduler._elastic_pp_drain_due()


def test_drained_fence_commits_route_replies_and_defers_cleanup():
    scheduler = _FakeScheduler(rank=0)
    source = _pp2_topology(generation=0)
    target = PPTopology.from_layer_counts([2, 2, 0, 4], generation=1)
    transition = DynamicPPTransition.build(source, target)
    coordinator = _FakeMigrationCoordinator(transition)
    scheduler._pp_install_topology(source)
    scheduler.elastic_pp_migration_coordinator = coordinator
    scheduler._elastic_pp_draining = True
    scheduler._elastic_pp_pending_req = SimpleNamespace(http_worker_ipc="control")
    scheduler.rust_server = None
    sender = _FakeSender()
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=sender)
    scheduler.mbs = [None, None, None, None]
    scheduler.last_rank_comm_queue = deque()
    scheduler.pp_outputs = None
    scheduler.send_req_work = []
    scheduler.send_proxy_work = []
    scheduler.send_output_work = []

    scheduler._elastic_pp_finalize_if_drained()

    assert scheduler._pp_topology() == target
    assert not scheduler._elastic_pp_draining
    assert scheduler._elastic_pp_pending_req is None
    assert len(sender.outputs) == 1
    assert sender.outputs[0][0].success
    assert sender.outputs[0][0].migration_phase == "committed"
    assert coordinator.phase == ElasticPPMigrationPhase.CLEANING
    assert coordinator.cleanup_event.recorded_stream == "schedule-stream"


def test_reconfigure_request_starts_prepare_without_blocking_scheduler_ring():
    scheduler = _FakeScheduler(rank=0)
    source = _pp2_topology(generation=0)
    seed_target = PPTopology.from_layer_counts([3, 1, 0, 4], generation=1)
    coordinator = _FakeMigrationCoordinator(
        DynamicPPTransition.build(source, seed_target),
        phase=ElasticPPMigrationPhase.IDLE,
    )
    scheduler._pp_install_topology(source)
    scheduler.elastic_pp_migration_coordinator = coordinator
    scheduler._elastic_pp_pending_req = None
    request = ReconfigureElasticPPReqInput(layer_counts=[2, 2, 0, 4])

    response = scheduler.handle_reconfigure_elastic_pp(request)

    assert response is None
    assert coordinator.begun
    assert coordinator.transition.target.generation == 1
    assert scheduler._elastic_pp_pending_req is request


def test_reconfigure_rejects_invalid_partition_before_prepare():
    scheduler = _FakeScheduler(rank=0)
    source = _pp2_topology(generation=0)
    target = PPTopology.from_layer_counts([3, 1, 0, 4], generation=1)
    coordinator = _FakeMigrationCoordinator(
        DynamicPPTransition.build(source, target),
        phase=ElasticPPMigrationPhase.IDLE,
    )
    scheduler._pp_install_topology(source)
    scheduler.elastic_pp_migration_coordinator = coordinator
    scheduler._elastic_pp_pending_req = None

    response = scheduler.handle_reconfigure_elastic_pp(
        ReconfigureElasticPPReqInput(layer_counts=[2, 2, 0, 3])
    )

    assert not response.success
    assert "cover every model layer" in response.message
    assert not coordinator.begun


def test_autoscale_state_reports_topology_and_backlog():
    scheduler = _FakeScheduler(rank=0)
    topology = _pp2_topology(generation=3)
    target = PPTopology.from_layer_counts([2, 2, 0, 4], generation=4)
    scheduler._pp_install_topology(topology)
    scheduler.elastic_pp_migration_coordinator = _FakeMigrationCoordinator(
        DynamicPPTransition.build(topology, target),
        phase=ElasticPPMigrationPhase.IDLE,
    )
    scheduler.waiting_queue = [object(), object(), object()]
    scheduler.running_batch = SimpleNamespace(reqs=[object(), object()])

    state = scheduler.handle_get_elastic_pp_state(GetElasticPPStateReqInput())

    assert state.enabled
    assert state.generation == 3
    assert state.layer_counts == [4, 0, 0, 4]
    assert state.migration_phase == "idle"
    assert state.source_usable
    assert state.waiting_requests == 3
    assert state.running_requests == 2
