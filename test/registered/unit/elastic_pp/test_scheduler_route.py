from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from sglang.srt.elastic_pp.topology import PPTopology
from sglang.srt.managers.scheduler_pp_mixin import (
    PPBatchMetadata,
    SchedulerPPMixin,
)


class _FakePPGroup:
    def __init__(self, rank: int, size: int = 4):
        self.rank = rank
        self.size = size
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
    assert sent["tensor_dict"]["__pp_generation__"] == 5
    assert sent["tensor_dict"]["__msg_type__"] == "proxy"


def test_static_pp_send_path_is_unchanged_without_topology():
    scheduler = _FakeScheduler(rank=0)

    tensors = {"hidden_states": "payload"}
    scheduler._pp_send_dict_to_next_stage(tensors, msg_type="proxy")

    sent = scheduler.pp_group.sent[0]
    assert sent["dst"] is None
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

    scheduler.pp_group.received.append(
        {
            "__msg_type__": "proxy",
            "__pp_generation__": 4,
            "hidden_states": "stale",
        }
    )
    with pytest.raises(RuntimeError, match="generation mismatch"):
        scheduler._pp_recv_proxy_tensors()


def test_inactive_rank_neither_receives_proxy_nor_launches_forward():
    scheduler = _FakeScheduler(rank=1)
    scheduler._pp_install_topology(_pp2_topology())

    assert scheduler._pp_recv_proxy_tensors() is None
    metadata = [None]
    result, event = scheduler._pp_launch_or_skip_batch(
        0,
        object(),
        None,
        metadata,
        deque(),
    )

    assert result is None
    assert metadata == [PPBatchMetadata(can_run_cuda_graph=False)]
    assert event.recorded_stream == "schedule-stream"


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
