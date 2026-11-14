from collections import deque
from types import SimpleNamespace

import pytest

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("message", [None, [], ["request-id"]])
def test_pd_queue_fence_roundtrips_distinct_from_empty_consensus(monkeypatch, message):
    packets = deque()

    def send(tensor, dst, group=None):
        packets.append(tensor.clone())
        return SimpleNamespace(wait=lambda: None)

    def receive(tensor, src=None, group=None):
        packet = packets.popleft()
        assert packet.dtype == tensor.dtype and packet.shape == tensor.shape
        tensor.copy_(packet)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(common.dist, "isend", send)
    monkeypatch.setattr(common.dist, "irecv", receive)
    work = common.point_to_point_pyobj(message, rank=0, src=0, dst=1, async_send=True)
    # None must carry serialized bytes; an empty consensus retains its zero-byte
    # fast path. The real wire function, not a scheduler mock, handles both.
    assert len(work) == (1 if message == [] else 2)
    received = common.point_to_point_pyobj([], rank=1, src=0, dst=1)
    assert received == message
    if message is None:
        assert received is None
    for entry in work:
        entry.work.wait()
    assert not packets
