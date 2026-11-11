from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.elastic_pp.kv_patch import (
    ContinuousKVPatch,
    KVPatchBusy,
    TensorKVPatchCopier,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Ready:
    def synchronize(self):
        pass


class Pending:
    def __init__(self):
        self.ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class Copier:
    def __init__(self):
        self.batches = []
        self.synchronize = Mock()

    def submit(self, rows):
        event = Pending()
        self.batches.append((rows, event))
        return event


def publish(session, slot):
    epoch = session.allocate(slot, generation=1)
    session.publish(slot, epoch, ready=Ready(), generation=1)
    return epoch


def test_source_can_append_during_copy_and_final_fence_drains_real_deltas():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier, max_rows=1)
    publish(session, 1)
    assert session.submit(generation=1)
    publish(session, 2)
    assert not session.poll(generation=1)
    assert session.finish_at_fence(generation=1, max_rounds=1) == dict(
        rounds=2, rows_copied=2
    )
    assert [[row.slot for row in rows] for rows, _ in copier.batches] == [[1], [2]]
    assert session.dirty_rows() == ()


def test_free_is_blocked_until_copy_completes_and_reuse_requires_another_copy():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier)
    first = publish(session, 1)
    session.submit(generation=1)
    with pytest.raises(KVPatchBusy, match="defer"):
        session.free(1, first, generation=1)
    copier.batches[0][1].ready = True
    session.poll(generation=1)
    session.free(1, first, generation=1)
    second = publish(session, 1)
    assert second > first
    session.finish_at_fence(generation=1, max_rounds=1)
    assert [rows[0].incarnation for rows, _ in copier.batches] == [first, second]


@pytest.mark.parametrize(
    "operation", ["allocate", "publish", "free", "submit", "poll", "finish", "abort"]
)
def test_wrong_generation_has_no_copy_side_effect(operation):
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier)
    epoch = publish(session, 1)
    calls = {
        "allocate": lambda: session.allocate(2, generation=2),
        "publish": lambda: session.publish(1, epoch, ready=Ready(), generation=2),
        "free": lambda: session.free(1, epoch, generation=2),
        "submit": lambda: session.submit(generation=2),
        "poll": lambda: session.poll(generation=2),
        "finish": lambda: session.finish_at_fence(generation=2),
        "abort": lambda: session.abort(generation=2),
    }
    with pytest.raises(ValueError, match="generation mismatch"):
        calls[operation]()
    assert copier.batches == []
    assert len(session.dirty_rows()) == 1


def test_unpublished_write_cannot_disappear_at_fence():
    session = ContinuousKVPatch(generation=1, capacity=8, copier=Copier())
    session.allocate(1, generation=1)
    with pytest.raises(RuntimeError, match="unpublished"):
        session.finish_at_fence(generation=1)
    with pytest.raises(RuntimeError, match="fenced"):
        session.allocate(2, generation=1)
    session.abort(generation=1)


def test_stale_incarnation_and_mutable_row_writes_are_rejected():
    session = ContinuousKVPatch(generation=1, capacity=8, copier=Copier())
    epoch = publish(session, 1)
    with pytest.raises(ValueError, match="immutable"):
        session.publish(1, epoch, ready=Ready(), generation=1)
    session.free(1, epoch, generation=1)
    publish(session, 1)
    with pytest.raises(ValueError, match="stale"):
        session.free(1, epoch, generation=1)


def test_round_bound_can_abort_without_modifying_the_source():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier, max_rows=1)
    publish(session, 1)
    publish(session, 2)
    with pytest.raises(RuntimeError, match="converge"):
        session.finish_at_fence(generation=1, max_rounds=1)
    session.abort(generation=1)
    copier.synchronize.assert_called_once()


def test_copy_failure_retains_logical_pins_until_abort_drains_the_copier():
    copier = Copier()
    copier.submit = Mock(side_effect=RuntimeError("copy failed"))
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier)
    publish(session, 1)
    with pytest.raises(RuntimeError, match="copy failed"):
        session.submit(generation=1)
    assert session._pinned == {1}
    session.abort(generation=1)
    assert not session._pinned
    copier.synchronize.assert_called_once()


def test_tensor_copier_handles_packed_row_strides_and_releases_page_lease():
    source = {0: tuple(torch.arange(32.0).view(8, 4)[:, ::2] for _ in range(2))}
    target = {0: tuple(torch.zeros(8, 4)[:, ::2] for _ in range(2))}
    held = []

    @contextmanager
    def lease(rows):
        held.append("pin")
        try:
            yield
        finally:
            held.append("unpin")

    copier = TensorKVPatchCopier(source, target, page_lease=lease)
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier)
    publish(session, 1)
    publish(session, 4)
    session.finish_at_fence(generation=1, max_rounds=1)
    assert held == ["pin", "unpin"]
    for src, dst in zip(source[0], target[0]):
        assert torch.equal(src[[1, 4]], dst[[1, 4]])
        assert torch.count_nonzero(dst[[0, 2, 3, 5, 6, 7]]) == 0
