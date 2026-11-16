from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.elastic_pp.kv_patch import (
    ContinuousKVPatch,
    KVPatchBusy,
    KVRowStamp,
    TensorKVPatchCopier,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Ready:
    def synchronize(self):
        pass


@pytest.mark.parametrize("field", ["slot", "incarnation", "ready"])
def test_stamp_keeps_full_version_and_event_immutable(field):
    event = Ready()
    row = KVRowStamp(7, (1 << 63) + 5, event)
    with pytest.raises(AttributeError):
        setattr(row, field, None)
    assert row.slot == 7 and row.incarnation == (1 << 63) + 5
    assert row.ready is event


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


def test_bulk_snapshot_matches_scalar_stamps_and_preserves_copy_pins():
    scalar = ContinuousKVPatch(generation=1, capacity=16, copier=Copier(), max_rows=2)
    batch = ContinuousKVPatch(generation=1, capacity=16, copier=Copier(), max_rows=2)
    slots, event = [8, 2, 11], Ready()
    expected = {slot: scalar.allocate(slot, generation=1) for slot in slots}
    assert batch.allocate_many(iter(slots), generation=1) == expected
    for slot, epoch in expected.items():
        scalar.publish(slot, epoch, ready=event, generation=1)
    batch.publish_many(iter(expected.items()), ready=event, generation=1)
    assert batch.dirty_rows() == scalar.dirty_rows()
    assert batch.unacknowledged_rows == 3
    assert batch.submit(generation=1)
    with pytest.raises(KVPatchBusy):
        batch.free(8, expected[8], generation=1)
    batch.poll(generation=1, wait=True)
    assert batch.free(8, expected[8], generation=1)
    assert batch.allocate_many([8], generation=1) == {8: 2}
    with pytest.raises(ValueError, match="stale"):
        batch.publish_many([(8, 1)], ready=event, generation=1)
    batch.publish_many([(8, 2)], ready=event, generation=1)
    batch.finish_at_fence(generation=1)
    assert batch.unacknowledged_rows == 0


@pytest.mark.parametrize("slots", [[2, 2], [2, 8], [2, -1], [2, 1]])
def test_bulk_allocation_rejects_entire_invalid_batch_without_mutation(slots):
    session = ContinuousKVPatch(generation=1, capacity=8, copier=Copier())
    session.allocate(1, generation=1)
    with pytest.raises(ValueError):
        session.allocate_many(slots, generation=1)
    assert set(session._live) == {1}
    assert session.allocate(2, generation=1) == 1


@pytest.mark.parametrize(
    "invalid", ["duplicate", "stale", "missing", "not_ready", "republish"]
)
def test_bulk_publish_checks_every_row_before_publishing_any(invalid):
    session = ContinuousKVPatch(generation=1, capacity=8, copier=Copier())
    session.allocate_many([1, 2, 3], generation=1)
    event = Ready()
    rows = [(1, 1), (2, 1)]
    if invalid == "duplicate":
        rows.append((1, 1))
    elif invalid == "stale":
        rows.append((3, 99))
    elif invalid == "missing":
        rows.append((7, 1))
    elif invalid == "not_ready":
        event = None
    else:
        session.publish(3, 1, ready=event, generation=1)
        rows.append((3, 1))
    with pytest.raises(ValueError):
        session.publish_many(rows, ready=event, generation=1)
    assert session._live[1].ready is None and session._live[2].ready is None
    assert all(row.slot == 3 for row in session.dirty_rows())


def test_bulk_snapshot_preserves_high_incarnations_and_generation_fences():
    session = ContinuousKVPatch(generation=1, capacity=8, copier=Copier())
    session._epochs[3] = 1 << 63
    epochs = session.allocate_many([3], generation=1)
    assert epochs == {3: (1 << 63) + 1}
    with pytest.raises(ValueError, match="generation"):
        session.allocate_many([4], generation=2)
    with pytest.raises(ValueError, match="generation"):
        session.publish_many(epochs.items(), ready=Ready(), generation=2)
    session.publish_many(epochs.items(), ready=Ready(), generation=1)
    session.finish_at_fence(generation=1)
    with pytest.raises(RuntimeError, match="complete"):
        session.allocate_many([], generation=1)
    with pytest.raises(RuntimeError, match="complete"):
        session.publish_many([], ready=Ready(), generation=1)


def test_online_round_does_not_enumerate_full_live_set(monkeypatch):
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=10000, copier=copier, max_rows=2)
    for slot in range(1, 1000):
        session.allocate(slot, generation=1)
    for slot in (999, 7, 500):
        session.publish(slot, 1, ready=Ready(), generation=1)
    monkeypatch.setattr(session, "dirty_rows", lambda: pytest.fail("full-set scan"))
    assert session.submit(generation=1)
    assert [r.slot for r in copier.batches[-1][0]] == [999, 7]
    session.poll(generation=1, wait=True)
    assert session.submit(generation=1)
    assert [r.slot for r in copier.batches[-1][0]] == [500]
    session.poll(generation=1, wait=True)
    assert not session.submit(generation=1)
    assert session.unacknowledged_rows == 996
    session.abort(generation=1)


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


def test_unacknowledged_count_includes_unpublished_rows_and_reused_incarnations():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=8, copier=copier)
    first = publish(session, 1)
    second = session.allocate(2, generation=1)
    assert session.unacknowledged_rows == 2
    session.submit(generation=1)
    session.poll(generation=1, wait=True)
    assert session.unacknowledged_rows == 1
    session.free(1, first, generation=1)
    assert session.unacknowledged_rows == 1
    publish(session, 1)
    assert session.unacknowledged_rows == 2
    session.free(2, second, generation=1)
    assert session.unacknowledged_rows == 1
    session.finish_at_fence(generation=1)
    assert session.unacknowledged_rows == 0


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


def test_large_fence_scans_once_and_copies_every_incarnation_in_bounded_order():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=1025, copier=copier, max_rows=256)
    for slot in reversed(range(1025)):
        publish(session, slot)
    session.dirty_rows = Mock(wraps=session.dirty_rows)
    assert session.finish_at_fence(generation=1, max_rounds=5) == dict(rounds=5, rows_copied=1025)
    assert session.dirty_rows.call_count == 1
    assert [(row.slot, row.incarnation) for rows, _ in copier.batches for row in rows] == [(i, 1) for i in range(1025)]


def test_bounded_partial_finish_can_retry_without_duplicate_copy():
    copier = Copier()
    session = ContinuousKVPatch(generation=1, capacity=1025, copier=copier, max_rows=256)
    for slot in range(1025):
        publish(session, slot)
    with pytest.raises(RuntimeError, match="converge"):
        session.finish_at_fence(generation=1, max_rounds=4)
    assert session.phase == "fenced" and session.rows_copied == 1024
    assert session.finish_at_fence(generation=1, max_rounds=1) == dict(rounds=5, rows_copied=1025)
    assert [(row.slot, row.incarnation) for rows, _ in copier.batches for row in rows] == [(i, 1) for i in range(1025)]
