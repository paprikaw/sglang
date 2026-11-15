from types import SimpleNamespace as NS
import ast
from pathlib import Path
import threading

import pytest

from sglang.srt.elastic_pp.pd_registered_regions import PreparedDecodeRegions
from sglang.srt.elastic_pp.pd_registration import DecodeRegistrationTransaction


SOURCE = {(100, 10), (200, 10), (300, 10)}
TARGET_KV = {(100, 10), (400, 10)}
TARGET = TARGET_KV | {(500, 10)}


class Engine:
    def __init__(self):
        self.live = set(SOURCE)
        self.calls = []
        self.register_status = self.deregister_status = 0

    def batch_register(self, pointers, lengths):
        rows = set(zip(pointers, lengths))
        self.calls.append(("register", rows))
        assert not (rows & self.live), "duplicate registration"
        self.live.update(rows)
        return self.register_status

    def batch_deregister(self, pointers):
        self.calls.append(("deregister", set(pointers)))
        self.live = {r for r in self.live if r[0] not in pointers}
        return self.deregister_status


def test_preparation_does_not_touch_source_and_commit_registers_only_missing_aux():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    lease.prepare()
    assert engine.live == SOURCE | TARGET_KV
    assert engine.calls == [("register", {(400, 10)})]
    lease.suspend()
    assert engine.live == TARGET_KV
    lease.commit(TARGET)
    assert engine.live == TARGET
    assert engine.calls == [
        ("register", {(400, 10)}),
        ("deregister", {200, 300}),
        ("register", {(500, 10)}),
    ]
    assert lease.metrics["prepare_regions"] == 1
    assert lease.metrics["refresh_regions"] == 1


def test_prepare_abort_leaves_original_registrations_untouched():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    lease.prepare()
    lease.abort()
    lease.abort()
    assert engine.live == SOURCE
    assert engine.calls[-1] == ("deregister", {400})


def test_fenced_abort_restores_source_only_after_prepared_backing_abort():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    lease.prepare()
    lease.suspend()
    lease.abort()
    assert engine.live == {(100, 10)}
    lease.restore_source()
    assert engine.live == SOURCE


def test_target_drift_is_rejected_before_any_new_registration():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    lease.prepare()
    lease.suspend()
    calls = len(engine.calls)
    with pytest.raises(RuntimeError, match="differ from prepared"):
        lease.commit({(100, 10), (600, 10)})
    assert len(engine.calls) == calls


@pytest.mark.parametrize("target", [{(100, 11)}, {(105, 5)}, {(400, 10), (405, 10)}])
def test_different_overlapping_regions_are_rejected_before_prepare(target):
    engine = Engine()
    with pytest.raises(ValueError, match="overlap"):
        PreparedDecodeRegions(engine, SOURCE, target)
    assert not engine.calls


def test_retained_only_prepare_does_not_reregister_existing_pages():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, {(100, 10)})
    lease.prepare()
    assert not engine.calls
    lease.suspend()
    lease.commit({(100, 10)})
    assert engine.calls == [("deregister", {200, 300})]


def test_failed_prepare_records_attempts_for_cleanup_before_backing_unmap():
    engine = Engine()
    engine.register_status = -1
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    with pytest.raises(RuntimeError, match="registration failed"):
        lease.prepare()
    lease.abort()
    assert engine.live == SOURCE


def test_unknown_partial_source_deregistration_cannot_claim_safe_rollback():
    engine = Engine()
    lease = PreparedDecodeRegions(engine, SOURCE, TARGET_KV)
    lease.prepare()
    engine.deregister_status = -1
    with pytest.raises(RuntimeError, match="deregistration failed"):
        lease.suspend()
    with pytest.raises(RuntimeError, match="backing must stay mapped"):
        lease.abort()
    assert lease.phase == "suspending"


def scheduler():
    engine = Engine()
    manager = NS(
        engine=engine,
        kv_args="source",
        connection_pool={"old_peer": 1},
        connection_lock=threading.Lock(),
    )
    manager._registerable_regions = lambda: (
        SOURCE if manager.kv_args == "source" else TARGET
    )
    pool = NS(
        capacity_managed=True,
        serving_capacity=8,
        prepared_pd_registration_regions=lambda generation: TARGET_KV,
    )
    runner = NS(
        token_to_kv_pool=pool,
        server_args=NS(elastic_pp_transition_strategy="background"),
    )
    queue = NS(
        kv_manager=manager,
        queue=[],
        pending_reqs=[],
        retracted_queue=[],
        _build_kv_args=lambda: "target",
    )
    sched = NS(
        tp_worker=NS(model_runner=runner),
        disagg_decode_prealloc_queue=queue,
        disagg_decode_transfer_queue=NS(
            queue=[], has_pending_deferred_releases=lambda: False
        ),
    )
    return sched, engine, manager, queue


def test_live_pd_writer_allows_prepare_but_blocks_deregistration_and_publication():
    sched, engine, manager, queue = scheduler()
    sched.disagg_decode_transfer_queue.queue.append(object())
    transaction = DecodeRegistrationTransaction(sched)
    transaction.prepare_target(1)
    assert manager.kv_args == "source" and manager.connection_pool
    assert SOURCE <= engine.live
    with pytest.raises(RuntimeError, match="transfers must finish"):
        transaction.suspend()
    sched.disagg_decode_transfer_queue.queue.clear()
    transaction.suspend()
    assert not manager.connection_pool
    transaction.refresh()
    assert manager.kv_args == "target"
    assert engine.live == TARGET
    assert transaction.metrics()["phase"] == "committed"
    assert not transaction.suspended


def test_transaction_abort_before_fence_is_idempotent_and_checks_generation():
    sched, engine, _, _ = scheduler()
    transaction = DecodeRegistrationTransaction(sched)
    transaction.prepare_target(1)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        transaction.abort_prepared(2)
    transaction.abort_prepared(1)
    transaction.abort_prepared(1)
    assert engine.live == SOURCE


def test_transaction_abort_after_fence_restores_source_addresses():
    sched, engine, manager, queue = scheduler()
    transaction = DecodeRegistrationTransaction(sched)
    transaction.prepare_target(1)
    transaction.suspend()
    transaction.abort_prepared(1)
    queue._build_kv_args = lambda: "source"
    transaction.refresh()
    assert manager.kv_args == "source" and engine.live == SOURCE
    assert transaction.metrics()["phase"] == "restored"


def runner_method(name):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    module = ast.parse(path.read_text())
    cls = next(
        x
        for x in module.body
        if isinstance(x, ast.ClassDef) and x.name == "ModelRunner"
    )
    method = next(
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name
    )
    namespace = {}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_runner_registers_only_after_target_backing_preparation():
    events = []

    def prepare(**_kwargs):
        events.append("map")
        return frozenset({1})

    runner = NS(
        token_to_kv_pool=NS(prepare_elastic_pp_kv_layers=prepare),
        model_config=NS(num_hidden_layers=48),
        _elastic_pp_pd_registration=NS(
            prepare_target=lambda g: events.append(("register", g))
        ),
    )
    result = runner_method("prepare_elastic_pp_kv_layers")(
        runner, target=object(), generation=3
    )
    assert result == frozenset({1}) and events == ["map", ("register", 3)]


def test_failed_registration_abort_never_unmaps_prepared_backing():
    events = []

    def failed(_generation):
        raise RuntimeError("registration cleanup failed")

    runner = NS(
        token_to_kv_pool=NS(
            abort_elastic_pp_kv_layers=lambda **_kw: events.append("unmap")
        ),
        _elastic_pp_pd_registration=NS(abort_prepared=failed),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner_method("abort_elastic_pp_kv_layers")(runner, generation=3)
    assert not events
