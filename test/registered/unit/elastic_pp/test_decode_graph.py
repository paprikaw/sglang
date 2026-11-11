"""Graph generation safety and drained recapture without importing GPU kernels."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.elastic_pp.decode_graph import (
    ElasticPPDecodeGraphState,
    rebuild_decode_graph,
    retire_decode_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def runner(*, disabled=False, fail=False):
    state = ElasticPPDecodeGraphState()
    state.begin(0).update(status="ready", capture_seconds=0.1)
    state.observe(True)
    state.invalidate()
    mapping = torch.tensor([[0, 0], [17, 18]], dtype=torch.int32)
    mamba = torch.tensor([0, 3], dtype=torch.int64)
    result = NS(
        server_args=NS(disable_cuda_graph=disabled),
        _elastic_pp_decode_graph_state=state,
        req_to_token_pool=NS(
            req_to_token=mapping, get_mamba_indices=lambda ids: mamba[ids.long()]
        ),
        decode_cuda_graph_runner=object(),
        prefill_cuda_graph_runner=object(),
        captures=0,
    )

    def capture():
        assert result.decode_cuda_graph_runner is None
        assert result.prefill_cuda_graph_runner is None
        result.captures += 1
        if fail:
            raise RuntimeError("capture failed")
        result.decode_cuda_graph_runner = object()

    result.init_decode_cuda_graph = capture
    return result


def test_recapture_preserves_request_map_and_rejects_retired_replay():
    r = runner()
    state = r._elastic_pp_decode_graph_state
    with pytest.raises(RuntimeError, match="retired/unready"):
        state.observe(True)
    before = r.req_to_token_pool.req_to_token.clone()
    rebuild_decode_graph(r, generation=1, active=True)
    state.observe(True)
    state.observe(False)
    assert torch.equal(before, r.req_to_token_pool.req_to_token)
    assert r.captures == 1
    assert state.snapshot()[0]["replays"] == 1
    assert state.snapshot()[1]["status"] == "ready"
    assert state.snapshot()[1]["replays"] == 1
    assert state.snapshot()[1]["eager_decode"] == 1
    copy = state.snapshot()
    copy[0]["replays"] = -1
    assert state.snapshot()[0]["replays"] == 1


@pytest.mark.parametrize(
    "disabled,active,status", [(True, True, "disabled"), (False, False, "inactive")]
)
def test_disabled_or_zero_layer_rank_never_captures(disabled, active, status):
    r = runner(disabled=disabled)
    rebuild_decode_graph(r, generation=1, active=active)
    assert r.captures == 0
    assert r.decode_cuda_graph_runner is None
    assert r._elastic_pp_decode_graph_state.snapshot()[-1]["status"] == status


@pytest.mark.parametrize(
    "fault", ["capture", "missing_runner", "request_zero", "gdn_zero"]
)
def test_capture_failure_stays_failed_without_eager_success(fault):
    r = runner(fail=fault == "capture")
    if fault == "missing_runner":
        r.init_decode_cuda_graph = lambda: None
    if fault == "request_zero":
        r.req_to_token_pool.req_to_token[0, 0] = 1
    if fault == "gdn_zero":
        r.req_to_token_pool.get_mamba_indices = lambda ids: torch.ones_like(ids)
    with pytest.raises(RuntimeError):
        rebuild_decode_graph(r, generation=1, active=True)
    assert r.decode_cuda_graph_runner is None
    assert r._elastic_pp_decode_graph_state.snapshot()[-1]["status"] == "failed"


def test_same_partition_new_generation_is_recaptured_not_reused():
    r = runner()
    rebuild_decode_graph(r, generation=1, active=True)
    first = r.decode_cuda_graph_runner
    r._elastic_pp_decode_graph_state.invalidate()
    rebuild_decode_graph(r, generation=2, active=True)
    assert r.decode_cuda_graph_runner is not first
    assert r.captures == 2
    with pytest.raises(RuntimeError, match="generation must increase"):
        rebuild_decode_graph(r, generation=2, active=True)


@pytest.mark.parametrize("has_graph", [True, False])
def test_retirement_releases_backend_and_forgets_nonowning_pool_ids(
    monkeypatch, has_graph
):
    events = []

    @contextmanager
    def capture_scope():
        events.append("drain_borrow_arena")
        yield
        events.append("leave_scope")

    def reset_pool(value):
        assert value is None
        events.append("reset_pool")

    def reset_allocator(value):
        assert value is None
        events.append("reset_allocator")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner_utils.pool",
        NS(
            graph_pool_capture_scope=capture_scope,
            set_global_graph_memory_pool=reset_pool,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        NS(set_graph_pool_id=reset_allocator),
    )
    state = ElasticPPDecodeGraphState()
    state.begin(0).update(status="ready")
    r = NS(
        decode_cuda_graph_runner=(
            NS(backend=NS(cleanup=lambda: events.append("release_graphs")))
            if has_graph
            else None
        ),
        prefill_cuda_graph_runner=None,
        _elastic_pp_decode_graph_state=state,
    )
    retire_decode_graph(r)
    assert events == ["drain_borrow_arena"] + (
        ["release_graphs"] if has_graph else []
    ) + ["reset_pool", "reset_allocator", "leave_scope"]
    assert r.decode_cuda_graph_runner is None
    assert r.prefill_cuda_graph_runner is None
    assert state.snapshot()[-1]["status"] == "retired"


@pytest.mark.parametrize("rank,expected", [(0, True), (1, False), (2, True)])
def test_startup_skips_empty_rank_even_if_capture_helper_returns_eager(rank, expected):
    from test_runtime_backend_views import production_methods

    method = production_methods(
        "model_executor/model_runner.py", "ModelRunner", {"init_cuda_graphs"}
    )["init_cuda_graphs"]
    seen = []
    graph = object()

    def capture(**kwargs):
        seen.append(kwargs["capture_decode_cuda_graph"])
        return NS(
            eager_runner=object(),
            prefill=NS(runner=None),
            decode=NS(runner=graph, capture_time=0.25 if expected else 0),
            memory_usage={},
            time_usage={},
        )

    method.__globals__["capture_cuda_graphs"] = capture
    r = NS(
        server_args=NS(
            enable_elastic_pp=True,
            elastic_pp_initial_partition="2,0,2",
            disable_cuda_graph=False,
        ),
        ps=NS(pp_rank=rank),
    )
    method(r)
    assert seen == [expected]
    assert (r.decode_cuda_graph_runner is graph) == expected
    assert r._elastic_pp_decode_graph_state.snapshot()[0]["status"] == (
        "ready" if expected else "inactive"
    )


@pytest.mark.parametrize("fault", [None, "local", "peer"])
def test_coordinator_waits_for_graph_after_route_commit(fault):
    from test_coordinator import ElasticPPMigrationPhase, _coordinator, _Group

    group = _Group(fail_stage="decode_graph" if fault == "peer" else None)
    coordinator, transition, r, _ = _coordinator(group)
    events = []

    def capture(**kwargs):
        assert r.committed
        assert events == ["route"]
        assert group.stages[-1] == "commit"
        assert kwargs == {"generation": 1, "active": True}
        events.append("graph")
        if fault == "local":
            raise RuntimeError("capture failed")

    r.rebuild_elastic_pp_decode_graph = capture
    coordinator.begin(transition)
    coordinator.wait_for_phase(ElasticPPMigrationPhase.BULK_READY, timeout=2)
    if fault:
        with pytest.raises(RuntimeError, match="Decode Graph rebuild"):
            coordinator.finalize_at_fence(lambda _: events.append("route"))
        assert not coordinator.snapshot().source_usable
    else:
        coordinator.finalize_at_fence(lambda _: events.append("route"))
        assert events == ["route", "graph"]
