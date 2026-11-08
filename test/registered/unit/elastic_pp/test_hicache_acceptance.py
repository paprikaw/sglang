"""Reject stale, rank-incomplete and metadata-only GPU acceptance evidence."""

import ast
import copy
import functools
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def acceptance():
    path = (
        Path(__file__).resolve().parents[4]
        / "research/pp_elastic/hicache_transition_smoke.py"
    )
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "rank_states",
            "validate_transition_states",
            "validate_fenced_decode",
            "fresh_prefix_restores",
            "install_transition_hooks",
            "demote_cpu_prefixes",
            "correctness_prompts",
            "observe_request_launch",
            "validate_uninterrupted_decode",
        }
    ]
    namespace = {"time": time}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def states():
    return [
        {
            "transition_rank": rank,
            "transition_metrics": {
                "generation": 2,
                "phase": "done",
                "backing_policy": "overlap",
                "host": {"host_bytes_sent": 1024 if rank == 1 else 0},
            },
            "transition_fence_records": [
                {"generation": generation, "requests": [{"output_length": 16}]}
                for generation in (1, 2)
            ],
            "agentx_restores": [{"time": 11, "loaded_tokens": 512}],
        }
        for rank in range(3)
    ]


def test_complete_current_transition_and_fresh_restore_are_accepted(acceptance):
    evidence = states()
    acceptance["validate_transition_states"](evidence, 2, (0, 1, 2), "overlap")
    acceptance["validate_fenced_decode"](evidence, 1, (0, 2), 256)
    assert set(acceptance["fresh_prefix_restores"](evidence, 10, (0, 2), 512)) == {
        0,
        1,
        2,
    }


@pytest.mark.parametrize("fault", ["stale_generation", "duplicate_rank", "no_bytes"])
def test_old_or_incomplete_host_transfer_cannot_pass(acceptance, fault):
    evidence = states()
    if fault == "stale_generation":
        evidence[1]["transition_metrics"]["generation"] = 1
    elif fault == "duplicate_rank":
        evidence[1] = copy.deepcopy(evidence[0])
    else:
        evidence[1]["transition_metrics"]["host"]["host_bytes_sent"] = 0
    with pytest.raises(AssertionError):
        acceptance["validate_transition_states"](evidence, 2, (0, 1, 2), "overlap")


@pytest.mark.parametrize("length", [0, 256])
def test_completed_or_unstarted_request_cannot_prove_active_fence(acceptance, length):
    evidence = states()
    evidence[2]["transition_fence_records"][0]["requests"][0]["output_length"] = length
    with pytest.raises(AssertionError, match="actual generation fence"):
        acceptance["validate_fenced_decode"](evidence, 1, (0, 2), 256)


def test_old_positive_restore_on_one_active_rank_cannot_pass(acceptance):
    evidence = states()
    evidence[2]["agentx_restores"][0]["time"] = 9
    with pytest.raises(AssertionError, match="every active stage"):
        acceptance["fresh_prefix_restores"](evidence, 10, (0, 2), 512)


def test_logical_detach_without_measured_owner_release_cannot_pass(acceptance):
    evidence = states()
    for state in evidence:
        state["transition_metrics"].update(
            backing_policy="cpu_spill",
            source_backing={
                "release": {"detached_bytes": 1024},
                "cuda_release": {"allocated_drop_bytes": 0},
            },
        )
    with pytest.raises(AssertionError, match="CUDA owner release"):
        acceptance["validate_transition_states"](evidence, 2, (0, 1, 2), "cpu_spill")


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_installed_state_hook_uses_parallel_state_rank(acceptance, rank):
    live = NS(
        rid="active",
        output_ids=[1] * 16,
        finished=lambda: False,
        retraction_count=0,
        is_retracted=False,
    )
    done = NS(rid="finished", output_ids=[2] * 24, finished=lambda: True)
    calls = []

    class Scheduler:
        def __init__(self):
            # The actual scheduler keeps rank identity in ps, with no pp_rank alias.
            self.ps = NS(pp_rank=rank)

        def get_internal_state(self, *args, **kwargs):
            calls.append((args, kwargs))
            return NS(internal_state={"existing": True})

        def _elastic_pp_live_requests(self):
            return [live, done]

        def _record_batch_launch(self, batch):
            pass

    class Coordinator:
        def finalize_at_fence(self, publish):
            publish()
            return "committed"

        def transition_metrics(self):
            return {"generation": 1, "phase": "done"}

    acceptance["install_transition_hooks"](Scheduler, Coordinator)
    scheduler = Scheduler()
    coordinator = Coordinator()
    scheduler.elastic_pp_migration_coordinator = coordinator
    coordinator.cache_transition = NS(scheduler=scheduler)
    coordinator._transition = NS(target=NS(generation=1))
    published = []
    assert coordinator.finalize_at_fence(lambda: published.append(True)) == "committed"
    result = scheduler.get_internal_state("request", probe=True)
    assert published == [True]
    assert calls == [(("request",), {"probe": True})]
    assert result.internal_state == {
        "existing": True,
        "transition_rank": rank,
        "transition_request_execution_scope": "active model-launch batches; zero-layer passthrough excluded",
        "transition_request_execution": {},
        "transition_active_output_lengths": [16],
        "transition_metrics": {"generation": 1, "phase": "done"},
        "transition_fence_records": [
            {
                "generation": 1,
                "requests": [
                    {
                        "rid": "active",
                        "output_length": 16,
                        "retraction_count": 0,
                        "is_retracted": False,
                        "execution": {},
                    }
                ],
            }
        ],
    }


def test_failed_finalize_does_not_publish_successful_fence_record(acceptance):
    class Scheduler:
        def get_internal_state(self):
            return NS(internal_state={})

        def _elastic_pp_live_requests(self):
            return []

        def _record_batch_launch(self, batch):
            pass

    class Coordinator:
        def finalize_at_fence(self, publish):
            raise RuntimeError("prepare failed")

    acceptance["install_transition_hooks"](Scheduler, Coordinator)
    coordinator = Coordinator()
    coordinator.cache_transition = NS(scheduler=Scheduler())
    coordinator._transition = NS(target=NS(generation=1))
    with pytest.raises(RuntimeError, match="prepare failed"):
        coordinator.finalize_at_fence(lambda: None)
    assert not getattr(coordinator, "_smoke_fence_records", [])


@pytest.mark.parametrize(
    "initial_probe",
    [
        None,
        {
            "command": 2,
            "time": 0,
            "ongoing_writes": 0,
            "after": {"full": 0, "mamba": 0},
        },
    ],
)
@pytest.mark.parametrize("becomes_idle", [True, False])
def test_demotion_retries_real_probe_hook_with_fresh_rank_ack(
    acceptance, monkeypatch, initial_probe, becomes_idle
):
    tick = [100.0]
    clock = NS(
        time=lambda: tick[0],
        monotonic=lambda: tick[0],
        sleep=lambda delay: tick.__setitem__(0, tick[0] + delay),
    )
    monkeypatch.setitem(acceptance, "time", clock)

    class Cache:
        def __init__(self):
            self.cache_controller = object()
            self.ongoing_write_through = []
            self.full = 512
            self.mamba = 1

        def init_load_back(self, params):
            return [], None

        def writing_check(self, *, write_back):
            assert write_back

        def full_evictable_size(self):
            return self.full

        def mamba_evictable_size(self):
            return self.mamba

        def evict(self, params):
            self.full = self.mamba = 0

    class Scheduler:
        def __init__(self, rank):
            self.ps = NS(pp_rank=rank)
            self.tree_cache = Cache()
            self.attempts = 0
            self._agentx_cache_probe = copy.deepcopy(initial_probe)

        def get_internal_state(self):
            return NS(internal_state={"transition_rank": self.ps.pp_rank})

        def set_internal_state(self, req):
            raise AssertionError("diagnostic command must use the installed hook")

        def is_fully_idle(self):
            self.attempts += 1
            return becomes_idle and self.attempts > 1

    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers.scheduler", NS(Scheduler=Scheduler)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.io_struct",
        NS(SetInternalStateReqOutput=lambda **kw: NS(**kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.base_prefix_cache",
        NS(EvictParams=lambda **kw: NS(**kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.unified_radix_cache",
        NS(UnifiedRadixCache=Cache),
    )
    path = (
        Path(__file__).resolve().parents[4]
        / "research/pp_elastic/runtime_profiles/agentx_a100/cache_probe.py"
    )
    install = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "install"
    )
    namespace = {"_INSTALLED": False, "time": clock, "wraps": functools.wraps}
    exec(
        compile(ast.Module(body=[install], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["install"]()
    ranks = [Scheduler(rank) for rank in range(3)]
    commands = []

    def http(path, payload=None):
        if path == "/set_internal_state":
            commands.append(payload)
            req = NS(server_args=payload["server_args"])
            return [rank.set_internal_state(req).updated for rank in ranks]
        assert path == "/get_server_info"
        return {
            "internal_states": [
                rank.get_internal_state().internal_state for rank in ranks
            ]
        }

    if becomes_idle:
        result = acceptance["demote_cpu_prefixes"](http)
        assert len(commands) == 2
        assert result["acknowledged"] == [True] * 3
        assert all(
            probe["time"] >= result["commanded_at"] for probe in result["probes"]
        )
    else:
        with pytest.raises(TimeoutError, match="demotion did not finish"):
            acceptance["demote_cpu_prefixes"](http)
        assert len(commands) > 1


def test_correctness_queries_allow_full_common_prefix_in_actual_request_api(acceptance):
    from sglang.srt.managers.schedule_batch import Req

    prefix, active, common = acceptance["correctness_prompts"]()
    req = Req.__new__(Req)
    req.return_logprob = False
    req.logprob_start_len = -1
    assert common == 512
    assert len(prefix) == len(active) == common + 1
    assert Req._compute_max_prefix_len(req, common) == 511
    assert Req._compute_max_prefix_len(req, len(prefix)) == common
    assert Req._compute_max_prefix_len(req, len(active)) == common
    assert common % 64 == 0


@pytest.mark.parametrize("replay", [False, True])
def test_launch_observer_counts_real_modes_without_retaining_request(
    acceptance, replay
):
    import weakref
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    class Request:
        rid = "active"
        retraction_count = 0

    scheduler = NS()
    req = Request()
    owner = weakref.ref(req)
    batch = NS(reqs=[req], forward_mode=ForwardMode.EXTEND)
    acceptance["observe_request_launch"](scheduler, batch)
    batch.forward_mode = ForwardMode.DECODE
    acceptance["observe_request_launch"](scheduler, batch)
    if replay:
        req.retraction_count = 1
        batch.forward_mode = ForwardMode.EXTEND
        acceptance["observe_request_launch"](scheduler, batch)
    del batch, req
    assert owner() is None
    assert scheduler._smoke_request_execution == {
        "active": {
            "decode_batches": 1,
            "initial_prefill_batches": 1,
            "prefill_after_decode": int(replay),
            "max_retraction_count": int(replay),
        }
    }


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_rid",
        "response_retraction",
        "rank_retraction",
        "reprefill",
        "no_resumed_decode",
    ],
)
def test_exact_fenced_request_requires_uninterrupted_decode(acceptance, fault):
    evidence = states()
    for state in evidence:
        state["transition_request_execution"] = {
            "active": {
                "decode_batches": 255,
                "initial_prefill_batches": 1,
                "prefill_after_decode": 0,
                "max_retraction_count": 0,
            }
        }
        for record in state["transition_fence_records"]:
            record["requests"][0].update(
                rid="active",
                retraction_count=0,
                is_retracted=False,
                execution={"decode_batches": 16},
            )
    result = {
        "meta_info": {"id": "active", "num_retractions": 0},
        "output_ids": [1] * 256,
    }
    if fault == "wrong_rid":
        result["meta_info"]["id"] = "other"
    elif fault == "response_retraction":
        result["meta_info"]["num_retractions"] = 1
    elif fault == "rank_retraction":
        evidence[1]["transition_request_execution"]["active"][
            "max_retraction_count"
        ] = 1
    elif fault == "reprefill":
        evidence[2]["transition_request_execution"]["active"][
            "prefill_after_decode"
        ] = 1
    elif fault == "no_resumed_decode":
        evidence[0]["transition_request_execution"]["active"]["decode_batches"] = 16
    if fault is None:
        record = acceptance["validate_uninterrupted_decode"](
            evidence, 1, result, (0, 2), (0, 1, 2)
        )
        assert record["rid"] == "active"
        assert set(record["ranks"]) == {0, 1, 2}
    else:
        with pytest.raises(AssertionError):
            acceptance["validate_uninterrupted_decode"](
                evidence, 1, result, (0, 2), (0, 1, 2)
            )


def test_newly_active_stage_can_have_no_prefence_model_launches(acceptance):
    evidence = states()
    for state in evidence:
        state["transition_request_execution"] = {
            "active": {
                "decode_batches": 255,
                "prefill_after_decode": 0,
                "max_retraction_count": 0,
            }
        }
        for record in state["transition_fence_records"]:
            record["requests"][0].update(
                rid="active",
                retraction_count=0,
                is_retracted=False,
                execution={"decode_batches": 16},
            )
    evidence[1]["transition_fence_records"][0]["requests"][0]["execution"] = {}
    result = {
        "meta_info": {"id": "active", "num_retractions": 0},
        "output_ids": [1] * 256,
    }
    acceptance["validate_uninterrupted_decode"](evidence, 1, result, (0, 2), (0, 1, 2))
    evidence[1]["transition_request_execution"]["active"]["decode_batches"] = 16
    acceptance["validate_uninterrupted_decode"](evidence, 2, result, (0, 1, 2), (0, 2))
