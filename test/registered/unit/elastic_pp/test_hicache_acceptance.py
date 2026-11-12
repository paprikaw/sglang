"""Reject stale, rank-incomplete and metadata-only GPU acceptance evidence."""

import ast
import concurrent.futures as cf
import copy
import functools
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
            "install_diagnostic_fanout",
            "demote_cpu_prefixes",
            "correctness_prompts",
            "reference_qualification",
            "observe_request_launch",
            "validate_uninterrupted_decode",
            "state_evidence",
            "validate_decode_graph_states",
            "worker",
            "paired_overhead",
        }
    ]
    namespace = {"time": time, "math": math}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("warm", [[1, 9, 3], [1, 2], []])
def test_cold_warm_reference_qualification_rejects_divergence(acceptance, warm):
    result = acceptance["reference_qualification"](
        {"output_ids": [1, 2, 3]}, {"output_ids": warm}
    )
    assert not result["qualified"]
    assert result["first_difference"] is not None


def test_cold_warm_reference_qualification_requires_nonempty_exact_ids(acceptance):
    qualify = acceptance["reference_qualification"]
    assert qualify({"output_ids": [1, 2]}, {"output_ids": [1, 2]})["qualified"]
    assert not qualify({}, {})["qualified"]


@pytest.mark.parametrize("cached_class", [False, True])
@pytest.mark.parametrize("failure", ["post_publication", "control"])
@pytest.mark.parametrize("first_cleanup", ["verified", "unverified", "error"])
@pytest.mark.parametrize("supervised", [False, True])
def test_worker_retains_published_control_when_continuation_fails(
    acceptance,
    monkeypatch,
    tmp_path,
    cached_class,
    failure,
    first_cleanup,
    supervised,
    record_property,
    overhead_sample=False,
):
    """Exercise the actual worker's HTTP, future, failure and cleanup sequence."""
    control_seen = threading.Event()
    active_started = threading.Event()
    cleanup_started = threading.Event()
    cleanup_was_late = threading.Event()
    requests, saved, cleanup_calls = [], {}, []
    shared_diagnostics_before_cleanup = []

    def save(path, value):
        if not cleanup_calls:
            shared_diagnostics_before_cleanup.append(path.name)
        saved[path.name] = copy.deepcopy(value)

    def request_one(endpoint, unused, key, tokens, output, index):
        requests.append((len(tokens), output))
        if len(requests) == (6 if overhead_sample else 5):
            active_started.set()
            if failure == "control":
                if not cleanup_started.wait(1):
                    cleanup_was_late.set()
                raise RuntimeError("active HTTP closed by owned-child cleanup")
            if failure != "completion":
                assert control_seen.wait(
                    2
                ), "active request must fail after publication"
                raise RuntimeError("post-publication finish failure")
        payload = {
            "input_ids": tokens,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output,
                "ignore_eos": True,
            },
        }
        sys.modules["agentx_replay"].stream(endpoint, key, payload)
        return {
            "output_ids": [
                2 if cached_class and output == 256 and len(requests) >= 4 else 1
            ]
            * output,
            "meta_info": {
                "cached_tokens": 512,
                "num_retractions": 0,
                "cached_tokens_details": (
                    {"device": 0, "host": 512}
                    if len(requests) <= 6
                    else {"device": 512, "host": 0}
                ),
            },
            "e2e_ms": 100,
            "ttft_ms": 10,
            "tpot_ms": 1,
        }

    def http_json(port, path, key, payload, *, timeout):
        if path == "/set_internal_state":
            return [True] * 3
        if path == "/reconfigure_elastic_pp":
            control_seen.set()
            return {"success": True, "generation": payload["generation"]}
        assert path == "/get_server_info"
        if saved.get("worker_stage.json", {}).get("stage") == "transition_1":
            assert active_started.wait(2), "real executor must start the request"
            if failure == "control":
                raise RuntimeError("control authentication failed")
        return {
            "api_key": "must-not-save-top-level",
            "internal_states": [
                {
                    "transition_rank": rank,
                    "transition_active_output_lengths": [4],
                    "elastic_pp_state": {"migration_phase": "done"},
                    "transition_metrics": {
                        "prepare_seconds": 0.1,
                        "fence_seconds": 0.2,
                        "total_seconds": 0.4,
                    },
                    "enable_hierarchical_cache": True,
                    "disable_cuda_graph": True,
                    "api_key": "must-not-save-state-key",
                    "admin_api_key": "must-not-save-admin-key",
                    "server_args": {"other_secret": "must-not-save-nested-config"},
                    "agentx_cache_probe": {
                        "command": 2,
                        "time": time.time(),
                        "ongoing_writes": 0,
                        "after": {"full": 0, "mamba": 0},
                    },
                }
                for rank in range(3)
            ],
        }

    def cleanup(children):
        cleanup_calls.append(children)
        if len(cleanup_calls) == 1:
            if first_cleanup == "unverified":
                return {"cleanup_verified": False}
            if first_cleanup == "error":
                raise RuntimeError("first cleanup observation failed")
        cleanup_started.set()
        return {
            "cleanup_verified": True,
            "children_reaped": True,
            "process_groups_exited": True,
            "node_gpu_idle_observed": True,
            "timestamp": time.time(),
            "process_identities": (
                [child["identity"] for child in children] if supervised else []
            ),
        }

    monkeypatch.setitem(
        sys.modules,
        "agentx_replay",
        NS(stream=lambda *args: None, request_one=request_one),
    )
    for key, value in {
        "cf": cf,
        "os": os,
        "json": json,
        "signal": NS(SIGALRM=14, signal=lambda *args: None, alarm=lambda *args: None),
        "launch": lambda *args: {
            "process": NS(poll=lambda: None),
            "identity": {"pid": 1},
        },
        "h": NS(
            authenticate=lambda *args, **kw: {"lease_id": "cpu-fixture"},
            read_key=lambda cfg: "unit-test",
            save=save,
            cleanup=cleanup,
            http_json=http_json,
        ),
    }.items():
        monkeypatch.setitem(acceptance, key, value)
    config = {
        "layouts": [{"port": 1}],
        "strategy": "sync",
        "backing": "overlap",
        "initial_counts": [20, 0, 20],
        "grown_counts": [13, 13, 14],
        "overhead_sample": overhead_sample,
    }
    if supervised:
        scripts = Path(__file__).resolve().parents[4] / "research/pp_elastic"
        monkeypatch.syspath_prepend(str(scripts))
        import hicache_run_lifecycle as lifecycle

        config.update(
            id="actual-worker-cpu",
            lifecycle=dict(
                version=1,
                nonce="actual-worker-fixture",
                work_s=200,
                cleanup_s=240,
                guard_stop_s=270,
                total_s=300,
            ),
        )
        lifecycle.atomic_json(
            tmp_path / "lifecycle_budget.json",
            dict(
                experiment_id=config["id"],
                nonce=config["lifecycle"]["nonce"],
                policy=config["lifecycle"],
                started_at=time.time(),
            ),
        )
        original_save = lifecycle.LocalEvidence.save

        def local_save(self, name, value):
            assert not (tmp_path / "reference_request_contract.json").exists()
            saved[name] = copy.deepcopy(value)
            return original_save(self, name, value)

        monkeypatch.setattr(lifecycle.LocalEvidence, "save", local_save)
        if failure == "completion":
            original_complete = lifecycle.WorkerLifecycle.complete_work

            def cancelled_completion(self):
                self.cancelled.set()
                return original_complete(self)

            monkeypatch.setattr(
                lifecycle.WorkerLifecycle, "complete_work", cancelled_completion
            )
            # Isolate the final deadline verdict. State validators have their
            # own cases; this CPU HTTP fixture supplies no real GPU evidence.
            for name in (
                "validate_transition_states",
                "validate_uninterrupted_decode",
                "validate_fenced_decode",
                "fresh_prefix_restores",
            ):
                monkeypatch.setitem(acceptance, name, lambda *args: {})
    qualification_calls = []
    if cached_class:
        config.update(
            acceptance_class="cached_execution_continuity",
            cached_reference_evidence={
                "unit_test": "independently tested proof validator"
            },
        )
        monkeypatch.setitem(
            sys.modules,
            "hicache_reference_qualification",
            NS(
                qualify_cached_reference_evidence=lambda receipt, cfg: qualification_calls.append(
                    (receipt, cfg)
                )
                or {"evidence_sha256": {"raw_results": "bound-proof"}}
            ),
        )
    with pytest.raises(SystemExit) as exited:
        acceptance["worker"](tmp_path, config)
    assert exited.value.code == 1
    evidence = saved["results.json"]["transition"]
    assert evidence["passed"] is False
    expected_stage = (
        "transition_1_resume" if failure == "post_publication" else "transition_1"
    )
    expected_error = (
        "RuntimeError: post-publication finish failure"
        if failure == "post_publication"
        else "RuntimeError: control authentication failed"
    )
    if failure == "completion":
        expected_stage = "final_prefix_restore"
        expected_error = (
            "TimeoutError: Work did not finish within the Guard-relative budget"
        )
    assert evidence["failed_stage"] == expected_stage
    assert evidence["error"] == expected_error
    assert evidence["cleanup_verified"] is True
    assert len(cleanup_calls) == (1 if first_cleanup == "verified" else 2)
    assert all(len(children) == 1 for children in cleanup_calls)
    if first_cleanup == "error" and not supervised:
        assert evidence["early_cleanup_error"] == "RuntimeError"
    if supervised:
        assert not shared_diagnostics_before_cleanup
        ack = lifecycle.private_json(tmp_path / "cleanup_ack.json")
        assert ack["cleanup"]["cleanup_verified"]
        if first_cleanup == "error":
            assert ack["cleanup"]["lifecycle_cleanup_errors"] == ["RuntimeError"]
        archive = lifecycle.private_json(tmp_path / "local_evidence_archive.json")
        assert "reference_request_contract.json" in archive["sha256"]
    record_property(
        "shared_stage_writes_before_cleanup", len(shared_diagnostics_before_cleanup)
    )
    record_property("shared_contract_writes_before_cleanup", 0 if supervised else 4)
    completed_requests = 9 if overhead_sample else 7
    assert len(requests) == (completed_requests if failure == "completion" else 5)
    assert not cleanup_was_late.is_set(), "cleanup waited for the in-flight HTTP"
    expected_transitions = (
        [
            {
                "generation": 1,
                "control": {"success": True, "generation": 1},
                "control_returned_at": evidence["transitions"][0][
                    "control_returned_at"
                ],
                "continuation_verified": False,
            }
        ]
        if failure == "post_publication"
        else []
    )
    if failure == "completion":
        assert [row["generation"] for row in evidence["transitions"]] == [1, 2]
        assert all(row["continuation_verified"] for row in evidence["transitions"])
    else:
        assert evidence["transitions"] == expected_transitions
    if expected_transitions:
        assert evidence["transitions"][0]["control_returned_at"] > 0
    assert saved["worker_progress.json"] == {"completed": 0}
    for state in evidence["before"]["internal_states"]:
        assert state["enable_hierarchical_cache"] is True
        assert state["disable_cuda_graph"] is True
    assert "must-not-save" not in repr(saved)
    contract = json.loads((tmp_path / "reference_request_contract.json").read_text())
    assert contract["capture"] == "actual pre-serialization stream payload"
    assert len(contract["requests"]) == (
        completed_requests if failure == "completion" else 4
    )
    if overhead_sample:
        assert evidence["performance_sample"] is True
        assert len(evidence["overhead_baselines"]) == 2
        assert "paired_demotion" in evidence
        for record in evidence["transitions"]:
            assert record["overhead"]["samples_per_direction"] == 1
        payloads = [row["payload"] for row in contract["requests"]]
        assert all(payloads[index] == payloads[2] for index in (3, 4, 5, 6, 7))
    assert (
        tmp_path / "reference_request_contract.json"
    ).stat().st_mode & 0o777 == 0o600
    if cached_class:
        assert len(qualification_calls) == 1
        assert evidence["observed_cold_warm_exact_ids_equal"] is False
        assert evidence["cold_vs_warm_generation_qualified"] is False
        assert evidence["generation_qualified"] is False
        assert evidence["cached_execution_continuity_qualified"] is False
        assert (
            evidence["reference_state_review"]["evidence_sha256"]["raw_results"]
            == "bound-proof"
        )


def test_worker_rejects_completion_cancelled_after_assertions(
    acceptance, monkeypatch, tmp_path, record_property
):
    test_worker_retains_published_control_when_continuation_fails(
        acceptance,
        monkeypatch,
        tmp_path,
        cached_class=True,
        failure="completion",
        first_cleanup="verified",
        supervised=True,
        record_property=record_property,
    )


def test_paired_overhead_actual_worker_keeps_requests_and_late_failure_gate(
    acceptance, monkeypatch, tmp_path, record_property
):
    test_worker_retains_published_control_when_continuation_fails(
        acceptance,
        monkeypatch,
        tmp_path,
        cached_class=True,
        failure="completion",
        first_cleanup="verified",
        supervised=True,
        record_property=record_property,
        overhead_sample=True,
    )


@pytest.mark.parametrize("generation", [1, 2])
def test_paired_overhead_reports_signed_single_sample_ms(acceptance, generation):
    tier = {"device": 0, "host": 512} if generation == 1 else {"device": 512, "host": 0}
    baseline = {
        "output_ids": [1] * 256,
        "meta_info": {
            "cached_tokens": 512,
            "cached_tokens_details": tier,
            "num_retractions": 0,
        },
        "e2e_ms": 100,
        "ttft_ms": 10,
        "tpot_ms": 1,
    }
    active = copy.deepcopy(baseline)
    active.update(e2e_ms=120, ttft_ms=9)
    evidence = states()
    for state in evidence:
        state["transition_metrics"].update(
            prepare_seconds=0.1, fence_seconds=0.2, total_seconds=0.4
        )
    result = acceptance["paired_overhead"](baseline, active, evidence, generation)
    assert result["difference_ms"] == {"e2e_ms": 20, "ttft_ms": -1, "tpot_ms": 0}
    assert result["samples_per_direction"] == 1
    assert result["per_rank_ms"]["0"] == {
        "prepare_ms": 100,
        "fence_ms": 200,
        "total_ms": 400,
    }


@pytest.mark.parametrize(
    "fault", ["tier", "ids", "retraction", "missing", "nan", "coordinator"]
)
def test_paired_overhead_rejects_incomparable_or_missing_evidence(acceptance, fault):
    baseline = {
        "output_ids": [1] * 256,
        "meta_info": {
            "cached_tokens": 512,
            "cached_tokens_details": {"device": 0, "host": 512},
            "num_retractions": 0,
        },
        "e2e_ms": 100,
        "ttft_ms": 10,
        "tpot_ms": 1,
    }
    active = copy.deepcopy(baseline)
    evidence = states()
    for state in evidence:
        state["transition_metrics"].update(
            prepare_seconds=0.1, fence_seconds=0.2, total_seconds=0.4
        )
    if fault == "tier":
        active["meta_info"]["cached_tokens_details"] = {"device": 512, "host": 0}
    elif fault == "ids":
        active["output_ids"][0] = 9
    elif fault == "retraction":
        active["meta_info"]["num_retractions"] = 1
    elif fault == "missing":
        active.pop("ttft_ms")
    elif fault == "nan":
        active["tpot_ms"] = float("nan")
    else:
        evidence[1]["transition_metrics"]["fence_seconds"] = None
    with pytest.raises(AssertionError):
        acceptance["paired_overhead"](baseline, active, evidence, 1)


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


def test_completed_background_host_payload_counts_when_fence_delta_is_empty(acceptance):
    evidence = states()
    for state in evidence:
        host = state["transition_metrics"]["host"]
        host["host_background_bytes_sent"] = host.pop("host_bytes_sent")
        host["host_bytes_sent"] = 0
    acceptance["validate_transition_states"](evidence, 2, (0, 1, 2), "overlap")


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


def test_private_startup_graph_walk_has_no_serving_fence_record(acceptance):
    class Scheduler:
        def get_internal_state(self):
            return NS(internal_state={})

        def _record_batch_launch(self, batch):
            pass

    class Coordinator:
        cache_transition = None

        def finalize_at_fence(self, publish):
            return publish()

    acceptance["install_transition_hooks"](Scheduler, Coordinator)
    coordinator = Coordinator()
    assert coordinator.finalize_at_fence(lambda: "startup") == "startup"
    assert not getattr(coordinator, "_smoke_fence_records", [])


@pytest.mark.parametrize(
    "fault", [None, "runtime_capture", "unsealed", "missing_capture", "no_selection"]
)
def test_prepared_graph_acceptance_rejects_recapture_and_incomplete_bank(
    acceptance, fault
):
    states = []
    for rank in range(3):
        states.append(
            {
                "transition_rank": rank,
                "transition_metrics": {
                    "decode_graph": [
                        {"generation": 0, "replays": int(rank != 1)},
                        {
                            "generation": 1,
                            "status": "ready",
                            "replays": 3,
                            "capture_seconds": 0.0,
                            "prepared": True,
                        },
                    ],
                    "prepared_decode_graphs": {
                        "sealed": True,
                        "runtime_captures": 0,
                        "startup_runner_captures": 1 if rank == 1 else 2,
                        "runtime_selections": 1,
                    },
                },
            }
        )
    metrics = states[2]["transition_metrics"]
    if fault == "runtime_capture":
        metrics["decode_graph"][1]["capture_seconds"] = 0.8
    elif fault == "unsealed":
        metrics["prepared_decode_graphs"]["sealed"] = False
    elif fault == "missing_capture":
        metrics["prepared_decode_graphs"]["startup_runner_captures"] = 1
    elif fault == "no_selection":
        metrics["prepared_decode_graphs"]["runtime_selections"] = 0
    if fault:
        with pytest.raises(AssertionError, match="prepared|bank"):
            acceptance["validate_decode_graph_states"](
                states, 1, (0, 2), (0, 1, 2), prepared=True
            )
    else:
        acceptance["validate_decode_graph_states"](
            states, 1, (0, 2), (0, 1, 2), prepared=True
        )


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize(
    "fault",
    [None, "resident", "reuse", "no_transfer", "export", "fallback", "unverified"],
)
@pytest.mark.parametrize("weight_source", ["legacy", "cpu_cache", "cpu_prepacked"])
def test_nonresident_graph_acceptance_requires_real_transfer_and_eviction(
    acceptance, generation, fault, weight_source
):
    senders = {0, 2} if generation == 1 else {1}
    targets = (0, 1, 2) if generation == 1 else (0, 2)
    states = []
    for rank in range(3):
        metrics = {
            "decode_graph": [
                {"generation": 0, "replays": int(rank != 1)},
                {
                    "generation": 1,
                    "status": "ready",
                    "replays": 3,
                    "capture_seconds": 0.0,
                    "prepared": True,
                },
            ],
            "prepared_decode_graphs": {
                "sealed": True,
                "runtime_captures": 0,
                "startup_runner_captures": 1 if rank == 1 else 2,
                "runtime_selections": 1 if rank == 1 else generation,
                "weight_policy": "offload",
                "inactive_registered_weight_bytes": 0,
                "weight_backing": {
                    "kind": "vmm_nonresident_weights",
                    "inactive_mapped_physical_bytes": 0,
                },
            },
            "prepared_weight_bytes_reused": 0,
            "background_weight_bytes_sent": 4096 if rank in senders else 0,
        }
        if weight_source in {"cpu_cache", "cpu_prepacked"}:
            receives = rank in ({1} if generation == 1 else {0, 2})
            metrics.update(
                weight_staging=weight_source,
                background_weight_bytes_sent=0,
                weight_staging_metrics=dict(
                    kind=(
                        "prepared_cpu_cache"
                        if weight_source == "cpu_prepacked"
                        else "canonical_cpu_cache"
                    ),
                    cache_key="initial-bank",
                    cache_key_matches_initial_load=receives,
                    gpu_export_bytes=0,
                    gloo_weight_bytes=0,
                    fallback_count=0,
                    target_weight_bytes=4096 if receives else 0,
                    checkpoint_bytes=4096 if receives else 0,
                    checkpoint_tensors=2 if receives else 0,
                    checkpoint_load_seconds=0.1 if receives else 0,
                    layers_loaded=[1] if receives else [],
                    storage_verified=receives,
                    prepared_cpu_cache_hit=receives,
                    cpu_prepare_seconds=0,
                    canonical_checkpoint_read_bytes=0,
                    h2d_bytes=4096 if receives else 0,
                    h2d_wall_seconds=0.1 if receives else 0,
                    h2d_stream_seconds=0.09 if receives else 0,
                    prepared_cpu_cache_bytes=4096,
                    prepared_cpu_cache_cap_bytes=8192,
                ),
            )
        if generation == 2:
            metrics["decode_graph"].append(
                {
                    "generation": 2,
                    "status": "ready" if rank in targets else "inactive",
                    "replays": 3 if rank in targets else 0,
                    "capture_seconds": 0.0,
                    "prepared": True,
                }
            )
        states.append({"transition_rank": rank, "transition_metrics": metrics})
    metrics = states[next(iter(senders))]["transition_metrics"]
    if fault == "resident":
        metrics["prepared_decode_graphs"]["weight_backing"][
            "inactive_mapped_physical_bytes"
        ] = 1
    elif fault == "reuse":
        metrics["prepared_weight_bytes_reused"] = 4096
    elif fault == "no_transfer":
        if weight_source in {"cpu_cache", "cpu_prepacked"}:
            receiver = next(
                row["transition_metrics"]
                for row in states
                if row["transition_rank"] not in senders
            )
            receiver["weight_staging_metrics"]["checkpoint_bytes"] = 0
        else:
            metrics["background_weight_bytes_sent"] = 0
    elif fault in {"export", "fallback", "unverified"}:
        if weight_source in {"cpu_cache", "cpu_prepacked"}:
            receiver = next(
                row["transition_metrics"]
                for row in states
                if row["transition_rank"] not in senders
            )
            key, value = {
                "export": ("gpu_export_bytes", 4),
                "fallback": ("fallback_count", 1),
                "unverified": ("storage_verified", False),
            }[fault]
            receiver["weight_staging_metrics"][key] = value
        else:
            metrics["background_weight_bytes_sent"] = 0
    if fault:
        with pytest.raises(AssertionError, match="weights|transfer"):
            acceptance["validate_decode_graph_states"](
                states,
                generation,
                (0, 2) if generation == 1 else (0, 1, 2),
                targets,
                prepared=True,
            )
    else:
        acceptance["validate_decode_graph_states"](
            states,
            generation,
            (0, 2) if generation == 1 else (0, 1, 2),
            targets,
            prepared=True,
        )


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


@pytest.fixture
def diagnostic_ipc(acceptance, tmp_path, monkeypatch):
    """Run the production channel constructor and control communicator setup."""
    import logging

    import zmq

    from sglang.srt.managers import tokenizer_control_mixin as control
    from sglang.srt.managers.scheduler_components.ipc_channels import (
        SchedulerIpcChannels,
    )

    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/managers/scheduler.py"
    scheduler_class = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_ipc_channels"
    )
    observation = NS(enable_metrics=False, load_snapshot_publish_interval=0.1)
    namespace = {
        "SchedulerIpcChannels": SchedulerIpcChannels,
        "get_observability": lambda: observation,
        "envs": NS(SGLANG_TEST_SCRIPTED_RUNTIME=NS(get=lambda: False)),
        "create_load_snapshot_writer": lambda *args, **kwargs: None,
        "logger": logging.getLogger(__name__),
        "rank_consensus_checker": NS(shutdown=lambda: None),
    }
    release = next(
        node
        for node in scheduler_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "release_host_resources"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method, release], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )

    class Scheduler:
        init_ipc_channels = namespace["init_ipc_channels"]

        release_host_resources = namespace["release_host_resources"]

    class Tokenizer(control.TokenizerControlMixin):
        def __init__(self):
            self.sent = []
            self._result_dispatcher = control.TypeBasedDispatcher([])
            self._dispatch_to_scheduler = self.sent.append

        def auto_create_handle_loop(self):
            pass

    acceptance["install_diagnostic_fanout"](Scheduler, Tokenizer)
    monkeypatch.setattr(control, "get_parallel", lambda: NS(dp_size=1))
    context = zmq.Context(1)
    receiver = context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVTIMEO, 2000)
    receiver_port = receiver.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{receiver_port}"
    ports = NS(
        scheduler_input_ipc_name=f"ipc://{tmp_path}/scheduler.sock",
        rpc_ipc_name=f"ipc://{tmp_path}/rpc.sock",
        tokenizer_ipc_name=endpoint,
    )
    workers = []
    for rank in range(3):
        scheduler = Scheduler()
        scheduler.ps = NS(
            pp_size=3,
            tp_size=1,
            dp_size=1,
            pp_rank=rank,
            attn_tp_rank=0,
            attn_cp_rank=0,
            dp_rank=0,
        )
        scheduler.skip_tokenizer_init = True
        scheduler.hisparse_coordinator = None
        scheduler.decode_offload_manager = None
        scheduler.tree_cache = NS(release_host_resources=lambda: None)
        scheduler.init_ipc_channels(ports)
        workers.append(scheduler)
    args = NS(
        pp_size=3,
        tp_size=1,
        dp_size=1,
        nnodes=1,
        enable_elastic_pp_autoscaling=False,
    )
    tokenizer = Tokenizer()
    try:
        tokenizer.init_communicators(args)
        yield NS(
            workers=workers,
            tokenizer=tokenizer,
            receiver=receiver,
            args=args,
            scheduler_class=Scheduler,
            tokenizer_class=Tokenizer,
            ports=ports,
            production_init_ipc=namespace["init_ipc_channels"],
            production_release=namespace["release_host_resources"],
        )
    finally:
        contexts = set()
        for scheduler in workers:
            try:
                scheduler.release_host_resources()
            except RuntimeError:
                pass
            channels = scheduler.ipc_channels
            for name in (
                "recv_from_tokenizer",
                "recv_from_rpc",
                "send_to_tokenizer",
                "send_to_detokenizer",
            ):
                channel = getattr(channels, name)
                socket = getattr(channel, "socket", channel)
                if socket is not None:
                    contexts.add(socket.context)
                    socket.close(linger=0)
        for owned in contexts:
            owned.term()
        receiver.close(linger=0)
        context.term()


@pytest.mark.asyncio
async def test_three_resident_diagnostic_replies_reach_real_communicators(
    diagnostic_ipc,
):
    import asyncio

    from sglang.srt.managers.io_struct import (
        GetInternalStateReq,
        GetInternalStateReqOutput,
        SetInternalStateReq,
        SetInternalStateReqOutput,
        sock_recv,
    )

    fixture = diagnostic_ipc
    tokenizer = fixture.tokenizer
    # Other controls still wait for one DP reply, including topology publication.
    assert tokenizer.reconfigure_elastic_pp_communicator._fan_out == 1
    assert tokenizer.get_elastic_pp_state_communicator._fan_out == 1
    assert tokenizer.get_internal_state_communicator._fan_out == 3
    assert tokenizer.set_internal_state_communicator._fan_out == 3
    for command in ("get", "set"):
        if command == "get":
            task = asyncio.create_task(tokenizer.get_internal_state())
            request = GetInternalStateReq(http_worker_ipc="return-route")
        else:
            request = SetInternalStateReq(
                server_args={"agentx_cache_cmd": 2}, http_worker_ipc="return-route"
            )
            task = asyncio.create_task(tokenizer.set_internal_state(request))
        await asyncio.sleep(0)
        assert tokenizer.sent
        for ordinal, rank in enumerate((0, 2, 1)):
            output = (
                GetInternalStateReqOutput(internal_state={"transition_rank": rank})
                if command == "get"
                else SetInternalStateReqOutput(updated=rank != 1)
            )
            fixture.workers[rank].ipc_channels.send_to_tokenizer.send_output(
                output, request
            )
            reply = sock_recv(fixture.receiver)
            assert reply.http_worker_ipc == "return-route"
            tokenizer._result_dispatcher(reply)
            await asyncio.sleep(0)
            if ordinal < 2:
                assert (
                    not task.done()
                ), "partial PP replies must not complete the HTTP result"
        result = await asyncio.wait_for(task, 2)
        if command == "get":
            assert {state["transition_rank"] for state in result} == {0, 1, 2}
        else:
            assert result == [True, True, False]


def test_diagnostic_sender_keeps_other_pp_outputs_on_original_route(diagnostic_ipc):
    from sglang.srt.managers.io_struct import (
        AbortReq,
        GetElasticPPStateReqOutput,
        ReconfigureElasticPPReqOutput,
        sock_recv,
    )

    fixture = diagnostic_ipc
    controls = [
        AbortReq(rid="ordinary"),
        GetElasticPPStateReqOutput(
            enabled=True,
            generation=0,
            layer_counts=[20, 0, 20],
            migration_phase="idle",
            source_usable=True,
            waiting_requests=0,
            running_requests=0,
        ),
        ReconfigureElasticPPReqOutput(
            success=True,
            message="ok",
            old_generation=0,
            new_generation=1,
            old_layer_counts=[20, 0, 20],
            new_layer_counts=[13, 13, 14],
            migration_phase="committed",
            source_usable=True,
        ),
    ]
    for worker in fixture.workers[1:]:
        assert worker.ipc_channels.send_to_tokenizer.normal.socket is None
        for output in controls:
            worker.ipc_channels.send_to_tokenizer.send_output(output)
    assert not fixture.receiver.poll(50)
    for output in controls:
        fixture.workers[0].ipc_channels.send_to_tokenizer.send_output(output)
        assert sock_recv(fixture.receiver) == output


def test_private_reply_sockets_close_even_after_host_cleanup_failure(diagnostic_ipc):
    worker = diagnostic_ipc.workers[1]
    socket = worker._smoke_diagnostic_socket
    context = worker._smoke_diagnostic_context

    def failed_release():
        raise RuntimeError("host release failed")

    worker.tree_cache.release_host_resources = failed_release
    with pytest.raises(RuntimeError, match="host release failed"):
        worker.release_host_resources()
    assert socket.closed and context.closed
    assert worker._smoke_diagnostic_socket is None


@pytest.mark.parametrize(
    "field,value", [("pp_size", 2), ("tp_size", 2), ("dp_size", 2), ("nnodes", 2)]
)
def test_private_reply_fanout_rejects_unqualified_topologies(
    diagnostic_ipc, field, value
):
    fixture = diagnostic_ipc
    args = NS(**vars(fixture.args))
    setattr(args, field, value)
    with pytest.raises(ValueError, match="one-node PP3/TP1/DP1"):
        fixture.tokenizer_class().init_communicators(args)
    if field != "nnodes":
        worker = fixture.scheduler_class()
        worker.ps = NS(pp_size=3, tp_size=1, dp_size=1)
        setattr(worker.ps, field, value)
        with pytest.raises(ValueError, match="PP3/TP1/DP1"):
            worker.init_ipc_channels(fixture.ports)


@pytest.mark.parametrize("failure_point", ["socket", "channels"])
def test_private_reply_construction_failure_closes_retained_context(
    acceptance, diagnostic_ipc, monkeypatch, failure_point
):
    import dataclasses

    from sglang.srt.utils import network

    fixture = diagnostic_ipc
    contexts = []
    sockets = []
    original_socket = network.get_zmq_socket

    def socket(context, *args):
        contexts.append(context)
        if failure_point == "socket":
            raise RuntimeError("socket construction failed")
        result = original_socket(context, *args)
        sockets.append(result)
        return result

    def failed_replace(*args, **kwargs):
        raise RuntimeError("channel publication failed")

    monkeypatch.setattr(network, "get_zmq_socket", socket)
    if failure_point == "channels":
        monkeypatch.setattr(dataclasses, "replace", failed_replace)

    class Scheduler:
        init_ipc_channels = fixture.production_init_ipc
        release_host_resources = fixture.production_release

    class Tokenizer:
        def init_communicators(self, args):
            pass

    acceptance["install_diagnostic_fanout"](Scheduler, Tokenizer)
    worker = Scheduler()
    worker.ps = NS(
        pp_size=3, tp_size=1, dp_size=1, pp_rank=1, attn_tp_rank=0, attn_cp_rank=0
    )
    worker.skip_tokenizer_init = True
    with pytest.raises(RuntimeError) as retained:
        worker.init_ipc_channels(fixture.ports)
    assert retained.value.__traceback__ is not None
    assert len(contexts) == 1 and contexts[0].closed
    assert all(socket.closed for socket in sockets)
    assert not hasattr(worker, "_smoke_diagnostic_context")
    assert worker.ipc_channels.send_to_tokenizer.socket is None
