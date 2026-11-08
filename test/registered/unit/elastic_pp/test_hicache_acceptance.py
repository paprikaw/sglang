"""Reject stale, rank-incomplete and metadata-only GPU acceptance evidence."""

import ast
import copy
from pathlib import Path

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
        }
    ]
    namespace = {}
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
