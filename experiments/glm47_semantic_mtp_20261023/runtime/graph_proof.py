"""CPU-only proof of actual Full Decode Graph replay for real B4 requests.

Install before serving requests. ``snapshot`` returns JSON-compatible counters;
the caller owns receipt writing. No tensor inspection, CUDA event, clock read,
synchronization, log write or numerical readback occurs in the timed path.

The three correlated observations are deliberately separate: a true
ModelRunnerOutput.can_run_graph flag alone is not evidence of backend replay.
"""
from __future__ import annotations

import copy
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from typing import Any

_FRAME = ContextVar("glm_b4_graph_proof", default=None)
_STATE = None
_INSTALL_TOKEN = object()
_MAX_COHORTS = 64
_MAX_GROUPS = 32


def _default_cohort(rids):
    """Recognize this harness's warm/timed request IDs, never capture dummies."""
    if not rids or not all(isinstance(rid, str) for rid in rids):
        return None
    bases = []
    for rid in rids:
        if not rid.startswith(("warm-", "timed-", "control-", "profile-")):
            return None
        base, separator, index = rid.rpartition("-")
        if not separator or not index.isdecimal() or not 0 <= int(index) < 8:
            return None
        bases.append(base)
    return bases[0] if len(set(bases)) == 1 and len(set(rids)) == len(rids) else None


def _cpu_int(value):
    # Never call int(tensor), .item(), .cpu() or iterate a device tensor.
    return value if type(value) is int else None


def _rank_identity(getter):
    parallel = getter()
    return {
        key: _cpu_int(getattr(parallel, source, None))
        for key, source in (
            ("world_rank", "world_rank"), ("pp_rank", "pp_rank"),
            ("tp_rank", "tp_rank"), ("attn_dp_rank", "attn_dp_rank"),
        )
    }


def _capture_configuration(runner):
    batches = getattr(runner, "capture_bs", None)
    if not isinstance(batches, (tuple, list)) or not all(type(v) is int for v in batches):
        batches = None
    backend = getattr(runner, "backend", None)
    graphs = getattr(backend, "_graphs", None)
    keys = list(graphs) if isinstance(graphs, dict) else []
    sizes = [_cpu_int(getattr(key, "size", None)) for key in keys]
    serialized_keys = []
    for key in keys:
        row = {}
        for field in ("size", "stream_idx", "variant_label", "dsa_variant"):
            value = getattr(key, field, None)
            # ShapeKey has exactly these primitive fields in the pinned source.
            # Never repr() an unknown object that might contain device tensors.
            row[field] = value if value is None or type(value) in (str, int, bool) else "unsupported_nonprimitive"
        serialized_keys.append(row)
    return {
        "capture_bs": list(batches) if batches is not None else None,
        "max_bs": _cpu_int(getattr(runner, "max_bs", None)),
        "captured_req_width": _cpu_int(getattr(runner, "captured_req_width", None)),
        "backend_type": type(backend).__name__,
        "captured_graph_sizes": sorted(set(size for size in sizes if size is not None)),
        "captured_graph_count": len(keys),
        "captured_graph_keys": serialized_keys,
    }


def _new_cohort():
    return {
        "decode_forwards": 0, "model_graph_true": 0, "model_graph_false": 0,
        "execute_successes": 0, "backend_replay_successes": 0,
        "verified_local_b4_replays": 0, "exceptions": 0,
        "batch_sizes": Counter(), "replay_key_sizes": Counter(),
        "violations": Counter(), "request_groups": {},
    }


def install(*, model_runner_cls=None, decode_graph_runner_cls=None,
            full_backend_cls=None, parallel_getter=None, cohort_resolver=None):
    """Install transparent hooks; injected fake classes permit CPU-only tests.

    ``cohort_resolver`` optionally accepts the immutable tuple of real request
    IDs and returns a cohort string or None. Default IDs end in request index
    0..7 and start with warm-/timed-/control-/profile-. Snapshot/reset must run
    at an idle receipt boundary, never from a concurrent management thread.
    """
    global _STATE
    if model_runner_cls is None:
        from sglang.srt.model_executor.model_runner import ModelRunner
        model_runner_cls = ModelRunner
    if decode_graph_runner_cls is None:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        decode_graph_runner_cls = DecodeCudaGraphRunner
    if full_backend_cls is None:
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
        full_backend_cls = FullCudaGraphBackend
    if parallel_getter is None:
        from sglang.srt.runtime_context import get_parallel
        parallel_getter = get_parallel
    classes = (model_runner_cls, decode_graph_runner_cls, full_backend_cls)
    installed = getattr(model_runner_cls, "_glm_decode_graph_proof", None)
    if installed is _INSTALL_TOKEN:
        if _STATE is None or _STATE["classes"] != classes:
            raise RuntimeError("Graph proof installation state mismatch")
        return {"installed": True, "already_installed": True, "cpu_only": True}
    if installed is not None:
        raise RuntimeError("A different Graph proof already wraps this runner")
    for cls, name in ((model_runner_cls, "forward"), (decode_graph_runner_cls, "execute"),
                      (full_backend_cls, "replay")):
        if not callable(getattr(cls, name, None)):
            raise TypeError(f"Missing reviewed method {cls.__name__}.{name}")
        if getattr(getattr(cls, name), "_glm_decode_graph_proof", None) is not None:
            raise RuntimeError(f"Graph proof already wraps {cls.__name__}.{name}")
    resolver = cohort_resolver or _default_cohort
    state = {"classes": classes, "rank": None, "cohorts": {}, "capture_configurations": [],
             "configuration_ids": {}, "ignored_decode_forwards": 0}
    _STATE = state
    original_forward = model_runner_cls.forward
    original_execute = decode_graph_runner_cls.execute
    original_replay = full_backend_cls.replay

    @wraps(original_forward)
    def forward(self, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_decode():
            return original_forward(self, forward_batch, *args, **kwargs)
        rids = getattr(forward_batch, "rids", None)
        ids = tuple(rids) if isinstance(rids, (list, tuple)) else ()
        cohort_name = resolver(ids)
        if not isinstance(cohort_name, str) or not cohort_name:
            state["ignored_decode_forwards"] += 1
            return original_forward(self, forward_batch, *args, **kwargs)
        if state["rank"] is None:
            state["rank"] = _rank_identity(parallel_getter)
        cohorts = state["cohorts"]
        if cohort_name not in cohorts:
            if len(cohorts) >= _MAX_COHORTS:
                raise RuntimeError("Graph proof cohort bound exceeded")
            cohorts[cohort_name] = _new_cohort()
        row = cohorts[cohort_name]
        group = "|".join(sorted(ids))
        if group not in row["request_groups"]:
            if len(row["request_groups"]) >= _MAX_GROUPS:
                raise RuntimeError("Graph proof request-group bound exceeded")
            row["request_groups"][group] = {"request_ids": list(sorted(ids)), "forwards": 0,
                                             "verified_replays": 0}
        group_row = row["request_groups"][group]
        batch_size = _cpu_int(getattr(forward_batch, "batch_size", None))
        row["decode_forwards"] += 1
        row["batch_sizes"][str(batch_size)] += 1
        group_row["forwards"] += 1
        frame = {"row": row, "batch": forward_batch, "execute": 0, "replay": 0,
                 "invalid": [], "batch_size": batch_size, "real_count": len(ids)}
        token = _FRAME.set(frame)
        try:
            result = original_forward(self, forward_batch, *args, **kwargs)
        except BaseException:
            row["exceptions"] += 1
            raise
        finally:
            _FRAME.reset(token)
        graph_flag = getattr(result, "can_run_graph", None)
        row["model_graph_true" if graph_flag is True else "model_graph_false"] += 1
        reasons = frame["invalid"]
        if graph_flag is not True:
            reasons.append("model_can_run_graph_not_true")
        if frame["execute"] != 1:
            reasons.append("execute_count_not_one")
        if frame["replay"] != 1:
            reasons.append("backend_replay_count_not_one")
        if batch_size != 4 or len(ids) != 4:
            reasons.append("real_local_batch_not_four")
        if reasons:
            row["violations"].update(set(reasons))
        else:
            row["verified_local_b4_replays"] += 1
            group_row["verified_replays"] += 1
        return result

    @wraps(original_execute)
    def execute(self, forward_batch, *args, **kwargs):
        frame = _FRAME.get()
        if frame is None:
            return original_execute(self, forward_batch, *args, **kwargs)
        if forward_batch is not frame["batch"]:
            frame["invalid"].append("execute_forward_batch_identity_mismatch")
        result = original_execute(self, forward_batch, *args, **kwargs)
        frame["execute"] += 1
        frame["row"]["execute_successes"] += 1
        if not isinstance(getattr(self, "backend", None), full_backend_cls):
            frame["invalid"].append("backend_not_full_cuda_graph")
        key = getattr(self, "_replay_graph_key", None)
        key_size = _cpu_int(getattr(key, "size", None))
        if key_size != 4 or _cpu_int(getattr(self, "bs", None)) != 4:
            frame["invalid"].append("replay_key_or_padded_batch_not_four")
        if id(self) not in state["configuration_ids"]:
            configuration = _capture_configuration(self)
            state["configuration_ids"][id(self)] = len(state["capture_configurations"])
            state["capture_configurations"].append(configuration)
        configuration = state["capture_configurations"][state["configuration_ids"][id(self)]]
        if (configuration["capture_bs"] != [4] or configuration["max_bs"] != 4
                or configuration["captured_req_width"] != 1
                or configuration["captured_graph_sizes"] != [4]):
            frame["invalid"].append("capture_configuration_not_b4_width_one")
        return result

    @wraps(original_replay)
    def replay(self, shape_key, static_forward_batch, **kwargs):
        frame = _FRAME.get()
        result = original_replay(self, shape_key, static_forward_batch, **kwargs)
        if frame is not None:
            frame["replay"] += 1
            frame["row"]["backend_replay_successes"] += 1
            size = _cpu_int(getattr(shape_key, "size", None))
            frame["row"]["replay_key_sizes"][str(size)] += 1
            if size != 4:
                frame["invalid"].append("actual_backend_replay_key_not_four")
            if static_forward_batch is not frame["batch"]:
                frame["invalid"].append("backend_forward_batch_identity_mismatch")
        return result

    for method in (forward, execute, replay):
        method._glm_decode_graph_proof = _INSTALL_TOKEN
    model_runner_cls.forward = forward
    decode_graph_runner_cls.execute = execute
    full_backend_cls.replay = replay
    model_runner_cls._glm_decode_graph_proof = _INSTALL_TOKEN
    return {"installed": True, "already_installed": False, "cpu_only": True,
            "model_output_field": "can_run_graph", "local_batch_size_required": 4,
            "actual_backend_replay_observed": True}


def snapshot(reset=False):
    """Return a detached receipt; optionally reset cohort counters at idle.

    Capture/rank identity remains retained across resets. No files are written.
    ``passed`` requires every recognized real Decode forward to have exactly
    one actual Full Graph replay and the exact B4/width1 capture configuration.
    Numerical correctness is intentionally not claimed by this receipt.
    """
    if _STATE is None:
        return {"installed": False, "passed": False, "cohorts": {}}
    state = _STATE
    cohorts = copy.deepcopy(state["cohorts"])
    for row in cohorts.values():
        for field in ("batch_sizes", "replay_key_sizes", "violations"):
            row[field] = dict(row[field])
        row["request_groups"] = list(row["request_groups"].values())
        row["passed"] = bool(row["decode_forwards"] > 0
                             and row["verified_local_b4_replays"] == row["decode_forwards"]
                             and not row["violations"] and not row["exceptions"])
    result = {
        "schema_version": 1, "installed": True, "rank": copy.deepcopy(state["rank"]),
        "capture_configurations": copy.deepcopy(state["capture_configurations"]),
        "cohorts": cohorts, "ignored_decode_forwards": state["ignored_decode_forwards"],
        "passed": bool(cohorts) and all(row["passed"] for row in cohorts.values()),
        "proof_boundary": "CPU identities and successful native FullCudaGraphBackend.replay submission; no GPU completion or numerical correctness claim",
        "model_output_field": "can_run_graph", "gpu_synchronizations_added": 0,
        "cuda_events_added": 0, "device_tensor_readbacks_added": 0,
    }
    if reset:
        state["cohorts"].clear()
        state["ignored_decode_forwards"] = 0
    return result
