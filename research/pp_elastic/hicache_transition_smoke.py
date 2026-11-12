"""Guard-only <=300s Qwen HiCache PP2 -> PP3 -> PP2 correctness check.

Shares the audited CPU-cache/owned-process launch helpers with gpu_smoke.
The optional overhead sample pairs each transition with one unchanged-layout request.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import gpu_smoke as base

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
h = base.h
sys.path.insert(0, str(ROOT / "runtime_profiles/agentx_a100"))


def rank_states(states):
    ranks = {state["transition_rank"]: state for state in states}
    if len(states) != 3 or set(ranks) != {0, 1, 2}:
        raise AssertionError(
            "transition evidence must contain each resident rank once: "
            f"received {len(states)} replies with ranks "
            f"{[state.get('transition_rank') for state in states]}"
        )
    return ranks


def paired_overhead(baseline, active, states, generation):
    """One observation per direction, with the source cache tier held fixed."""
    expected_tier = (
        {"device": 0, "host": 512} if generation == 1 else {"device": 512, "host": 0}
    )
    if generation not in (1, 2):
        raise AssertionError("unknown overhead direction")
    if len(baseline.get("output_ids", ())) != 256 or baseline[
        "output_ids"
    ] != active.get("output_ids"):
        raise AssertionError("paired output IDs differ")
    measured = {}
    for label, result in (("baseline", baseline), ("transition", active)):
        meta = result.get("meta_info", {})
        if (
            meta.get("cached_tokens") != 512
            or meta.get("cached_tokens_details") != expected_tier
            or meta.get("num_retractions") != 0
        ):
            raise AssertionError("paired cache tier or retraction count differs")
        measured[label] = {}
        for name in ("e2e_ms", "ttft_ms", "tpot_ms"):
            value = result.get(name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise AssertionError("paired request timing is missing or invalid")
            measured[label][name] = value
    measured["difference_ms"] = {
        name: measured["transition"][name] - measured["baseline"][name]
        for name in measured["baseline"]
    }
    measured["per_rank_ms"] = {}
    for rank, state in rank_states(states).items():
        metrics = state["transition_metrics"]
        timings = {}
        for name in ("prepare_seconds", "fence_seconds", "total_seconds"):
            value = metrics.get(name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise AssertionError("coordinator timing is missing or invalid")
            timings[name.replace("_seconds", "_ms")] = value * 1000
        measured["per_rank_ms"][str(rank)] = timings
    measured.update(
        samples_per_direction=1,
        source_counts=[20, 0, 20] if generation == 1 else [13, 13, 14],
        target_counts=[13, 13, 14] if generation == 1 else [20, 0, 20],
        cache_tier=expected_tier,
        timing_scope="client stream E2E/TTFT/mean TPOT; coordinator prepare/fence/through cleanup",
        unmeasured=[
            "token interval pause",
            "isolated transfer",
            "isolated restore",
            "strict peak memory",
        ],
    )
    return measured


def validate_transition_states(states, generation, source_ranks, backing):
    ranks = rank_states(states)
    for state in ranks.values():
        metrics = state.get("transition_metrics", {})
        if metrics.get("generation") != generation or metrics.get("phase") != "done":
            raise AssertionError(
                "transition metrics are stale or cleanup is incomplete"
            )
        if metrics.get("backing_policy") != backing:
            raise AssertionError("transition backing differs from the frozen config")
    sent = sum(
        state["transition_metrics"].get("host", {}).get("host_bytes_sent", 0)
        for state in ranks.values()
    )
    if sent <= 0:
        raise AssertionError("no CPU-only host payload moved during this transition")
    if backing == "cpu_spill":
        for rank in source_ranks:
            source = ranks[rank]["transition_metrics"].get("source_backing", {})
            if source.get("release", {}).get("detached_bytes", 0) <= 0:
                raise AssertionError(
                    "source backing was not detached on an active rank"
                )
            if source.get("cuda_release", {}).get("allocated_drop_bytes", 0) <= 0:
                raise AssertionError("no measured CUDA owner release on an active rank")


def validate_fenced_decode(states, generation, source_ranks, output_length):
    ranks = rank_states(states)
    for rank in source_ranks:
        records = [
            record
            for record in ranks[rank].get("transition_fence_records", [])
            if record["generation"] == generation
        ]
        if len(records) != 1 or not any(
            0 < req["output_length"] < output_length for req in records[0]["requests"]
        ):
            raise AssertionError(
                "no active Decode state at the actual generation fence"
            )


def fresh_prefix_restores(states, started_at, active_ranks, prefix_length):
    ranks = rank_states(states)
    restores = {
        rank: [
            event
            for event in state.get("agentx_restores", [])
            if event["time"] >= started_at
        ]
        for rank, state in ranks.items()
    }
    if not all(
        any(event.get("loaded_tokens", 0) >= prefix_length for event in restores[rank])
        for rank in active_ranks
    ):
        raise AssertionError(
            "final prefix lacks fresh CPU loadback on every active stage"
        )
    return restores


def validate_uninterrupted_decode(
    states, generation, result, source_ranks, target_ranks
):
    ranks = rank_states(states)
    meta = result.get("meta_info", {})
    rid = meta.get("id")
    if not isinstance(rid, str) or not rid or meta.get("num_retractions") != 0:
        raise AssertionError(
            "active response lacks request identity or has retractions"
        )
    fences = {}
    for rank, state in ranks.items():
        records = [
            record
            for record in state.get("transition_fence_records", [])
            if record["generation"] == generation
        ]
        if len(records) != 1:
            raise AssertionError("missing unique request fence for this generation")
        requests = [req for req in records[0]["requests"] if req["rid"] == rid]
        if len(requests) != 1:
            raise AssertionError("active response does not match the fenced request")
        request = requests[0]
        if rank in source_ranks and not (
            0 < request["output_length"] < len(result["output_ids"])
        ):
            raise AssertionError("matched request was not in Decode at its fence")
        if (
            request.get("retraction_count") != 0
            or request.get("is_retracted") is not False
        ):
            raise AssertionError("fenced request has retracted")
        execution = state.get("transition_request_execution", {}).get(rid, {})
        if (
            execution.get("max_retraction_count") != 0
            or execution.get("prefill_after_decode") != 0
        ):
            raise AssertionError(
                "active request retracted or re-prefilled after Decode"
            )
        if rank in target_ranks and execution.get("decode_batches", 0) <= request.get(
            "execution", {}
        ).get("decode_batches", 0):
            raise AssertionError(
                "target stage has no resumed Decode launch for the request"
            )
        fences[rank] = {"at_fence": request, "after": execution}
    return {"rid": rid, "generation": generation, "ranks": fences}


def observe_request_launch(scheduler, batch):
    executions = getattr(scheduler, "_smoke_request_execution", None)
    if executions is None:
        executions = scheduler._smoke_request_execution = {}
    for req in batch.reqs:
        execution = executions.setdefault(
            req.rid,
            {
                "decode_batches": 0,
                "initial_prefill_batches": 0,
                "prefill_after_decode": 0,
                "max_retraction_count": 0,
            },
        )
        if batch.forward_mode.is_decode():
            execution["decode_batches"] += 1
        elif batch.forward_mode.is_extend():
            key = (
                "prefill_after_decode"
                if execution["decode_batches"]
                else "initial_prefill_batches"
            )
            execution[key] += 1
        execution["max_retraction_count"] = max(
            execution["max_retraction_count"], req.retraction_count
        )


def correctness_prompts():
    common_prefix = [1000 + i % 160 for i in range(512)]
    active_prefix = [2000 + i % 160 for i in range(512)]
    # Req reserves the final input token, so 513 input permits the full 512
    # cached tokens and their aligned recurrent checkpoint to be matched.
    return common_prefix + [1160], active_prefix + [2160], len(common_prefix)


def reference_qualification(cold, warm):
    cold_ids, warm_ids = cold.get("output_ids") or [], warm.get("output_ids") or []
    first_difference = next(
        (
            index
            for index, pair in enumerate(zip(cold_ids, warm_ids))
            if pair[0] != pair[1]
        ),
        min(len(cold_ids), len(warm_ids)) if len(cold_ids) != len(warm_ids) else None,
    )
    return {
        "qualified": bool(cold_ids) and cold_ids == warm_ids,
        "comparison": "exact token IDs; identical input and greedy sampling",
        "first_difference": first_difference,
        "cold_output_tokens": len(cold_ids),
        "warm_output_tokens": len(warm_ids),
    }


def demote_cpu_prefixes(http):
    deadline = time.monotonic() + 15
    while True:
        commanded_at = time.time()
        answer = http("/set_internal_state", {"server_args": {"agentx_cache_cmd": 2}})
        states = rank_states(http("/get_server_info")["internal_states"])
        probes = [state.get("agentx_cache_probe") or {} for state in states.values()]
        acknowledged = (
            isinstance(answer, list)
            and len(answer) == len(states)
            and all(value is True for value in answer)
        )
        if acknowledged and all(
            probe.get("command") == 2
            and probe.get("time", 0) >= commanded_at
            and probe.get("ongoing_writes") == 0
            and probe.get("after") == {"full": 0, "mamba": 0}
            for probe in probes
        ):
            return {
                "commanded_at": commanded_at,
                "acknowledged": answer,
                "probes": probes,
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU backup and GPU demotion did not finish")
        time.sleep(0.05)


def state_evidence(info):
    # get_server_info also exposes ServerArgs. Persist only the diagnostics
    # needed by this check, rather than credentials or unrelated configuration.
    fields = (
        "transition_rank",
        "transition_fence_records",
        "transition_metrics",
        "transition_request_execution_scope",
        "transition_request_execution",
        "transition_active_output_lengths",
        "agentx_cache_probe",
        "agentx_restores",
        "elastic_pp_state",
        "memory_usage",
        "enable_hierarchical_cache",
        "disable_radix_cache",
        "disable_cuda_graph",
        "hicache_mem_layout",
        "hicache_io_backend",
        "hicache_host_memory_mode",
        "elastic_pp_transition_strategy",
        "elastic_pp_transition_backing",
        "enable_elastic_pp_kv_patch",
        "elastic_pp_kv_patch_transport",
    )
    return {
        "internal_states": [
            {key: state[key] for key in fields if key in state}
            for state in info.get("internal_states", [])
        ]
    }


def validate_decode_graph_states(
    states, generation, source_ranks, target_ranks, *, prepared=False
):
    ranked = rank_states(states)
    cached = [
        state["transition_metrics"]
        for state in ranked.values()
        if state["transition_metrics"].get("weight_staging")
        in {"cpu_cache", "cpu_prepacked"}
    ]
    if cached and (
        len(cached) != len(ranked)
        or len({m.get("weight_staging_metrics", {}).get("cache_key") for m in cached})
        != 1
    ):
        raise AssertionError("CPU cache weights came from inconsistent rank banks")
    for rank, state in ranked.items():
        history = state["transition_metrics"]["decode_graph"]
        entries = {row["generation"]: row for row in history}
        if set(entries) != set(range(generation + 1)):
            raise AssertionError("missing elastic Decode Graph generations")
        if rank in source_ranks and entries[generation - 1]["replays"] <= 0:
            raise AssertionError("source Decode Graph did not actually replay")
        target = entries[generation]
        if rank in target_ranks:
            if target["status"] != "ready" or target["replays"] <= 0:
                raise AssertionError("target Decode Graph did not actually replay")
            if target["capture_seconds"] is None:
                raise AssertionError("missing target Graph capture timing")
            if prepared and (
                target["capture_seconds"] != 0.0 or not target.get("prepared")
            ):
                raise AssertionError("runtime target Graph was not prepared at startup")
        elif target["status"] != "inactive" or target["replays"]:
            raise AssertionError("inactive rank retained a runnable Decode Graph")
        if prepared:
            bank = state["transition_metrics"].get("prepared_decode_graphs") or {}
            if (
                not bank.get("sealed")
                or bank.get("runtime_captures") != 0
                or bank.get("startup_runner_captures") != (1 if rank == 1 else 2)
                or bank.get("runtime_selections") != (1 if rank == 1 else generation)
            ):
                raise AssertionError(
                    "prepared Graph bank capture/selection evidence incomplete"
                )
            if bank.get("weight_policy") == "offload":
                backing = bank.get("weight_backing", {})
                if (
                    backing.get("kind") != "vmm_nonresident_weights"
                    or backing.get("inactive_mapped_physical_bytes", -1) != 0
                ):
                    raise AssertionError(
                        "inactive candidate weights still occupy physical GPU pages"
                    )
                metrics = state["transition_metrics"]
                if metrics.get("prepared_weight_bytes_reused", -1) != 0:
                    raise AssertionError(
                        "nonresident weights were reported as resident reuse"
                    )
                if bank.get("inactive_registered_weight_bytes", -1) != 0:
                    raise AssertionError("inactive worker retains spare model weights")
                if metrics.get("weight_staging") in {"cpu_cache", "cpu_prepacked"}:
                    reload = metrics.get("weight_staging_metrics", {})
                    if (
                        reload.get("kind")
                        != (
                            "prepared_cpu_cache"
                            if metrics["weight_staging"] == "cpu_prepacked"
                            else "canonical_cpu_cache"
                        )
                        or not reload.get("cache_key")
                        or metrics.get("background_weight_bytes_sent", -1) != 0
                        or reload.get("gpu_export_bytes", -1) != 0
                        or reload.get("gloo_weight_bytes", -1) != 0
                        or reload.get("fallback_count", -1) != 0
                    ):
                        raise AssertionError(
                            "CPU cache weights used an export or fallback path"
                        )
                    incoming = {1} if generation == 1 else {0, 2}
                    if (
                        metrics["weight_staging"] == "cpu_prepacked"
                        and rank in incoming
                    ):
                        if (
                            not reload.get("prepared_cpu_cache_hit")
                            or reload.get("cpu_prepare_seconds", -1) != 0
                            or reload.get("canonical_checkpoint_read_bytes", -1) != 0
                            or reload.get("h2d_bytes", 0) <= 0
                            or reload.get("h2d_wall_seconds", 0) <= 0
                            or reload.get("h2d_stream_seconds", 0) <= 0
                            or not 0
                            < reload.get("prepared_cpu_cache_bytes", 0)
                            <= reload.get("prepared_cpu_cache_cap_bytes", 0)
                        ):
                            raise AssertionError(
                                "runtime CPU prepacked weights were not reused within budget"
                            )
                    if rank in incoming and (
                        reload.get("target_weight_bytes", 0) <= 0
                        or reload.get("checkpoint_bytes", 0) <= 0
                        or reload.get("checkpoint_tensors", 0) <= 0
                        or not reload.get("cache_key_matches_initial_load")
                        or not reload.get("storage_verified")
                        or not reload.get("layers_loaded")
                        or reload.get("checkpoint_load_seconds", 0) <= 0
                    ):
                        raise AssertionError(
                            "CPU cache target weights were not fully restored"
                        )
                    if rank not in incoming and (
                        reload.get("target_weight_bytes", -1) != 0
                        or reload.get("checkpoint_bytes", -1) != 0
                        or reload.get("layers_loaded") != []
                    ):
                        raise AssertionError(
                            "CPU cache reload touched non-target weights"
                        )
                elif (
                    rank in ({0, 2} if generation == 1 else {1})
                    and metrics.get("background_weight_bytes_sent", 0) <= 0
                ):
                    raise AssertionError(
                        "nonresident transition did not actually transfer source weights"
                    )


def install_hooks():
    spec = importlib.util.spec_from_file_location(
        "transition_native_adapter", base.NATIVE / "a100_runtime_adapter.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    sys.path.insert(0, str(base.NATIVE))
    adapter.install()
    import runtime_setup

    runtime_setup.install()
    sys.path.remove(str(base.NATIVE))
    import cache_probe

    cache_probe.install()
    from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

    install_diagnostic_fanout(Scheduler, TokenizerControlMixin)
    install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator)
    if os.environ.get("PP_HICACHE_BASELINE_PROBE") == "1":
        import hicache_baseline_probe

        hicache_baseline_probe.install_hooks()


def install_diagnostic_fanout(Scheduler, TokenizerControlMixin):
    """Return private diagnostic replies from all three resident PP workers.

    Normal scheduler IPC and control communicators return only the first PP
    worker's reply per DP replica. Requests reach the next PP worker after the
    local handler returns, so a collective inside the handler would deadlock.
    This opt-in, single-node TP1/DP1 smoke uses independent reply sockets for
    only get/set internal state; inference and topology control keep their
    normal output route and fanout.
    """
    from dataclasses import replace

    import zmq

    from sglang.srt.managers.io_struct import (
        GetInternalStateReqOutput,
        SetInternalStateReqOutput,
    )
    from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
    from sglang.srt.utils.network import get_zmq_socket

    diagnostic_types = (GetInternalStateReqOutput, SetInternalStateReqOutput)

    class DiagnosticSender(SenderWrapper):
        def __init__(self, normal, diagnostic):
            super().__init__(normal.socket)
            self.normal = normal
            self.diagnostic = diagnostic

        def send_output(self, output, recv_obj=None):
            sender = (
                self.diagnostic if isinstance(output, diagnostic_types) else self.normal
            )
            return sender.send_output(output, recv_obj)

    original_ipc = Scheduler.init_ipc_channels

    def init_ipc(self, port_args):
        if (self.ps.pp_size, self.ps.tp_size, self.ps.dp_size) != (3, 1, 1):
            raise ValueError("private PP diagnostic replies require PP3/TP1/DP1")
        original_ipc(self, port_args)
        if self.ps.pp_rank == 0:
            return
        context = zmq.Context(1)
        try:
            sender = SenderWrapper(
                get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, False)
            )
            channels = replace(
                self.ipc_channels,
                send_to_tokenizer=DiagnosticSender(
                    self.ipc_channels.send_to_tokenizer, sender
                ),
            )
        except BaseException:
            context.destroy(linger=0)
            raise
        self._smoke_diagnostic_context = context
        self._smoke_diagnostic_socket = sender.socket
        self.ipc_channels = channels

    Scheduler.init_ipc_channels = init_ipc
    original_release = Scheduler.release_host_resources

    def release(self):
        try:
            return original_release(self)
        finally:
            socket = getattr(self, "_smoke_diagnostic_socket", None)
            if socket is not None:
                socket.close(linger=0)
                self._smoke_diagnostic_context.term()
                self._smoke_diagnostic_socket = None
                self._smoke_diagnostic_context = None

    Scheduler.release_host_resources = release
    original_communicators = TokenizerControlMixin.init_communicators

    def init_communicators(self, server_args):
        if (
            server_args.pp_size,
            server_args.tp_size,
            server_args.dp_size,
            server_args.nnodes,
        ) != (3, 1, 1, 1):
            raise ValueError(
                "private PP diagnostic replies require one-node PP3/TP1/DP1"
            )
        original_communicators(self, server_args)
        self.get_internal_state_communicator.set_fan_out(3)
        self.set_internal_state_communicator.set_fan_out(3)

    TokenizerControlMixin.init_communicators = init_communicators


def install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator):
    original_launch = Scheduler._record_batch_launch

    def launch(self, batch):
        result = original_launch(self, batch)
        observe_request_launch(self, batch)
        return result

    Scheduler._record_batch_launch = launch
    original_finalize = ElasticPPMigrationCoordinator.finalize_at_fence

    def finalize(self, publish):
        # The startup-only, empty-state Graph preparation walk has no cache
        # adapter or user requests. Only record public serving transitions.
        if self.cache_transition is None:
            return original_finalize(self, publish)
        scheduler = self.cache_transition.scheduler
        record = {
            "generation": self._transition.target.generation,
            "requests": [
                {
                    "rid": req.rid,
                    "output_length": len(req.output_ids),
                    "retraction_count": req.retraction_count,
                    "is_retracted": req.is_retracted,
                    "execution": dict(
                        getattr(scheduler, "_smoke_request_execution", {}).get(
                            req.rid, {}
                        )
                    ),
                }
                for req in scheduler._elastic_pp_live_requests()
                if not req.finished()
            ],
        }
        result = original_finalize(self, publish)
        self._smoke_fence_records = getattr(self, "_smoke_fence_records", []) + [record]
        return result

    ElasticPPMigrationCoordinator.finalize_at_fence = finalize
    previous_get = Scheduler.get_internal_state

    def get_state(self, *args, **kwargs):
        result = previous_get(self, *args, **kwargs)
        result.internal_state["transition_rank"] = self.ps.pp_rank
        coordinator = getattr(self, "elastic_pp_migration_coordinator", None)
        if coordinator is not None:
            result.internal_state["transition_fence_records"] = getattr(
                coordinator, "_smoke_fence_records", []
            )
            result.internal_state["transition_metrics"] = (
                coordinator.transition_metrics()
            )
        result.internal_state["transition_request_execution_scope"] = (
            "active model-launch batches; zero-layer passthrough excluded"
        )
        result.internal_state["transition_request_execution"] = {
            rid: dict(value)
            for rid, value in getattr(self, "_smoke_request_execution", {}).items()
        }
        result.internal_state["transition_active_output_lengths"] = sorted(
            len(req.output_ids)
            for req in self._elastic_pp_live_requests()
            if not req.finished()
        )
        return result

    Scheduler.get_internal_state = get_state


if os.environ.get("PP_HICACHE_SMOKE_ENGINE") == "1":
    install_hooks()


def engine_entry(cfg, row):
    import sglang.srt.server_args as module

    prepare = module.prepare_server_args

    def args(argv):
        return prepare(
            argv
            + (
                [
                    "--enable-elastic-pp-kv-patch",
                    "--elastic-pp-kv-patch-transport",
                    cfg.get("kv_patch_transport", "cpu"),
                ]
                if cfg.get("live_kv_patch")
                else []
            )
            + (
                ["--cuda-graph-bs-decode", "1", "2", "4"]
                if cfg.get("cuda_graph")
                else ["--disable-cuda-graph"]
            )
            + (
                ["--elastic-pp-profile-weight-load"]
                if cfg.get("profile_native_weight_load")
                else []
            )
            + (
                ["--elastic-pp-hicache-fast-copy"]
                if cfg.get("hicache_fast_copy")
                else []
            )
            + (
                [
                    "--elastic-pp-graph-partitions",
                    ";".join(
                        ",".join(map(str, counts))
                        for counts in (cfg["initial_counts"], cfg["grown_counts"])
                    ),
                    "--elastic-pp-prepared-weight-policy",
                    cfg.get("prepared_weight_policy", "offload"),
                    "--elastic-pp-hicache-allocator",
                    cfg.get("hicache_allocator", "mmap"),
                ]
                if cfg.get("prepared_graphs")
                else []
            )
            + [
                "--enable-elastic-pp",
                "--elastic-pp-initial-partition",
                cfg["initial_partition"],
                "--elastic-pp-transition-strategy",
                cfg["strategy"],
                "--elastic-pp-weight-staging",
                cfg.get("weight_staging", "legacy"),
                "--elastic-pp-cpu-weight-cache-gib",
                str(cfg.get("cpu_weight_cache_gib", 24)),
                "--elastic-pp-transition-backing",
                cfg.get("backing", "overlap"),
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--max-mamba-cache-size",
                "16",
                "--context-length",
                str(cfg.get("context_length", 2048)),
                "--hicache-size",
                "1",
                "--chunked-prefill-size",
                "1024",
            ]
        )

    module.prepare_server_args = args
    base.engine_entry(cfg, row)


def launch(run, cfg, row, key):
    # Reuse exact-owned PGID/log attestation, changing only the entrypoint.
    original_environment, original_file = base.environment, base.__file__

    def environment(cfg, row, run):
        env = original_environment(cfg, row, run)
        env.update(INTEGRATION_SERVER="0", PP_HICACHE_SMOKE_ENGINE="1")
        if cfg.get("kv_patch_transport") == "nixl":
            env["PYTHONPATH"] = cfg["nixl_root"] + os.pathsep + env["PYTHONPATH"]
            env.update(UCX_PROTO_INFO="y", UCX_LOG_LEVEL="info")
        if cfg.get("baseline_probe"):
            env.update(
                PP_HICACHE_BASELINE_PROBE="1",
                PP_HICACHE_PROBE_DIR=str(run / "baseline_probe"),
            )
        return env

    base.environment, base.__file__ = environment, __file__
    try:
        return base.launch(run, cfg, row, key)
    finally:
        base.environment, base.__file__ = original_environment, original_file


def launch_foundation_probe(run, cfg, row, key):
    """Own the primitive probe under the same exact-node experiment lease."""
    env = base.environment(cfg, row, run)
    env.update(
        CUDA_VISIBLE_DEVICES="3",
        PYTHONPATH=os.pathsep.join(
            [str(ENGINE / "python"), cfg["kv_foundation_probe"]["root"]]
        ),
        KVCACHED_LAYER_STACKING="true",
        KVCACHED_PAGE_PREALLOC_ENABLED="false",
    )
    proc = subprocess.Popen(
        [cfg["python"], str(ROOT / "kv_foundation_gpu_probe.py"), "--run", str(run)],
        env=env,
        cwd=ENGINE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    identity = h.process_record(proc.pid)
    if not identity or identity["pgid"] != proc.pid or identity["session"] != proc.pid:
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("Could not establish owned primitive-probe process group")
    thread = threading.Thread(
        target=h.stream_log,
        args=(proc.stdout, str(run / "component_gpu.log"), key),
        daemon=True,
    )
    thread.start()
    return dict(process=proc, identity=identity, log_thread=thread)


def worker(run, cfg):
    if cfg.get("baseline_probe"):
        from hicache_baseline_probe import worker as baseline_worker

        return baseline_worker(run, cfg)
    import agentx_replay

    active_tokens = cfg.get("active_output_tokens", 256)

    authentication_started = time.time()
    authentication = h.authenticate(run, cfg, os.environ, local=True)
    key = h.read_key(cfg)
    row = cfg["layouts"][0]
    started = time.monotonic()
    children, evidence = (
        [],
        {
            "strategy": cfg["strategy"],
            "backing": cfg.get("backing", "overlap"),
            "passed": False,
            "resident_ranks": 3,
            "gpu_released": False,
            "performance_sample": bool(cfg.get("overhead_sample", False)),
            "cuda_graph_enabled": bool(cfg.get("cuda_graph", False)),
            "diagnostic_reply_scope": "private one-node PP3/TP1/DP1 get/set state replies",
            "acceptance_class": cfg.get(
                "acceptance_class", "cold_warm_exact_generation"
            ),
            "cold_vs_warm_generation_qualified": False,
            "generation_qualified": False,
            "cached_execution_continuity_qualified": False,
        },
    )
    original_stream = agentx_replay.stream
    request_contracts = []
    owned_cleanup = None
    lifecycle = diagnostics = None
    phase_timings = [
        {
            "phase": "initial_authentication",
            "started_at": authentication_started,
            "elapsed_s": time.time() - authentication_started,
        }
    ]
    if cfg.get("lifecycle"):
        from hicache_run_lifecycle import LocalEvidence, WorkerLifecycle, atomic_json

        diagnostics = LocalEvidence(run)
        lifecycle = WorkerLifecycle(run, cfg, authentication, children, h.cleanup)
        atomic_json(
            run / "local_evidence_location.json",
            {
                "experiment_id": cfg["id"],
                "nonce": cfg["lifecycle"]["nonce"],
                "worker": lifecycle.worker["worker"],
                "path": str(diagnostics.root),
            },
        )

    def check_cancelled():
        if lifecycle is not None and lifecycle.cancelled.is_set():
            raise RuntimeError("This run was cancelled; owned cleanup is in progress")

    def cleanup_owned():
        nonlocal owned_cleanup
        if owned_cleanup is None or not owned_cleanup.get("cleanup_verified", False):
            owned_cleanup = (
                lifecycle.cleanup() if lifecycle is not None else h.cleanup(children)
            )
        return owned_cleanup

    def measured_stream(endpoint, key, payload, timeout):
        if cfg.get("migration_profile"):
            events = []
            result = original_stream(
                endpoint, key, payload, min(timeout, 45), event_sink=events
            )
            result["completion_events"] = events
            result["completion_event_scope"] = (
                "client arrival time in ms since request; cumulative token count; "
                "multi-token chunks do not expose individual token times"
            )
            return result
        return original_stream(endpoint, key, payload, min(timeout, 45))

    def captured_stream(endpoint, key, payload, timeout=600):
        check_cancelled()
        request_contracts.append(
            {"stage": evidence.get("stage"), "payload": json.loads(json.dumps(payload))}
        )
        contract = {
            "capture": "actual pre-serialization stream payload",
            "requests": request_contracts,
        }
        if diagnostics is not None:
            diagnostics.save("reference_request_contract.json", contract)
            return measured_stream(endpoint, key, payload, timeout)
        target = run / "reference_request_contract.json"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                contract,
                stream,
            )
        return measured_stream(endpoint, key, payload, timeout)

    agentx_replay.stream = captured_stream

    def expired(*_):
        raise TimeoutError(
            "225-second transition deadline; 75 seconds reserved for cleanup"
        )

    if lifecycle is None:
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(225)

    def checkpoint(stage):
        evidence["stage"] = stage
        phase_timings.append({"phase": stage, "started_at": time.time()})
        try:
            value = {
                "stage": stage,
                "timestamp": time.time(),
                "elapsed_seconds": time.monotonic() - started,
            }
            if diagnostics is not None:
                diagnostics.save("worker_stage.json", value)
            else:
                h.save(run / "worker_stage.json", value)
        except OSError:
            # Diagnostic output must never prevent owned-process cleanup.
            pass

    def request(tokens, output):
        check_cancelled()
        result = agentx_replay.request_one(
            dict(host="127.0.0.1", port=row["port"], ep=1), None, key, tokens, output, 0
        )
        if len(result.get("output_ids") or ()) != output:
            raise AssertionError("generated token IDs missing or incomplete")
        return result

    def http(path, payload=None, timeout=30):
        check_cancelled()
        auth_started = time.time()
        h.authenticate(run, cfg, os.environ, local=True)
        phase_timings.append(
            {
                "phase": "control_authentication",
                "path": path,
                "started_at": auth_started,
                "elapsed_s": time.time() - auth_started,
            }
        )
        result = h.http_json(row["port"], path, key, payload, timeout=timeout)
        return state_evidence(result) if path == "/get_server_info" else result

    try:
        if evidence["acceptance_class"] == "cached_execution_continuity":
            from hicache_reference_qualification import (
                qualify_cached_reference_evidence,
            )

            qualification_started = time.time()
            evidence["reference_state_review"] = qualify_cached_reference_evidence(
                cfg["cached_reference_evidence"], cfg
            )
            phase_timings.append(
                {
                    "phase": "complete_reference_qualification",
                    "started_at": qualification_started,
                    "elapsed_s": time.time() - qualification_started,
                }
            )
        elif evidence["acceptance_class"] == "same_process_warm_execution_continuity":
            if not cfg.get("prepared_graphs"):
                raise AssertionError(
                    "same-process warm protocol requires prepared Graphs"
                )
            evidence["reference_scope"] = (
                "Fresh same-version warmed reference; exact transition continuity only. "
                "No prior state-review receipt or cold/warm generation qualification."
            )
        elif evidence["acceptance_class"] != "cold_warm_exact_generation":
            raise AssertionError("unknown reference acceptance class")
        if cfg.get("kv_foundation_probe"):
            if lifecycle is None:
                raise RuntimeError("The foundation probe requires supervised cleanup")
            checkpoint("component_gpu_probe")
            child = lifecycle.launch(
                lambda: launch_foundation_probe(run, cfg, row, key)
            )
            deadline = time.monotonic() + 45
            while child["process"].poll() is None:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Primitive GPU probe exceeded 45 seconds")
                time.sleep(0.1)
            child["log_thread"].join(timeout=2)
            if child["process"].returncode != 0:
                raise RuntimeError("Primitive GPU probe failed; see component_gpu.log")
            component = json.loads((run / "component_gpu.json").read_text())
            if not component.get("passed"):
                raise AssertionError("Primitive GPU acceptance was not recorded")
            evidence["component_gpu"] = component
        checkpoint("startup")
        if lifecycle is None:
            engine_child = launch(run, cfg, row, key)
            children.append(engine_child)
        else:
            engine_child = lifecycle.launch(lambda: launch(run, cfg, row, key))
        while True:
            if engine_child["process"].poll() is not None:
                raise RuntimeError("engine exited during startup")
            try:
                http("/get_server_info", timeout=2)
                break
            except (OSError, ValueError):
                time.sleep(0.25)
        prefix, active_prompt, common_prefix_length = correctness_prompts()
        evidence["workload"] = {
            "common_prefix_tokens": common_prefix_length,
            "fixed_continuation_tokens": 1,
            "prefix_input_tokens": len(prefix),
            "active_input_tokens": len(active_prompt),
            "active_output_tokens": active_tokens,
            "reference_cache_state": "warmed at the same prefix-match boundary",
        }
        if lifecycle is not None:
            lifecycle.admit("references", 150 if cfg.get("overhead_sample") else 105)
        checkpoint("prefix_warmup")
        cold_prefix = request(prefix, 24)
        checkpoint("prefix_reference")
        reference = request(prefix, 24)
        checkpoint("active_warmup")
        cold_active = request(active_prompt, active_tokens)
        checkpoint("active_reference")
        active_reference = request(active_prompt, active_tokens)
        evidence["references"] = {
            "cold_prefix": cold_prefix,
            "prefix": reference,
            "cold_active": cold_active,
            "active": active_reference,
        }
        evidence["reference_qualification"] = {
            "prefix": reference_qualification(cold_prefix, reference),
            "active": reference_qualification(cold_active, active_reference),
        }
        cold_warm_equal = all(
            item["qualified"] for item in evidence["reference_qualification"].values()
        )
        evidence["observed_cold_warm_exact_ids_equal"] = cold_warm_equal
        if evidence["acceptance_class"] in (
            "cached_execution_continuity",
            "same_process_warm_execution_continuity",
        ):
            for warmed_reference in (reference, active_reference):
                if (
                    warmed_reference.get("meta_info", {}).get("cached_tokens") != 512
                    or warmed_reference.get("meta_info", {}).get("num_retractions") != 0
                ):
                    raise AssertionError(
                        "warmed reference did not use the reviewed boundary"
                    )
        elif not cold_warm_equal:
            raise AssertionError(
                "cold/warm reference output IDs differ before transition"
            )
        else:
            evidence["cold_vs_warm_generation_qualified"] = True
            evidence["generation_qualified"] = True

        checkpoint("initial_cpu_demotion")
        evidence["initial_demotion"] = demote_cpu_prefixes(http)
        evidence["before"] = http("/get_server_info")
        evidence["transitions"] = []
        for generation, counts in enumerate(
            (cfg["grown_counts"], cfg["initial_counts"]), 1
        ):
            baseline = None
            if cfg.get("overhead_sample"):
                checkpoint(f"overhead_reference_{generation}")
                lifecycle.admit(
                    f"overhead_reference_{generation}", 105 if generation == 1 else 52
                )
                baseline = request(active_prompt, active_tokens)
                evidence.setdefault("overhead_baselines", []).append(baseline)
                if baseline["output_ids"] != active_reference["output_ids"]:
                    raise AssertionError(
                        "no-transition baseline changed warmed output IDs"
                    )
                if generation == 1:
                    evidence["paired_demotion"] = demote_cpu_prefixes(http)
            checkpoint(f"transition_{generation}")
            if lifecycle is not None:
                lifecycle.admit(
                    f"transition_{generation}",
                    (
                        (80 if cfg.get("overhead_sample") else 60)
                        if generation == 1
                        else 32
                    ),
                )
            with cf.ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    active = pool.submit(request, active_prompt, active_tokens)
                    deadline = time.monotonic() + 15
                    while True:
                        active_info = http("/get_server_info", timeout=3)
                        lengths = [
                            length
                            for state in active_info["internal_states"]
                            for length in state.get(
                                "transition_active_output_lengths", []
                            )
                        ]
                        if any(length >= 4 for length in lengths):
                            break
                        if active.done() or time.monotonic() >= deadline:
                            raise RuntimeError(
                                "no live Decode progress observed before transition"
                            )
                        time.sleep(0.02)
                    control_started = time.perf_counter()
                    transition = http(
                        "/reconfigure_elastic_pp",
                        {"layer_counts": counts, "generation": generation},
                        timeout=60,
                    )
                    record = {
                        "generation": generation,
                        "control": transition,
                        "control_returned_at": time.time(),
                        "continuation_verified": False,
                    }
                    if cfg.get("migration_profile"):
                        record.update(
                            control_started_perf_counter=control_started,
                            control_finished_perf_counter=time.perf_counter(),
                        )
                    evidence["transitions"].append(record)
                    if not transition.get("success"):
                        raise RuntimeError("topology transition failed")
                    checkpoint(f"transition_{generation}_resume")
                    result = active.result(timeout=45)
                    record["active"] = result
                except BaseException:
                    # Executor exit waits for HTTP: stop this run's model first.
                    for _ in range(2):
                        try:
                            if cleanup_owned().get("cleanup_verified", False):
                                break
                        except BaseException as cleanup_error:
                            evidence["early_cleanup_error"] = type(
                                cleanup_error
                            ).__name__
                    raise
            if result["output_ids"] != active_reference["output_ids"]:
                raise AssertionError("active output IDs changed across transition")
            # Commit response precedes deferred cleanup; wait until next transition is admissible.
            deadline = time.monotonic() + 15
            while True:
                states = http("/get_server_info")["internal_states"]
                if all(
                    state["elastic_pp_state"]["migration_phase"] == "done"
                    for state in states
                ):
                    record["states"] = states
                    validate_transition_states(
                        states,
                        generation,
                        (0, 2) if generation == 1 else (0, 1, 2),
                        cfg.get("backing", "overlap"),
                    )
                    if cfg.get("cuda_graph"):
                        validate_decode_graph_states(
                            states,
                            generation,
                            (0, 2) if generation == 1 else (0, 1, 2),
                            (0, 1, 2) if generation == 1 else (0, 2),
                            prepared=cfg.get("prepared_graphs", False),
                        )
                        record["decode_graph_verified"] = True
                    if cfg.get("live_kv_patch"):
                        patches = [
                            state["transition_metrics"]
                            .get("kv", {})
                            .get("live_kv_patch", {})
                            for state in rank_states(states).values()
                        ]
                        copied = [p for p in patches if p.get("total_rows", 0)]
                        if not copied or any(
                            not p.get("complete") or p.get("error") for p in copied
                        ):
                            raise AssertionError("online KV snapshot was not completed")
                        if (
                            sum(p.get("online_submissions", 0) for p in copied) <= 0
                            or sum(p.get("online_completed_rows", 0) for p in copied)
                            <= 0
                        ):
                            raise AssertionError(
                                "no KV rows completed during source inference"
                            )
                        record["live_kv_patch_verified"] = True
                        if cfg.get("kv_patch_transport") == "nixl":
                            transports = [
                                s["transition_metrics"]["kv"]["nixl"]
                                for s in rank_states(states).values()
                            ]
                            if any(
                                not t.get("enabled")
                                or t["endpoint"].get("warm_channels") != 2
                                for t in transports
                            ):
                                raise AssertionError(
                                    "candidate NIXL endpoints were not prewarmed"
                                )
                            applied = sum(
                                t["receiver"].get("online_applied_rows", 0)
                                for t in transports
                            )
                            sealed = sum(
                                t["receiver"].get("sealed_sources", 0)
                                for t in transports
                            )
                            if applied <= 0 or sealed <= 0:
                                raise AssertionError(
                                    "no online GPU application or final NIXL seal"
                                )
                            record["nixl_verified"] = dict(
                                online_applied_rows=applied,
                                sealed_sources=sealed,
                                actual_rdma_qualified=False,
                            )
                    record["continuation"] = validate_uninterrupted_decode(
                        states,
                        generation,
                        result,
                        (0, 2) if generation == 1 else (0, 1, 2),
                        (0, 1, 2) if generation == 1 else (0, 2),
                    )
                    record["continuation_verified"] = True
                    if baseline is not None:
                        record["overhead"] = paired_overhead(
                            baseline, result, states, generation
                        )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("transition cleanup did not complete")
                time.sleep(0.05)
        fence_states = http("/get_server_info")["internal_states"]
        for generation, source_ranks in ((1, (0, 2)), (2, (0, 1, 2))):
            validate_fenced_decode(
                fence_states, generation, source_ranks, active_tokens
            )
        evidence["fence_states"] = fence_states
        checkpoint("final_cpu_demotion")
        evidence["final_demotion"] = demote_cpu_prefixes(http)
        checkpoint("final_prefix_restore")
        restore_start = time.time()
        restored = request(prefix, 24)
        if restored["output_ids"] != reference["output_ids"]:
            raise AssertionError("CPU-only prefix continuation changed")
        info = http("/get_server_info")
        restores = fresh_prefix_restores(
            info["internal_states"], restore_start, (0, 2), common_prefix_length
        )
        evidence["final_prefix_restore_events"] = restores
        evidence.update(passed=True, restored=restored, after=info)
        if lifecycle is not None:
            evidence["work_completed_at"] = lifecycle.complete_work()
    except BaseException as exc:
        evidence["passed"] = False
        evidence["failed_stage"] = evidence.get("stage", "startup")
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        agentx_replay.stream = original_stream
        if lifecycle is None:
            signal.alarm(0)
        checkpoint("cleanup")
        cleanup = cleanup_owned()
        if lifecycle is not None:
            evidence["stage_admissions"] = list(lifecycle.admissions)
        elapsed = time.monotonic() - started
        evidence.update(
            elapsed_seconds=elapsed,
            cleanup_verified=cleanup.get("cleanup_verified", False),
        )
        evidence["passed"] = (
            evidence["passed"]
            and cleanup.get("cleanup_verified", False)
            and elapsed <= 300
        )
        evidence["cached_execution_continuity_qualified"] = (
            evidence["passed"]
            and evidence["acceptance_class"] == "cached_execution_continuity"
        )
        evidence["stage"] = "done" if evidence["passed"] else "failed"
        if diagnostics is not None:
            diagnostics.save("phase_timings.json", phase_timings)
            try:
                evidence["local_diagnostic_sha256"] = diagnostics.archive()
            except BaseException as exc:
                evidence.update(passed=False, archive_error=type(exc).__name__)
        else:
            h.save(run / "worker_cleanup.json", cleanup)
        evidence["cached_execution_continuity_qualified"] = (
            evidence["passed"]
            and evidence["acceptance_class"] == "cached_execution_continuity"
        )
        evidence["same_process_warm_continuity_verified"] = (
            evidence["passed"]
            and evidence["acceptance_class"] == "same_process_warm_execution_continuity"
        )
        h.save(run / "results.json", {"transition": evidence})
        h.save(run / "worker_progress.json", {"completed": int(evidence["passed"])})
        if lifecycle is not None:
            lifecycle.close()
    raise SystemExit(0 if evidence["passed"] else 1)


def prepare(args):
    # Freeze audited node cache and Guard sources using the existing helper.
    if args.kv_foundation_root and not args.supervised_lifecycle:
        raise ValueError("The foundation probe requires supervised cleanup")
    args.cases = "pp2"
    base.prepare(args)
    run = args.run.resolve()
    cfg = json.loads((run / "config.json").read_text())
    model = json.loads((Path(cfg["model_path"]) / "config.json").read_text())
    layers = model.get("text_config", model)["num_hidden_layers"]
    split = layers // 2
    initial = [split, 0, layers - split]
    third = layers // 3
    grown = [third, third, layers - 2 * third]
    row = cfg["layouts"][0]
    row.update(id="transition", pp=3, gpus=[0, 1, 2], port=24831, dist_port=24931)
    (run / "pp2").rename(run / "transition")
    use = row["usage_logs"][0]
    use.update(
        id=run.name + ".transition", expected_ranks=[f"PP{i} TP0" for i in range(3)]
    )
    cfg.update(
        cases=["transition"],
        strategy=args.strategy,
        backing=args.backing,
        live_kv_patch=getattr(args, "live_kv_patch", False),
        kv_patch_transport=getattr(args, "kv_patch_transport", "cpu"),
        cuda_graph=getattr(args, "cuda_graph", False),
        prepared_graphs=getattr(args, "prepared_graphs", False),
        prepared_weight_policy=getattr(args, "prepared_weight_policy", "offload"),
        weight_staging=getattr(args, "weight_staging", "legacy"),
        cpu_weight_cache_gib=getattr(args, "cpu_weight_cache_gib", 24),
        profile_native_weight_load=getattr(args, "profile_native_weight_load", False),
        migration_profile=getattr(args, "migration_profile", False),
        hicache_allocator=getattr(args, "hicache_allocator", "mmap"),
        hicache_fast_copy=getattr(args, "hicache_fast_copy", False),
        active_output_tokens=getattr(args, "active_output_tokens", 256),
        context_length=(
            4096 if getattr(args, "active_output_tokens", 256) > 1024 else 2048
        ),
        nixl_root=(
            str(args.nixl_root.resolve()) if getattr(args, "nixl_root", None) else None
        ),
        initial_counts=initial,
        initial_partition=",".join(map(str, initial)),
        grown_counts=grown,
        engine_entrypoint=str(Path(__file__).resolve()),
        baseline_probe=args.baseline_probe,
        overhead_sample=getattr(args, "overhead_sample", False),
        acceptance_class=(
            "cached_execution_continuity"
            if args.cached_reference_evidence
            else (
                "same_process_warm_execution_continuity"
                if args.same_process_warm_reference
                else "cold_warm_exact_generation"
            )
        ),
    )
    if getattr(args, "supervised_lifecycle", False):
        if args.baseline_probe:
            raise ValueError(
                "Supervised cancellation is for the transition acceptance run"
            )
        cfg["lifecycle"] = dict(
            version=1,
            nonce=uuid.uuid4().hex,
            work_s=args.work_seconds,
            cleanup_s=args.cleanup_seconds,
            guard_stop_s=args.guard_stop_seconds,
            total_s=300,
        )
    if args.cached_reference_evidence:
        from hicache_reference_qualification import qualify_cached_reference_evidence

        cfg["cached_reference_evidence"] = json.loads(
            args.cached_reference_evidence.read_text()
        )
        qualify_cached_reference_evidence(cfg["cached_reference_evidence"], cfg)
    if args.kv_foundation_root:
        root = args.kv_foundation_root.resolve()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip()
        lock = json.loads((ROOT / "dependencies.lock.json").read_text())["kvcached"]
        if (
            tree != lock["patched_tree"]
            or subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
            ).strip()
        ):
            raise RuntimeError(
                "The primitive probe requires the locked clean dependency"
            )
        binaries = list((root / "kvcached").glob("vmm_ops*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Expected one pinned KVCacheD native extension")
        cfg["kv_foundation_probe"] = dict(
            root=str(root), tree=tree, native=base.desc(binaries[0])
        )
    cfg["model_cache"]["usage_logs"] = {args.node: [use]}
    if cfg["weight_staging"] == "cpu_prepacked":
        if not cfg["prepared_graphs"] or cfg["prepared_weight_policy"] != "offload":
            raise ValueError(
                "CPU prepacking requires startup Graphs and nonresident weights"
            )
        if cfg["cpu_weight_cache_gib"] <= 0:
            raise ValueError("CPU prepacking needs a positive per-rank cap")
        # This admission pool is a conservative combined upper bound, not a
        # larger HiCache allocation: the engine retains the same 1 GiB/rank
        # HiCache configuration plus the separately capped CPU weight buffers.
        cfg["prepared_cpu_weight_reserve_bytes"] = (
            3 * cfg["cpu_weight_cache_gib"] * 1024**3
        )
        cfg["model_cache"]["hicache_reserve_bytes_per_node"][args.node] += cfg[
            "prepared_cpu_weight_reserve_bytes"
        ]
    h.save(run / "config.json", cfg)
    spec = json.loads((run / "dashboard_spec.json").read_text())
    spec.update(
        title="Qwen HiCache elastic PP: resident PP2 to PP3 to PP2",
        model_cache=cfg["model_cache"],
        port=args.dashboard_port,
    )
    spec["tasks"][0].update(
        expected=1,
        label="HiCache active state + CPU-only prefixes",
        controller_script=str(Path(__file__).resolve()),
    )
    spec["frozen_inputs"] = [
        x for x in spec["frozen_inputs"] if x["path"] != str(run / "config.json")
    ]
    spec["frozen_inputs"] += [
        base.desc(run / "config.json"),
        base.desc(__file__),
        base.desc(ENGINE / "python/sglang/srt/elastic_pp/hicache_transition.py"),
    ]
    if getattr(args, "supervised_lifecycle", False):
        spec["frozen_inputs"] += [
            base.desc(ROOT / name)
            for name in ("hicache_run_lifecycle.py", "hicache_guard_run.py")
        ]
    if args.kv_foundation_root:
        spec["frozen_inputs"] += [
            base.desc(ROOT / "kv_foundation_gpu_probe.py"),
            *(
                base.desc(ENGINE / "python/sglang/srt/elastic_pp" / name)
                for name in ("graph_bank.py", "kv_patch.py")
            ),
        ]
    if cfg["live_kv_patch"]:
        spec["frozen_inputs"] += [
            base.desc(ENGINE / "python/sglang/srt/elastic_pp" / name)
            for name in ("kv_patch.py", "live_kv_patch.py")
        ]
    if cfg["weight_staging"] in {"cpu_cache", "cpu_prepacked"}:
        spec["frozen_inputs"] += [
            base.desc(ENGINE / "python/sglang/srt/weight_cache" / name)
            for name in (
                "cpu_canonical.py",
                "cpu_protocol.py",
                "cpu_canonical_loader.py",
            )
        ]
    if cfg["cuda_graph"]:
        spec["frozen_inputs"] += [
            base.desc(ENGINE / path)
            for path in (
                "python/sglang/srt/elastic_pp/decode_graph.py",
                "python/sglang/srt/elastic_pp/coordinator.py",
                "python/sglang/srt/elastic_pp/kv_transport.py",
                "python/sglang/srt/model_executor/model_runner.py",
                "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
                "python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py",
                "python/sglang/srt/model_executor/runner_utils/pool.py",
                "python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py",
                "python/sglang/srt/elastic_pp/prepared_decode_graph.py",
                "python/sglang/srt/elastic_pp/weight_backing.py",
                "python/sglang/srt/elastic_pp/weight_staging.py",
                "python/sglang/srt/elastic_pp/cpu_weight_reload.py",
                "python/sglang/srt/elastic_pp/prepared_cpu_weights.py",
                "python/sglang/srt/cuda_vmm_utils.py",
                "python/sglang/srt/elastic_pp/model.py",
                "python/sglang/srt/elastic_pp/model_wrapper.py",
                "python/sglang/srt/mem_cache/memory_pool.py",
                "python/sglang/srt/mem_cache/elastic_hybrid_pool.py",
                "python/sglang/srt/managers/scheduler.py",
                "python/sglang/srt/server_args.py",
            )
        ]
    if cfg["kv_patch_transport"] == "nixl":
        spec["frozen_inputs"] += [
            base.desc(ENGINE / "python/sglang/srt/elastic_pp" / name)
            for name in ("nixl_transport.py", "nixl_kv_patch.py")
        ]
        root = Path(cfg["nixl_root"])
        for pattern in (
            "nixl/_api.py",
            "nixl/__init__.py",
            "nixl_cu12/_api.py",
            "nixl_cu12/_bindings*.so",
            "nixl_cu12.libs/libucp*.so*",
            "nixl_cu12.libs/nixl/libplugin_UCX.so",
        ):
            matched = list(root.glob(pattern))
            if not matched:
                raise ValueError(f"NIXL dependency fingerprint missing {pattern}")
            spec["frozen_inputs"] += [base.desc(path) for path in matched]
    if args.baseline_probe:
        spec["title"] = "Qwen HiCache static generation-zero qualification diagnostic"
        spec["tasks"][0]["label"] = "Static cold256/warm8 state and sampler diagnostic"
        spec["frozen_inputs"].append(base.desc(ROOT / "hicache_baseline_probe.py"))
    if args.cached_reference_evidence:
        spec["title"] = (
            "Qwen HiCache PP transitions: exact warmed-cache execution continuity"
        )
        spec["tasks"][0][
            "label"
        ] = "512-token warmed boundary; cold/warm generation unqualified"
        spec["frozen_inputs"] += [
            base.desc(ROOT / "hicache_reference_qualification.py"),
            base.desc(args.cached_reference_evidence),
            *(
                base.desc(Path(item["path"]))
                for item in cfg["cached_reference_evidence"]["evidence"].values()
            ),
            *(
                base.desc(ENGINE / item)
                for item in cfg["cached_reference_evidence"]["production_sources"]
            ),
            base.desc(Path(cfg["model_path"]) / "config.json"),
        ]
    if args.same_process_warm_reference:
        spec["title"] = "Startup-prepared Decode Graphs: same-version warmed continuity"
        spec["tasks"][0][
            "label"
        ] = "PP2↔PP3; startup captures only; no cold/warm generation qualification"
    h.save(run / "dashboard_spec.json", spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine-entry", action="store_true")
    parser.add_argument("--layout")
    parser.add_argument("--baseline-probe", action="store_true")
    parser.add_argument("--overhead-sample", action="store_true")
    parser.add_argument("--kv-foundation-root", type=Path)
    parser.add_argument("--dashboard-port", type=int, default=24760)
    parser.add_argument("--cached-reference-evidence", type=Path)
    parser.add_argument("--supervised-lifecycle", action="store_true")
    parser.add_argument("--work-seconds", type=float, default=250)
    parser.add_argument("--cleanup-seconds", type=float, default=285)
    parser.add_argument("--guard-stop-seconds", type=float, default=290)
    parser.add_argument("--tag")
    parser.add_argument("--node", default="spartan-gpgpu168")
    parser.add_argument("--job", default="33463293")
    parser.add_argument(
        "--python",
        default="/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python",
    )
    parser.add_argument(
        "--guard-scripts",
        default="/home/bxb1/.codex/skills/llm-experiment-guard/scripts",
    )
    parser.add_argument("--backfill-state", required=False)
    parser.add_argument(
        "--node-spec",
        type=Path,
        default=Path(
            "/data/gpfs/projects/punim2715/vllm_workbench/artifacts/source_integration/sglang-pp-elastic-20261105/gpu-r9/cache/node.json"
        ),
    )
    parser.add_argument(
        "--strategy", choices=["sync", "background", "batched"], default="sync"
    )
    parser.add_argument("--live-kv-patch", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--prepared-graphs", action="store_true")
    parser.add_argument(
        "--prepared-weight-policy",
        choices=["offload", "reuse", "recopy"],
        default="offload",
    )
    parser.add_argument("--migration-profile", action="store_true")
    parser.add_argument(
        "--weight-staging",
        choices=["cpu_cache", "cpu_prepacked", "legacy", "pinned"],
        default="pinned",
    )
    parser.add_argument("--cpu-weight-cache-gib", type=int, default=24)
    parser.add_argument("--profile-native-weight-load", action="store_true")
    parser.add_argument("--hicache-fast-copy", action="store_true")
    parser.add_argument(
        "--hicache-allocator", choices=["mmap", "pytorch"], default="mmap"
    )
    parser.add_argument("--same-process-warm-reference", action="store_true")
    parser.add_argument("--active-output-tokens", type=int, default=256)
    parser.add_argument("--kv-patch-transport", choices=["cpu", "nixl"], default="cpu")
    parser.add_argument("--nixl-root", type=Path)
    parser.add_argument(
        "--backing", choices=["overlap", "cpu_spill"], default="overlap"
    )
    args = parser.parse_args()
    if not 256 <= args.active_output_tokens <= 2048:
        parser.error("active output tokens must be between 256 and 2048")
    if args.overhead_sample and args.active_output_tokens != 256:
        parser.error("the existing paired overhead protocol requires 256 output tokens")
    if args.cuda_graph and args.backing != "overlap":
        parser.error("Decode Graph requires overlap backing")
    if args.prepared_graphs and not args.cuda_graph:
        parser.error("prepared Graphs require --cuda-graph")
    if args.same_process_warm_reference and (
        not args.prepared_graphs
        or args.cached_reference_evidence
        or args.baseline_probe
    ):
        parser.error(
            "same-process warm reference is a separate prepared-Graph protocol"
        )
    if args.kv_patch_transport == "nixl" and (
        not args.live_kv_patch or not args.nixl_root
    ):
        parser.error("NIXL requires --live-kv-patch and an isolated --nixl-root")
    if args.live_kv_patch and (
        args.strategy != "background" or args.backing != "overlap"
    ):
        parser.error("live KV patch requires background/overlap")
    if args.overhead_sample and not (
        args.supervised_lifecycle
        and args.cached_reference_evidence
        and args.strategy == "sync"
        and args.backing == "cpu_spill"
    ):
        parser.error(
            "overhead sample requires supervised sync/cpu_spill and reviewed warm evidence"
        )
    if args.baseline_probe and args.cached_reference_evidence:
        parser.error(
            "static diagnostic and cached execution continuity are separate classes"
        )
    if args.backing == "cpu_spill" and args.strategy == "background":
        parser.error("CPU-spill backing requires sync or batched transition")
    if args.prepare:
        prepare(args)
        return
    cfg = json.loads((args.run / "config.json").read_text())
    if args.engine_entry:
        engine_entry(cfg, cfg["layouts"][0])
    elif args.worker:
        worker(args.run, cfg)
    else:
        if cfg.get("lifecycle"):
            from hicache_run_lifecycle import controller

            if base.git("rev-parse", "HEAD") != cfg["engine_commit"] or base.git(
                "status", "--porcelain", "--untracked-files=no"
            ):
                raise RuntimeError("Frozen source changed")
            raise SystemExit(controller(args.run, cfg, h, __file__))
        else:
            base.__file__ = __file__
            base.controller(args.run, cfg)


if __name__ == "__main__":
    main()
