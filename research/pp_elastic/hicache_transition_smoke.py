"""Guard-only <=300s Qwen HiCache PP2 -> PP3 -> PP2 correctness check.

Shares the audited CPU-cache/owned-process launch helpers with gpu_smoke.
This is not a performance benchmark or a toy-model substitution.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import gpu_smoke as base

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
h = base.h
sys.path.insert(0, str(ROOT / "runtime_profiles/agentx_a100"))


def rank_states(states):
    ranks = {state["transition_rank"]: state for state in states}
    if len(states) != 3 or set(ranks) != {0, 1, 2}:
        raise AssertionError("transition evidence must contain each resident rank once")
    return ranks


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
    from sglang.srt.managers.scheduler import Scheduler

    from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator

    install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator)


def install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator):
    original_finalize = ElasticPPMigrationCoordinator.finalize_at_fence

    def finalize(self, publish):
        scheduler = self.cache_transition.scheduler
        record = {
            "generation": self._transition.target.generation,
            "requests": [
                {"rid": req.rid, "output_length": len(req.output_ids)}
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
            + [
                "--enable-elastic-pp",
                "--elastic-pp-initial-partition",
                cfg["initial_partition"],
                "--elastic-pp-transition-strategy",
                cfg["strategy"],
                "--elastic-pp-transition-backing",
                cfg.get("backing", "overlap"),
                "--disable-cuda-graph",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--max-mamba-cache-size",
                "16",
                "--context-length",
                "2048",
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
        return env

    base.environment, base.__file__ = environment, __file__
    try:
        return base.launch(run, cfg, row, key)
    finally:
        base.environment, base.__file__ = original_environment, original_file


def worker(run, cfg):
    import agentx_replay

    h.authenticate(run, cfg, os.environ, local=True)
    key = h.read_key(cfg)
    row = cfg["layouts"][0]
    started = time.monotonic()
    children, evidence = [], {
        "strategy": cfg["strategy"],
        "backing": cfg.get("backing", "overlap"),
        "passed": False,
        "resident_ranks": 3,
        "gpu_released": False,
        "performance_sample": False,
        "cuda_graph_enabled": False,
    }
    original_stream = agentx_replay.stream
    agentx_replay.stream = lambda endpoint, key, payload, timeout=600: original_stream(
        endpoint, key, payload, min(timeout, 45)
    )

    def expired(*_):
        raise TimeoutError(
            "165-second workload deadline; cleanup reserved inside 300 seconds"
        )

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(165)

    def checkpoint(stage):
        evidence["stage"] = stage
        try:
            h.save(
                run / "worker_stage.json",
                {
                    "stage": stage,
                    "timestamp": time.time(),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        except OSError:
            # Diagnostic output must never prevent owned-process cleanup.
            pass

    def request(tokens, output):
        result = agentx_replay.request_one(
            dict(host="127.0.0.1", port=row["port"], ep=1), None, key, tokens, output, 0
        )
        if len(result.get("output_ids") or ()) != output:
            raise AssertionError("generated token IDs missing or incomplete")
        return result

    def http(path, payload=None, timeout=30):
        h.authenticate(run, cfg, os.environ, local=True)
        return h.http_json(row["port"], path, key, payload, timeout=timeout)

    try:
        checkpoint("startup")
        children.append(launch(run, cfg, row, key))
        while True:
            if children[0]["process"].poll() is not None:
                raise RuntimeError("engine exited during startup")
            try:
                http("/get_server_info", timeout=2)
                break
            except (OSError, ValueError):
                time.sleep(0.25)
        prefix = [1000 + i % 160 for i in range(512)]
        active_prompt = [2000 + i % 160 for i in range(512)]
        checkpoint("prefix_reference")
        reference = request(prefix, 24)
        checkpoint("prefix_warmup")
        request(prefix, 24)
        checkpoint("active_reference")
        active_reference = request(active_prompt, 256)

        def demote():
            deadline = time.monotonic() + 15
            while True:
                answer = http(
                    "/set_internal_state", {"server_args": {"agentx_cache_cmd": 2}}
                )
                states = http("/get_server_info")["internal_states"]
                probes = [state.get("agentx_cache_probe", {}) for state in states]
                if all(
                    probe.get("ongoing_writes") == 0
                    and probe.get("after") == {"full": 0, "mamba": 0}
                    for probe in probes
                ):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU backup and GPU demotion did not finish")
                time.sleep(0.05)
            if (
                not isinstance(answer, list)
                or not answer
                or not all(x is True for x in answer)
            ):
                raise RuntimeError("idle CPU backup/demotion not acknowledged")

        checkpoint("initial_cpu_demotion")
        demote()
        evidence["before"] = http("/get_server_info")
        evidence["transitions"] = []
        for generation, counts in enumerate(
            (cfg["grown_counts"], cfg["initial_counts"]), 1
        ):
            checkpoint(f"transition_{generation}")
            with cf.ThreadPoolExecutor(max_workers=1) as pool:
                active = pool.submit(request, active_prompt, 256)
                deadline = time.monotonic() + 15
                while True:
                    active_info = http("/get_server_info", timeout=3)
                    lengths = [
                        length
                        for state in active_info["internal_states"]
                        for length in state.get("transition_active_output_lengths", [])
                    ]
                    if any(length >= 4 for length in lengths):
                        break
                    if active.done() or time.monotonic() >= deadline:
                        raise RuntimeError(
                            "no live Decode progress observed before transition"
                        )
                    time.sleep(0.02)
                transition = http(
                    "/reconfigure_elastic_pp",
                    {"layer_counts": counts, "generation": generation},
                    timeout=60,
                )
                if not transition.get("success"):
                    raise RuntimeError("topology transition failed")
                result = active.result(timeout=45)
            if result["output_ids"] != active_reference["output_ids"]:
                raise AssertionError("active output IDs changed across transition")
            evidence["transitions"].append(dict(control=transition, active=result))
            # Commit response precedes deferred cleanup; wait until next transition is admissible.
            deadline = time.monotonic() + 15
            while True:
                states = http("/get_server_info")["internal_states"]
                if all(
                    state["elastic_pp_state"]["migration_phase"] == "done"
                    for state in states
                ):
                    evidence["transitions"][-1]["states"] = states
                    validate_transition_states(
                        states,
                        generation,
                        (0, 2) if generation == 1 else (0, 1, 2),
                        cfg.get("backing", "overlap"),
                    )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("transition cleanup did not complete")
                time.sleep(0.05)
        fence_states = http("/get_server_info")["internal_states"]
        for generation, source_ranks in ((1, (0, 2)), (2, (0, 1, 2))):
            validate_fenced_decode(fence_states, generation, source_ranks, 256)
        evidence["fence_states"] = fence_states
        checkpoint("final_cpu_demotion")
        demote()
        checkpoint("final_prefix_restore")
        restore_start = time.time()
        restored = request(prefix, 24)
        if restored["output_ids"] != reference["output_ids"]:
            raise AssertionError("CPU-only prefix continuation changed")
        info = http("/get_server_info")
        restores = fresh_prefix_restores(
            info["internal_states"], restore_start, (0, 2), len(prefix)
        )
        evidence["final_prefix_restore_events"] = restores
        evidence.update(passed=True, restored=restored, after=info)
    except BaseException as exc:
        evidence["failed_stage"] = evidence.get("stage", "startup")
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        checkpoint("cleanup")
        cleanup = h.cleanup(children)
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
        evidence["stage"] = "done" if evidence["passed"] else "failed"
        h.save(run / "worker_cleanup.json", cleanup)
        h.save(run / "results.json", {"transition": evidence})
        h.save(run / "worker_progress.json", {"completed": int(evidence["passed"])})
    raise SystemExit(0 if evidence["passed"] else 1)


def prepare(args):
    # Freeze audited node cache and Guard sources using the existing helper.
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
        initial_counts=initial,
        initial_partition=",".join(map(str, initial)),
        grown_counts=grown,
        engine_entrypoint=str(Path(__file__).resolve()),
    )
    cfg["model_cache"]["usage_logs"] = {args.node: [use]}
    h.save(run / "config.json", cfg)
    spec = json.loads((run / "dashboard_spec.json").read_text())
    spec.update(
        title="Qwen HiCache elastic PP: resident PP2 to PP3 to PP2",
        model_cache=cfg["model_cache"],
    )
    spec["tasks"][0].update(
        expected=1,
        label="HiCache active state + CPU-only prefixes",
        controller_script=str(Path(__file__).resolve()),
    )
    spec["frozen_inputs"] = [
        x for x in spec["frozen_inputs"] if x["path"] != str(run / "config.json")
    ]
    spec["frozen_inputs"] += [base.desc(run / "config.json"), base.desc(__file__)]
    h.save(run / "dashboard_spec.json", spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine-entry", action="store_true")
    parser.add_argument("--layout")
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
    parser.add_argument(
        "--backing", choices=["overlap", "cpu_spill"], default="overlap"
    )
    args = parser.parse_args()
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
        base.__file__ = __file__
        base.controller(args.run, cfg)


if __name__ == "__main__":
    main()
