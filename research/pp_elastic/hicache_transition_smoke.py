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
        result.internal_state["transition_rank"] = self.pp_rank
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
        reference = request(prefix, 24)
        request(prefix, 24)
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

        demote()
        evidence["before"] = http("/get_server_info")
        evidence["transitions"] = []
        for generation, counts in enumerate(
            (cfg["grown_counts"], cfg["initial_counts"]), 1
        ):
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
                    if (
                        sum(
                            state.get("transition_metrics", {})
                            .get("host", {})
                            .get("host_bytes_sent", 0)
                            for state in states
                        )
                        <= 0
                    ):
                        raise AssertionError(
                            "no CPU-only host payload moved during transition"
                        )
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("transition cleanup did not complete")
                time.sleep(0.05)
        fence_states = http("/get_server_info")["internal_states"]
        for generation, source_ranks in ((1, (0, 2)), (2, (0, 1, 2))):
            for state in fence_states:
                if state["transition_rank"] not in source_ranks:
                    continue
                record = next(
                    r
                    for r in state["transition_fence_records"]
                    if r["generation"] == generation
                )
                if not any(0 < r["output_length"] < 256 for r in record["requests"]):
                    raise AssertionError("no active Decode state at actual fence")
        evidence["fence_states"] = fence_states
        demote()
        restore_start = time.time()
        restored = request(prefix, 24)
        if restored["output_ids"] != reference["output_ids"]:
            raise AssertionError("CPU-only prefix continuation changed")
        info = http("/get_server_info")
        restores = {
            state["transition_rank"]: [
                event
                for event in state.get("agentx_restores", [])
                if event["time"] >= restore_start
            ]
            for state in info["internal_states"]
        }
        if not all(
            any(
                event.get("loaded_tokens", 0) >= len(prefix)
                for event in restores.get(rank, [])
            )
            for rank in (0, 2)
        ):
            raise AssertionError(
                "final prefix lacks fresh CPU loadback on both active stages"
            )
        evidence["final_prefix_restore_events"] = restores
        evidence.update(passed=True, restored=restored, after=info)
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
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
    parser.add_argument("--guard-scripts", required=False)
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
    args = parser.parse_args()
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
