"""One real, full80 EP8 streaming load plus finite-output functional check."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from stream_common import ROOT, OLD, config, environment, save, expected_ranks
from stream_auth import authenticate
from timing import extract_weight_load

BOOTSTRAP = "import json,os,sys\np=json.loads(sys.stdin.readline())\nos.environ.update(p['env'])\nos.execv(p['command'][0],p['command'])\n"


def call(url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def probe(url, rank):
    payload = {"text": "Hello! Please introduce yourself briefly.",
               "sampling_params": {"max_new_tokens": 4, "temperature": 0, "ignore_eos": True},
               "return_logprob": True, "logprob_start_len": -1, "data_parallel_rank": rank}
    value = call(url + "/generate", payload, timeout=180)
    meta = value.get("meta_info", {})
    probs = meta.get("output_token_logprobs", [])
    finite = bool(probs) and all(isinstance(row[0], (int, float)) and math.isfinite(row[0]) for row in probs)
    return {"request": payload, "response": value,
            "passed": meta.get("completion_tokens") == 4 and finite,
            "timing_kind": "functional_probe_not_performance_benchmark"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    run = parser.parse_args().run
    cfg = config(run)
    authenticate(run, cfg)
    keys = ("EXPERIMENT_GUARD_LEASE_RECEIPT", "EXPERIMENT_GUARD_ALLOCATION_FILE",
            "EXPERIMENT_GUARD_ACQUIRED_NODES", "EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS")
    guard_env = {key: os.environ[key] for key in keys}
    children, cleanups, results, timings, failure = [], [], [], [], None
    cancelled = {"signal": 0}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, _frame: cancelled.update(signal=number))
    started, started_epoch, ready_elapsed = time.monotonic(), time.time(), None
    save(run / "task/progress.json", {"timestamp": started_epoch, "stage": "loading",
                                      "controller_pid": os.getpid(), "valid_points": 0,
                                      "process_start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                                      "request_completed": 0, "request_total": 8, "progress_percent": 0})
    try:
        for rank, node in enumerate(cfg["nodes"]):
            command = [cfg["python"], str(Path(cfg["guard_scripts"]) / "model_cache_adapter.py"), "exec-local",
                       "--bindings", guard_env["EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS"], "--",
                       cfg["python"], str(ROOT / "runtime/model_node.py"), "--run", str(run), "--node-rank", str(rank)]
            log = (run / cfg["layout"] / (node + ".ssh.log")).open("x")
            proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
                                     shlex.join(["/usr/bin/python3", "-c", BOOTSTRAP])],
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, text=True)
            children.append((node, proc, log))
            proc.stdin.write(json.dumps({"env": environment(cfg, rank, guard_env), "command": command}) + "\n")
            proc.stdin.flush()
        url = f"http://{cfg['nodes'][0]}:{cfg['api_port']}"
        while True:
            if cancelled["signal"] or any(proc.poll() is not None for _, proc, _ in children):
                raise RuntimeError("Cancelled or worker exited before readiness")
            authenticate(run, cfg, frozen=False)
            try:
                call(url + "/health_generate", timeout=5)
                ready_elapsed = time.monotonic() - started
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            if time.monotonic() - started >= cfg["timeout_s"]:
                raise TimeoutError("Streaming model did not become ready")
            time.sleep(2)
        capacity_dir=run/cfg['layout']/'capacity'
        deadline=time.monotonic()+120
        while len(list(capacity_dir.glob('rank*.json'))) != 8:
            if any(p.poll() is not None for _,p,_ in children): raise RuntimeError('Worker exited during capacity collection')
            if time.monotonic()>deadline: raise TimeoutError('Missing capacity receipts')
            time.sleep(.5)
        results=[json.loads((capacity_dir/f'rank{rank}.json').read_text()) for rank in range(8)]
        save(run/cfg['layout']/'capacities.json',results)
        from measure import measure_all
        results=measure_all(url,run,cfg)
    except BaseException as exc:
        failure = {"exception": type(exc).__name__, "message": str(exc)}
    finally:
        for _, proc, _ in children:
            if proc.poll() is None:
                try:
                    proc.stdin.write("stop\n"); proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        sys.path.append(str(Path(cfg["guard_scripts"])))
        from artifact_visibility import wait_for_stable_json
        for node, proc, log in children:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            log.close()
            try:
                row, visibility = wait_for_stable_json(run / cfg["layout"] / node / "exit.json", timeout_s=20)
                cleanups.append(row)
                save(run / cfg["layout"] / (node + ".exit_visibility.json"), visibility)
            except Exception as exc:
                cleanups.append({"node": node, "owned_children_exited": False, "error": str(exc)})
        for rank, node in enumerate(cfg["nodes"]):
            try:
                metric = extract_weight_load((run / cfg["layout"] / node / "server.log").read_text(), expected_ranks(rank))
                timings.append({"node": node, **metric})
            except (OSError, ValueError) as exc:
                failure = failure or {"message": "Incomplete common load timing", "node": node, "error": str(exc)}
    clean = len(cleanups) == 2 and all(row.get("owned_children_exited") and row.get("gpu_cleanup", {}).get("passed") for row in cleanups)
    passed = failure is None and len(results) == 2*len(cfg["prefix_lengths"]) and clean
    save(run / "task/results.json", {"passed": passed, "failure": failure, "results": results,
         "full_model_generation_completed": passed, "capacity_calibration_completed": (len(list((run/cfg["layout"]/"capacity").glob("rank*.json"))) == 8), "inference_performance_completed": passed, "effective_num_hidden_layers": 80,
         "weight_source": "hy3_bounded_streaming", "persistent_weight_banks_modified": False,
         "performance_kind": "cached_native_hicache_inference",
         "model_runner_weight_load": timings,
         "slowest_rank_weight_load_s": max((row["slowest_rank_s"] for row in timings), default=None),
         "process_launch_to_api_ready_s": ready_elapsed, "launch_epoch": started_epoch,
         "guard_postcheck_required": True})
    save(run / "task/progress.json", {"timestamp": time.time(), "stage": "complete" if passed else "failed",
                                      "valid_points": len(results),
                                      "request_completed": len(results), "request_total": 8,
                                      "progress_percent": 100 if passed else 0})
    save(run / "task/cleanup.json", {"timestamp": time.time(), "experiment_id": cfg["id"], "task_id": "streaming",
                                    "passed": clean, "gpu_idle": clean, "model_processes_exited": clean,
                                    "gpu_resources_released": clean, "nodes": cleanups})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
