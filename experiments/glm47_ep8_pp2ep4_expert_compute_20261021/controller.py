"""Guard child for the one-node, one-GPU EP4 local expert profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

from common import ROOT, save
from guard_authorization import (
    GUARD_KEYS,
    authorization_signature,
    close_remote_workers,
    process_ticks,
    validate_guard,
    visible_json,
)


def main(run: Path) -> int:
    config = json.loads((run / "config.json").read_text())
    receipt = validate_guard(
        run, config, os.environ, run / "visibility/controller", require_parent=True
    )
    node = config["nodes"][0]
    task = run / "profile/task"
    output = run / "profile"
    output.mkdir(exist_ok=True)
    cancelled = []
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda number, frame: cancelled.append(number))

    payload = {
        "guard_env": {key: os.environ[key] for key in GUARD_KEYS},
        "model_cache_bindings": os.environ["EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS"],
        "lease_id": receipt["lease_id"],
        "node": node,
        "arm": "profile",
        "capacity": config["cases"]["profile"]["ep_capacity"],
        "node_rank": 0,
        "controller_pid": os.getpid(),
        "controller_start_ticks": process_ticks(os.getpid()),
        "controller_host": socket.gethostname(),
    }
    command = [
        "python3",
        str(ROOT / "model_node.py"),
        "--run",
        str(run),
        "--arm",
        "profile",
        "--node-rank",
        "0",
        "--controller-stdin",
    ]
    passed = False
    error = None
    with (output / "remote.log").open("w") as log:
        process = subprocess.Popen(
            ["ssh", "-o", "ConnectTimeout=5", node, shlex.join(command)],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            process.stdin.write(
                json.dumps(
                    {
                        "payload": payload,
                        "signature": authorization_signature(
                            payload, receipt["token"]
                        ),
                    }
                )
                + "\n"
            )
            process.stdin.flush()
            deadline = time.monotonic() + 1100
            while True:
                if cancelled:
                    raise InterruptedError("Guard/user cancellation")
                if process.poll() is not None:
                    raise RuntimeError("Remote profile exited before success")
                if time.monotonic() > deadline:
                    raise TimeoutError("Expert profile exceeded its deadline")
                result_path = output / "result.json"
                if result_path.exists():
                    result = visible_json(
                        result_path, run / "visibility/controller", timeout=10
                    )
                    if result.get("passed") is not None:
                        if result.get("passed") is not True:
                            raise RuntimeError("Expert profile result failed")
                        if len(result.get("rows", [])) != config["expected_points"]:
                            raise RuntimeError("Expert profile point count differs")
                        passed = True
                        break
                time.sleep(0.5)
        except BaseException as exc:
            error = {"type": type(exc).__name__, "error": str(exc)}
            save(run / "failure.json", error)
        finally:
            save(output / "STOP", {"timestamp": time.time()})
            close_remote_workers([process])

    end = visible_json(
        output / "node0_end.json", run / "visibility/controller", timeout=30
    )
    cleaned = all(
        end.get(key) is True
        for key in ("worker_exited", "owned_processes_exited", "gpu_idle")
    )
    save(
        task / "progress.json",
        {
            "timestamp": time.time(),
            "stage": "complete" if passed and cleaned else "failed",
            "valid_points": config["expected_points"] if passed else 0,
            "note": "GLM EP8 versus PP2xEP4 equal-work compute",
            "failure": error,
        },
    )
    save(
        task / "cleanup.json",
        {
            "timestamp": time.time(),
            "passed": cleaned,
            "model_processes_exited": cleaned,
            "gpu_idle": end["gpu_idle"],
            "experiment_id": config["id"],
            "task_id": "profile",
            "evidence": [end],
        },
    )
    return 0 if passed and cleaned else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().run.resolve()))
