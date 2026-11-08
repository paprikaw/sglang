"""Task-local budget around an unchanged dashboard_guard run entrypoint."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from hicache_run_lifecycle import (
    atomic_json,
    load_budget,
    private_json,
    request_cancel,
    validate_ack,
)


def registry_owner(run, cfg):
    sys.path.insert(0, cfg["guard_scripts"])
    module = importlib.import_module("gpu_registry")
    receipt = private_json(Path(run) / "dashboard_guard/gpu_registry_lease.json")
    spec_hash = hashlib.sha256(
        (Path(run) / "dashboard_spec.json").read_bytes()
    ).hexdigest()
    if receipt["experiment_id"] != cfg["id"] or receipt["spec_sha256"] != spec_hash:
        raise RuntimeError("Refusing another run's Guard receipt")
    return module, module.SharedGpuRegistry(receipt["registry_dir"]), receipt


def lease_status(registry, receipt):
    state = json.loads(registry.state_path.read_text())
    matches = [
        row
        for key in ("leases", "history")
        for row in state.get(key, [])
        if row["lease_id"] == receipt["lease_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("Guard receipt has no unique registry state")
    row = matches[0]
    return {
        key: row[key]
        for key in (
            "lease_id",
            "state",
            "cleanup_verified",
            "ended_at",
            "cleanup_grace_expires_at",
        )
        if key in row
    }


def remote_ack(run, cfg, budget):
    _module, _registry, receipt = registry_owner(run, cfg)
    worker = private_json(Path(run) / "worker_identity.json")
    ack = private_json(Path(run) / "cleanup_ack.json")
    return validate_ack(ack, worker, cfg, budget, lease_id=receipt["lease_id"]), ack


def supervise(run, cfg, command, *, poll_s=0.1):
    """Cancel the run, not Guard; preserve the original atomic lease protocol."""
    run = Path(run)
    origin = time.time()
    atomic_json(
        run / "lifecycle_budget.json",
        {
            "experiment_id": cfg["id"],
            "nonce": cfg["lifecycle"]["nonce"],
            "policy": cfg["lifecycle"],
            "started_at": origin,
        },
    )
    budget = load_budget(run, cfg)
    cancelled = threading.Event()
    handlers = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        handlers[number] = signal.signal(number, lambda *_: cancelled.set())
    log = (run / "guard_console.log").open("x")
    os.chmod(run / "guard_console.log", 0o600)
    child = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    outcome = {
        "started_at": origin,
        "canonical_guard_unchanged": True,
        "command": command,
    }
    emergency_heartbeat = None
    owner = None
    cancel_written = False

    def cancel(reason):
        try:
            request_cancel(run, cfg, reason)
            outcome["cancel_requested_at"] = time.time()
            return True
        except OSError as error:
            # Preserve Guard's heartbeat and the absolute settlement deadline
            # even when the cancellation file cannot be persisted.
            outcome["cancel_request_error_type"] = type(error).__name__
            return False

    try:
        while True:
            now = time.time()
            if (
                cancelled.is_set() or now >= origin + cfg["lifecycle"]["work_s"]
            ) and not cancel_written:
                cancel_written = cancel(
                    (
                        "supervisor cancellation"
                        if cancelled.is_set()
                        else "work deadline"
                    ),
                )
            if child.poll() is not None:
                # If Guard itself failed, keep its authenticated lease alive
                # while the remote worker handles the same bounded cancellation.
                try:
                    owner = registry_owner(run, cfg)
                    module, registry, receipt = owner
                    row = lease_status(registry, receipt)
                    if row["state"] == "active":
                        emergency_heartbeat = module.LeaseHeartbeat(
                            registry, receipt
                        ).start()
                        cancel("Guard exited before lease settlement")
                        while time.time() < origin + cfg["lifecycle"]["cleanup_s"]:
                            try:
                                verified, ack = remote_ack(run, cfg, budget)
                                outcome["remote_acknowledgement"] = ack
                                break
                            except (OSError, ValueError, KeyError, RuntimeError):
                                time.sleep(poll_s)
                        else:
                            verified = False
                        registry.release(
                            receipt, verified, "task-local settlement after Guard exit"
                        )
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    outcome["settlement_error_type"] = type(error).__name__
                break
            if now >= origin + cfg["lifecycle"]["guard_stop_s"]:
                # Establish quarantine before stopping the only heartbeat owner.
                outcome["forced_stop"] = True
                try:
                    owner = registry_owner(run, cfg)
                    _module, registry, receipt = owner
                    row = lease_status(registry, receipt)
                    if row["state"] == "active":
                        registry.release(
                            receipt, False, "task-local absolute supervisor deadline"
                        )
                    outcome["forced_settlement"] = lease_status(registry, receipt)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    outcome["settlement_error_type"] = type(error).__name__
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)
                break
            time.sleep(poll_s)
    finally:
        if emergency_heartbeat is not None:
            emergency_heartbeat.stop()
            outcome["emergency_heartbeat_stopped"] = (
                not emergency_heartbeat.thread.is_alive()
            )
            outcome["emergency_heartbeat_error"] = (
                type(emergency_heartbeat.error).__name__
                if emergency_heartbeat.error
                else None
            )
        for number, handler in handlers.items():
            signal.signal(number, handler)
        log.close()
    outcome.update(
        guard_returncode=child.returncode,
        ended_at=time.time(),
        elapsed_s=time.time() - origin,
    )
    try:
        verified, ack = remote_ack(run, cfg, budget)
        outcome.update(remote_cleanup_verified=verified, remote_acknowledgement=ack)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        outcome.update(
            remote_cleanup_verified=False, remote_ack_error_type=type(error).__name__
        )
    try:
        _module, registry, receipt = owner or registry_owner(run, cfg)
        outcome["lease"] = lease_status(registry, receipt)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        outcome["lease_error_type"] = type(error).__name__
    outcome["normal_lease_release_verified"] = (
        outcome.get("lease", {}).get("state") == "released"
        and outcome.get("lease", {}).get("cleanup_verified") is True
    )
    outcome["passed"] = bool(
        child.returncode == 0
        and outcome["remote_cleanup_verified"]
        and outcome["normal_lease_release_verified"]
        and not outcome.get("forced_stop")
        and outcome["elapsed_s"] <= cfg["lifecycle"]["total_s"]
    )
    atomic_json(run / "supervisor_outcome.json", outcome)
    return 0 if outcome["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cfg = private_json(args.run / "config.json")
    guard = Path(cfg["guard_scripts"]) / "dashboard_guard.py"
    command = [
        sys.executable,
        str(guard),
        "run",
        "--spec",
        str(args.run / "dashboard_spec.json"),
        "--exit-after-run",
        "--",
        cfg["python"],
        str(Path(__file__).with_name("hicache_transition_smoke.py")),
        "--run",
        str(args.run),
    ]
    raise SystemExit(supervise(args.run, cfg, command))


if __name__ == "__main__":
    main()
