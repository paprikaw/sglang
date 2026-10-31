"""Own a subprocess before fallible model setup; cleanup is not scientific success.

This CPU-only helper does not acquire/release leases, alter registry state, load
models, evict caches, or signal processes outside the supplied owned PID tree.
The caller supplies the existing lease authentication and supervision backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time


def _write_new_json(path, value):
    """Publish once, atomically; never replace a failed or accepted receipt."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".exit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _control_reader():
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN | select.POLLHUP | select.POLLERR)

    def read():
        if not poll.poll(200):
            return None
        line = sys.stdin.readline()
        return line.strip() if line else "controller_eof"

    return read


def run_owned_worker(*, command, environment, exit_path, log_path, node,
                     layout, timeout_s, prepare, verify_alive, supervision,
                     control_reader=None):
    """Return (status, receipt), including post-exit proof on startup failures.

    ``prepare`` and model startup run inside the cleanup scope. Both successful
    and failed child exits stop the owned tree and obtain a fresh GPU query.
    A clean failed startup still returns a nonzero status. Unknown cleanup never
    becomes success. Existing receipts/logs are not overwritten.
    """
    exit_path, log_path = Path(exit_path), Path(log_path)
    if exit_path.exists() or exit_path.is_symlink():
        raise FileExistsError(exit_path)
    if timeout_s <= 0:
        raise ValueError("Positive owned-worker timeout required")
    owner = supervision.process_record(os.getpid())
    if owner is None:
        raise RuntimeError("Cannot authenticate owner PID/start ticks")
    started_at = time.time()
    deadline = time.monotonic() + timeout_s
    child, log, owned, failure = None, None, {}, None
    reason, phase = "", "owner_setup"
    interrupted = {"signal": None}
    previous_handlers = {}
    try:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[number] = signal.signal(
                number, lambda n, _frame: interrupted.update(signal=n))
        if not supervision.enable_subreaper():
            raise RuntimeError("Owned descendant cleanup requires subreaper")
        phase = "preparing"
        prepare()
        if interrupted["signal"]:
            raise InterruptedError("Interrupted before model launch")
        if time.monotonic() >= deadline:
            raise TimeoutError("Preparation exhausted worker deadline")
        read_control = control_reader if control_reader is not None else _control_reader()
        log = log_path.open("a")
        phase = "launching"
        child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        phase = "running"
        while True:
            supervision.track_descendants(owner["pid"], owner["ticks"], owned)
            if child.poll() is not None:
                reason = "child_exited"
                break
            if interrupted["signal"]:
                reason = "signal"
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            verify_alive()
            action = read_control()
            if action is not None:
                reason = action
                break
    except BaseException as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)[:2000]}
        reason = "startup_failed" if child is None else "worker_failed"
    finally:
        clean, cleanup_error = False, None
        try:
            clean = (not supervision.alive_owned(owned) if child is None else
                     supervision.stop_owned(child, owned, owner["pid"], owner["ticks"]))
        except BaseException as error:
            cleanup_error = {"type": type(error).__name__, "message": str(error)[:2000]}
        if log is not None:
            log.close()
        try:
            probe = supervision.gpu_cleanup_query()
        except BaseException as error:
            probe = {"passed": False, "error_type": type(error).__name__}
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        receipt = {"node": node, "layout": layout, "timestamp": time.time(),
                   "owner_started_at": started_at,
                   "owner": {"pid": owner["pid"], "ticks": owner["ticks"]},
                   "reason": reason, "child_started": child is not None,
                   "returncode": child.returncode if child else None,
                   "failure": failure, "cleanup_error": cleanup_error,
                   "owned_children_exited": clean, "gpu_cleanup": probe,
                   "owned": supervision.owned_metadata(owned)}
        _write_new_json(exit_path, receipt)
    normal_stop = reason in ("stop", "controller_eof")
    ok = failure is None and clean and probe.get("passed") is True and normal_stop
    return (0 if ok else 1), receipt
