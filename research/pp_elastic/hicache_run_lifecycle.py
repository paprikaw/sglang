"""Private HiCache smoke-run cancellation; this module never claims a lease."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid


CHECKS = (
    "children_reaped",
    "process_groups_exited",
    "node_gpu_idle_observed",
    "cleanup_verified",
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def private_json(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise RuntimeError(
                "Lifecycle evidence must be a private owned regular file"
            )
        return json.load(stream)


def process_identity():
    data = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": os.getpid(),
        "uid": os.getuid(),
        "start_ticks": int(data[19]),
        "pgid": os.getpgrp(),
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def load_budget(run, cfg):
    budget = private_json(Path(run) / "lifecycle_budget.json")
    policy = cfg["lifecycle"]
    if budget["experiment_id"] != cfg["id"] or budget["nonce"] != policy["nonce"]:
        raise RuntimeError("Lifecycle budget belongs to another run")
    if budget["policy"] != policy:
        raise RuntimeError("Lifecycle budget differs from frozen policy")
    limits = [policy[k] for k in ("work_s", "cleanup_s", "guard_stop_s", "total_s")]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in limits):
        raise ValueError("Lifecycle limits must be finite")
    if not 0 < limits[0] < limits[1] < limits[2] < limits[3] <= 300:
        raise ValueError("Lifecycle needs separate work, cleanup and supervisor limits")
    if not math.isfinite(budget["started_at"]):
        raise ValueError("Invalid lifecycle origin")
    return budget


def request_cancel(run, cfg, reason):
    atomic_json(
        Path(run) / "cancel_request.json",
        {
            "experiment_id": cfg["id"],
            "nonce": cfg["lifecycle"]["nonce"],
            "requested_at": time.time(),
            "reason": reason,
        },
    )


def validate_ack(ack, worker, cfg, budget, *, lease_id=None):
    for value in (ack, worker):
        if value["experiment_id"] != cfg["id"] or value["nonce"] != budget["nonce"]:
            raise RuntimeError("Stale or foreign worker cleanup evidence")
    if ack["worker"] != worker["worker"] or ack["lease_id"] != worker["lease_id"]:
        raise RuntimeError("Cleanup acknowledgement changed worker or lease identity")
    if lease_id is not None and ack["lease_id"] != lease_id:
        raise RuntimeError("Cleanup acknowledgement belongs to another Guard lease")
    identity = worker["worker"]
    if not (
        identity["pid"] > 0
        and identity["start_ticks"] > 0
        and identity["boot_id"]
        and identity["hostname"]
    ):
        raise RuntimeError("Worker process identity is incomplete")
    start, end = ack["cleanup_started_at"], ack["cleanup_finished_at"]
    if not budget["started_at"] <= start <= end <= time.time() + 1:
        raise RuntimeError("Cleanup acknowledgement is not fresh for this run")
    proof = ack["cleanup"]
    verified = all(proof.get(key) is True for key in CHECKS)
    if verified and end > budget["started_at"] + budget["policy"]["cleanup_s"]:
        raise RuntimeError("Verified cleanup exceeded this run's absolute deadline")
    completed = ack.get("work_completed_at")
    if (
        completed is not None
        and not budget["started_at"]
        <= completed
        <= budget["started_at"] + budget["policy"]["work_s"]
    ):
        raise RuntimeError("Completed work exceeds this run's absolute work deadline")
    if verified and (
        proof.get("timestamp", 0) < start
        or proof.get("timestamp", math.inf) > end
        or proof.get("process_identities") != ack["owned_children"]
    ):
        raise RuntimeError("Cleanup proof is stale or omits owned children")
    return verified


class LocalEvidence:
    """Persist request diagnostics locally; archive after model cleanup."""

    def __init__(self, run):
        self.run = Path(run)
        self.root = Path(tempfile.mkdtemp(prefix="pp-hicache-" + self.run.name + "-"))
        os.chmod(self.root, 0o700)
        self.timings = []

    def save(self, name, value):
        started = time.monotonic()
        atomic_json(self.root / name, value)
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.timings.append(
            {"operation": name, "local_write_s": time.monotonic() - started}
        )

    def archive(self):
        files = {}
        for source in sorted(self.root.iterdir()):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = self.run / source.name
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            shutil.copyfile(source, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Local diagnostic archive changed content")
            files[source.name] = digest
        atomic_json(
            self.run / "local_evidence_archive.json",
            {"sha256": files, "timings": self.timings},
        )
        shutil.rmtree(self.root)
        return files


class WorkerLifecycle:
    """One cleanup owner independent of the request/executor thread."""

    def __init__(self, run, cfg, authentication, children, cleanup, hard_exit=os._exit):
        self.run, self.cfg = Path(run), cfg
        self.budget = load_budget(run, cfg)
        self.children, self.cleanup_function = children, cleanup
        self.hard_exit = hard_exit
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.ack_lock = threading.Lock()
        self.cancelled, self.done, self.closed = (threading.Event() for _ in range(3))
        self.cleanup_thread = None
        self.proof = None
        self.reason = "normal completion"
        self.work_completed_at = None
        self.admissions = []
        self.worker = {
            "experiment_id": cfg["id"],
            "nonce": self.budget["nonce"],
            "lease_id": authentication["lease_id"],
            "worker": process_identity(),
        }
        atomic_json(self.run / "worker_identity.json", self.worker)
        # The hard deadline must also survive a blocked cancellation-file reader.
        self.deadline_timer = threading.Timer(
            max(0, self.cleanup_deadline - time.time()), self.hard_exit, args=(124,)
        )
        self.deadline_timer.daemon = True
        self.deadline_timer.start()
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()

    def complete_work(self):
        now = time.time()
        if (
            self.cancelled.is_set()
            or (self.run / "cancel_request.json").exists()
            or now >= self.budget["started_at"] + self.budget["policy"]["work_s"]
        ):
            raise TimeoutError("Work did not finish within the Guard-relative budget")
        self.work_completed_at = now
        return self.work_completed_at

    def admit(self, phase, remaining_required_s):
        now = time.time()
        remaining = self.budget["started_at"] + self.budget["policy"]["work_s"] - now
        record = dict(
            phase=phase,
            timestamp=now,
            remaining_work_s=remaining,
            required_remaining_s=remaining_required_s,
            admitted=remaining >= remaining_required_s,
        )
        self.admissions.append(record)
        if not record["admitted"] or self.cancelled.is_set():
            self.begin_cleanup(
                "Insufficient remaining work budget for " + phase, cancelled=True
            )
            raise TimeoutError("Insufficient remaining work budget for " + phase)

    @property
    def cleanup_deadline(self):
        return self.budget["started_at"] + self.budget["policy"]["cleanup_s"]

    def launch(self, function):
        # A cancellation cannot acknowledge an empty list and then admit a child.
        with self.lock:
            if (
                self.run / "cancel_request.json"
            ).exists() or time.time() >= self.budget["started_at"] + self.budget[
                "policy"
            ][
                "work_s"
            ]:
                self.begin_cleanup("cancelled before launch", cancelled=True)
            if self.cancelled.is_set() or self.cleanup_thread is not None:
                raise RuntimeError("Run cancelled before model launch")
            child = function()
            self.children.append(child)
            return child

    def _ack(self, proof, started, owned):
        ack = dict(self.worker)
        ack.update(
            cleanup_started_at=started,
            cleanup_finished_at=time.time(),
            owned_children=owned,
            reason=self.reason,
            cleanup=proof,
            work_completed_at=self.work_completed_at,
        )
        validate_ack(ack, self.worker, self.cfg, self.budget)
        atomic_json(self.run / "worker_cleanup.json", proof)
        atomic_json(self.run / "cleanup_ack.json", ack)

    def _finish(self, proof, started, owned):
        # The absolute-deadline watchdog and ordinary cleanup publish once.
        with self.ack_lock:
            if self.done.is_set():
                return
            if time.time() >= self.cleanup_deadline:
                proof = dict(proof, **{key: False for key in CHECKS})
            self._ack(proof, started, owned)
            self.proof = proof
            self.done.set()

    def _clean(self):
        started = time.time()
        with self.lock:
            owned = [child["identity"] for child in self.children]
            proof = None
            errors = []
            for _ in range(2):
                if time.time() >= self.cleanup_deadline:
                    break
                try:
                    proof = self.cleanup_function(self.children)
                    if all(proof.get(key) is True for key in CHECKS):
                        break
                except BaseException as error:
                    errors.append(type(error).__name__)
            if proof is None:
                proof = {key: False for key in CHECKS}
            proof = dict(proof, lifecycle_cleanup_errors=errors)
            try:
                self._finish(proof, started, owned)
            except BaseException as error:
                proof = {key: False for key in CHECKS}
                proof["lifecycle_cleanup_errors"] = errors + [type(error).__name__]
                self._finish(proof, started, owned)

    def begin_cleanup(self, reason, *, cancelled=False):
        if cancelled:
            self.cancelled.set()
        # This short lock never waits for the cleanup/launch lock.
        with self.start_lock:
            if self.cleanup_thread is None:
                self.reason = reason
                self.cleanup_thread = threading.Thread(target=self._clean, daemon=True)
                self.cleanup_thread.start()

    def cleanup(self):
        self.begin_cleanup(self.reason)
        self.done.wait(max(0, self.cleanup_deadline - time.time()))
        if self.proof is None:
            return {key: False for key in CHECKS}
        return self.proof

    def _watch(self):
        while not self.closed.wait(0.05):
            if (
                self.work_completed_at is None
                and time.time()
                >= self.budget["started_at"] + self.budget["policy"]["work_s"]
            ):
                self.begin_cleanup("work deadline", cancelled=True)
            request = self.run / "cancel_request.json"
            if request.exists() and not self.cancelled.is_set():
                try:
                    value = private_json(request)
                    if (
                        value["experiment_id"] != self.cfg["id"]
                        or value["nonce"] != self.budget["nonce"]
                    ):
                        raise RuntimeError("Foreign cancellation request")
                    reason = value["reason"]
                except BaseException as error:
                    reason = "invalid cancellation: " + type(error).__name__
                self.begin_cleanup(reason, cancelled=True)
            if time.time() >= self.cleanup_deadline:
                # Do not wait for a hung cleanup or filesystem writer here.
                # Absence of a fresh acknowledgement means quarantine.
                self.hard_exit(124)
                return

    def close(self):
        self.closed.set()
        self.deadline_timer.cancel()
        self.deadline_timer.join(timeout=1)
        self.watcher.join(timeout=1)


def controller(run, cfg, helpers, entrypoint):
    """Keep SSH and Guard alive until this worker's cleanup is acknowledged."""
    run = Path(run)
    authentication = helpers.authenticate(run, cfg, os.environ)
    budget = load_budget(run, cfg)
    environment = ["env"] + [key + "=" + os.environ[key] for key in helpers.GUARD_KEYS]
    command = environment + [
        cfg["python"],
        str(entrypoint),
        "--worker",
        "--run",
        str(run),
    ]
    child = subprocess.Popen(
        ["ssh", "-oBatchMode=yes", cfg["node"], shlex.join(command)],
        start_new_session=True,
    )
    cancellation = threading.Event()
    old_handlers = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        old_handlers[number] = signal.signal(number, lambda *_: cancellation.set())
    worker, ack, error = None, None, None
    cancelled = False
    next_authentication = time.monotonic() + 2
    identity = process_identity()
    try:
        atomic_json(
            run / "task/progress.json",
            {
                "experiment_id": cfg["id"],
                "task_id": "regression",
                "timestamp": time.time(),
                "stage": "running",
                "valid_points": 0,
                "expected": 1,
                "controller_pid": identity["pid"],
                "process_start_ticks": identity["start_ticks"],
            },
        )
    except OSError as failure:
        # A diagnostic write must not abandon an already-started remote worker.
        error = type(failure).__name__
        cancellation.set()
    try:
        while time.time() < budget["started_at"] + budget["policy"]["cleanup_s"]:
            try:
                if (
                    cancellation.is_set()
                    or time.time() >= budget["started_at"] + budget["policy"]["work_s"]
                ):
                    if not (run / "cancel_request.json").exists():
                        request_cancel(
                            run, cfg, "controller cancellation or work deadline"
                        )
                if (run / "cancel_request.json").exists():
                    cancelled = True
                if not cancelled and time.monotonic() >= next_authentication:
                    helpers.authenticate(run, cfg, os.environ)
                    next_authentication = time.monotonic() + 2
                if worker is None and (run / "worker_identity.json").exists():
                    candidate = private_json(run / "worker_identity.json")
                    if candidate["lease_id"] != authentication["lease_id"]:
                        raise RuntimeError("Remote worker authenticated another lease")
                    worker = candidate
                if worker is not None and (run / "cleanup_ack.json").exists():
                    candidate = private_json(run / "cleanup_ack.json")
                    validate_ack(
                        candidate,
                        worker,
                        cfg,
                        budget,
                        lease_id=authentication["lease_id"],
                    )
                    ack = candidate
                    if child.poll() is not None:
                        break
                if child.poll() is not None and ack is None and not cancelled:
                    # A disconnected local SSH client is not remote cleanup proof.
                    request_cancel(run, cfg, "SSH exited before remote acknowledgement")
                    cancelled = True
            except (OSError, ValueError, KeyError, RuntimeError) as failure:
                error = type(failure).__name__
                cancelled = True
                try:
                    if not (run / "cancel_request.json").exists():
                        request_cancel(run, cfg, "controller failed: " + error)
                except OSError:
                    # The worker also enforces the same frozen absolute cutoff.
                    # Keep waiting for its acknowledgement if shared writes fail.
                    pass
            time.sleep(0.1)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=1)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    verified = False
    if worker is not None:
        try:
            ack = private_json(run / "cleanup_ack.json")
            verified = validate_ack(
                ack, worker, cfg, budget, lease_id=authentication["lease_id"]
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            ack = None
    proof = ack["cleanup"] if ack else {key: False for key in CHECKS}
    results = (
        private_json(run / "results.json") if (run / "results.json").exists() else {}
    )
    completed = results.get("transition", {}).get("work_completed_at")
    cancellation_after_work = False
    if cancelled and completed is not None:
        try:
            cancellation_after_work = (
                private_json(run / "cancel_request.json")["requested_at"] >= completed
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            # A missing cancellation record cannot establish work completion.
            pass
    passed = bool(
        verified
        and error is None
        and (not cancelled or cancellation_after_work)
        and child.returncode == 0
        and results.get("transition", {}).get("passed") is True
        and completed == ack.get("work_completed_at")
    )
    # Canonical Guard requires cleanup evidence newer than final progress.
    atomic_json(
        run / "task/progress.json",
        {
            "experiment_id": cfg["id"],
            "task_id": "regression",
            "timestamp": time.time(),
            "stage": "complete" if passed else "failed",
            "valid_points": int(passed),
            "expected": 1,
        },
    )
    atomic_json(
        run / "task/cleanup.json",
        {
            "experiment_id": cfg["id"],
            "task_id": "regression",
            "timestamp": time.time(),
            "passed": verified,
            "gpu_idle": verified and proof["node_gpu_idle_observed"],
            "model_processes_exited": verified and proof["process_groups_exited"],
            "remote_evidence": proof,
            "remote_acknowledgement": ack,
            "ssh_returncode": child.returncode,
            "error_type": error,
        },
    )
    return 0 if passed else 1
