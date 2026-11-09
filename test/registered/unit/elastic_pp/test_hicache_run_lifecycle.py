"""CPU process/lease tests; CPU stand-ins never attest actual GPU idleness."""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[4] / "research/pp_elastic"
sys.path.insert(0, str(SCRIPTS))
import hicache_run_lifecycle as lifecycle
import hicache_guard_run as supervisor

GUARD = Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts")


def configuration(run, work=0.4, cleanup=3, stop=5, total=8):
    (run / "task").mkdir()
    (run / "dashboard_guard").mkdir()
    cfg = dict(
        id="cpu-" + run.name,
        python=sys.executable,
        node="spartan-gpgpu999",
        guard_scripts=str(GUARD),
        lifecycle=dict(
            version=1,
            nonce=os.urandom(16).hex(),
            work_s=work,
            cleanup_s=cleanup,
            guard_stop_s=stop,
            total_s=total,
        ),
    )
    lifecycle.atomic_json(run / "config.json", cfg)
    lifecycle.atomic_json(
        run / "dashboard_spec.json",
        dict(
            id=cfg["id"],
            port=1,
            resource_priority="normal",
            gpu_cleanup_grace_s=45,
            tasks=[dict(id="cpu", nodes=[cfg["node"]])],
        ),
    )
    lifecycle.atomic_json(
        run / "lifecycle_budget.json",
        dict(
            experiment_id=cfg["id"],
            nonce=cfg["lifecycle"]["nonce"],
            policy=cfg["lifecycle"],
            started_at=time.time(),
        ),
    )
    return cfg


def wait_file(path, timeout=5):
    limit = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= limit:
            raise TimeoutError(str(path))
        time.sleep(0.02)
    return lifecycle.private_json(path)


def proof(children):
    return dict.fromkeys(lifecycle.CHECKS, True) | dict(
        timestamp=time.time(),
        process_identities=[x["identity"] for x in children],
        cpu_only_gpu_idle_standin=True,
    )


@pytest.mark.parametrize(
    "fault", ["nonce", "lease", "worker", "timestamp", "children", "late"]
)
def test_ack_rejects_stale_or_foreign_evidence(tmp_path, fault):
    cfg = configuration(tmp_path)
    budget = lifecycle.load_budget(tmp_path, cfg)
    worker = dict(
        experiment_id=cfg["id"],
        nonce=budget["nonce"],
        lease_id="own",
        worker=lifecycle.process_identity(),
    )
    ack = dict(
        worker,
        cleanup_started_at=time.time(),
        cleanup_finished_at=time.time(),
        owned_children=[],
        cleanup=proof([]),
    )
    ack["cleanup_finished_at"] = time.time()
    if fault == "nonce":
        ack["nonce"] = "previous-run"
    elif fault == "lease":
        ack["lease_id"] = "different"
        worker = dict(worker, lease_id="different")
    elif fault == "worker":
        ack["worker"] = dict(ack["worker"], start_ticks=1)
    elif fault == "timestamp":
        ack["cleanup"]["timestamp"] = budget["started_at"] - 1
    elif fault == "children":
        ack["owned_children"] = [{"pid": 123}]
    else:
        budget["policy"]["cleanup_s"] = 0.00001
    with pytest.raises(RuntimeError):
        lifecycle.validate_ack(ack, worker, cfg, budget, lease_id="own")


def test_cancel_and_child_registration_are_one_transaction(tmp_path):
    cfg = configuration(tmp_path, work=3, cleanup=5, stop=6, total=8)
    entered, registered = threading.Event(), threading.Event()
    children, calls = [], []

    def clean(owned):
        calls.append([x["identity"] for x in owned])
        for child in owned:
            child["process"].terminate()
            child["process"].wait(timeout=2)
        return proof(owned)

    runner = lifecycle.WorkerLifecycle(
        tmp_path, cfg, {"lease_id": "own"}, children, clean
    )

    def launch():
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
        entered.set()
        assert registered.wait(2)
        return {"process": child, "identity": {"pid": child.pid}}

    thread = threading.Thread(target=runner.launch, args=(launch,))
    thread.start()
    try:
        assert entered.wait(2)
        lifecycle.request_cancel(
            tmp_path, cfg, "CPU cancellation during Popen registration"
        )
        time.sleep(0.1)
        assert not runner.done.is_set()
        registered.set()
        thread.join(2)
        assert runner.done.wait(2)
        assert calls == [[children[0]["identity"]]]
        assert runner.cleanup()["cleanup_verified"]
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.launch(lambda: pytest.fail("launch after acknowledgement"))
    finally:
        registered.set()
        thread.join(2)
        runner.close()
        for child in children:
            if child["process"].poll() is None:
                child["process"].kill()
                child["process"].wait(2)


def test_preexisting_cancel_prevents_any_child(tmp_path):
    cfg = configuration(tmp_path)
    lifecycle.request_cancel(tmp_path, cfg, "cancel before worker exists")
    runner = lifecycle.WorkerLifecycle(tmp_path, cfg, {"lease_id": "own"}, [], proof)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.launch(lambda: pytest.fail("cancelled worker launched a child"))
        assert runner.done.wait(2)
        assert (
            lifecycle.private_json(tmp_path / "cleanup_ack.json")["owned_children"]
            == []
        )
    finally:
        runner.close()


def test_remaining_budget_rejection_starts_same_absolute_cleanup(tmp_path):
    cfg = configuration(tmp_path)
    runner = lifecycle.WorkerLifecycle(tmp_path, cfg, {"lease_id": "own"}, [], proof)
    deadline = runner.cleanup_deadline
    try:
        with pytest.raises(TimeoutError, match="Insufficient remaining"):
            runner.admit("references", 100)
        assert runner.done.wait(2)
        assert runner.cancelled.is_set()
        assert runner.cleanup_deadline == deadline
        assert not runner.admissions[0]["admitted"]
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.launch(lambda: pytest.fail("Rejected phase launched work"))
    finally:
        runner.close()


@pytest.mark.parametrize("policy", [(250, 285, 290, 301), (250, 240, 290, 300)])
def test_invalid_whole_run_budget_is_rejected(tmp_path, policy):
    cfg = configuration(tmp_path, *policy)
    with pytest.raises(ValueError):
        lifecycle.load_budget(tmp_path, cfg)


def test_local_diagnostics_preserve_content_and_avoid_shared_writes(
    tmp_path, monkeypatch
):
    store = lifecycle.LocalEvidence(tmp_path)
    payload = {
        "capture": "actual pre-serialization stream payload",
        "requests": [
            {
                "stage": "active_reference",
                "payload": {
                    "input_ids": list(range(513)),
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 256,
                        "ignore_eos": True,
                    },
                },
            }
        ],
    }
    writes = []
    original = lifecycle.atomic_json

    def save(path, value):
        writes.append(Path(path))
        return original(path, value)

    monkeypatch.setattr(lifecycle, "atomic_json", save)
    for stage in (
        "prefix_warmup",
        "prefix_reference",
        "active_warmup",
        "active_reference",
    ):
        store.save("worker_stage.json", {"stage": stage})
        store.save("reference_request_contract.json", payload)
    assert all(path.parent == store.root for path in writes)
    assert not list(tmp_path.iterdir())
    expected = hashlib.sha256(
        (store.root / "reference_request_contract.json").read_bytes()
    ).hexdigest()
    files = store.archive()
    assert files["reference_request_contract.json"] == expected
    assert (
        lifecycle.private_json(tmp_path / "reference_request_contract.json") == payload
    )
    assert (
        len(lifecycle.private_json(tmp_path / "local_evidence_archive.json")["timings"])
        == 8
    )


def test_local_archive_rejects_corrupted_copy(tmp_path, monkeypatch):
    store = lifecycle.LocalEvidence(tmp_path)
    store.save("contract.json", {"exact": [1, 2, 3]})
    original = lifecycle.shutil.copyfile

    def corrupt(source, destination):
        original(source, destination)
        Path(destination).write_bytes(b"corrupted")

    monkeypatch.setattr(lifecycle.shutil, "copyfile", corrupt)
    try:
        with pytest.raises(RuntimeError, match="changed content"):
            store.archive()
    finally:
        lifecycle.shutil.rmtree(store.root)


# These fixtures launch actual OS processes and the original registry/heartbeat.
# The cleanup GPU-idle field is explicitly a CPU stand-in, never GPU validation.
WORKER = r"""
import json, os, pathlib, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import hicache_run_lifecycle as m
run, mode, delay = pathlib.Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
cfg = m.private_json(run/'config.json')
receipt = m.private_json(run/'dashboard_guard/gpu_registry_lease.json')
sys.path.insert(0, cfg['guard_scripts'])
import gpu_registry
registry = gpu_registry.SharedGpuRegistry(receipt['registry_dir'])
registry.assert_owned(receipt, cfg['id'], receipt['spec_sha256'])
children = []
def clean(owned):
    if mode == 'hang':
        while True: time.sleep(.05)
    time.sleep(delay)
    if mode == 'failure': return dict.fromkeys(m.CHECKS, False)
    for child in owned:
        child['process'].terminate()
        child['process'].wait(timeout=2)
    return dict.fromkeys(m.CHECKS, True) | dict(timestamp=time.time(), process_identities=[x['identity'] for x in owned], cpu_only_gpu_idle_standin=True)
if mode == 'watch_hang': m.WorkerLifecycle._watch = lambda self: time.sleep(90)
worker = m.WorkerLifecycle(run, cfg, receipt, children, clean)
def launch():
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'], start_new_session=True)
    ticks = int(pathlib.Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19])
    identity = dict(pid=child.pid, start_ticks=ticks)
    m.atomic_json(run/'cpu_child.json', identity)
    return dict(process=child, identity=identity)
completed = None
try:
    worker.launch(launch)
    m.atomic_json(run/'worker_ready.json', dict(pid=os.getpid()))
    if mode == 'complete':
        completed = worker.complete_work()
    else:
        while not worker.cancelled.wait(.02): pass
except RuntimeError:
    if not worker.cancelled.is_set(): raise
cleanup = worker.cleanup()
m.atomic_json(run/'results.json', dict(transition=dict(passed=mode=='complete' and cleanup['cleanup_verified'], work_completed_at=completed)))
worker.close()
sys.exit(0 if mode == 'complete' else 1)
"""

GUARD_FIXTURE = r"""
import hashlib, json, os, pathlib, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import hicache_run_lifecycle as m
run, mode, delay = pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = m.private_json(run/'config.json')
sys.path.insert(0, cfg['guard_scripts'])
import gpu_registry
spec_path = run/'dashboard_spec.json'
spec = m.private_json(spec_path)
registry = gpu_registry.SharedGpuRegistry(run/'registry')
receipt, _ = registry.claim(spec, hashlib.sha256(spec_path.read_bytes()).hexdigest(), [cfg['node']], run/'dashboard_guard/gpu_registry_lease.json', ttl_s=15)
heartbeat = gpu_registry.LeaseHeartbeat(registry, receipt).start()
if mode in ('guard_hang', 'guard_ignore'):
    def stop(*_):
        state = json.loads(registry.state_path.read_text())
        row = next(x for x in state['leases'] if x['lease_id']==receipt['lease_id'])
        m.atomic_json(run/'guard_terminated.json', dict(state=row['state'], timestamp=time.time()))
        if mode == 'guard_hang': raise SystemExit(1)
    signal.signal(signal.SIGTERM, stop)
    while True: time.sleep(.05)
worker = subprocess.Popen([sys.executable, str(run/'cpu_worker.py'), sys.argv[1], str(run), mode, delay], start_new_session=True)
if mode == 'guard_exit': raise SystemExit(1)
code = worker.wait()
try:
    identity = m.private_json(run/'worker_identity.json')
    ack = m.private_json(run/'cleanup_ack.json')
    verified = m.validate_ack(ack, identity, cfg, m.load_budget(run,cfg), lease_id=receipt['lease_id'])
except (OSError, ValueError, KeyError, RuntimeError): verified = False
state = registry.release(receipt, verified, 'CPU fixture settlement after remote acknowledgement')
m.atomic_json(run/'guard_settled.json', dict(state=state['state'], timestamp=time.time(), heartbeat_error=type(heartbeat.error).__name__ if heartbeat.error else None))
heartbeat.stop()
sys.exit(code)
"""


def terminate_cpu_child(run):
    path = run / "cpu_child.json"
    if path.exists():
        identity = lifecycle.private_json(path)
        stat = Path(f"/proc/{identity['pid']}/stat")
        if (
            stat.exists()
            and int(stat.read_text().rsplit(")", 1)[1].split()[19])
            == identity["start_ticks"]
        ):
            os.kill(identity["pid"], signal.SIGKILL)


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "cancel",
        "cancel_write",
        "failure",
        "hang",
        "watch_hang",
        "guard_hang",
        "guard_ignore",
        "guard_exit",
        "final_read_delay",
    ],
)
def test_supervisor_real_process_and_atomic_registry(tmp_path, monkeypatch, mode):
    # cancel crosses a real 15s TTL; all other cases use short absolute budgets.
    delay = 15.6 if mode == "cancel" else 0.05
    cfg = configuration(
        tmp_path,
        work=1,
        cleanup=21 if mode == "cancel" else 3,
        stop=24 if mode == "cancel" else 5,
        total=28 if mode == "cancel" else 9,
    )
    worker = tmp_path / "cpu_worker.py"
    guard = tmp_path / "cpu_guard.py"
    worker.write_text(WORKER)
    guard.write_text(GUARD_FIXTURE)
    if mode == "final_read_delay":
        original_ack = supervisor.remote_ack

        def delayed_final_ack(*args):
            result = original_ack(*args)
            time.sleep(cfg["lifecycle"]["total_s"] + 0.1)
            return result

        monkeypatch.setattr(supervisor, "remote_ack", delayed_final_ack)
    if mode == "cancel_write":

        def failed_cancel(*_args):
            raise OSError("CPU injected cancellation write failure")

        monkeypatch.setattr(supervisor, "request_cancel", failed_cancel)
    before = time.monotonic()
    try:
        code = supervisor.supervise(
            tmp_path,
            cfg,
            [
                sys.executable,
                str(guard),
                str(SCRIPTS),
                str(tmp_path),
                "complete" if mode == "final_read_delay" else mode,
                str(delay),
            ],
        )
        outcome = lifecycle.private_json(tmp_path / "supervisor_outcome.json")
        if mode != "final_read_delay":
            assert time.monotonic() - before < cfg["lifecycle"]["total_s"]
        else:
            assert outcome["elapsed_s"] > cfg["lifecycle"]["total_s"]
        assert outcome["passed"] == (mode == "complete")
        assert code == (0 if mode == "complete" else 1)
        if mode in {
            "complete",
            "cancel",
            "cancel_write",
            "guard_exit",
            "final_read_delay",
        }:
            assert outcome["normal_lease_release_verified"]
            ack = outcome["remote_acknowledgement"]
            if mode != "guard_exit":
                settled = lifecycle.private_json(tmp_path / "guard_settled.json")
                assert ack["cleanup_finished_at"] <= settled["timestamp"]
                assert settled["heartbeat_error"] is None
            else:
                assert outcome["emergency_heartbeat_stopped"]
            if mode == "cancel":
                assert ack["cleanup_finished_at"] - outcome["started_at"] > 15
                assert (
                    lifecycle.private_json(tmp_path / "cancel_request.json")[
                        "requested_at"
                    ]
                    < ack["cleanup_finished_at"]
                )
                assert (
                    ack["cleanup_started_at"]
                    >= outcome["started_at"] + cfg["lifecycle"]["work_s"]
                )
            if mode == "cancel_write":
                assert outcome["cancel_request_error_type"] == "OSError"
                assert not (tmp_path / "cancel_request.json").exists()
                assert (
                    ack["cleanup_started_at"]
                    >= outcome["started_at"] + cfg["lifecycle"]["work_s"]
                )
        else:
            assert not outcome["normal_lease_release_verified"]
            assert outcome["lease"]["state"] == "quarantined"
            assert outcome["lease"]["cleanup_grace_expires_at"] > outcome["ended_at"]
            _module, registry, receipt = supervisor.registry_owner(tmp_path, cfg)
            contender = lifecycle.private_json(tmp_path / "dashboard_spec.json")
            contender["id"] += "-contender"
            with pytest.raises(_module.RegistryConflict):
                registry.claim(
                    contender,
                    "f" * 64,
                    [cfg["node"]],
                    tmp_path / "competitor.json",
                    ttl_s=15,
                )
            if mode == "failure":
                identity = lifecycle.private_json(tmp_path / "cpu_child.json")
                assert Path(f"/proc/{identity['pid']}").exists()
            if mode in {"guard_hang", "guard_ignore"}:
                assert outcome["forced_stop"]
                assert (
                    lifecycle.private_json(tmp_path / "guard_terminated.json")["state"]
                    == "quarantined"
                )
    finally:
        terminate_cpu_child(tmp_path)


@pytest.mark.parametrize("disconnect", [False, True])
@pytest.mark.parametrize("write_failure", [None, "progress", "cancel"])
def test_controller_requires_remote_ack_even_when_ssh_exits(
    tmp_path, monkeypatch, disconnect, write_failure
):
    cfg = configuration(tmp_path, work=1, cleanup=3, stop=5, total=8)
    sys.path.insert(0, str(GUARD))
    import gpu_registry

    registry = gpu_registry.SharedGpuRegistry(tmp_path / "registry")
    spec_path = tmp_path / "dashboard_spec.json"
    receipt, _ = registry.claim(
        lifecycle.private_json(spec_path),
        hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        [cfg["node"]],
        tmp_path / "dashboard_guard/gpu_registry_lease.json",
        ttl_s=15,
    )
    heartbeat = gpu_registry.LeaseHeartbeat(registry, receipt).start()
    worker = tmp_path / "cpu_worker.py"
    worker.write_text(WORKER)
    ssh = tmp_path / "ssh"
    # An actual short-lived transport spawns a detached actual worker, then exits.
    command = [
        sys.executable,
        str(worker),
        str(SCRIPTS),
        str(tmp_path),
        "cancel" if disconnect or write_failure else "complete",
        "0.1",
    ]
    ssh.write_text(
        f"#!{sys.executable}\nimport subprocess,sys,os,json\n"
        f"p=subprocess.Popen({command!r}, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"with os.fdopen(os.open({str(tmp_path/'transport.json')!r},os.O_WRONLY|os.O_CREAT,0o600),'w') as f: json.dump(dict(pid=p.pid),f)\n"
        + ("" if disconnect else "sys.exit(p.wait())\n")
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    helper = SimpleNamespace(
        GUARD_KEYS=(),
        authenticate=lambda *_a, **_k: registry.assert_owned(
            receipt, cfg["id"], receipt["spec_sha256"]
        ),
    )
    if write_failure == "progress":
        original = lifecycle.atomic_json
        writes = []

        def failed_progress(path, value):
            if Path(path) == tmp_path / "task/progress.json" and not writes:
                writes.append(path)
                raise OSError("CPU injected first progress write failure")
            return original(path, value)

        monkeypatch.setattr(lifecycle, "atomic_json", failed_progress)
    elif write_failure == "cancel":

        def failed_cancel(*_args):
            raise OSError("CPU injected all cancellation writes failure")

        monkeypatch.setattr(lifecycle, "request_cancel", failed_cancel)
    try:
        assert lifecycle.controller(tmp_path, cfg, helper, worker) == int(
            bool(disconnect or write_failure)
        )
        cleanup = lifecycle.private_json(tmp_path / "task/cleanup.json")
        assert cleanup["passed"]
        # Exercise the canonical release gate, not a fixture interpretation of it.
        import dashboard_guard

        authorization = dashboard_guard.cleanup_authorization(
            {
                "id": cfg["id"],
                "runtime_dir": str(tmp_path / "dashboard_guard"),
                "tasks": [{"id": "regression", "root": str(tmp_path / "task")}],
            }
        )
        assert authorization == [{"task_id": "regression", "mode": "exited"}]
        assert (
            cleanup["remote_acknowledgement"]["worker"]["pid"]
            == wait_file(tmp_path / "transport.json")["pid"]
        )
        if write_failure:
            assert cleanup["error_type"] == "OSError"
        if write_failure == "cancel":
            assert not (tmp_path / "cancel_request.json").exists()
        elif disconnect:
            assert (
                cleanup["remote_acknowledgement"]["cleanup_started_at"]
                >= lifecycle.private_json(tmp_path / "cancel_request.json")[
                    "requested_at"
                ]
            )
    finally:
        terminate_cpu_child(tmp_path)
        registry.release(receipt, True, "CPU-only test child cleanup complete")
        heartbeat.stop()
