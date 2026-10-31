"""CPU-test and atomically freeze old-native EP locality communication phases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import (
    CAPACITY, CUDA, DEEP, DEEP_COMMIT, DEEP_TAG, DEFAULT_RUN, EXPERTS,
    GUARD, GUARD_SCRIPTS, HIDDEN, ITERATIONS, NATIVE_SHA, CACHED_NATIVE_RECEIPT,
    NATIVE_SO, NODES, PYTHON, ROOT, TOPK, TRACE_SHIM, HOST_COMMIT,
    TRACE_SHIM_SHA, WARMUP, save, sha,
)
from helpers import validate_config, phase_definitions

sys.path.insert(0, str(GUARD_SCRIPTS))
import model_cache_adapter  # noqa: E402

TEMPLATE = Path("/home/bxb1/experiment-guard/dashboards/templates/parallel-tasks-v5")


def verify_pinned_stack():
    prefix = ["git", "-C", str(DEEP)]
    head = subprocess.check_output(prefix + ["rev-parse", "HEAD"], text=True).strip()
    assert head == DEEP_COMMIT
    assert subprocess.check_output(prefix + ["rev-parse", DEEP_TAG + "^{}"], text=True).strip() == head
    assert not subprocess.check_output(prefix + ["status", "--porcelain"], text=True).strip()
    assert sha(TRACE_SHIM) == TRACE_SHIM_SHA
    assert sha(NATIVE_SO) == NATIVE_SHA
    assert Path(PYTHON).is_file() and GUARD.is_file() and TEMPLATE.is_dir()


def run_cpu_tests() -> dict:
    commands = [
        [sys.executable, "-m", "compileall", "-q", str(ROOT)],
        [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT/'tests'), "-v"],
        [sys.executable, "-c", "from prepare import build_config; from node import phase_for; c=build_config(\"cpu_smoke\"); [phase_for(c,p[\"id\"]) for p in c[\"phases\"]]; print(\"CPU layout/config smoke PASS\")"],
    ]
    records = []
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        records.append({"command": command, "returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            raise RuntimeError(f"CPU test failed: {command}\n{completed.stdout}\n{completed.stderr}")
    return {"passed": True, "gpu_execution": False, "records": records, "timestamp": time.time()}


def build_config(experiment_id):
    config = {
        'id': experiment_id, 'scope': 'hy3_serving_shape_pure_communication',
        'nodes': list(NODES), 'candidate_nodes': list(NODES), 'node_count': 2,
        'hidden': HIDDEN, 'experts': EXPERTS, 'topk': TOPK, 'capacity': CAPACITY,
        'warmup': WARMUP, 'iterations': ITERATIONS, 'dtype': 'bfloat16',
        'phases': phase_definitions(NODES),
        'workloads': [
            {'id': 'decode', 'source_tokens_per_rank': 1, 'cuda_graph': False},
            {'id': 'prefill', 'source_tokens_per_rank': 128, 'cuda_graph': False}],
        'model_loaded': False, 'expert_compute': False,
        'transport': {'nic_handler': 'cpu_host_memory', 'network': 'RoCE',
            'qp_depth': 4096, 'pair_only': True, 'pcie_p2p_max_tokens': 0,
            'local_payload': 'NVLink pair; cross-pair RDMA', 'cross_node': 'RDMA'},
        'deep_ep': {'worktree': str(DEEP), 'commit': DEEP_COMMIT, 'tag': DEEP_TAG},
        'native': {'path': str(NATIVE_SO), 'sha256': NATIVE_SHA, 'reused': True,
            'reused_receipt': str(CACHED_NATIVE_RECEIPT)},
        'runtime': {'python': PYTHON, 'cuda': CUDA, 'trace_shim': str(TRACE_SHIM)},
        'causal_limits': ['Matched source rows/rank and receive assignments/rank; EP8 has twice the global tokens per call',
            'EP4 and EP8 expert ownership, QPs, group width and topology differ; not a pure wire-latency ablation',
            'Balanced synthetic routes; no model, no GEMM, no HiCache and no full-model PP speedup claim']}
    validate_config(config)
    return config


def frozen_dependency_paths():
    files = subprocess.check_output(['git', '-C', str(DEEP), 'ls-files'], text=True).splitlines()
    paths = [DEEP / file for file in files if (DEEP / file).is_file()]
    nvlib = Path(PYTHON).parents[1] / 'lib/python3.12/site-packages/nvidia/nvshmem/lib'
    paths += [NATIVE_SO, CACHED_NATIVE_RECEIPT, TRACE_SHIM,
        nvlib / 'libnvshmem_host.so.3', nvlib / 'nvshmem_transport_ibgda.so.3']
    receipt = json.loads(CACHED_NATIVE_RECEIPT.read_text())
    assert receipt['passed'] and receipt['native_sha256'] == NATIVE_SHA
    assert receipt['source_commit'] == HOST_COMMIT and receipt['cuda_override_commit'] == DEEP_COMMIT
    for key in ('host', 'cuda_override'):
        obj=Path(receipt[key+'_path']);assert sha(obj)==receipt[key+'_sha256'];paths.append(obj)
    for item in receipt.get('reused_cuda_objects', receipt.get('reused_objects', [])):
        path = Path(item['path'])
        assert sha(path) == item['sha256'], 'Cached native object identity drift'
        paths.append(path)
    paths = list(dict.fromkeys(paths))
    assert all(path.is_file() for path in paths)
    return paths


def prepare(run: Path) -> dict:
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"immutable runtime already exists: {run}")
    if run != DEFAULT_RUN:
        raise ValueError("This pinned build output requires its declared DEFAULT_RUN")
    verify_pinned_stack()
    test_receipt = run_cpu_tests()
    config = build_config(run.name)
    staging = run.with_name(f".{run.name}.prepare.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        frozen_harness = staging / "frozen/harness"
        shutil.copytree(ROOT, frozen_harness, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        save(staging / "config.json", config)
        save(staging / "cpu_test_receipt.json", test_receipt)
        final_harness = run / "frozen/harness"
        controller = final_harness / "controller.py"
        command = ["python3", str(controller), "--run", str(run)]
        frozen = []
        for staged_path in sorted(path for path in frozen_harness.rglob("*") if path.is_file()):
            relative = staged_path.relative_to(frozen_harness)
            frozen.append({"path": str(final_harness / relative), "sha256": sha(staged_path)})
        frozen.append({"path": str(run / "config.json"), "sha256": sha(staging / "config.json")})
        for path in frozen_dependency_paths():
            frozen.append({"path": str(path.resolve()), "sha256": sha(path)})
        if len({entry["path"] for entry in frozen}) != len(frozen):
            raise RuntimeError("duplicate frozen input paths")
        source = {"worktree": str(DEEP), "commit": DEEP_COMMIT, "tag": DEEP_TAG,
                  "role": "deepep_native_source"}
        contract = {
            "policy": "communication_only",
            "reason": "Hy3 nativec340; node-local EP4 on065/167 and cross-node EP8; source1/128 per rank, eager; no model or expert compute",
            "benchmark": "deepep_identity_expert",
            "entrypoint": {"path": str(controller), "sha256": sha(frozen_harness / "controller.py")},
            "command": command,
        }
        spec = {
            "schema_version": 1, "id": config["id"],
            "title": "Hy3-shaped A100 EP4 local versus EP8 cross-node dispatch/combine",
            "engine": "deepep", "protocol": "task-events-v2", "template": "parallel-tasks-v5",
            "features": ["parallel-tasks", "task-progress", "task-details", "gpu-tab", "guard"],
            "gpu_telemetry": {"enabled": False, "poll_interval_s": 10, "timeout_s": 10},
            "port": 24743, "max_active_nodes": 2, "gpu_lease_ttl_s": 90,
            "cleanup_wait_s": 90, "runtime_dir": str(run / "dashboard_guard"),
            "gpu_registry_dir": "/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry",
            "sources": [source], "frozen_inputs": frozen, "model_cache": contract,
            "workload": {"kind": "synthetic_identity_expert_communication",
                         "model_loaded": False, "expert_compute": False},
            "tasks": [{
                "id": "local", "label": "Hy3 matched-source dispatch/combine",
                "root": str(run / "task"), "expected": len(config["phases"]), "nodes": list(NODES),
                "candidate_nodes": list(NODES), "controller_script": str(controller),
                "progress_file": "progress.json", "cleanup_file": "cleanup.json",
            }],
        }
        save(staging / "frozen_inputs.json", {"files": frozen})
        save(staging / "dashboard_spec.json", spec)
        save(staging / "task/progress.json", {
            "timestamp": time.time(), "stage": "queued", "valid_points": 0,
            "note": "Hy3 c340；065/167节点内EP4与跨节点EP8；每卡1/128 tokens，eager，各3次；无模型计算",
        })
        validation_spec = json.loads(json.dumps(spec).replace(str(run), str(staging)))
        model_cache_adapter.validate_contract(validation_spec, launch=True)
        staging.rename(run)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    model_cache_adapter.validate_contract(json.loads((run / "dashboard_spec.json").read_text()), launch=True)
    prepared = subprocess.run(["python3", str(GUARD), "prepare", "--spec", str(run / "dashboard_spec.json")],
                              capture_output=True, text=True)
    save(run / "dashboard_prepare.json", {
        "returncode": prepared.returncode, "stdout": prepared.stdout, "stderr": prepared.stderr,
    })
    if prepared.returncode:
        raise RuntimeError(f"dashboard preparation failed: {prepared.stdout}\n{prepared.stderr}")
    launch = ["python3", str(GUARD), "run", "--spec", str(run / "dashboard_spec.json"), "--", *command]
    result = {"prepared": True, "run": str(run), "launch_command": launch,
              "frozen_files": len(frozen), "cpu_tests_passed": True, "gpu_execution": False}
    save(run / "prepare_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    print(json.dumps(prepare(parser.parse_args().run), indent=2))
