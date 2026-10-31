"""Freeze one streaming arm without allocating GPUs or touching CPU banks."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from stream_common import ROOT, OLD, ENGINE, PYTHON, MODEL, qualified, sha, save, expected_ranks


def descriptor(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Frozen regular dependency required: " + str(path))
    return {"path": str(path), "sha256": sha(path)}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--guard-scripts", required=True, type=Path)
    parser.add_argument("--job-ids", required=True, nargs=2)
    parser.add_argument("--layout", choices=("ep8","pp2ep4"), required=True)
    parser.add_argument("--timeout-s", default=3600, type=int)
    args = parser.parse_args()
    run, guard = args.run, args.guard_scripts.resolve()
    if (not run.is_absolute() or run.exists() or not run.is_relative_to(ROOT)
            or not all(value.isdecimal() for value in args.job_ids) or args.timeout_s <= 0):
        raise ValueError("Fresh absolute run directory, exact job IDs and positive timeout required")
    sys.path.insert(0, str(guard))
    import model_cache_adapter
    import hy3_streaming_policy as policy
    import cpu_memory_budget
    head = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(ENGINE), "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("Engine tracked source must be committed before freezing")
    source = descriptor(OLD / "download/qualification_source_manifest.json")
    portable = descriptor(OLD / "download/portable_source_identity.json")
    completed = json.loads((OLD / "download/completion.json").read_text())
    if (completed["download_status"] != "complete" or not completed["tensor_index_verified"]
            or source["sha256"] != completed["source_proof_manifest_sha256"]):
        raise RuntimeError("Verified complete checkpoint source required")
    run.mkdir(mode=0o700, parents=True)
    (run / "task").mkdir(mode=0o700)
    nodes = ["spartan-gpgpu065", "spartan-gpgpu167"]
    cfg = {"schema": "hy3-stream-cached-v1", "id": run.name,
           "run_dir": str(run), "python": PYTHON, "engine_root": str(ENGINE),
           "engine_commit": head, "nodes": nodes, "jobs": dict(zip(nodes, args.job_ids)),
           "guard_scripts": str(guard), "effective_num_hidden_layers": 80,
           "layouts": {k: qualified.LAYOUTS[k] for k in ("ep8","pp2ep4")}, "layout": args.layout,
           "api_port": free_port(), "dist_port": free_port(), "timeout_s": args.timeout_s}
    save(run / "config.json", cfg)
    (run / args.layout / "capacity").mkdir(parents=True, mode=0o700)
    node_specs, usage_logs = {}, {}
    for rank, node in enumerate(nodes):
        base = run / args.layout / node
        base.mkdir(mode=0o700, parents=True)
        value = {"node": node, "expected_job_id": cfg["jobs"][node], "gpus": 4,
                 "python": PYTHON, "engine_root": str(ENGINE), "engine_commit": head,
                 "staging_ledger": f"/tmp/{os.getuid()}-{run.name}-staging/registry.json",
                 "staging_bytes": policy.STAGING, "rank_staging_limit_bytes": policy.RANK_STAGING,
                 "workspace_bytes": policy.WORKSPACE, "headroom_bytes": policy.FLOOR,
                 "effective_num_hidden_layers": 80}
        path = run / (node + ".json")
        save(path, value)
        node_specs[node] = descriptor(path)
        usage_logs[node] = [{"id": "stream-" + node, "path": str(base / "usage.jsonl"),
                             "expected_ranks": expected_ranks(rank)}]
    entrypoint = descriptor(ROOT / "runtime/controller.py")
    worker = descriptor(ROOT / "runtime/model_node.py")
    command = ["/home/bxb1/conda/bin/python", entrypoint["path"], "--run", str(run)]
    dep_paths = {*guard.glob("*.py"), *(ROOT / "runtime").glob("*.py"),
                 *(ROOT / "streaming").glob("*.py"), OLD / "runtime/common.py",
                 OLD / "runtime/supervision.py"}
    # The unchanged published Guard also requires its original sibling
    # streaming inputs to remain frozen. Freeze those in addition to the v2
    # implementation used by PYTHONPATH; do not mutate the published policy.
    dep_paths.update((policy.TASK_ROOT / "streaming").glob("*.py"))
    dep_paths.update(Path(d["path"]) for d in cpu_memory_budget.dependency_identity()["files"])
    dependencies = [descriptor(path) for path in sorted(dep_paths)]
    # Reuse the already frozen immutable engine file descriptors. Engine HEAD
    # and tracked-clean checks above, plus runtime authentication hashes, still apply.
    previous = json.loads((OLD / "node_local/ep8_layers80_reuse_r1_20261019/dashboard_spec.json").read_text())
    frozen = {row["path"]: row for row in previous["frozen_inputs"]
              if Path(row["path"]).is_relative_to(ENGINE)}
    for row in [*dependencies, source, portable, descriptor(run / "config.json"),
                descriptor(Path(MODEL) / "config.json"), descriptor(Path(MODEL) / "model.safetensors.index.json"),
                descriptor(qualified.NATIVE), descriptor(qualified.SHIM), *node_specs.values()]:
        frozen[row["path"]] = row
    contract = {"policy": policy.POLICY, "provider": policy.PROVIDER,
                "support_profile": policy.PROFILE, "model_manifest": source,
                "portable_source_identity": portable, "entrypoint": entrypoint,
                "worker_entrypoint": worker, "command": command,
                "usage_logs": usage_logs, "node_specs": node_specs, "dependencies": dependencies}
    spec = {"schema_version": 1, "id": run.name, "engine": "sglang",
            "title": "Hy3 BF16 cached EP8 / PP2xEP4",
            "protocol": "task-events-v2", "template": "parallel-tasks-v5",
            "features": ["parallel-tasks", "task-progress", "guard", "cpu-model-cache"],
            "port": free_port(), "max_active_nodes": 2, "gpu_lease_ttl_s": 90,
            "cleanup_wait_s": 90, "runtime_dir": str(run / "dashboard_guard"),
            "gpu_registry_dir": "/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry",
            "workload": {"kind": "hy3_cached_streaming_capacity", "model_loaded": True,
                         "effective_num_hidden_layers": 80, "full_model": True,
                         "batch_size": 8, "new_output_tokens": 4, "hicache": True, "native_host_pool_max_bytes_per_node": 64_400_000_000,
                         "performance_benchmark": False},
            "model_cache": contract, "frozen_inputs": sorted(frozen.values(), key=lambda row: row["path"]),
            "sources": [{"worktree": str(ENGINE), "commit": head}],
            "tasks": [{"id": "streaming", "label": "Full80 streamed weights and native HiCache",
                       "root": str(run / "task"), "expected": 2, "nodes": nodes,
                       "controller_script": entrypoint["path"]}]}
    save(run / "dashboard_spec.json", spec)
    model_cache_adapter.validate_contract(spec, launch=True)
    print(json.dumps({"run": str(run), "spec": str(run / "dashboard_spec.json"),
                      "command": command, "frozen_files": len(frozen),
                      "gpu_launched": False, "weight_banks_modified": False}))


if __name__ == "__main__":
    main()
