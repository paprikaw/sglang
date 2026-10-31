"""Freeze the authorized two-node CPU bank replacement, without node mutation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT.parent / "hy3_layer_window_bank_20261019"
WORKSPACE = ROOT.parents[1]
OLD = WORKSPACE / "experiments/hy3_a100_qualification_20261018"
PREVIOUS = OLD / "node_local/ep8_layers80_reuse_r1_20261019"
ENGINE = WORKSPACE / ".engine-worktrees/hy3-a100-qualification-20261018"
PYTHON = "/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python"
NODES = ["spartan-gpgpu167", "spartan-gpgpu169"]
GIB = 1024**3


def descriptor(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    # New run roots are immutable inputs; never overwrite a previous run.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--guard-scripts", type=Path, required=True)
    parser.add_argument("--job-ids", nargs=2, required=True)
    args = parser.parse_args()
    if not args.run.is_absolute():
        raise ValueError("An absolute run path is required")
    run = args.run.resolve(strict=False)
    guard = args.guard_scripts.resolve(strict=True)
    if not run.is_absolute() or run.exists() or not run.is_relative_to(ROOT):
        raise ValueError("A fresh absolute run directory under this experiment is required")
    if not all(value.isdecimal() for value in args.job_ids):
        raise ValueError("Exact currently RUNNING booking IDs are required")
    head = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "-C", str(ENGINE), "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise RuntimeError("Engine tracked source must be immutable before freezing")
    sys.path.insert(0, str(ENGINE / "python/sglang/srt/weight_cache"))
    import cpu_node_local
    sys.path.insert(0, str(guard))
    import cpu_memory_budget
    import model_cache_adapter
    import hy3_layer_window_pair_policy as policy

    run.mkdir(parents=True, mode=0o700)
    (run / "task").mkdir(mode=0o700)
    source = descriptor(OLD / "download/qualification_source_manifest.json")
    portable = descriptor(OLD / "download/portable_source_identity.json")
    inventory = descriptor(PREPARED / "planning/results/memory_bank_inventory.json")
    provider = descriptor(PREPARED / "runtime/layer_window_provider.py")
    node_specs = {}
    for rank, (node, job) in enumerate(zip(NODES, args.job_ids)):
        ledger = json.loads(Path("/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/cpu_weight_daemons/registry.json").read_text())
        prior = next((r for r in ledger["entries"] if r["node"]==node),None)
        out = run / "nodes" / node
        out.mkdir(parents=True, mode=0o700)
        value = {"node": node, "node_rank": rank, "expected_job_id": job,
            "python": PYTHON, "engine_root": str(ENGINE), "source_commit": head,
            "model_path": "/data/gpfs/projects/punim2565/models/Hy3-BF16", "socket_path": "/tmp/sglang-cpu-layer-weights-15540/cache.sock",
            "existing_bank": ({"daemon_spec":descriptor(PREPARED / "runs/prepare_r3/nodes" / node / "new_daemon_spec.json"),
                              "manifest":descriptor(PREPARED / "runs/prepare_r3/nodes" / node / "layer_window_manifest.json")} if rank==1 else None),
            "stale_entry": None,
            "provider_module": provider, "output_path": str(out / "result.json"),
            "min_window_layers": 40, "max_window_layers": 80,
            "headroom_bytes": 48 * GIB, "workspace_bytes": 64 * GIB,
            "streaming_staging_bytes": 8 * GIB, "fill_scratch_bytes": 128 * 1024**2,
            "metadata_reserve_bytes": 4 * GIB, "hicache_reserve_bytes": 0,
            "chunk_bytes": 8 * 1024**2}
        if rank==0 and prior:
            stale_path=run / (node+".prior_ledger.json")
            save(stale_path,prior);value["stale_entry"]=descriptor(stale_path)
        path = run / (node + ".json")
        save(path, value)
        node_specs[node] = descriptor(path)
    dependencies = {*guard.glob("*.py"), *(ROOT / "runtime").glob("*.py"),
                    *(PREPARED / "memory_bank_candidate/src/sglang_memory_bank").glob("*.py")}
    dependencies.update(Path(d["path"]) for d in cpu_memory_budget.dependency_identity()["files"])
    contract = {"policy": policy.POLICY, "provider": policy.PROVIDER, "support_profile": policy.PROFILE,
        "model_manifest": source, "portable_source_identity": portable, "inventory": inventory,
        "worker_entrypoint": provider, "node_specs": node_specs,
        "dependencies": [descriptor(p) for p in sorted(dependencies)],
        "replacement_authorization": {"replace_ready_cache": False, "scope": "167_169_hy3_new_or_reuse"},
        "cpu_memory_budget": {"schema_version": 1, "kv_banks_per_node": {node: [] for node in NODES}},
        "hicache_reserve_bytes_per_node": {node: 0 for node in NODES}}
    previous_spec = json.loads((PREVIOUS / "dashboard_spec.json").read_bytes())
    frozen = {d["path"]: d for d in previous_spec["frozen_inputs"] if Path(d["path"]).is_relative_to(ENGINE)}
    extra = [source, portable, inventory, provider, *contract["dependencies"], *node_specs.values()]
    for node in NODES:
        n=json.loads((run/(node+".json")).read_text())
        extra += list(n["existing_bank"].values()) if n["existing_bank"] else []
        if n["stale_entry"]:extra.append(n["stale_entry"])
    extra += [descriptor(Path("/data/gpfs/projects/punim2565/models/Hy3-BF16") / name)
              for name in ("config.json", "model.safetensors.index.json")]
    for item in extra:
        frozen[item["path"]] = item
    spec = {"schema_version": 1, "id": run.name, "engine": "sglang",
        "title": "Hy3 preserve169 and prepare167 CPU layer-bank",
        "protocol": "task-events-v2", "template": "parallel-tasks-v5",
        "features": ["parallel-tasks", "task-progress", "guard", "cpu-model-cache"],
        "port": free_port(), "max_active_nodes": 2, "gpu_lease_ttl_s": 90,
        "cleanup_wait_s": 90, "runtime_dir": str(run / "dashboard_guard"),
        "gpu_registry_dir": "/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry",
        "workload": {"kind": "hy3_layer_window_bank_preparation", "model_loaded": False,
            "attention_compute": False, "expert_compute": False, "gpu_used": False},
        "model_cache": contract, "frozen_inputs": sorted(frozen.values(), key=lambda r:r["path"]),
        "sources": [{"worktree": str(ENGINE), "commit": head}],
        "tasks": [{"id": "layer-bank", "label": "Prepare immutable CPU layer windows",
                   "root": str(run / "task"), "expected": 2, "nodes": NODES}]}
    save(run / "dashboard_spec.json", spec)
    model_cache_adapter.validate_contract(spec, launch=True)
    print(json.dumps({"spec": str(run / "dashboard_spec.json"), "policy": policy.POLICY,
                      "frozen_files": len(frozen), "node_mutation": False, "gpu_used": False}))


if __name__ == "__main__":
    main()
