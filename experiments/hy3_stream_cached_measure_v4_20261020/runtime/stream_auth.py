"""Exact lease, source and dedicated bounded-stream admission checks."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from stream_common import sha


def private_json(path):
    path = Path(path)
    info = path.lstat()
    if (not path.is_absolute() or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError("Private UID-owned regular authorization file required")
    return json.loads(path.read_text())


def authenticate(run, cfg, *, local=False, frozen=True):
    sys.path.insert(0, cfg["guard_scripts"])
    import gpu_registry
    import model_cache_adapter as adapter
    import hy3_streaming_policy as policy
    run = Path(run)
    spec_path = run / "dashboard_spec.json"
    spec = private_json(spec_path)
    receipt_path = Path(os.environ["EXPERIMENT_GUARD_LEASE_RECEIPT"])
    receipt = gpu_registry.load_receipt(receipt_path)
    binding, _ = adapter._load_binding(os.environ["EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS"])
    contract = adapter.validate_contract(spec, launch=True)
    if (cfg["run_dir"] != str(run)
            or binding.get("policy") != policy.POLICY or contract.get("policy") != policy.POLICY
            or binding.get("provider") != policy.PROVIDER or contract.get("provider") != policy.PROVIDER
            or binding.get("support_profile") != policy.PROFILE
            or contract.get("support_profile") != policy.PROFILE
            or binding.get("experiment_id") != cfg["id"]
            or binding.get("contract_sha256") != adapter._json_sha256(contract)
            or binding.get("spec_sha256") != adapter._json_sha256(spec)
            or binding.get("lease_receipt_sha256") != sha(receipt_path)
            or binding.get("lease_id") != receipt.get("lease_id")):
        raise RuntimeError("Streaming admission differs from exact frozen run")
    registry = gpu_registry.SharedGpuRegistry(receipt["registry_dir"])
    lease = registry.assert_owned(receipt, cfg["id"], sha(spec_path))
    registry.assert_owned_nodes(receipt, cfg["nodes"])
    if (time.time() - lease["heartbeat_at"] > lease["ttl_s"]
            or sorted(binding["acquired_nodes"]) != sorted(cfg["nodes"])):
        raise RuntimeError("Exact-node lease is not live")
    if frozen:
        head = subprocess.check_output(["git", "-C", cfg["engine_root"], "rev-parse", "HEAD"], text=True).strip()
        if head != cfg["engine_commit"] or subprocess.run(
                ["git", "-C", cfg["engine_root"], "diff", "--quiet", "HEAD", "--"], check=False).returncode:
            raise RuntimeError("Hy3 engine revision or tracked source differs")
        for descriptor in spec["frozen_inputs"]:
            if sha(descriptor["path"]) != descriptor["sha256"]:
                raise RuntimeError("Frozen input changed: " + descriptor["path"])
    if local:
        node = os.uname().nodename.split(".")[0]
        if node not in cfg["nodes"] or not re.search(
                r"(?:^|/)job_" + re.escape(cfg["jobs"][node]) + r"(?:/|$)",
                Path("/proc/self/cgroup").read_text(), re.MULTILINE):
            raise RuntimeError("Worker is not in exact booked node/job")
        if Path(sys.executable).resolve() != Path(cfg["python"]).resolve():
            raise RuntimeError("Worker interpreter differs")
        descriptor = contract["node_specs"][node]
        node_spec = private_json(descriptor["path"])
        if (sha(descriptor["path"]) != descriptor["sha256"]
                or node_spec["engine_commit"] != cfg["engine_commit"]
                or node_spec["expected_job_id"] != cfg["jobs"][node]
                or os.environ.get("EXPERIMENT_GUARD_HY3_MODEL_IDENTITY") != contract["model_manifest"]["sha256"]
                or os.environ.get("EXPERIMENT_GUARD_MODEL_CACHE_ADMISSION_ID") != binding["admission_id"]):
            raise RuntimeError("Worker source/admission identity differs")
    return lease, binding
