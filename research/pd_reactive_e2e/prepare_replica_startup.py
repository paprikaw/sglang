"""Freeze a paired PP2 startup probe against the prior replica server configuration."""

import argparse
import copy
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys

import run as harness

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1]
q, h = harness.q, harness.h
GIB = 1024**3


def prepare(run, prior, nodes, overlay, port, prepared_replica=False):
    nodes = list(nodes)
    if len(set(nodes)) != 2:
        raise ValueError("Expected different Prefill and Decode nodes")
    job_output = subprocess.check_output(
        ["squeue", "--me", "--states=R", "--noheader", "--format=%N|%i"], text=True
    )
    jobs = dict(
        line.strip().split("|") for line in job_output.splitlines() if "|" in line
    )
    for node in nodes:
        if node not in jobs:
            raise ValueError(f"No running booking on {node}")
    rev = subprocess.check_output(
        ["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(ENGINE), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise ValueError("Freeze the engine commit first")
    tag = subprocess.check_output(
        [
            "git",
            "-C",
            str(ENGINE),
            "describe",
            "--tags",
            "--exact-match",
            "--match",
            "experiment/sglang/qwen122b-sllmstore-*",
            rev,
        ],
        text=True,
    ).strip()
    config = json.loads((prior / "config.json").read_text())
    spec = json.loads((prior / "dashboard_spec.json").read_text())
    originals = config["servers"]
    prior_engine = Path(config["engine_root"])
    run.mkdir(mode=0o700, parents=True)
    for name in ("cache", "task", "servers", "node_cleanup", "status_stacks"):
        (run / name).mkdir()
    prep = run / "replica-preparation"
    if prepared_replica:
        prep.mkdir(mode=0o700)
    key = run / "admin.key"
    key.write_text(secrets.token_urlsafe(32))
    key.chmod(0o600)
    cache = spec["model_cache"]
    # Two PP2 replicas each allocate two 32-GiB rank caches, including
    # transient old/new pools; allow 128 GiB for CPU weight offload. The
    # 122B official host pool, rounded to 32-MiB chunks, is <120 GiB.
    # The prepared comparison has one 136-GiB Store used sequentially. Its
    # peak preparation includes two independent <=64-GiB HiCache banks, two
    # <=64-GiB sealed stage exports, <=16 GiB of dense staging and 8 GiB for
    # transient copies. The raw control retires before export. Guard adds its
    # separate headroom and canonical-weight reservation in both modes.
    cache.update(
        usage_logs={n: [] for n in nodes},
        node_specs={},
        hicache_reserve_bytes_per_node={
            nodes[0]: 160 * GIB,
            nodes[1]: (416 if prepared_replica else 512) * GIB,
        },
        cpu_memory_budget=dict(
            schema_version=1, kv_banks_per_node={n: [] for n in nodes}
        ),
    )
    frozen = []
    for node in nodes:
        old_node = originals["p" if node == nodes[0] else "d1"]["node"]
        nc = json.loads((prior / "cache" / (old_node + ".json")).read_text())
        nc.update(node=node, expected_job_id=jobs[node])
        path = run / "cache" / (node + ".json")
        h.save(path, nc)
        cache["node_specs"][node] = q.describe(path)
        frozen.append(path)
    servers = {}
    for sid, source, node, port_offset in [
        ("p", "p", nodes[0], 1),
        ("d0", "d0", nodes[1], 11),
        ("d_raw" if prepared_replica else "d_cpu", "d1", nodes[1], 21),
        ("d_store", "d1", nodes[1], 31),
    ]:
        cfg = copy.deepcopy(originals[source])
        cfg.update(
            id=run.name,
            node=node,
            job_id=jobs[node],
            engine_root=str(ENGINE),
            code_root=str(ENGINE / "research/pp_elastic"),
            engine_commit=rev,
            engine_tag=tag,
            server_id=sid,
            model_cache=cache,
            admin_key_file=str(key),
            usage_audit_hostname=socket.gethostname().split(".")[0],
            port=port + port_offset,
            dist_port=port + port_offset + 1,
            bootstrap_port=port + 45,
            store_overlay=str(overlay),
        )
        cfg["launch_env"].update(ELASTIC_PD_STATUS_STACKS=str(run / "status_stacks"))
        if sid != "p":
            cfg["launch_env"]["KVCACHED_IPC_NAME"] = run.name + "-" + sid
            cfg["launch_env"]["SGLANG_PP2_STARTUP_PROBE"] = "1"
        if sid in ("d_store", "d_raw"):
            cfg["sllmstore_root"] = "/tmp/" + run.name + "-weights"
        if prepared_replica and sid in ("d0", "d_store"):
            cfg["launch_env"].update(
                SGLANG_REPLICA_ENGINE_COMMIT=rev,
                SGLANG_REPLICA_MARLIN_SHA256=q.describe(
                    ENGINE
                    / "research/pp_elastic/runtime_profiles/agentx_a100/fp8_marlin_adapter.py"
                )["sha256"],
            )
            name = (
                "SGLANG_REPLICA_EXPORT_DIR"
                if sid == "d0"
                else "SGLANG_REPLICA_PREPARED_DIR"
            )
            cfg["launch_env"][name] = str(prep)
        use = dict(
            id=run.name + "." + sid,
            path=str(run / "cache" / (sid + ".log")),
            expected_ranks=["PP0 TP0", "PP1 TP0"],
        )
        cfg["layouts"][0]["usage_logs"] = [use]
        cache["usage_logs"][node].append(use)
        servers[sid] = cfg
    node_cfgs = {
        node: copy.deepcopy(next(v for v in servers.values() if v["node"] == node))
        for node in nodes
    }
    for n, cfg in node_cfgs.items():
        cfg["layouts"] = [
            layout
            for s in servers.values()
            if s["node"] == n
            for layout in s["layouts"]
        ]
    config.update(
        id=run.name,
        servers=servers,
        node_configs=node_cfgs,
        model_cache=cache,
        client_node=nodes[1],
        admin_key_file=str(key),
        engine_root=str(ENGINE),
        engine_commit=rev,
        prepared_replica=prepared_replica,
        store_reserve_bytes=(136 if prepared_replica else 128) * GIB,
        spare_hicache_reserve_bytes=64 * GIB if prepared_replica else 0,
        scope="two fresh fixed PP2 starts, one measured request each",
    )
    # Native status polling, Graph B1/B2/B4, 32-GiB-per-replica HiCache flags
    # and the existing D process are identical for both arms.
    h.save(run / "config.json", config)
    spec.update(
        id=run.name,
        title="122B PP2 Decode replica weight loading",
        port=port,
        runtime_dir=str(run / "dashboard_guard"),
        tasks=[
            dict(
                id="comparison",
                label="Two fresh PP2 startups",
                expected=1,
                root=str(run / "task"),
                nodes=nodes,
                controller_script=str(HERE / "replica_startup.py"),
            )
        ],
    )
    for item in spec["sources"]:
        if item["role"] == "inference_engine":
            item.update(commit=rev, worktree=str(ENGINE), tag=tag)
    # The immutable source checks protect the entire engine. Pin the startup
    # controller, relevant runtime files, official native binary and Python API.
    for e in spec["frozen_inputs"]:
        path = Path(e["path"])
        if path.is_relative_to(prior_engine):
            path = ENGINE / path.relative_to(prior_engine)
        elif path.is_relative_to(prior):
            continue
        if path.is_file():
            frozen.append(path)
    frozen.extend(
        [
            run / "config.json",
            HERE / "replica_startup.py",
            HERE / "prepare_replica_startup.py",
            ENGINE / "research/pp_elastic/startup_timing_probe.py",
            ENGINE / "python/sglang/srt/weight_cache/sllmstore_stage.py",
        ]
    )
    if prepared_replica:
        frozen.extend(
            [
                ENGINE / "python/sglang/srt/weight_cache/replica_preparation.py",
                ENGINE / "python/sglang/srt/elastic_pp/replica_host_cache.py",
            ]
        )
    native = overlay / "sllm_store"
    frozen.extend(
        p for p in native.rglob("*") if p.is_file() and p.suffix in (".py", ".so")
    )
    frozen.append((native / "libglog.so").resolve())
    sys.path.insert(0, str(q.GUARD))
    import cpu_memory_budget

    frozen.extend(
        Path(e["path"]) for e in cpu_memory_budget.dependency_identity()["files"]
    )
    spec["frozen_inputs"] = [q.describe(p) for p in dict.fromkeys(frozen)]
    h.save(run / "dashboard_spec.json", spec)
    # This measurement requires a real P→D handoff. Check for compatible
    # prior persisted states, then warm the exact prefix on the running P.
    registry = Path(
        "/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json"
    )
    entries = json.loads(registry.read_text())["entries"]
    values = entries.values() if isinstance(entries, dict) else entries
    matches = [
        e
        for e in values
        if e.get("compatibility", e.get("spec", {}))
        .get("engine", {})
        .get("source_commit")
        == rev
    ]
    h.save(
        run / "checkpoint_lookup.json",
        dict(
            registry=str(registry),
            entry_count=len(entries),
            exact_engine_candidates=len(matches),
            engine_commit=rev,
            prefill_nodes=[nodes[0]],
            protocol="same live HiCache on P, fresh D for each arm",
        ),
    )
    cmd = [
        q.PYTHON,
        str(q.GUARD / "dashboard_guard.py"),
        "run",
        "--spec",
        str(run / "dashboard_spec.json"),
        "--resource-choice",
        "auto",
        "--exit-after-run",
        "--",
        q.PYTHON,
        str(HERE / "replica_startup.py"),
        "--run",
        str(run),
    ]
    h.save(
        run / "preparation.json",
        dict(
            command=cmd,
            engine=rev,
            overlay=str(overlay),
            frozen_count=len(spec["frozen_inputs"]),
        ),
    )
    print(run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prior", required=True, type=Path)
    p.add_argument("--nodes", nargs=2, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--port", type=int, default=30720)
    p.add_argument("--prepared-replica", action="store_true")
    a = p.parse_args()
    prepare(
        a.run.resolve(),
        a.prior.resolve(),
        a.nodes,
        a.overlay.resolve(),
        a.port,
        a.prepared_replica,
    )
