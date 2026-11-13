"""One Guard-owned 122B/A100 P2 + D(PP2->PP3) qualification.

Not an autoscaling benchmark. Admission is quiescent during each synchronous
transition. The physical metadata workers stay resident; a separate process
tests actual reuse of the retired middle GPU, with residual memory reported.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
PROFILE = ROOT / "runtime_profiles/agentx_a100"
sys.path.insert(0, str(PROFILE))
import process_helpers as h

GUARD = Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts")
PRIOR = Path("/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/agentx_pd_reactive_burst_20261113/runs/p2d2_start_r1")
PYTHON = "/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python"


def save(path, value):
    h.save(path, value)


def progress(run, root, stage, valid=0):
    identity = h.process_record(os.getpid())
    save(run / "task/progress.json", dict(experiment_id=root["id"],
        task_id="qualification", timestamp=time.time(), stage=stage,
        controller_pid=os.getpid(), controller_start_ticks=identity["start_ticks"],
        process_start_ticks=identity["start_ticks"], valid_points=valid, expected=1))


def describe(path):
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def memory_size_json(value):
    """Pool byte counters can contain NumPy scalars; keep HTTP diagnostics JSON-safe."""
    if isinstance(value, (tuple, list)):
        return [memory_size_json(item) for item in value]
    return int(value)


def prepare(run, *, roundtrip=False):
    run.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ENGINE), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("Commit the qualification candidate before preparing a run")
    old = json.loads((PRIOR / "config.json").read_text())
    key = run / "admin.key"
    key.write_text(secrets.token_hex(32)); key.chmod(0o600)
    spec = json.loads((PRIOR / "dashboard_spec.json").read_text())
    spec.update(id=run.name, port=29570, runtime_dir=str(run / "dashboard_guard"),
                gpu_lease_ttl_s=30, title="122B A100 P/D retained-HiCache PP2→PP3 qualification")
    cache = spec["model_cache"]
    cache["usage_logs"] = {}
    cache["node_specs"] = {}
    cache["hicache_reserve_bytes_per_node"] = {}
    nodes = {}
    for index, (node, previous) in enumerate(old["node_configs"].items()):
        cfg = dict(previous)
        pp = 2 if index == 0 else 3
        role = "prefill" if index == 0 else "decode"
        use = dict(id=f"{run.name}.{role}", path=str(run / "cache" / f"{role}.log"),
                   expected_ranks=[f"PP{rank} TP0" for rank in range(pp)])
        row = dict(id=role, replicas=1, pp=pp, tp=1, ep=1, dp=1,
                   gpu_groups=[list(range(pp))], usage_logs=[use])
        cfg.update(id=run.name, engine_root=str(ENGINE), engine_commit=revision,
                   guard_scripts=str(GUARD), admin_key_file=str(key), code_root=str(ROOT),
                   usage_audit_hostname=socket.gethostname().split('.')[0],
                   role=role, layouts=[row], port=29581 + index * 10,
                   dist_port=29582 + index * 10, bootstrap_port=29579)
        nodes[node] = cfg
        node_spec = json.loads((PRIOR / "cache" / f"{node}.json").read_text())
        node_path = run / "cache" / f"{node}.json"
        save(node_path, node_spec)
        cache["node_specs"][node] = describe(node_path)
        cache["usage_logs"][node] = [use]
        # Host cache + CPU KV/GDN spill staging share an explicit reservation.
        cache["hicache_reserve_bytes_per_node"][node] = 32 * 1024**3
    for cfg in nodes.values():
        cfg["model_cache"] = cache
    config = dict(id=run.name, python=PYTHON, node_configs=nodes,
                  admin_key_file=str(key), engine_commit=revision,
                  engine_root=str(ENGINE), output_tokens=2048,
                  admission="quiescent P/D transfers during synchronous fences",
                  logical_gpu_counts=[4, 5, 4] if roundtrip else [4, 5],
                  resident_worker_gpus=5, reserved_gpus=8,
                  roundtrip=roundtrip, gpu_reuse_probe_gib=64 if roundtrip else 0)
    save(run / "config.json", config)
    spec["tasks"] = [dict(id="qualification", label="Request/cache/reuse acceptance",
        root=str(run / "task"), expected=1, nodes=list(nodes), controller_script=str(Path(__file__).resolve()))]
    spec["sources"] = [dict(role="inference_engine", worktree=str(ENGINE), commit=revision),
                        spec["sources"][-1]]
    paths = [run / "config.json", *(run / "cache").glob("*.json")]
    tracked = subprocess.check_output(["git", "-C", str(ENGINE), "ls-files", "python/sglang", "research/pp_elastic"], text=True).splitlines()
    paths.extend(ENGINE / p for p in tracked if p.endswith((".py", ".json")))
    paths.extend(GUARD.glob("*.py"))
    bank = Path("/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank/src/sglang_memory_bank")
    paths.extend(bank / name for name in ("__init__.py", "memory_budget.py", "memory_bank.py"))
    spec["frozen_inputs"] = [describe(p) for p in dict.fromkeys(paths)]
    save(run / "dashboard_spec.json", spec)
    save(run / "preparation.json", dict(command=[PYTHON, str(GUARD / "dashboard_guard.py"),
        "run", "--spec", str(run / "dashboard_spec.json"), "--resource-choice", "auto",
        "--", PYTHON, str(Path(__file__).resolve()), "--run", str(run)]))
    print(json.dumps({"run": str(run), "source": revision, "frozen_files": len(paths)}))


def node_config(root):
    return root["node_configs"][socket.gethostname().split('.')[0]]


def environment(cfg):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("SGLANG_", "QWEN_A100_", "MOTIVATION_", "PP_HICACHE_")):
            env.pop(key)
    env.update(cfg["launch_env"])
    env.update(PYTHONPATH=os.pathsep.join([str(ENGINE / "python"), str(ROOT), str(PROFILE),
        cfg["mooncake_dependency_path"]]), CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["layouts"][0]["gpu_groups"][0])),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="1", QWEN_A100_FP8_MOE_MARLIN="1",
        QWEN_A100_FP8_MOE_MARLIN_QUALIFICATION="1", ELASTIC_PD_ROLE=cfg["role"])
    return env


def install_hooks():
    import fp8_marlin_adapter
    fp8_marlin_adapter.install()
    from sglang.srt.model_executor.model_runner import ModelRunner
    old_load = ModelRunner.load_model

    def load(runner, *args, **kwargs):
        class RankFilter(logging.Filter):
            def filter(self, record):
                if str(record.msg).startswith("CPU weight cache load:"):
                    record.msg = f"[PP{runner.ps.pp_rank} TP{runner.ps.tp_rank}] " + record.msg
                return True
        log = logging.getLogger("sglang.srt.weight_cache.cpu_canonical_loader")
        label = RankFilter()
        log.addFilter(label)
        try:
            result = old_load(runner, *args, **kwargs)
        finally:
            log.removeFilter(label)
        local = [(n, m) for n, m in runner.model.named_modules()
                 if hasattr(m, "_qwen_a100_fp8_marlin_geometry")]
        count = runner.model.end_layer - runner.model.start_layer
        if len(local) != count:
            raise RuntimeError(f"Incomplete native FP8 coverage: {len(local)} != {count}")
        print("ELASTIC_PD_FP8_COVERAGE", json.dumps(dict(pp_rank=runner.ps.pp_rank,
              layers=len(local), start=runner.model.start_layer, end=runner.model.end_layer)), flush=True)
        return result
    ModelRunner.load_model = load
    if os.environ["ELASTIC_PD_ROLE"] == "decode":
        import cache_probe
        cache_probe.install()
        import hicache_transition_smoke as probe
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
        from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
        probe.install_diagnostic_fanout(Scheduler, TokenizerControlMixin)
        probe.install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator)
        old_state = Scheduler.get_internal_state

        def state(scheduler, *args, **kwargs):
            import torch
            result = old_state(scheduler, *args, **kwargs)
            pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
            def bytes_of(values):
                storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                            for t in values if t.is_cuda and t.numel()}
                return sum(storages.values())
            result.internal_state["pd_release_memory"] = dict(
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                free_bytes=torch.cuda.mem_get_info()[0], total_bytes=torch.cuda.mem_get_info()[1],
                model_bytes=bytes_of(list(scheduler.tp_worker.model_runner.model.parameters()) +
                                    list(scheduler.tp_worker.model_runner.model.buffers())),
                kv_bytes=memory_size_json(pool.get_kv_size_bytes()), pid=os.getpid())
            return result
        Scheduler.get_internal_state = state


if os.environ.get("ELASTIC_PD_ROLE"):
    install_hooks()


def engine_entry(run, cfg):
    h.authenticate(run, cfg, os.environ, local=True)
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    argv = ["--model-path", cfg["model_path"], "--host", "0.0.0.0", "--port", str(cfg["port"]),
        "--dist-init-addr", f"127.0.0.1:{cfg['dist_port']}", "--tp-size", "1", "--ep-size", "1",
        "--pp-size", str(cfg["layouts"][0]["pp"]), "--dtype", "bfloat16", "--quantization", "fp8",
        "--kv-cache-dtype", "bfloat16", "--context-length", "8192", "--max-total-tokens", "8192",
        "--max-running-requests", "4", "--max-mamba-cache-size", "16", "--mamba-ssm-dtype", "float32",
        "--mamba-max-states-per-path", "1", "--mem-fraction-static", ".90", "--page-size", "64",
        "--chunked-prefill-size", "512", "--max-prefill-tokens", "1024", "--pp-max-micro-batch-size", "4",
        "--pp-async-batch-depth", "0", "--attention-backend", "triton", "--linear-attn-backend", "triton",
        "--moe-runner-backend", "marlin", "--moe-a2a-backend", "none", "--disable-cuda-graph",
        "--skip-server-warmup", "--json-model-override-args", '{"language_model_only":true}',
        "--enable-cache-report", "--enable-hierarchical-cache", "--hicache-size", "8",
        "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel",
        "--hicache-mem-layout", "page_first", "--disaggregation-mode", cfg["role"],
        "--disaggregation-transfer-backend", "mooncake", "--disaggregation-bootstrap-port", str(cfg["bootstrap_port"]),
        "--disaggregation-ib-device", "mlx5_bond_0"]
    if cfg["role"] == "decode":
        argv += ["--enable-elastic-pp", "--elastic-pp-disagg-decode", "--elastic-pp-initial-partition", "24,0,24",
                 "--elastic-pp-transition-strategy", "sync", "--elastic-pp-transition-backing", "cpu_spill",
                 "--disaggregation-decode-enable-radix-cache", "--disaggregation-decode-extra-slots", "4"]
    args = prepare_server_args(argv)
    args.enable_multimodal = False
    args.api_key = args.admin_api_key = h.read_key(cfg)
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def reuse_probe(run, cfg):
    h.authenticate(run, cfg, os.environ, local=True)
    import torch
    before = torch.cuda.mem_get_info()
    block = torch.empty(64 * 1024**3, dtype=torch.uint8, device="cuda")
    block.fill_(19)
    matrix = torch.ones(1024, 1024, device="cuda")
    answer = matrix @ matrix
    torch.cuda.synchronize()
    assert torch.all(answer == 1024).item()
    save(run / "gpu_reuse_started.json", dict(passed=True, pid=os.getpid(),
         node=cfg["node"], physical_gpu=1, independent_process=True,
         allocated_bytes=torch.cuda.memory_allocated(), free_before_bytes=before[0],
         free_during_bytes=torch.cuda.mem_get_info()[0], timestamp=time.time()))
    time.sleep(5)
    save(run / "gpu_reuse_done.json", dict(passed=True, timestamp=time.time()))


def worker(run, root):
    cfg = node_config(root)
    packet = json.loads(sys.stdin.readline())
    os.environ.update(packet["environment"])
    h.authenticate(run, cfg, os.environ, local=True)
    key = h.read_key(cfg)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    children = []
    directory = run / "nodes" / cfg["node"]
    try:
        topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                  text=True, check=True, timeout=20)
        save(directory / "topology.json", dict(stdout=topology.stdout))
        command = [PYTHON, str(GUARD / "model_cache_adapter.py"), "exec-local", "--bindings",
            os.environ[h.GUARD_KEYS[3]], "--", PYTHON, str(Path(__file__).resolve()), "--engine", "--run", str(run)]
        proc = subprocess.Popen(command, env=environment(cfg), cwd=ENGINE, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        identity = h.process_record(proc.pid)
        assert identity and identity["pgid"] == proc.pid
        thread = threading.Thread(target=h.stream_log, args=(proc.stdout,
            cfg["layouts"][0]["usage_logs"][0]["path"], h.read_key(cfg)), daemon=True)
        thread.start()
        children.append(dict(process=proc, identity=identity, log_thread=thread))
        save(directory / "worker_identity.json", identity)
        deadline = time.monotonic() + 1200
        while not stop.is_set():
            if proc.poll() is not None:
                raise RuntimeError("Engine exited during startup")
            try:
                h.http_json(cfg["port"], "/get_server_info", h.read_key(cfg), timeout=2)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError("Engine startup deadline")
                stop.wait(1)
        save(directory / "ready.json", dict(passed=True, timestamp=time.time()))
        while not stop.wait(.5):
            h.authenticate(run, cfg, os.environ, local=True)
            if proc.poll() is not None:
                raise RuntimeError("Engine exited while serving")
            if cfg["role"] == "decode" and (run / "probe.trigger").exists() and len(children) == 1:
                probe_env = dict(os.environ, **cfg["launch_env"], CUDA_VISIBLE_DEVICES="1")
                probe = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "--probe", "--run", str(run)],
                    env=probe_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                probe_thread = threading.Thread(target=h.stream_log,
                    args=(probe.stdout, str(directory / "probe.log"), key), daemon=True)
                probe_thread.start()
                children.append(dict(process=probe, identity=h.process_record(probe.pid),
                                     log_thread=probe_thread))
            if (run / "stop.json").exists():
                break
    except BaseException as error:
        save(directory / "failed.json", dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        result = h.cleanup(children)
        save(directory / "cleanup.json", result)
        if not result["cleanup_verified"]:
            raise RuntimeError("Owned GPU cleanup failed")


def controller(run, root):
    import agentx_replay as replay
    import hicache_transition_smoke as probe
    cfgs = list(root["node_configs"].values())
    def stop(*_):
        raise KeyboardInterrupt("Guard requested owned cleanup")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for cfg in cfgs:
        h.authenticate(run, cfg, os.environ)
    key = h.read_key(cfgs[0])
    endpoints = [dict(host=cfg["node"], port=cfg["port"], ep=1,
                     bootstrap_port=cfg["bootstrap_port"]) for cfg in cfgs]
    p, d = endpoints
    workers = []
    results = dict(passed=False, static_reference_only=False, transitions=[])
    def health():
        for cfg, worker_process in zip(cfgs, workers):
            h.authenticate(run, cfg, os.environ)
            if worker_process.poll() is not None:
                raise RuntimeError("Remote worker exited: " + cfg["node"])
    def http(path, payload=None):
        return replay.http(d["host"], d["port"], path, key, payload, 360)
    def state(label):
        info = http("/get_server_info")
        saved = probe.state_evidence(info)
        for source, target in zip(info["internal_states"], saved["internal_states"]):
            target["pd_release_memory"] = source.get("pd_release_memory")
        save(run / f"state.{label}.json", saved)
        return saved["internal_states"]
    def wait_file(path, seconds=60):
        deadline = time.monotonic() + seconds
        while not path.exists():
            health()
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(.25)
        subprocess.run([PYTHON, str(GUARD / "artifact_visibility.py"), str(path),
            "--timeout-s", "20", "--receipt", str(path.with_name(path.stem + ".visibility.json"))],
            check=True, stdout=subprocess.DEVNULL, timeout=25)
    try:
        progress(run, root, "starting")
        packet = json.dumps(dict(environment={k: os.environ[k] for k in h.GUARD_KEYS})) + "\n"
        for cfg in cfgs:
            directory = run / "nodes" / cfg["node"]
            directory.mkdir(parents=True, exist_ok=True)
            command = [PYTHON, str(Path(__file__).resolve()), "--worker", "--run", str(run)]
            remote = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", cfg["node"], shlex.join(command)],
                stdin=subprocess.PIPE, stdout=(directory / "controller.log").open("wb"), stderr=subprocess.STDOUT)
            remote.stdin.write(packet.encode()); remote.stdin.flush()
            workers.append(remote)
        for cfg in cfgs:
            wait_file(run / "nodes" / cfg["node"] / "ready.json", 1200)
        progress(run, root, "reference_and_roundtrip")
        tokens = [1000 + (i * 31) % 20000 for i in range(512)]
        results["warmup"] = replay.request_one(p, d, key, tokens, 1, 0, True)
        prompt = tokens + [198]
        results["reference"] = replay.request_one(p, d, key, prompt, root["output_tokens"], 0, True)
        save(run / "results.partial.json", results)
        events = []
        original_stream = replay.stream
        def observed_stream(endpoint, key, payload, timeout=600, **kwargs):
            return original_stream(endpoint, key, payload, timeout, event_sink=events)
        replay.stream = observed_stream
        try:
            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(replay.request_one, p, d, key, prompt, root["output_tokens"], 0, True)
                transitions = [(1, [16, 16, 16])]
                if root.get("roundtrip"):
                    transitions.append((2, [24, 0, 24]))
                for generation, counts in transitions:
                    minimum = 8 if generation == 1 else events[-1][0] + 8
                    deadline = time.monotonic() + 90
                    while not events or events[-1][0] < minimum:
                        health()
                        if active.done():
                            raise RuntimeError("Request finished before its migration fence")
                        if time.monotonic() > deadline:
                            raise TimeoutError("No progressing active Decode request")
                        time.sleep(.02)
                    before = state(f"before{generation}")
                    started = time.time()
                    answer = http("/reconfigure_elastic_pp", dict(layer_counts=counts, generation=generation))
                    results["transitions"].append(dict(generation=generation, started_at=started,
                        finished_at=time.time(), answer=answer, before=before))
                    save(run / "results.partial.json", results)
                    if not answer.get("success"):
                        raise RuntimeError("PP migration was rejected: " + str(answer))
                    deadline = time.monotonic() + 60
                    while True:
                        after = state(f"after{generation}")
                        if all(s.get("transition_metrics", {}).get("phase") == "done" for s in after):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError("Migration cleanup did not finish")
                        time.sleep(.1)
                    probe.validate_fenced_decode(after, generation, (0, 2) if generation == 1 else (0, 1, 2), root["output_tokens"])
                    results["transitions"][-1]["after"] = after
                if root.get("roundtrip"):
                    (run / "probe.trigger").touch(mode=0o600)
                    wait_file(run / "gpu_reuse_started.json", 90)
                results["active"] = active.result(timeout=240)
        finally:
            replay.stream = original_stream
        results["stream_events"] = events
        completed = state("completed")
        continuity_paths = [(1, (0, 2), (0, 1, 2))]
        if root.get("roundtrip"):
            continuity_paths.append((2, (0, 1, 2), (0, 2)))
        results["continuity"] = [probe.validate_uninterrupted_decode(
            completed, generation, results["active"], sources, targets
        ) for generation, sources, targets in continuity_paths]
        results["exact_output_match"] = results["reference"]["output_ids"] == results["active"]["output_ids"]
        if not results["exact_output_match"] or not results["active"]["output_ids"]:
            raise AssertionError("Transition output IDs differ from unchanged-layout warm reference")
        if root.get("roundtrip"):
            wait_file(run / "gpu_reuse_done.json", 60)
            results["gpu_reuse"] = json.loads((run / "gpu_reuse_started.json").read_text())
        results["after_handoff"] = replay.request_one(p, d, key, prompt, 32, 0, True)
        results["demote"] = probe.demote_cpu_prefixes(http)
        results["cpu_restore"] = replay.request_one(p, d, key, prompt, 32, 0, True)
        final = state("final")
        results["restore_ids_match"] = results["cpu_restore"]["output_ids"] == results["after_handoff"]["output_ids"]
        if not results["restore_ids_match"]:
            raise AssertionError("HiCache CPU restore changed output IDs")
        target_ranks = (0, 2) if root.get("roundtrip") else (0, 1, 2)
        restored = probe.fresh_prefix_restores(final, results["demote"]["commanded_at"], target_ranks, 512)
        results.update(passed=True, cpu_restores=restored, final_state=final,
                       whole_worker_gpu_exit=False,
                       independent_gpu_reuse_verified=bool(root.get("roundtrip")))
    except BaseException as error:
        results.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        save(run / "stop.json", dict(timestamp=time.time()))
        for remote in workers:
            remote.stdin.close()
        for remote in workers:
            try:
                remote.wait(timeout=90)
            except subprocess.TimeoutExpired:
                remote.terminate(); remote.wait(timeout=10)
        cleanups = []
        for cfg in cfgs:
            path = run / "nodes" / cfg["node"] / "cleanup.json"
            if path.exists():
                wait_file(path, 20)
                cleanups.append(json.loads(path.read_text()))
        cleanup_passed = len(cleanups) == len(cfgs) and all(
            c.get("cleanup_verified") for c in cleanups)
        try:
            if not cleanup_passed or not all(p.returncode == 0 for p in workers):
                raise RuntimeError("Owned remote execution/cleanup did not pass")
            for cfg in cfgs:
                for row in cfg["layouts"]:
                    save(run / "cache" / f"{cfg['role']}.usage_audit.json", h.verify_layout_usage(cfg, row, os.environ))
        except Exception as error:
            results.update(passed=False, final_audit_error=str(error))
        save(run / "results.json", results)
        progress(run, root, "complete" if results["passed"] else "failed", int(results["passed"]))
        save(run / "task/cleanup.json", dict(experiment_id=root["id"], task_id="qualification",
            passed=cleanup_passed, gpu_idle=cleanup_passed, model_processes_exited=cleanup_passed,
            timestamp=time.time(), worker_returncodes=[p.returncode for p in workers]))
    if not results["passed"]:
        raise RuntimeError("Qualification failed; preserved evidence at " + str(run))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    for option in ("prepare", "worker", "engine", "probe"):
        parser.add_argument("--" + option, action="store_true")
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.prepare:
        return prepare(run, roundtrip=args.roundtrip)
    root = json.loads((run / "config.json").read_text())
    if args.worker:
        return worker(run, root)
    if args.engine:
        return engine_entry(run, node_config(root))
    if args.probe:
        return reuse_probe(run, node_config(root))
    return controller(run, root)


if __name__ == "__main__":
    main()
