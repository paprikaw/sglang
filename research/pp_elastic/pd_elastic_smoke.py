"""One Guard-owned 122B/A100 P2 + D(PP2->PP3->PP2) qualification.

Not an autoscaling benchmark. Admission is quiescent during each
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
NIXL_ROOT = Path("/data/gpfs/projects/punim2715/vllm_workbench/.dependency-runtimes/sglang-nixl-cu12-1.5.2")
KVCACHED_ROOT = Path("/data/gpfs/projects/punim2715/vllm_workbench/.dependency-worktrees/kvcached-stream-mapping-20261114")


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


def prepare(run, optimized=True, diagnostic_prepare=False, native_prepare=False,
            lazy_cuda_modules=False, kvcached_capacity=False):
    run.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ENGINE), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("Commit the qualification candidate before preparing a run")
    old = json.loads((PRIOR / "config.json").read_text())
    key = run / "admin.key"
    key.write_text(secrets.token_hex(32)); key.chmod(0o600)
    spec = json.loads((PRIOR / "dashboard_spec.json").read_text())
    spec.update(id=run.name, port=29670 if kvcached_capacity else 29470, runtime_dir=str(run / "dashboard_guard"),
                gpu_lease_ttl_s=30, title="122B A100 P/D PP2→PP3→PP2 qualification")
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
                   role=role, layouts=[row], port=29481 + index * 10,
                   dist_port=29482 + index * 10, bootstrap_port=29479)
        cfg.update(optimized_elastic_pp=optimized and role == "decode",
                   cuda_graph=optimized and role == "decode", hicache_size_gib=8,
                   prepare_diagnostic_dir=str(run / "diagnostics") if diagnostic_prepare and role == "decode" else None,
                   native_prepare_dir=str(run / "native_stacks") if native_prepare and role == "decode" else None,
                   cpu_weight_cache_gib=64 if optimized and role == "decode" else 0,
                   nixl_root=str(NIXL_ROOT) if optimized and role == "decode" else None)
        cfg["launch_env"] = dict(cfg["launch_env"])
        cfg["kvcached_capacity"] = kvcached_capacity
        if kvcached_capacity:
            cfg.update(port=29681 + index * 10, dist_port=29682 + index * 10,
                       bootstrap_port=29679)
        if kvcached_capacity and role == "decode":
            cfg["kvcached_root"] = str(KVCACHED_ROOT)
            cfg["launch_env"].update(KVCACHED_LAYER_STACKING="true",
                KVCACHED_PAGE_PREALLOC_ENABLED="false", KVCACHED_CONTIGUOUS_LAYOUT="true",
                KVCACHED_PHYSICAL_BLOCK_SIZE_MB="2", KVCACHED_MIN_RESERVED_PAGES="0",
                KVCACHED_MAX_RESERVED_PAGES="0", KVCACHED_GPU_DIRECT_RDMA_REQUIRED="1")
        if optimized and role == "decode" and not lazy_cuda_modules:
            # First-use NIXL gather kernels on the dormant rank must not load
            # CUDA modules while the real CPU weight restore is in flight.
            cfg["launch_env"].update(CUDA_MODULE_LOADING="EAGER",
                                     CUDA_MODULE_DATA_LOADING="EAGER")
        nodes[node] = cfg
        node_spec = json.loads((PRIOR / "cache" / f"{node}.json").read_text())
        node_path = run / "cache" / f"{node}.json"
        save(node_path, node_spec)
        cache["node_specs"][node] = describe(node_path)
        cache["usage_logs"][node] = [use]
        # Host cache + CPU KV/GDN spill staging share an explicit reservation.
        cache["hicache_reserve_bytes_per_node"][node] = (240 if optimized and role == "decode" else 32) * 1024**3
    for cfg in nodes.values():
        cfg["model_cache"] = cache
    config = dict(id=run.name, python=PYTHON, node_configs=nodes,
                  admin_key_file=str(key), engine_commit=revision,
                  engine_root=str(ENGINE), output_tokens=512,
                  admission="quiescent P/D transfers during PP transitions",
                  optimized_elastic_pp=optimized,
                  diagnostic_prepare=diagnostic_prepare,
                  native_prepare=native_prepare,
                  eager_cuda_modules=optimized and not lazy_cuda_modules,
                  logical_gpu_counts=[4, 5, 4], resident_worker_gpus=5,
                  reserved_gpus=8, gpu_reuse_probe_gib=64)
    config["deterministic_inference"] = True
    config["temporary_prefill_diagnostics"] = False
    config["kvcached_capacity"] = kvcached_capacity
    if kvcached_capacity:
        config["capacity_frontier"] = [8192, 12288, 8192]
        config["capacity_probe_input_tokens"] = 9216
    save(run / "config.json", config)
    spec["tasks"] = [dict(id="qualification", label="Request/cache/reuse acceptance",
        root=str(run / "task"), expected=1, nodes=list(nodes), controller_script=str(Path(__file__).resolve()))]
    spec["sources"] = [dict(role="inference_engine", worktree=str(ENGINE), commit=revision),
                        spec["sources"][-1]]
    paths = [run / "config.json", *(run / "cache").glob("*.json")]
    tracked = subprocess.check_output(["git", "-C", str(ENGINE), "ls-files", "python/sglang", "research/pp_elastic"], text=True).splitlines()
    paths.extend(ENGINE / p for p in tracked if p.endswith((".py", ".json")))
    paths.extend(GUARD.glob("*.py"))
    if optimized:
        for pattern in ("nixl/_api.py", "nixl/__init__.py", "nixl_cu12/_api.py",
                        "nixl_cu12/_bindings*.so", "nixl_cu12.libs/libucp*.so*",
                        "nixl_cu12.libs/nixl/libplugin_UCX.so"):
            matched = list(NIXL_ROOT.glob(pattern))
            if not matched:
                raise RuntimeError("Missing qualified NIXL dependency: " + pattern)
            paths.extend(matched)
    if kvcached_capacity:
        lock = json.loads((ROOT / "dependencies.lock.json").read_text())["kvcached"]
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=KVCACHED_ROOT, text=True).strip()
        if tree != lock["patched_tree"] or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=KVCACHED_ROOT, text=True).strip():
            raise RuntimeError("KVCacheD dependency must match the clean source lock")
        binaries = list((KVCACHED_ROOT / "kvcached").glob("vmm_ops*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Expected one compiled KVCacheD extension")
        dependency_files = subprocess.check_output(["git", "ls-files"], cwd=KVCACHED_ROOT, text=True).splitlines()
        paths.extend(KVCACHED_ROOT / p for p in dependency_files if p.endswith((".py", ".cpp", ".h", ".hpp")))
        paths.extend(binaries)
        spec["sources"].append(dict(role="kv_backing", worktree=str(KVCACHED_ROOT),
                                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=KVCACHED_ROOT, text=True).strip()))
        save(run / "kvcached_dependency.json", dict(tree=tree, native=describe(binaries[0]),
             build_node="spartan-gpgpu167", native_stream_mapping_abi=1,
             native_rdma_mapping_abi=1))
    bank = Path("/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank/src/sglang_memory_bank")
    paths.extend(bank / name for name in ("__init__.py", "memory_budget.py", "memory_bank.py"))
    spec["frozen_inputs"] = [describe(p) for p in dict.fromkeys(paths)]
    save(run / "dashboard_spec.json", spec)
    save(run / "preparation.json", dict(command=[PYTHON, str(GUARD / "dashboard_guard.py"),
        "run", "--spec", str(run / "dashboard_spec.json"), "--resource-choice", "auto",
        "--exit-after-run", "--", PYTHON, str(Path(__file__).resolve()), "--run", str(run)]))
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
    if cfg.get("optimized_elastic_pp"):
        env["PYTHONPATH"] = cfg["nixl_root"] + os.pathsep + env["PYTHONPATH"]
        env.update(UCX_PROTO_INFO="y", UCX_LOG_LEVEL="info")
    if cfg.get("kvcached_root"):
        env["PYTHONPATH"] = cfg["kvcached_root"] + os.pathsep + env["PYTHONPATH"]
        env.pop("KVCACHED_AUTOPATCH", None)
    if cfg.get("prepare_diagnostic_dir"):
        env["ELASTIC_PD_PREPARE_DIAGNOSTICS"] = cfg["prepare_diagnostic_dir"]
    if cfg.get("native_prepare_dir"):
        env["ELASTIC_PD_NATIVE_STACKS"] = cfg["native_prepare_dir"]
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
        import ctypes
        driver = ctypes.CDLL("libcuda.so.1")
        mode = ctypes.c_int()
        status = driver.cuModuleGetLoadingMode(ctypes.byref(mode))
        if status:
            raise RuntimeError(f"cuModuleGetLoadingMode failed: {status}")
        print("ELASTIC_PD_CUDA_LOADING", json.dumps(dict(pp_rank=runner.ps.pp_rank,
              driver_module_mode=mode.value,
              module=os.environ.get("CUDA_MODULE_LOADING"),
              data=os.environ.get("CUDA_MODULE_DATA_LOADING"))), flush=True)
        return result
    ModelRunner.load_model = load
    if os.environ["ELASTIC_PD_ROLE"] == "decode":
        import cache_probe
        cache_probe.install()
        if os.environ.get("ELASTIC_PD_STATIC_DECODE") == "1":
            # Ordinary PP2 replicas do not have the qualification's PP3-only
            # diagnostic sockets or elastic transition objects.
            return
        import hicache_transition_smoke as probe
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
        from sglang.srt.elastic_pp.coordinator import ElasticPPMigrationCoordinator
        probe.install_diagnostic_fanout(Scheduler, TokenizerControlMixin)
        probe.install_transition_hooks(Scheduler, ElasticPPMigrationCoordinator)
        original_copy = ElasticPPMigrationCoordinator._precopy_immutable_weights

        def copy_with_progress(coordinator, transition):
            adapter = coordinator.cache_transition
            if adapter is None:
                return original_copy(coordinator, transition)
            scheduler = adapter.scheduler

            def snapshot():
                return dict(monotonic=time.monotonic(), wall=time.time(), requests={
                    req.rid: len(req.output_ids)
                    for req in scheduler._elastic_pp_live_requests()
                    if not req.finished()
                })

            before = snapshot()
            try:
                return original_copy(coordinator, transition)
            finally:
                record = dict(generation=transition.target.generation,
                              rank=coordinator.rank, before=before, after=snapshot())
                coordinator._smoke_weight_copy_progress = getattr(
                    coordinator, "_smoke_weight_copy_progress", []) + [record]

        ElasticPPMigrationCoordinator._precopy_immutable_weights = copy_with_progress
        if os.environ.get("ELASTIC_PD_PREPARE_DIAGNOSTICS"):
            import prepare_stall_diagnostics
            prepare_stall_diagnostics.install(ElasticPPMigrationCoordinator,
                os.environ["ELASTIC_PD_PREPARE_DIAGNOSTICS"])
        if os.environ.get("ELASTIC_PD_NATIVE_STACKS"):
            import prepare_native_stacks
            prepare_native_stacks.install(ElasticPPMigrationCoordinator,
                os.environ["ELASTIC_PD_NATIVE_STACKS"])
        old_state = Scheduler.get_internal_state

        def state(scheduler, *args, **kwargs):
            import torch
            result = old_state(scheduler, *args, **kwargs)
            coordinator = getattr(scheduler, "elastic_pp_migration_coordinator", None)
            result.internal_state["transition_weight_copy_progress"] = getattr(
                coordinator, "_smoke_weight_copy_progress", [])
            # High-frequency policy observations must not enter CUDA allocator
            # inspection. Keep the qualification's explicit memory audit opt-in.
            if os.environ.get("ELASTIC_PD_LIGHTWEIGHT_STATUS") == "1":
                return result
            pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
            full = getattr(pool, "full_kv_pool", pool)
            if getattr(full, "capacity_managed", False):
                allocator = scheduler.token_to_kv_pool_allocator
                result.internal_state["pd_kv_capacity"] = dict(
                    **full.kvcached_physical_stats(), free_tokens=allocator.available_size(),
                    max_live_slot=max(allocator._live_page_ids, default=0),
                    scheduler_capacity=scheduler.max_total_num_tokens,
                    preallocator_capacity=scheduler.disagg_decode_prealloc_queue.max_total_num_tokens)
            def bytes_of(values):
                storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                            for t in values if t.is_cuda and t.numel()}
                return sum(storages.values())
            result.internal_state["pd_release_memory"] = dict(
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                free_bytes=torch.cuda.mem_get_info()[0], total_bytes=torch.cuda.mem_get_info()[1],
                model_bytes=bytes_of(list(scheduler.tp_worker.model_runner.model.parameters()) +
                                    list(scheduler.tp_worker.model_runner.model.buffers())),
                kv_bytes=memory_size_json(pool.get_kv_size_bytes()), pid=os.getpid(),
                hicache_alignment=getattr(scheduler, "_pp_pd_hicache_fence_alignment", None))
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
        "--max-running-requests", "1" if cfg["role"] == "decode" else "4", "--max-mamba-cache-size", "16", "--mamba-ssm-dtype", "float32",
        "--mamba-max-states-per-path", "1", "--mem-fraction-static", ".90", "--page-size", "64",
        "--chunked-prefill-size", "512", "--max-prefill-tokens", "1024", "--pp-max-micro-batch-size", "1" if cfg["role"] == "decode" else "4",
        "--pp-async-batch-depth", "0", "--attention-backend", "triton", "--linear-attn-backend", "triton",
        "--moe-runner-backend", "marlin", "--moe-a2a-backend", "none",
        "--enable-deterministic-inference",
        "--skip-server-warmup", "--json-model-override-args", '{"language_model_only":true}',
        "--enable-cache-report", "--enable-hierarchical-cache", "--hicache-size", "8",
        "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel",
        "--hicache-mem-layout", "page_first", "--disaggregation-mode", cfg["role"],
        "--disaggregation-transfer-backend", "mooncake", "--disaggregation-bootstrap-port", str(cfg["bootstrap_port"]),
        "--disaggregation-ib-device", "mlx5_bond_0"]
    if cfg["role"] == "decode":
        argv += ["--enable-elastic-pp", "--elastic-pp-disagg-decode", "--elastic-pp-initial-partition", "24,0,24",
                 "--disaggregation-decode-enable-radix-cache", "--disaggregation-decode-extra-slots", "4"]
    if cfg.get("kvcached_capacity"):
        for option, value in (("--context-length", "16384"), ("--max-total-tokens", "12288"),
                              ("--page-size", "1"), ("--chunked-prefill-size", "4096"),
                              ("--max-prefill-tokens", "8192")):
            argv[argv.index(option) + 1] = value
        if cfg["role"] == "decode":
            argv += ["--elastic-pp-kv-backend", "kvcached", "--elastic-pp-kv-capacities", "8192;12288"]
    if cfg.get("optimized_elastic_pp"):
        argv += ["--elastic-pp-transition-strategy", "background", "--elastic-pp-transition-backing", "overlap",
                 "--elastic-pp-graph-partitions", "24,0,24;16,16,16",
                 "--elastic-pp-prepared-weight-policy", "offload", "--cuda-graph-bs-decode", "1",
                 "--elastic-pp-weight-staging", "cpu_prepacked",
                 "--elastic-pp-cpu-weight-cache-gib", str(cfg["cpu_weight_cache_gib"]),
                 "--elastic-pp-hicache-fast-copy", "--elastic-pp-hicache-batch-io",
                 "--elastic-pp-hicache-background", "--elastic-pp-hicache-shared-host",
                 "--enable-elastic-pp-kv-patch", "--elastic-pp-kv-patch-transport", "nixl",
                 "--elastic-pp-kv-base-precopy"]
    else:
        argv += ["--disable-cuda-graph"]
        if cfg["role"] == "decode":
            argv += ["--elastic-pp-transition-strategy", "sync", "--elastic-pp-transition-backing", "cpu_spill"]
    save(run / "nodes" / cfg["node"] / "effective_launch.json",
         dict(engine_commit=cfg["engine_commit"], argv=argv,
              optimized_elastic_pp=cfg.get("optimized_elastic_pp", False)))
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
            target["pd_kv_capacity"] = source.get("pd_kv_capacity")
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
    def capacity_evidence(states, expected):
        if not root.get("kvcached_capacity"):
            return None
        ranks = probe.rank_states(states)
        rows = [s.get("pd_kv_capacity") or {} for s in ranks.values()]
        if len(rows) != 3 or any(
            r.get("serving_capacity") != expected or r.get("scheduler_capacity") != expected
            or r.get("preallocator_capacity") != expected or not r.get("physical_layer_partitioned")
            or r.get("virtual_capacity") != 12288 for r in rows
        ):
            raise AssertionError("KV capacity was not published to every live admission owner")
        return dict(capacity=expected, mapped_bytes=sum(r["mapped_memory_bytes"] for r in rows),
                    ranks={str(rank): s["pd_kv_capacity"] for rank, s in ranks.items()})
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
        prompt = tokens + [20000 + i for i in range(64)]
        # D can cache a longer prefix after its first completed Decode than P
        # cached during the original Prefill. Warm that boundary before taking
        # the reference, so a cache-path difference is not blamed on migration.
        results["full_prefix_warmup"] = replay.request_one(p, d, key, prompt, 64, 0, True)
        results["reference"] = replay.request_one(p, d, key, prompt, root["output_tokens"], 0, True)
        results["static_repeat"] = replay.request_one(p, d, key, prompt, 64, 0, True)
        results["static_repeat_match"] = (
            results["static_repeat"]["output_ids"] == results["reference"]["output_ids"][:64]
        )
        save(run / "results.partial.json", results)
        for name in ("reference", "static_repeat"):
            if results[name]["meta_info"].get("cached_tokens") != len(prompt):
                raise AssertionError("Warm reference must use the complete D prefix: " + name)
        if not results["static_repeat_match"]:
            raise AssertionError("Unchanged-layout warm repetitions differ before migration")
        events = []
        original_stream = replay.stream
        def observed_stream(endpoint, key, payload, timeout=600, **kwargs):
            anchor = dict(wall=time.time(), perf_counter=time.perf_counter())
            save(run / "active_stream_clock.json", anchor)
            result = original_stream(endpoint, key, payload, timeout, event_sink=events)
            if root.get("diagnostic_prepare"):
                result["diagnostic_clock_anchor"] = anchor
                result["diagnostic_stream_started"] = result["started"]
            return result
        replay.stream = observed_stream
        try:
            with cf.ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(replay.request_one, p, d, key, prompt, root["output_tokens"], 0, True)
                for generation, counts in ((1, [16, 16, 16]), (2, [24, 0, 24])):
                    minimum = 8 if generation == 1 or not events else events[-1][0] + 8
                    deadline = time.monotonic() + 90
                    while not events or events[-1][0] < minimum:
                        health()
                        if active.done():
                            raise RuntimeError("Request finished before its migration fence")
                        if time.monotonic() > deadline:
                            raise TimeoutError("No progressing active Decode request")
                        time.sleep(.02)
                    before = state(f"before{generation}")
                    if root.get("kvcached_capacity") and generation == 1:
                        results["initial_capacity"] = capacity_evidence(before, 8192)
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
                    if root.get("optimized_elastic_pp"):
                        probe.validate_transition_states(after, generation,
                            (0, 2) if generation == 1 else (0, 1, 2), "overlap", shared_host=True)
                    results["transitions"][-1]["after"] = after
                    if root.get("kvcached_capacity"):
                        results["transitions"][-1]["capacity"] = capacity_evidence(after, 12288 if generation == 1 else 8192)
                        if generation == 1:
                            results["active_grow"] = active.result(timeout=240)
                            save(run / "stream_events.grow.json", events)
                            # A real P→D request must consume slots that the old
                            # PP2 ceiling could not admit. No cold-prefix timing
                            # is compared with the warm reference.
                            large = [30000 + (i * 7) % 15000 for i in range(root["capacity_probe_input_tokens"])]
                            results["new_capacity_request"] = replay.request_one(p, d, key, large, 32, 0, True)
                            used = state("new_capacity_used")
                            proof = capacity_evidence(used, 12288)
                            if not all(r["max_live_slot"] > 8192 for r in proof["ranks"].values()):
                                raise AssertionError("No live/cache slot used the additional KV capacity")
                            if proof["mapped_bytes"] <= results["initial_capacity"]["mapped_bytes"]:
                                raise AssertionError("Capacity growth did not increase physical KV backing")
                            results["grown_capacity"] = proof
                            results["pre_shrink_demotion"] = probe.demote_cpu_prefixes(http)
                            results["pre_shrink_rewarm"] = replay.request_one(p, d, key, prompt, 64, 0, True)
                            events.clear()
                            active = executor.submit(replay.request_one, p, d, key, prompt, root["output_tokens"], 0, True)
                        else:
                            shrunk = results["transitions"][-1]["capacity"]
                            if shrunk["mapped_bytes"] >= results["grown_capacity"]["mapped_bytes"]:
                                raise AssertionError("Shrink did not return any physical KV pages")
                            if shrunk["ranks"]["1"]["mapped_memory_bytes"] != 0:
                                raise AssertionError("Retired GPU retained KVCacheD physical pages")
                (run / "probe.trigger").touch(mode=0o600)
                wait_file(run / "gpu_reuse_started.json", 90)
                results["active"] = active.result(timeout=240)
        finally:
            replay.stream = original_stream
            save(run / "stream_events.partial.json", events)
        results["stream_events"] = events
        completed = state("completed")
        if root.get("optimized_elastic_pp"):
            probe.validate_decode_graph_states(completed, 2, (0, 1, 2), (0, 2), prepared=True)
        results["continuity"] = [probe.validate_uninterrupted_decode(
            completed, generation, results.get("active_grow", results["active"]) if generation == 1 else results["active"], sources, targets
        ) for generation, sources, targets in (
            (1, (0, 2), (0, 1, 2)), (2, (0, 1, 2), (0, 2))) ]
        results["exact_output_match"] = results["reference"]["output_ids"] == results["active"]["output_ids"]
        if "active_grow" in results:
            results["exact_output_match"] &= results["reference"]["output_ids"] == results["active_grow"]["output_ids"]
        results["warm_start_match"] = all(
            results["reference"][role].get("cached_tokens")
            == results["active"][role].get("cached_tokens")
            for role in ("prefill_meta", "meta_info")
        )
        if not results["warm_start_match"]:
            raise AssertionError("Reference and transition request used different warm prefixes")
        if not results["exact_output_match"] or not results["active"]["output_ids"]:
            raise AssertionError("Transition output IDs differ from unchanged-layout warm reference")
        wait_file(run / "gpu_reuse_done.json", 60)
        results["gpu_reuse"] = json.loads((run / "gpu_reuse_started.json").read_text())
        results["after_handoff"] = replay.request_one(p, d, key, prompt, 32, 0, True)
        results["demote"] = probe.demote_cpu_prefixes(http)
        results["cpu_restore"] = replay.request_one(p, d, key, prompt, 32, 0, True)
        final = state("final")
        results["restore_ids_match"] = results["cpu_restore"]["output_ids"] == results["after_handoff"]["output_ids"]
        if not results["restore_ids_match"]:
            raise AssertionError("HiCache CPU restore changed output IDs")
        restored = probe.fresh_prefix_restores(final, results["demote"]["commanded_at"], (0, 2), 512)
        results.update(passed=True, cpu_restores=restored, final_state=final,
                       whole_worker_gpu_exit=False, independent_gpu_reuse_verified=True)
    except BaseException as error:
        results.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        # A repeated cancellation must not interrupt owned cleanup and lose its
        # receipts after the first signal has already unwound the controller.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        save(run / "results.json", results)
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
    parser.add_argument("--sync-baseline", action="store_true")
    parser.add_argument("--diagnostic-prepare", action="store_true")
    parser.add_argument("--native-prepare", action="store_true")
    parser.add_argument("--lazy-cuda-modules", action="store_true")
    parser.add_argument("--kvcached-capacity", action="store_true")
    for option in ("prepare", "worker", "engine", "probe"):
        parser.add_argument("--" + option, action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.prepare:
        return prepare(run, optimized=not args.sync_baseline, diagnostic_prepare=args.diagnostic_prepare,
                       native_prepare=args.native_prepare, lazy_cuda_modules=args.lazy_cuda_modules,
                       kvcached_capacity=args.kvcached_capacity)
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
