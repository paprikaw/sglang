"""One real P→D request per new PP2 replica. No workload replay."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import run as harness

h, q = harness.h, harness.q
PYTHON, GUARD = q.PYTHON, q.GUARD
HERE = Path(__file__).resolve()
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def store_service(run, config):
    import asyncio
    import grpc
    from sllm_store import server
    from sllm_store.client import SllmStoreClient
    from sllm_store.proto import storage_pb2_grpc
    from sglang.srt.weight_cache.cpu_canonical import model_view
    from sglang.srt.weight_cache.sllmstore_stage import prepare_indices

    cfg = config["servers"]["d_store"]
    h.authenticate(run, cfg, os.environ, local=True)
    root = Path(cfg["sllmstore_root"])
    with model_view(
        cfg["model_path"],
        os.environ["SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET"],
        os.environ["SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT"],
    ) as (view, bank):
        manifest = prepare_indices(root, view, bank, [[0, 24], [24, 48]])
        try:

            async def serve():
                chunk = 32 << 20
                pool_bytes = sum(
                    ((root / "store" / n / "tensor.data_0").stat().st_size + chunk - 1)
                    // chunk
                    * chunk
                    for n in manifest["parts"]
                )
                if pool_bytes > config["store_reserve_bytes"]:
                    raise ValueError(
                        "Official Store host pool exceeds the frozen RAM reservation"
                    )
                service = server.StorageServicer(
                    str(root / "store"), pool_bytes, 4, chunk, True
                )
                rpc = grpc.aio.server(options=(("grpc.so_reuseport", 0),))
                storage_pb2_grpc.add_StorageServicer_to_server(service, rpc)
                if rpc.add_insecure_port("127.0.0.1:8073") != 8073:
                    raise RuntimeError("SllmStore loopback listener unavailable")
                await rpc.start()

                def prewarm():
                    client = SllmStoreClient()
                    # Native ToHost emits this receipt only after its read threads finish.
                    with (run / "store.log").open(errors="replace") as log:
                        for name in manifest["parts"]:
                            size = (
                                (root / "store" / name / "tensor.data_0").stat().st_size
                            )
                            if client.register_model(
                                name
                            ) != size or not client.load_into_cpu(name):
                                raise RuntimeError(
                                    f"Store host registration/load failed for {name}"
                                )
                            until = time.monotonic() + 90
                            marker = f"Finished loading model {name} from disk"
                            while not any(marker in line for line in log):
                                if time.monotonic() > until:
                                    raise TimeoutError(
                                        f"No host-ready receipt for {name}"
                                    )
                                time.sleep(0.08)

                try:
                    await asyncio.to_thread(prewarm)
                    h.save(
                        run / "store_ready.json",
                        dict(passed=True, host_pool_bytes=pool_bytes, **manifest),
                    )
                    while not (run / "stop_store").exists():
                        await asyncio.sleep(0.4)
                finally:
                    await rpc.stop(1)

            asyncio.run(serve())
        finally:
            shutil.rmtree(root)


def store_worker(run, config):
    cfg = config["servers"]["d_store"]
    os.environ.update(json.loads(sys.stdin.readline())["environment"])
    h.authenticate(run, cfg, os.environ, local=True)
    env = q.environment(cfg)
    env.update(
        GLOG_logtostderr="1",
        PYTHONPATH=cfg["store_overlay"] + os.pathsep + env["PYTHONPATH"],
    )
    args = [
        PYTHON,
        str(GUARD / "model_cache_adapter.py"),
        "exec-local",
        "--bindings",
        os.environ[h.GUARD_KEYS[3]],
        "--",
        PYTHON,
        str(HERE),
        "--run",
        str(run),
        "--store-service",
    ]
    with (run / "store.log").open("wb", buffering=0) as log:
        child = subprocess.Popen(
            args,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ident = h.process_record(child.pid)

        def copy_log():
            for line in child.stdout:
                log.write(line)

        writer = threading.Thread(target=copy_log, daemon=True)
        writer.start()
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        threading.Thread(
            target=lambda: (sys.stdin.read(), stop.set()), daemon=True
        ).start()
        try:
            while not stop.wait(0.5):
                if child.poll() is not None:
                    raise RuntimeError(f"Store exited with {child.returncode}")
        finally:
            (run / "stop_store").touch()
            try:
                child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                pass
            cleanup = h.cleanup(
                [dict(process=child, identity=ident, log_thread=writer)],
                require_node_gpu_idle=False,
            )
            h.save(run / "store_cleanup.json", cleanup)


def events(log, marker, room):
    values = []
    for line in log.read_text(errors="replace").splitlines():
        if marker + " " in line:
            item = json.loads(line.split(marker + " ", 1)[1])
            if item.get("room") == room:
                values.append(item)
    return values


def request(prefill, decode, key, tokens):
    import agentx_replay as replay

    room = secrets.randbelow(2**61) + 1
    payload = dict(
        input_ids=tokens,
        stream=True,
        return_logprob=True,
        logprob_start_len=len(tokens) - 1,
        bootstrap_room=room,
        bootstrap_host=prefill["node"],
        bootstrap_port=prefill["bootstrap_port"],
        disagg_prefill_dp_rank=0,
        sampling_params=dict(temperature=0, max_new_tokens=8, ignore_eos=True),
    )

    def send_prefill():
        reply = replay.http(
            prefill["node"],
            prefill["port"],
            "/generate",
            key,
            dict(payload, stream=False),
            120,
        )
        return reply, time.perf_counter()

    timeline = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pre = pool.submit(send_prefill)
        reply = replay.stream(
            dict(host=decode["node"], port=decode["port"]),
            key,
            payload,
            timeout=120,
            event_sink=timeline,
        )
        p, p_returned = pre.result()
    if (
        "error" in p
        or p["meta_info"].get("completion_tokens") != 1
        or len(reply["output_ids"]) != 8
    ):
        raise RuntimeError("Incomplete Prefill or Decode response")
    first = next(t for count, t in timeline if count >= 1)
    first_decode = next(t for count, t in timeline if count >= 2)
    return dict(
        room=room,
        events=timeline,
        prefill_returned=p_returned,
        first_stream=reply["started"] + first / 1000,
        first_decode=reply["started"] + first_decode / 1000,
        output=reply["output_ids"],
        prefill_cache=p["meta_info"].get("cached_tokens_details"),
        prefill_hit=p["meta_info"].get("cached_tokens"),
        decode_cache=reply["meta_info"].get("cached_tokens_details"),
    )


def measure(run, config):
    workers, handles = {}, []
    reports, errors = [], []
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    packet = (
        json.dumps(dict(environment={k: os.environ[k] for k in h.GUARD_KEYS})) + "\n"
    )
    key = Path(config["admin_key_file"]).read_text().strip()
    servers = config["servers"]
    ident = h.process_record(os.getpid())

    def progress(stage):
        h.save(
            run / "task/progress.json",
            dict(
                experiment_id=config["id"],
                task_id="comparison",
                timestamp=time.time(),
                stage=stage,
                controller_pid=os.getpid(),
                controller_start_ticks=ident["start_ticks"],
                process_start_ticks=ident["start_ticks"],
                valid_points=int(len(reports) == 2),
                expected=1,
            ),
        )

    def spawn(sid):
        cfg = servers["d_store" if sid == "store" else sid]
        script, args = (
            (HERE, ["--store-worker"])
            if sid == "store"
            else (HERE.with_name("run.py"), ["--worker", "--server", sid])
        )
        directory = run / "servers" / sid
        directory.mkdir(parents=True, exist_ok=True)
        log = (directory / "worker.log").open("wb")
        handles.append(log)
        cmd = [PYTHON, str(script), "--run", str(run), *args]
        proc = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", cfg["node"], shlex.join(cmd)],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers[sid] = proc
        proc.stdin.write(packet.encode())
        proc.stdin.flush()

    def read_visible(path):
        proc = subprocess.run(
            [
                PYTHON,
                str(GUARD / "artifact_visibility.py"),
                str(path),
                "--timeout-s",
                "12",
                "--receipt",
                str(path.with_suffix(".visibility.json")),
            ],
            capture_output=True,
            timeout=18,
        )
        if proc.returncode:
            raise RuntimeError(f"Invalid remote artifact {path.name}")
        return json.loads(path.read_text())

    def wait_ready(sid):
        path = (
            run / "store_ready.json"
            if sid == "store"
            else run / "servers" / sid / "ready.json"
        )
        deadline = time.monotonic() + 600
        while not path.exists():
            if stopped.is_set() or time.monotonic() > deadline:
                raise RuntimeError(f"Stopped or timed out waiting for {sid}")
            for name, proc in workers.items():
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited before {sid} became ready")
            time.sleep(0.10)
        return read_visible(path)

    def retire(sid):
        proc = workers.pop(sid)
        proc.stdin.close()
        try:
            proc.wait(timeout=95)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=12)
        path = (
            run / "store_cleanup.json"
            if sid == "store"
            else run / "servers" / sid / "cleanup.json"
        )
        if not read_visible(path)["cleanup_verified"]:
            raise RuntimeError(f"Cleanup failed for {sid}")

    try:
        for cfg in config["node_configs"].values():
            h.authenticate(run, cfg, os.environ)
        progress("warm Prefill, resident Decode and host weight pools")
        for sid in ("p", "d0", "store"):
            spawn(sid)
        for sid in ("p", "d0", "store"):
            wait_ready(sid)
        tokens = [1000 + i % 123 for i in range(4608)]
        anchor = request(servers["p"], servers["d0"], key, tokens)
        anchor_hash = hashlib.sha256(json.dumps(anchor["output"]).encode()).hexdigest()
        for sid in ("d_cpu", "d_store"):
            # The same P is warm, and the old D remains serving on the same GPUs.
            warm = request(servers["p"], servers["d0"], key, tokens)
            if warm["output"] != anchor["output"]:
                raise RuntimeError("Reference outputs changed")
            progress("trigger " + sid)
            trigger_unix = time.time()
            start = time.perf_counter()
            spawn(sid)
            ready = wait_ready(sid)
            routable = time.perf_counter()
            result = request(servers["p"], servers[sid], key, tokens)
            log = Path(servers[sid]["layouts"][0]["usage_logs"][0]["path"])
            receipt_deadline = time.monotonic() + 8
            while True:
                handoff = events(log, "PP2_STARTUP_HANDOFF", result["room"])
                graph = events(log, "PP2_STARTUP_GRAPH", result["room"])
                if {x["rank"] for x in handoff} == {0, 1} and {
                    x["rank"] for x in graph
                } == {0, 1}:
                    break
                if time.monotonic() >= receipt_deadline:
                    break
                time.sleep(0.1)
            valid = (
                result["output"] == anchor["output"]
                and {x["rank"] for x in handoff} == {0, 1}
                and {x["rank"] for x in graph} == {0, 1}
            )
            row = dict(
                method=sid,
                passed=valid,
                trigger_unix=trigger_unix,
                engine_ready_unix=ready["unix"],
                trigger_to_routable_s=routable - start,
                trigger_to_first_sse_s=result["first_stream"] - start,
                trigger_to_first_decode_s=result["first_decode"] - start,
                prefill_reply_s=result["prefill_returned"] - start,
                room=result["room"],
                handoff=handoff,
                graph=graph,
                events=result["events"],
                prefill_hit=result["prefill_hit"],
                prefill_cache=result["prefill_cache"],
                decode_cache=result["decode_cache"],
                reference_output_sha256=anchor_hash,
                output_sha256=hashlib.sha256(
                    json.dumps(result["output"]).encode()
                ).hexdigest(),
            )
            reports.append(row)
            h.save(run / "measurements.json", reports)
            if not valid:
                raise RuntimeError(
                    f"Real output, two PP handoffs, or two Graph replays differ for {sid}"
                )
            retire(sid)
        if reports[0]["prefill_hit"] != reports[1]["prefill_hit"]:
            raise RuntimeError("Warm Prefill hit counts differ")
        h.save(run / "completed.json", dict(passed=True, measurements=reports))
        progress("complete")
    finally:
        for sid in reversed(list(workers)):
            try:
                retire(sid)
            except Exception as exc:
                errors.append(f"{sid}: {exc}")
        for log in handles:
            log.close()
        for node in config["node_configs"]:
            try:
                check = subprocess.run(
                    [
                        "ssh",
                        "-T",
                        "-o",
                        "BatchMode=yes",
                        node,
                        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=20,
                )
                if check.stdout.strip():
                    errors.append(f"{node} has CUDA processes after cleanup")
            except Exception as exc:
                errors.append(str(exc))
        h.save(
            run / "task/cleanup.json",
            dict(
                experiment_id=config["id"],
                task_id="comparison",
                timestamp=time.time(),
                passed=not errors,
                model_processes_exited=not errors,
                gpu_idle=not errors,
                errors=errors,
            ),
        )
        if errors:
            raise RuntimeError(str(errors))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--store-worker", action="store_true")
    cli.add_argument("--store-service", action="store_true")
    args = cli.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    if args.store_worker:
        store_worker(args.run, config)
    elif args.store_service:
        store_service(args.run, config)
    else:
        try:
            measure(args.run, config)
        except BaseException as exc:
            h.save(
                args.run / "failure.json",
                dict(error=repr(exc), traceback=traceback.format_exc()),
            )
            raise
