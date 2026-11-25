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


async def unload_store_weights(client, names):
    import asyncio

    # The gRPC server receiving these blocking client calls uses this event loop.
    for name in names:
        if await asyncio.to_thread(client.unload_from_cpu, name) is False:
            raise RuntimeError(f"Could not retire raw Store weights: {name}")


async def wait_for_weight_release(root, key, timeout=90):
    import asyncio
    from sglang.srt.weight_cache.replica_preparation import read_packet

    started = time.monotonic()
    names = [f"weights-released-{rank}.json" for rank in (0, 1)]
    while not all((root / name).is_file() for name in names):
        if time.monotonic() - started >= timeout:
            raise TimeoutError("Donor stage weight backing was not released")
        await asyncio.sleep(0.15)
    for name in names:
        read_packet(root, name, key)
    return time.monotonic() - started


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
        prepared = config.get("prepared_replica", False)
        manifest = prepare_indices(root, view, bank, [[0, 24], [24, 48]])
        chunk = 32 << 20

        def rounded_size(names):
            return sum(
                ((root / "store" / n / "tensor.data_0").stat().st_size + chunk - 1)
                // chunk
                * chunk
                for n in names
            )

        pool_bytes = (
            config["store_reserve_bytes"]
            if prepared
            else rounded_size(manifest["parts"])
        )
        if pool_bytes > config["store_reserve_bytes"]:
            raise ValueError("Official Store exceeds its admitted pinned budget")

        def prewarm(names):
            if rounded_size(names) > pool_bytes:
                raise ValueError(
                    "Official Store cannot keep this entire representation warm"
                )
            client = SllmStoreClient()
            with (run / "store.log").open(errors="replace") as log:
                for name in names:
                    size = (root / "store" / name / "tensor.data_0").stat().st_size
                    if client.register_model(name) != size or not client.load_into_cpu(
                        name
                    ):
                        raise RuntimeError(f"Store registration/load failed for {name}")
                    deadline = time.monotonic() + 90
                    marker = f"Finished loading model {name} from disk"
                    while not any(marker in line for line in log):
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"No host-ready receipt for {name}")
                        time.sleep(0.08)

        async def serve():
            from sglang.srt.elastic_pp.replica_host_cache import prepare_spare
            from sglang.srt.weight_cache.replica_preparation import (
                prepare_dense_files,
                preparation_key,
                publish,
                register_stage_files,
            )

            service = server.StorageServicer(
                str(root / "store"), pool_bytes, 4, chunk, True
            )
            rpc = grpc.aio.server(options=(("grpc.so_reuseport", 0),))
            storage_pb2_grpc.add_StorageServicer_to_server(service, rpc)
            if rpc.add_insecure_port("127.0.0.1:8073") != 8073:
                raise RuntimeError("SllmStore loopback listener unavailable")
            await rpc.start()
            spare, fds = None, []
            try:
                started = time.monotonic()
                await asyncio.to_thread(prewarm, manifest["parts"])
                h.save(
                    run / "store_ready.json",
                    dict(
                        passed=True,
                        host_pool_bytes=pool_bytes,
                        prewarm_s=time.monotonic() - started,
                        **manifest,
                    ),
                )
                while not (run / "stop_store").exists():
                    if prepared and spare is None and (run / "prepare-packed").exists():
                        # No raw consumer remains. Reuse the same pinned region
                        # and the same native Store server in both measurements.
                        started = time.monotonic()
                        await unload_store_weights(SllmStoreClient(), manifest["parts"])
                        preparation = run / "replica-preparation"
                        key = preparation_key(preparation, bank["group_key"])
                        parts = register_stage_files(root, preparation, key)
                        parts += prepare_dense_files(
                            root, view, bank, manifest["partitions"], fds, 16 << 30
                        )
                        used = rounded_size(parts)
                        await asyncio.to_thread(prewarm, parts)
                        for fd in fds:
                            os.close(fd)
                        fds.clear()
                        publish(preparation / "weights-consumed.json", dict(key=key))
                        release_s = await wait_for_weight_release(preparation, key)
                        spare = prepare_spare(
                            preparation, key, config["spare_hicache_reserve_bytes"]
                        )
                        h.save(
                            run / "store_packed_ready.json",
                            dict(
                                passed=True,
                                host_pool_bytes=pool_bytes,
                                rounded_payload_bytes=used,
                                release_before_spare_s=release_s,
                                preparation_s=time.monotonic() - started,
                            ),
                        )
                    await asyncio.sleep(0.2)
            finally:
                for fd in fds:
                    os.close(fd)
                if spare is not None:
                    spare.close()
                await rpc.stop(1)

        try:
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


def prepare_history(run, servers, key, tokens, anchor, wait_for_packed):
    import agentx_replay as replay

    d = servers["d0"]
    root = run / "replica-preparation"
    if request(servers["p"], d, key, tokens)["output"] != anchor["output"]:
        raise RuntimeError("Donor output changed before prefix snapshot")
    start = time.perf_counter()
    deadline = start + 480
    while time.perf_counter() < deadline:
        ok = replay.http(
            d["node"],
            d["port"],
            "/set_internal_state",
            key,
            {"server_args": {"agentx_cache_cmd": 4}},
            420,
        )
        if donor_published(ok, root, "weights"):
            break
        time.sleep(0.25)
    else:
        raise TimeoutError("Donor failed to export both processed Marlin stages")
    export_wall_s = time.perf_counter() - start
    (run / "prepare-packed").touch(exist_ok=False)
    packed = wait_for_packed()
    start = time.perf_counter()
    deadline = start + 75
    while time.perf_counter() < deadline:
        ok = replay.http(
            d["node"],
            d["port"],
            "/set_internal_state",
            key,
            {"server_args": {"agentx_cache_cmd": 3}},
            45,
        )
        if donor_published(ok, root, "history"):
            break
        time.sleep(0.2)
    else:
        raise TimeoutError("Donor did not publish a drained, host-backed KV/GDN prefix")
    history = [json.loads((root / f"history-{r}.json").read_text()) for r in (0, 1)]
    if (
        history[0]["nodes"] != history[1]["nodes"]
        or min(history[0]["rows"].values()) < 1
    ):
        raise ValueError("Donor ranks disagree on the historical prefix")
    weights = [json.loads((root / f"weights-{r}.json").read_text()) for r in (0, 1)]
    spare = json.loads((root / "spare.json").read_text())
    snapshot_wall_s = time.perf_counter() - start
    for rank in (0, 1):
        released = json.loads((root / f"weights-released-{rank}.json").read_text())
        if released["key"] != spare["key"]:
            raise RuntimeError("Donor weight staging release has a foreign identity")
    h.save(
        run / "replica_preparation_cost.json",
        dict(
            donor_weight_export_wall_s=export_wall_s,
            store_conversion_and_prewarm_s=packed["preparation_s"],
            prefix_snapshot_wall_s=snapshot_wall_s,
            weight_release_before_spare_s=packed["release_before_spare_s"],
            prefix_payload_bytes=sum(x["bytes"] for x in history),
            historical_rows=history[0]["rows"],
            prefix_copy_rank_s=[x["seconds"] for x in history],
            marlin_export_rank_s=[x["seconds"] for x in weights],
            marlin_export_bytes=sum(x["bytes"] for x in weights),
            spare_hicache_bytes=spare["bytes"],
            spare_hicache_allocation_s=spare["seconds"],
            official_store_pinned_bytes=json.loads(
                (run / "store_ready.json").read_text()
            )["host_pool_bytes"],
        ),
    )


def donor_published(reply, root, kind):
    # The fixed PP2 endpoint exposes one public control reply. Both stage
    # writers publish their receipts only after all their copies are complete.
    return (
        isinstance(reply, list)
        and True in reply
        and all((root / f"{kind}-{rank}.json").is_file() for rank in (0, 1))
    )


def qualify_history(run, servers, key, tokens, anchor, row):
    target = servers["d_store"]
    log = Path(target["layouts"][0]["usage_logs"][0]["path"])
    deadline = time.monotonic() + 8
    while True:
        hits = events(log, "PP2_HISTORICAL_RESTORE", row["room"])
        events_by_rank = [[e for e in hits if e["rank"] == r] for r in (0, 1)]
        if all(events_by_rank) or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    row["first_request_historical_restore"] = events_by_rank
    h.save(
        run / "first_request_history.json", dict(room=row["room"], ranks=events_by_rank)
    )
    if len(events_by_rank) != 2 or any(
        not any(
            e["loaded_tokens"] > 0
            and e["host_gdn_to_device_cache"]
            and not e["copy_mamba_to_request"]
            for e in rank
        )
        for rank in events_by_rank
    ):
        raise RuntimeError("First target request lacks two-rank KV/GDN CPU restore")
    # Independent writes must preserve both caches and the original output.
    branch = tokens[:-64] + [1337] * 64
    newer = request(servers["p"], target, key, branch)
    donor = request(servers["p"], servers["d0"], key, branch)
    original = request(servers["p"], servers["d0"], key, tokens)
    if newer["output"] != donor["output"] or original["output"] != anchor["output"]:
        raise RuntimeError("Cross-replica prefix branch or donor reference changed")
    row["independent_branch_outputs_match"] = True


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
            run / ("store_ready.json" if sid == "store" else "store_packed_ready.json")
            if sid in ("store", "packed")
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
        prepared = config.get("prepared_replica", False)
        for sid in ("p", "d0"):
            spawn(sid)
        if not prepared:
            spawn("store")
        for sid in ("p", "d0"):
            wait_ready(sid)
        if prepared:
            spawn("store")
        wait_ready("store")
        tokens = [1000 + i % 123 for i in range(4608)]
        anchor = request(servers["p"], servers["d0"], key, tokens)
        anchor_hash = hashlib.sha256(json.dumps(anchor["output"]).encode()).hexdigest()
        for sid in ("d_raw" if prepared else "d_cpu", "d_store"):
            if prepared and sid == "d_store":
                prepare_history(
                    run, servers, key, tokens, anchor, lambda: wait_ready("packed")
                )
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
            if prepared and sid == "d_store":
                qualify_history(run, servers, key, tokens, anchor, row)
                h.save(run / "measurements.json", reports)
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
