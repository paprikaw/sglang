"""One small PP migration fixture, both directions, with matched static controls."""
import concurrent.futures as cf
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

BENCH = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/benchmarks/prefill_pp_migration_v1")
CROSS_ROOM = 1827110000


def install_trigger_probe():
    """Arm early, dispatch on the scheduler thread when the crossing KV read is active."""
    from sglang.srt.managers.io_struct import ReconfigureElasticPPReqInput
    from sglang.srt.managers.scheduler import Scheduler

    native = Scheduler.process_input_requests

    def dispatch(scheduler, requests):
        if scheduler.ps.pp_rank == 0:
            for message in list(requests):
                if isinstance(message, ReconfigureElasticPPReqInput):
                    if getattr(scheduler, "_armed_migration", None):
                        raise RuntimeError("Two overlapping migration probes")
                    scheduler._armed_migration = message
                    requests.remove(message)
            armed = getattr(scheduler, "_armed_migration", None)
            if armed is not None:
                fence = scheduler.disagg_prefill_bootstrap_queue.kv_manager.prefill_transfer_fence
                if fence.sending(CROSS_ROOM + armed.generation):
                    requests.append(armed)
                    scheduler._armed_migration = None
        return native(scheduler, requests)

    Scheduler.process_input_requests = dispatch


def measure(run, p, d, key, wait_file, replay, save):
    spec = importlib.util.spec_from_file_location("frozen_migration_workload", BENCH / "workload.py")
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    fingerprints = json.loads((BENCH / "fixtures.json").read_text())
    previous = {}
    observations = {}
    transition_waits = {}
    for generation, direction in enumerate(("up", "down"), 1):
        fixture = workload.case("small", direction)
        frozen = next(row for row in fingerprints if row["direction"] == direction and row["size"] == "small")
        canonical = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(canonical).hexdigest() == frozen["sha256"]
        for migrate in (False, True):
            name = f"{direction}-{'migration' if migrate else 'control'}"
            for endpoint in (p, d):
                replay.http(endpoint["host"], endpoint["port"], "/flush_cache?timeout=60", key)
            for history in fixture["histories"]:
                replay.request_one(p, d, key, history, 1, 0, True, request_timeout_s=90)
            room_base = 1827000000 + 10000 * generation + (1000 if migrate else 0)
            samples = run / "samples" / name
            samples.mkdir(parents=True, exist_ok=True)
            def send(row, room):
                prefix = [] if row["history"] is None else fixture["histories"][row["history"]]
                result = replay.request_one(p, d, key, prefix + row["suffix_ids"], row["max_new_tokens"],
                                          0, True, request_timeout_s=60, bootstrap_room=room)
                save(samples / f"{room}.json", result)
                return result

            with cf.ThreadPoolExecutor(max_workers=90) as pool:
                started = time.monotonic()
                requests = {}
                action = None
                for second in range(-30, 31):
                    time.sleep(max(0, started + second + 30 - time.monotonic()))
                    if second == -1 and migrate:
                        action = pool.submit(replay.http, p["host"], p["port"], "/reconfigure_elastic_pp",
                                             key, dict(generation=generation, layer_counts=fixture["target"]), 70)
                    if second == 0:
                        crossing_room = CROSS_ROOM + generation if migrate else room_base + 99
                        requests["crossing"] = pool.submit(send, fixture["crossing_request"], crossing_room)
                    requests[str(second)] = pool.submit(send, workload.probe(second), room_base + second + 31)
                    for pending in [*requests.values(), *([action] if action else [])]:
                        if pending.done() and pending.exception() is not None:
                            raise pending.exception()
                if action:
                    answer = action.result(timeout=65)
                    save(run / f"{name}.transition.json", answer)
                    if not answer["success"] or answer["new_layer_counts"] != fixture["target"]:
                        raise RuntimeError("Prefill transition failed or wrong target layout")
                rows = {i: f.result(timeout=65) for i, f in requests.items()}
            save(run / f"{name}.requests.json", rows)
            if any(len(row.get("output_ids") or []) != 32 for row in rows.values()):
                raise ValueError("Missing generated tokens")
            if not migrate:
                previous = rows
            else:
                if any(r["output_ids"] != previous[i]["output_ids"] for i,r in rows.items()):
                    raise ValueError("Migration changed a generated token")
                if rows["crossing"]["prefill_meta"].get("cached_tokens", 0) < 2048:
                    raise ValueError("Crossing request lost its warm history")
                text = (run / "cache/prefill.log").read_text(errors="replace")
                trigger = re.search(rf"Elastic Prefill trigger rank=0 generation={generation} at_ns=(\d+)",text)
                transfers = re.findall(rf"Elastic Prefill KV transport rank=0 room={crossing_room} start_ns=(\d+) end_ns=(\d+) bytes=(\d+) rc=0", text)
                overlap = [(int(a),int(b),int(n)) for a,b,n in transfers if trigger and
                           int(a) <= int(trigger[1]) < int(b) and int(n)>0]
                save(run / f"{name}.overlap.json", dict(overlap=overlap,trigger_ns=trigger[1] if trigger else None))
                if not overlap:
                    raise ValueError("No positive-byte KV transport spanned the actual migration trigger")
                observations[name] = dict(verified_output_requests=len(rows), overlap=overlap)
            # All D workers are on the same host. Measure from first destination
            # publication to the last required receiver's transfer completion.
            text = (run / "cache/decode.log").read_text(errors="replace")
            waits = {}
            for second in range(0,31):
                room = room_base + second + 31
                starts = re.findall(rf"Migration probe D posted rank=\d+ room={room} at_ns=(\d+)",text)
                ends = re.findall(rf"Migration probe D ready rank=\d+ room={room} at_ns=(\d+)",text)
                if len(starts)!=2 or len(ends)!=2:
                    raise ValueError(f"Incomplete D handoff timestamps at {name}, second {second}")
                waits[second] = (max(map(int,ends))-min(map(int,starts)))/1e6
            transition_waits[name] = waits
            save(run / "handoff_ms.json", transition_waits)
    (run / "probe.trigger").touch(mode=0o600)
    wait_file(run / "gpu_reuse_done.json", 90)
    return dict(passed=True,observations=observations,
                independent_reuse=json.loads((run / "gpu_reuse_started.json").read_text()),
                handoff_ms=transition_waits)
