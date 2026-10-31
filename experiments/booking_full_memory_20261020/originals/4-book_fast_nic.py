#!/usr/bin/env python3
"""Two bounded A100 bookings with an allocation-time NIC gate; no GPU work."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
NAME = "A100-fastNIC"
MAX_ATTEMPTS = 3
# These hosts were measured with PCIe 3.0 NICs. Excluding the entire chassis
# family is conservative: it does NOT claim every host was measured slow.
SLOW_FAMILY = {f"spartan-gpgpu{i:03d}" for i in range(132, 144)}


def run(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True,
                          timeout=45).stdout.strip()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def queue():
    rows = run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%N"])
    return [dict(zip(("id", "name", "state", "nodes"), row.split("|", 3)))
            for row in rows.splitlines() if row]


def node_pool(rows, nic):
    lines = run(["sinfo", "-h", "-N", "-p", "deeplearn",
                 "-o", "%N|%f|%G"]).splitlines()
    inventory = {}
    for line in lines:
        name, features, gres = line.split("|", 2)
        inventory[name] = {"features": features.split(","), "gres": gres}
    held = set()
    for job in rows:
        if job["state"] in {"RUNNING", "COMPLETING", "CONFIGURING"} and job["nodes"]:
            held.update(run(["scontrol", "show", "hostnames", job["nodes"]]).splitlines())
    pool = {n for n, info in inventory.items()
            if "dlg5" in info["features"] and "gpu:A100:4" in info["gres"]}
    pool -= held
    if nic == "connectx6-pcie4":
        pool -= SLOW_FAMILY
    return sorted(pool), sorted(set(inventory) - pool), sorted(held)


def command(request, record_dir, slot, test_only=False):
    excluded = sorted(set(request["excluded_nodes"]) | set(request["rejected_nodes"]))
    argv = ["sbatch", "--parsable", "--job-name=" + NAME,
            "--partition=deeplearn", "--qos=gpgpudeeplearn", "--account=punim2715",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=700000",
            "--gres=gpu:A100:4", "--constraint=dlg5", "--time=" + request["time"],
            "--chdir=" + str(ROOT), "--output=" + str(record_dir / "job_%j.out"),
            "--comment=fast-nic-" + request["request_id"] + "-slot" + str(slot)]
    if excluded:
        argv.append("--exclude=" + ",".join(excluded))
    if test_only:
        argv.append("--test-only")
    argv += [str(ROOT / "fast_nic_hold.sh"), "--nic", request["nic"],
             "--record-dir", str(record_dir)]
    return argv


def status(path):
    data = json.loads(path.read_text())
    ids = [entry["job_id"] for slot in data["slots"] for entry in slot["attempts"]]
    print(json.dumps({"request": str(path), "job_ids": ids}, indent=2))
    if ids:
        print(run(["squeue", "--start", "--jobs=" + ",".join(ids),
                   "-o", "%.12i %.12T %.20S %.24N %R"]))
    for job_id in ids:
        report = path.parent / ("attestation_" + job_id + ".json")
        if report.exists():
            print(report.read_text())


def retry_candidates(request, record_dir, rows):
    """Only replace failed NIC gates after Slurm confirms the old job exited."""
    live = {row["id"] for row in rows}
    slots = []
    for slot in request["slots"]:
        if not slot["attempts"]:
            slots.append(slot["slot"])
            continue
        previous = slot["attempts"][-1]
        if previous["job_id"] in live or len(slot["attempts"]) >= MAX_ATTEMPTS:
            continue
        report = record_dir / ("attestation_" + previous["job_id"] + ".json")
        if not report.exists():
            continue
        attestation = json.loads(report.read_text())
        if attestation.get("passed") is not False:
            continue
        accounting = run(["sacct", "-X", "-n", "-P", "-j", previous["job_id"],
                          "-o", "JobIDRaw,State,ExitCode"])
        root_rows = [r.split("|") for r in accounting.splitlines()
                     if r.split("|")[0] == previous["job_id"]]
        if not any(r[1].startswith("FAILED") for r in root_rows):
            continue
        node = attestation.get("node")
        if node:
            request["rejected_nodes"] = sorted(set(request["rejected_nodes"]) | {node})
        slots.append(slot["slot"])
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nic", choices=("connectx6-pcie4", "rdma"), default="connectx6-pcie4")
    parser.add_argument("--count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--time", default="1-00:00:00", help="Slurm duration; default 24 hours")
    parser.add_argument("--submit", action="store_true", help="Actually submit; default is preview")
    parser.add_argument("--test-only", action="store_true", help="Ask Slurm for a forecast without submitting")
    parser.add_argument("--status", type=Path, help="Read this request.json and scheduler status")
    parser.add_argument("--retry", type=Path, help="Retry only NIC-rejected jobs; max 3 attempts per slot")
    args = parser.parse_args()
    if args.submit and args.test_only:
        parser.error("--submit and --test-only are mutually exclusive")
    if args.status:
        status(args.status.resolve())
        return
    with (ROOT / ".fast-nic-booking.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = queue()
        if args.retry:
            path = args.retry.resolve()
            request = json.loads(path.read_text())
            record_dir = path.parent
            slots = retry_candidates(request, record_dir, rows)
        else:
            if any(row["name"] == NAME for row in rows):
                raise RuntimeError("An active fast-NIC request exists; use --status/--retry, not another booking.")
            if args.submit and any(row["state"] == "PENDING" and row["name"].startswith("A100") for row in rows):
                raise RuntimeError("Pending A100 bookings exist; inspect them before adding nodes.")
            pool, excluded, held = node_pool(rows, args.nic)
            if len(pool) < args.count:
                raise RuntimeError("Insufficient eligible distinct nodes")
            request_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
            record_dir = ROOT / "outputs" / ("fast_nic_" + request_id)
            path = record_dir / "request.json"
            request = {"request_id": request_id, "created_at": dt.datetime.now().astimezone().isoformat(),
                       "nic": args.nic, "time": args.time, "count": args.count,
                       "candidate_nodes_unverified_until_gate": pool, "excluded_nodes": excluded,
                       "held_nodes_excluded": held, "rejected_nodes": [],
                       "max_attempts_per_slot": MAX_ATTEMPTS,
                       "slots": [{"slot": i + 1, "attempts": []} for i in range(args.count)]}
            slots = list(range(1, args.count + 1))
        print(json.dumps({"request": str(path), "nic": request["nic"], "time": request["time"],
                          "candidate_nodes": request["candidate_nodes_unverified_until_gate"],
                          "slots_to_submit": slots}, indent=2), flush=True)
        if args.submit:
            record_dir.mkdir(parents=True, exist_ok=True)
            save(path, request)
        for slot_num in slots:
            argv = command(request, record_dir, slot_num, args.test_only)
            print(shlex.join(argv), flush=True)
            if args.test_only:
                result = subprocess.run(argv, text=True, capture_output=True, timeout=45)
                print(result.stdout + result.stderr, flush=True)
                if result.returncode:
                    raise RuntimeError("Slurm test-only failed")
            elif args.submit:
                result = run(argv)
                job_id = result.split(";")[0]
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise RuntimeError("Unexpected sbatch response: " + result)
                request["slots"][slot_num - 1]["attempts"].append(
                    {"job_id": job_id, "submitted_at": dt.datetime.now().astimezone().isoformat()})
                save(path, request)
                print("Submitted " + job_id, flush=True)
        if args.submit:
            status(path)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
