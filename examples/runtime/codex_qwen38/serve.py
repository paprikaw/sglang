"""Launch SGLang without putting the local API credential in process arguments."""
import json
import logging
import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NODE = "spartan-gpgpu142"
JOB = "29526906"


def preflight():
    if socket.gethostname().split(".")[0] != NODE:
        raise SystemExit(f"Run this launcher on {NODE} only")
    jobs = subprocess.check_output([
        "/apps/slurm/latest/bin/squeue", "-h", "-j", JOB,
        "-o", "%u %T %N",
    ], text=True).strip()
    if jobs != f"bxb1 RUNNING {NODE}":
        raise SystemExit(f"Expected running booking is unavailable: {jobs}")
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True).strip()
    if active:
        raise SystemExit("Node already has GPU compute processes; refusing to co-locate")
    manifest = json.loads((ROOT / "model_snapshot.json").read_text())
    if manifest["shards"] != 18:
        raise SystemExit("Model download is incomplete")


if __name__ == "__main__":
    os.umask(0o077)
    preflight()
    key = (ROOT / "api.key").read_text().strip()
    # Some SGLang versions log the ServerArgs repr, which includes api_key.
    factory = logging.getLogRecordFactory()

    def redacted_record(*args, **kwargs):
        record = factory(*args, **kwargs)
        if key in record.getMessage():
            record.msg = record.getMessage().replace(key, "[REDACTED]")
            record.args = ()
        return record

    logging.setLogRecordFactory(redacted_record)
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = json.loads((ROOT / "server_args.json").read_text())
    server_args = prepare_server_args(args)
    server_args.api_key = key
    (ROOT / "server.pid").write_text(str(os.getpid()) + "\n")
    print(f"SGLang service PID={os.getpid()} node={NODE} GPU=0", flush=True)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
        (ROOT / "server.pid").unlink(missing_ok=True)
