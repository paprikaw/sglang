"""Manage named tmux terminals; never allocate nodes or touch other services."""
import argparse
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NODE = "spartan-gpgpu142"
SOCKET = "qwen38-codex"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]


def remote(args, check=True):
    return subprocess.run(SSH + [NODE, shlex.join(args)], check=check)


def tmux(*args):
    return ["tmux", "-L", SOCKET, *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status", "tunnel", "router"])
    action = parser.parse_args().action
    if action == "router":
        if subprocess.run(tmux("has-session", "-t", "router"), stderr=subprocess.DEVNULL).returncode == 0:
            print("Router terminal already exists")
            return
        subprocess.run(tmux("new-session", "-d", "-s", "router", "bash " + shlex.quote(str(ROOT / "run_router.sh"))), check=True)
        print("Started loopback-only model router in named terminal")
        return
    if action == "start":
        if remote(tmux("has-session", "-t", "server"), check=False).returncode == 0:
            print("Server terminal already exists")
        else:
            remote(tmux("new-session", "-d", "-s", "server", "bash " + shlex.quote(str(ROOT / "run_logged_server.sh"))))
            print("Started named server terminal on " + NODE)
        return
    if action == "tunnel":
        if subprocess.run(tmux("has-session", "-t", "tunnel"), stderr=subprocess.DEVNULL).returncode == 0:
            print("Tunnel terminal already exists")
            return
        command = SSH + ["-N", "-T", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-L", "127.0.0.1:30038:127.0.0.1:30038", NODE]
        subprocess.run(tmux("new-session", "-d", "-s", "tunnel", shlex.join(command)), check=True)
        print("Started loopback-only SSH tunnel in named terminal")
        return
    if action == "stop":
        # Ctrl-C goes only to this service's named terminal, not other Python jobs.
        remote(tmux("send-keys", "-t", "server", "C-c"), check=False)
        subprocess.run(tmux("kill-session", "-t", "tunnel"), check=False)
        print("Stop requested; run status to confirm GPU processes have exited")
        return
    remote(tmux("list-sessions"), check=False)
    remote(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"])
    subprocess.run(tmux("list-sessions"), check=False)
    key = (ROOT / "api.key").read_text().strip()
    req = urllib.request.Request("http://127.0.0.1:30038/health", headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            print("API health:", response.status)
    except Exception as exc:
        print("API not ready:", str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
