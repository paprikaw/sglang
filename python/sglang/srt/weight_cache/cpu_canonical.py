"""Topology-independent node-local safetensors bank (stdlib only).

Run this file directly to build/status a bank. A bank stores complete immutable
checkpoint files, not TP/PP-prepared tensors. The normal loader still slices
and prepares weights, but reads sealed RAM files instead of shared storage.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time

try:
    from .cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key
except ImportError:
    from cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key


def _stat_identity(path: Path) -> dict:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"canonical shard is not a nonempty regular file: {path}")
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def describe_model(model_root: str | Path) -> dict:
    """Bind raw bytes to immutable local-file identities, never parallel layout."""
    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("canonical cache requires a local model directory")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("canonical cache requires local safetensors shards")
    metadata = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            data = path.read_bytes()
            metadata[name] = hashlib.sha256(data).hexdigest()
            if name.endswith("index.json"):
                index = json.loads(data)
                indexed = set(index.get("weight_map", {}).values())
                available = {file.name for file in files}
                if not indexed or indexed != available:
                    raise ValueError(
                        "safetensors index contains missing/invalid shards"
                    )
    if "config.json" not in metadata:
        raise ValueError("canonical model requires config.json")
    shards = [{"name": file.name, **_stat_identity(file)} for file in files]
    fingerprint = {
        "format": "sglang-canonical-safetensors-v1",
        "model_root": str(root),
        "metadata_sha256": metadata,
        "shards": shards,
    }
    # Device/inode are observed node-locally. GPFS mount identities can differ
    # between nodes; the invariant is same-node reuse across TP/PP layouts.
    group_key = fingerprint_key(fingerprint)
    items = []
    for shard in shards:
        identity = {
            "format": "sglang-canonical-shard-v1",
            "group_key": group_key,
            "shard": shard,
        }
        items.append({"key": fingerprint_key(identity), "fingerprint": identity})
    return {
        "model_root": str(root),
        "group_key": group_key,
        "fingerprint": fingerprint,
        "items": items,
        "shards": shards,
        "total_bytes": sum(shard["size"] for shard in shards),
        "shard_count": len(shards),
    }


def _check_daemon(client: CpuWeightCacheClient, source_commit: str) -> dict:
    status = client.status()
    if not source_commit or status["source_commit"] != source_commit:
        raise CpuWeightCacheError("source", "canonical daemon source commit differs")
    if "canonical_groups_v1" not in status.get("features", []):
        raise CpuWeightCacheError("protocol", "daemon lacks atomic canonical groups")
    return status


def bank_status(model_root: str, socket_path: str, source_commit: str) -> dict:
    bank = describe_model(model_root)
    with CpuWeightCacheClient(socket_path) as client:
        status = _check_daemon(client, source_commit)
        fetched = client.get_group(bank["items"])
        ready = fetched["state"] == "ready"
        if ready:
            try:
                _validate_members(bank, fetched)
            finally:
                client.release(fetched["reader_token"])
    return {
        "passed": ready,
        "ready": ready,
        "group_key": bank["group_key"],
        "total_bytes": bank["total_bytes"],
        "shard_count": bank["shard_count"],
        "missing_keys": fetched.get("missing_keys", []),
        "daemon_pid": status["pid"],
        "daemon_used_bytes": status["used_bytes"],
    }


def _validate_members(bank: dict, fetched: dict) -> None:
    if len(fetched["entries"]) != bank["shard_count"]:
        raise CpuWeightCacheError("fingerprint", "canonical bank shard count differs")
    for item, shard, member in zip(bank["items"], bank["shards"], fetched["entries"]):
        manifest = member["manifest"] or {}
        digest = manifest.get("sha256", "")
        if (
            member["key"] != item["key"]
            or member["size"] != shard["size"]
            or manifest.get("kind") != "canonical_safetensors"
            or manifest.get("group_key") != bank["group_key"]
            or manifest.get("name") != shard["name"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise CpuWeightCacheError("fingerprint", "canonical shard metadata differs")


def _private_parent(socket_path: str) -> Path:
    parent = Path(socket_path).resolve().parent
    info = parent.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("canonical cache requires a private owned socket directory")
    return parent


def build_bank(
    model_root: str,
    socket_path: str,
    source_commit: str,
    *,
    chunk_bytes: int = 8 * 1024**2,
    progress=None,
) -> dict:
    if chunk_bytes <= 0 or chunk_bytes > 64 * 1024**2:
        raise ValueError("copy chunk must be within (0,64MiB]")
    started = time.monotonic()
    parent = _private_parent(socket_path)
    lock_fd = os.open(
        parent / "canonical-build.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    copied = reused = 0
    try:
        # Serialize all canonical builds per daemon; this is a local CPU-bank
        # construction lock, not an experiment/GPU admission precheck.
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        bank = describe_model(model_root)
        with CpuWeightCacheClient(socket_path) as client:
            status = _check_daemon(client, source_commit)
            if bank["total_bytes"] > status["max_bytes"]:
                raise CpuWeightCacheError(
                    "capacity", "complete model exceeds daemon cap"
                )
            acquired = []
            # Reserve the entire missing bank before reading checkpoint bytes.
            # Existing ready entries remain pinned throughout the build.
            for item, shard in zip(bank["items"], bank["shards"]):
                result = client.acquire(item["key"], shard["size"], item["fingerprint"])
                if result["state"] not in ("build", "ready"):
                    raise CpuWeightCacheError("busy", "another builder owns a shard")
                acquired.append((item, shard, result))
            for item, shard, result in acquired:
                if result["state"] == "ready":
                    reused += 1
                else:
                    path = Path(bank["model_root"]) / shard["name"]
                    expected = {k: v for k, v in shard.items() if k != "name"}
                    if _stat_identity(path) != expected:
                        raise CpuWeightCacheError(
                            "fingerprint", "source changed before copy"
                        )
                    digest = hashlib.sha256()
                    offset = 0
                    with path.open("rb", buffering=0) as source:
                        while True:
                            data = source.read(chunk_bytes)
                            if not data:
                                break
                            digest.update(data)
                            written = 0
                            while written < len(data):
                                count = os.pwrite(
                                    result["fd"], data[written:], offset + written
                                )
                                if count <= 0:
                                    raise OSError("short memfd write")
                                written += count
                            offset += len(data)
                        if hasattr(os, "posix_fadvise"):
                            os.posix_fadvise(
                                source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                            )
                    if offset != shard["size"] or _stat_identity(path) != expected:
                        raise CpuWeightCacheError(
                            "fingerprint", "source changed during copy"
                        )
                    client.publish(
                        item["key"],
                        result["token"],
                        item["fingerprint"],
                        {
                            "kind": "canonical_safetensors",
                            "group_key": bank["group_key"],
                            "name": shard["name"],
                            "sha256": digest.hexdigest(),
                        },
                        pin_after_publish=True,
                    )
                    copied += 1
                if progress:
                    progress(
                        {
                            "completed_shards": copied + reused,
                            "shard_count": bank["shard_count"],
                            "group_key": bank["group_key"],
                        }
                    )
            # A complete bank must be atomically discoverable before success.
            fetched = client.get_group(bank["items"])
            if fetched["state"] != "ready":
                raise CpuWeightCacheError("incomplete", "published bank is incomplete")
            _validate_members(bank, fetched)
            client.release(fetched["reader_token"])
        # Client close releases construction/read leases but leaves ready data.
        return {
            "passed": True,
            "ready": True,
            "group_key": bank["group_key"],
            "total_bytes": bank["total_bytes"],
            "shard_count": bank["shard_count"],
            "copied_shards": copied,
            "reused_shards": reused,
            "build_elapsed_s": time.monotonic() - started,
        }
    finally:
        os.close(lock_fd)


@contextlib.contextmanager
def model_view(model_root: str, socket_path: str, source_commit: str):
    """A normal-loader directory whose complete weight set is pinned in RAM."""
    bank = describe_model(model_root)
    parent = _private_parent(socket_path)
    with CpuWeightCacheClient(socket_path) as client:
        _check_daemon(client, source_commit)
        fetched = client.get_group(bank["items"])
        if fetched["state"] != "ready":
            raise CpuWeightCacheError(
                "missing", "complete canonical bank must be prebuilt"
            )
        view = None
        try:
            _validate_members(bank, fetched)
            view = Path(tempfile.mkdtemp(prefix="canonical-view-", dir=parent))
            names = {shard["name"] for shard in bank["shards"]}
            for source in Path(bank["model_root"]).iterdir():
                if source.is_file() and source.name not in names:
                    (view / source.name).symlink_to(source)
            for shard, member in zip(bank["shards"], fetched["entries"]):
                target = Path(member["path"])
                if not target.is_file() or target.stat().st_size != shard["size"]:
                    raise CpuWeightCacheError(
                        "missing", "canonical backing file disappeared"
                    )
                (view / shard["name"]).symlink_to(target)
            yield str(view), bank
        finally:
            if view is not None:
                shutil.rmtree(view)
            # All safetensors contexts/mappings must close before leaving view.
            # Retain canonical bytes for another topology or engine process.
            client.release(fetched["reader_token"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "status"))
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        if args.operation == "build":

            def progress(value):
                print(
                    "Canonical CPU bank copy: " + json.dumps(value),
                    file=sys.stderr,
                    flush=True,
                )

            result = build_bank(
                args.model_root, args.socket, args.source_commit, progress=progress
            )
        else:
            result = bank_status(args.model_root, args.socket, args.source_commit)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    except (CpuWeightCacheError, OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
