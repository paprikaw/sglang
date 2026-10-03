"""Bounded per-node CPU weight cache with sealed memfds and Unix FD passing.

Run this file directly with --socket/--max-bytes to avoid importing the serving
engine. A daemon owns weight snapshots independently of model-worker lifetime.
It never reads model files, imports CUDA, or changes an external/HiCache cache.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import secrets
import signal
import socket
import stat
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .cpu_protocol import (
        F_GET_SEALS,
        PROTOCOL_VERSION,
        REQUIRED_SEALS,
        create_memfd,
        default_socket_path,
        fingerprint_key,
        receive_message,
        send_message,
    )
except ImportError:
    from cpu_protocol import (  # Direct script execution on CPU-only nodes.
        F_GET_SEALS,
        PROTOCOL_VERSION,
        REQUIRED_SEALS,
        create_memfd,
        default_socket_path,
        fingerprint_key,
        receive_message,
        send_message,
    )


def process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def memory_headroom(pid: int) -> dict:
    """Read physical and all finite v2-cgroup ancestor limits for a peer."""
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    if available is None:
        raise RuntimeError("cannot observe node MemAvailable")
    observations = [{"path": "MemAvailable", "available_bytes": available}]
    lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    unified = next(
        (line.split(":", 2)[2] for line in lines if line.startswith("0::")), None
    )
    if unified is None:
        raise RuntimeError("CPU cache requires observable cgroup v2 memory limits")
    root = Path("/sys/fs/cgroup")
    current = root / unified.lstrip("/")
    while current == root or root in current.parents:
        maximum = current / "memory.max"
        usage = current / "memory.current"
        if maximum.exists() and usage.exists():
            raw_maximum = maximum.read_text().strip()
            if raw_maximum != "max":
                maximum_bytes = int(raw_maximum)
                current_bytes = int(usage.read_text())
                observations.append(
                    {
                        "path": str(current),
                        "limit_bytes": maximum_bytes,
                        "current_bytes": current_bytes,
                        "available_bytes": max(0, maximum_bytes - current_bytes),
                    }
                )
        if current == root:
            break
        current = current.parent
    return {
        "available_bytes": min(x["available_bytes"] for x in observations),
        "observations": observations,
    }


@dataclass
class Entry:
    key: str
    fingerprint: Any
    reserved_bytes: int
    fd: int
    owner: str | None
    build_token: str | None
    size: int = 0
    manifest: Any = None
    state: str = "building"
    readers: dict[str, str] = field(default_factory=dict)
    last_access: float = field(default_factory=time.monotonic)
    evict_pending: bool = False


class CacheFailure(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


class CpuWeightCacheDaemon:
    def __init__(
        self,
        socket_path: str,
        max_bytes: int,
        source_commit: str = "",
        headroom_bytes: int = 32 * 1024**3,
    ):
        if max_bytes <= 0 or headroom_bytes < 0:
            raise ValueError("positive byte budget and nonnegative headroom required")
        self.socket_path = Path(socket_path)
        self.max_bytes = max_bytes
        self.source_commit = source_commit
        self.headroom_bytes = headroom_bytes
        self.entries: dict[str, Entry] = {}
        self.group_readers: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.connections: set[socket.socket] = set()
        self.owners: dict[str, tuple[int, int, socket.socket]] = {}
        self.listener = None
        self.lock_fd = None
        self.threads: list[threading.Thread] = []

    def _total(self) -> int:
        return sum(
            e.reserved_bytes if e.state == "building" else e.size
            for e in self.entries.values()
        )

    def _remove(self, entry: Entry) -> None:
        del self.entries[entry.key]
        os.close(entry.fd)
        self.condition.notify_all()

    def _identity(self, request: dict) -> tuple[str, Any]:
        fingerprint = request.get("fingerprint")
        key = request.get("key")
        if (
            fingerprint is None
            or not isinstance(key, str)
            or key != fingerprint_key(fingerprint)
        ):
            raise CacheFailure(
                "fingerprint", "key must match the immutable fingerprint"
            )
        return key, fingerprint

    def _read(self, entry: Entry, owner: str) -> tuple[dict, int]:
        token = secrets.token_hex(24)
        # A duplicate of an O_RDWR FD is still writable. Reopen the sealed inode
        # read-only so consumers cannot accidentally request writable mappings.
        fd = os.open(f"/proc/self/fd/{entry.fd}", os.O_RDONLY | os.O_CLOEXEC)
        entry.readers[token] = owner
        entry.last_access = time.monotonic()
        return {
            "state": "ready",
            "key": entry.key,
            "reader_token": token,
            "size": entry.size,
            "manifest": entry.manifest,
        }, fd

    def handle(self, request: dict, owner: str, pid: int) -> tuple[dict, int | None]:
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise CacheFailure("protocol", "unsupported cache protocol version")
        op = request.get("op")
        with self.condition:
            if op == "status":
                used = sum(e.size for e in self.entries.values() if e.state == "ready")
                reserved = self._total() - used
                return {
                    "protocol_version": PROTOCOL_VERSION,
                    "uid": os.getuid(),
                    "pid": os.getpid(),
                    "start_ticks": process_start_ticks(os.getpid()),
                    "source_commit": self.source_commit,
                    "max_bytes": self.max_bytes,
                    "headroom_bytes": self.headroom_bytes,
                    "features": ["canonical_groups_v1"],
                    "used_bytes": used,
                    "reserved_bytes": reserved,
                    "current_bytes": used + reserved,
                    "entries": [
                        {
                            "key": e.key,
                            "state": e.state,
                            "size": e.size,
                            "reserved_bytes": e.reserved_bytes,
                            "readers": len(e.readers),
                            "evict_pending": e.evict_pending,
                        }
                        for e in self.entries.values()
                    ],
                }, None
            if op == "get_group":
                items = request.get("items")
                if (
                    not isinstance(items, list)
                    or not items
                    or len(items) > 4096
                    or not all(isinstance(item, dict) for item in items)
                ):
                    raise CacheFailure("group", "nonempty bounded shard list required")
                identities = [self._identity(item) for item in items]
                if len({key for key, _ in identities}) != len(identities):
                    raise CacheFailure("group", "duplicate group member")
                wait_s = min(max(float(request.get("wait_s", 0)), 0), 3600)
                deadline = time.monotonic() + wait_s
                while True:
                    unavailable = []
                    for key, fingerprint in identities:
                        entry = self.entries.get(key)
                        if entry and entry.fingerprint != fingerprint:
                            raise CacheFailure(
                                "fingerprint", "stored fingerprint differs"
                            )
                        if not entry or entry.state != "ready" or entry.evict_pending:
                            unavailable.append(key)
                    if not unavailable:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"state": "missing", "missing_keys": unavailable}, None
                    self.condition.wait(min(remaining, 0.25))
                    if self.stop_event.is_set():
                        raise CacheFailure("shutdown", "daemon is stopping")
                token = secrets.token_hex(24)
                keys = tuple(key for key, _ in identities)
                self.group_readers[token] = (owner, keys)
                members = []
                for key in keys:
                    entry = self.entries[key]
                    entry.readers[token] = owner
                    entry.last_access = time.monotonic()
                    members.append(
                        {
                            "key": key,
                            "size": entry.size,
                            "manifest": entry.manifest,
                            "path": f"/proc/{os.getpid()}/fd/{entry.fd}",
                        }
                    )
                return {
                    "state": "ready",
                    "reader_token": token,
                    "entries": members,
                }, None
            if op in ("acquire", "get"):
                key, fingerprint = self._identity(request)
                wait_s = min(max(float(request.get("wait_s", 0)), 0), 3600)
                deadline = time.monotonic() + wait_s
                while key in self.entries and self.entries[key].state == "building":
                    if self.entries[key].owner == owner:
                        raise CacheFailure(
                            "lease", "connection already owns this build"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"state": "building", "key": key}, None
                    self.condition.wait(min(remaining, 0.25))
                    if self.stop_event.is_set():
                        raise CacheFailure("shutdown", "daemon is stopping")
                entry = self.entries.get(key)
                if entry:
                    if entry.fingerprint != fingerprint:
                        raise CacheFailure("fingerprint", "stored fingerprint differs")
                    if entry.evict_pending:
                        return {"state": "retiring", "key": key}, None
                    return self._read(entry, owner)
                if op == "get":
                    return {"state": "missing", "key": key}, None
                expected = request.get("expected_bytes")
                if (
                    isinstance(expected, bool)
                    or not isinstance(expected, int)
                    or expected <= 0
                ):
                    raise CacheFailure("size", "positive integer reservation required")
                if expected > self.max_bytes:
                    raise CacheFailure("capacity", "model exceeds daemon byte limit")
                # Evict only complete, unborrowed snapshots. A reader or build
                # cannot be silently sacrificed to admit another model.
                victims = sorted(
                    (
                        e
                        for e in self.entries.values()
                        if e.state == "ready" and not e.readers
                    ),
                    key=lambda e: e.last_access,
                )
                if (
                    self._total() + expected - sum(e.size for e in victims)
                    > self.max_bytes
                ):
                    raise CacheFailure(
                        "capacity", "budget occupied by readers or builds"
                    )
                for victim in victims:
                    if self._total() + expected <= self.max_bytes:
                        break
                    self._remove(victim)
                if self._total() + expected > self.max_bytes:
                    raise CacheFailure(
                        "capacity", "budget occupied by readers or builds"
                    )
                # ftruncate is lazy. Add only the not-yet-faulted part of each
                # build reservation: existing pages already appear in current
                # cgroup/physical usage. Observe backing pages first, then usage;
                # concurrent writes can only make this estimate conservative.
                pending = sum(
                    max(0, e.reserved_bytes - os.fstat(e.fd).st_blocks * 512)
                    for e in self.entries.values()
                    if e.state == "building"
                )
                try:
                    headroom = memory_headroom(pid)
                    daemon_headroom = memory_headroom(os.getpid())
                except (OSError, RuntimeError, ValueError) as exc:
                    raise CacheFailure("headroom", str(exc)) from exc
                available = min(
                    headroom["available_bytes"], daemon_headroom["available_bytes"]
                )
                if available < expected + pending + self.headroom_bytes:
                    raise CacheFailure(
                        "headroom",
                        "insufficient observed CPU/cgroup memory: "
                        f"available={available}, needed="
                        f"{expected + pending + self.headroom_bytes}",
                    )
                fd = create_memfd("sglang-cpu-weight-cache")
                try:
                    os.ftruncate(fd, expected)
                except BaseException:
                    os.close(fd)
                    raise
                token = secrets.token_hex(24)
                self.entries[key] = Entry(key, fingerprint, expected, fd, owner, token)
                return {
                    "state": "build",
                    "key": key,
                    "token": token,
                    "size": expected,
                    "headroom": headroom,
                }, os.dup(fd)
            if op == "publish":
                key, fingerprint = self._identity(request)
                entry = self.entries.get(key)
                if (
                    not entry
                    or entry.state != "building"
                    or entry.owner != owner
                    or entry.build_token != request.get("token")
                ):
                    raise CacheFailure(
                        "lease", "publish requires the owning build lease"
                    )
                size = os.fstat(entry.fd).st_size
                if (
                    size <= 0
                    or size > entry.reserved_bytes
                    or size != request.get("used_bytes")
                ):
                    raise CacheFailure("size", "publication does not fit reservation")
                if (
                    fcntl.fcntl(entry.fd, F_GET_SEALS) & REQUIRED_SEALS
                    != REQUIRED_SEALS
                ):
                    raise CacheFailure(
                        "mutable", "publication requires a fully sealed memfd"
                    )
                if entry.fingerprint != fingerprint:
                    raise CacheFailure("fingerprint", "publication identity changed")
                entry.size = size
                entry.manifest = request.get("manifest")
                entry.state, entry.owner, entry.build_token = "ready", None, None
                entry.last_access = time.monotonic()
                self.condition.notify_all()
                if request.get("pin_after_publish"):
                    return self._read(entry, owner)
                return {"state": "ready", "key": key, "size": size}, None
            if op == "release":
                token = request.get("token")
                group = self.group_readers.get(token)
                if group and group[0] == owner:
                    del self.group_readers[token]
                    for key in group[1]:
                        entry = self.entries[key]
                        del entry.readers[token]
                        entry.evict_pending |= bool(request.get("evict"))
                        if entry.evict_pending and not entry.readers:
                            self._remove(entry)
                    self.condition.notify_all()
                    return {"state": "released", "shard_count": len(group[1])}, None
                for entry in list(self.entries.values()):
                    if entry.build_token == token and entry.owner == owner:
                        self._remove(entry)
                        return {"state": "aborted"}, None
                    if entry.readers.get(token) == owner:
                        del entry.readers[token]
                        entry.evict_pending |= bool(request.get("evict"))
                        if entry.evict_pending and not entry.readers:
                            self._remove(entry)
                            return {"state": "evicted"}, None
                        self.condition.notify_all()
                        return {
                            "state": "released",
                            "readers": len(entry.readers),
                        }, None
                raise CacheFailure("lease", "unknown lease for this connection")
            if op == "evict":
                entry = self.entries.get(request.get("key"))
                if not entry:
                    return {"state": "missing"}, None
                if entry.state != "ready" or entry.readers:
                    raise CacheFailure("busy", "cannot evict a build or current reader")
                self._remove(entry)
                return {"state": "evicted"}, None
            raise CacheFailure("operation", "unknown cache operation")

    def _connection(self, connection: socket.socket) -> None:
        owner = secrets.token_hex(24)
        try:
            pid, uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid():
                return
            start = process_start_ticks(pid)
            with self.condition:
                self.owners[owner] = (pid, start, connection)
            while not self.stop_event.is_set():
                request, supplied_fd = receive_message(connection)
                if supplied_fd is not None:
                    os.close(supplied_fd)
                    raise ValueError("clients cannot supply arbitrary descriptors")
                response_fd = None
                try:
                    if process_start_ticks(pid) != start:
                        raise CacheFailure("owner", "client process identity changed")
                    response, response_fd = self.handle(request, owner, pid)
                    send_message(connection, {"ok": True, **response}, response_fd)
                except CacheFailure as exc:
                    send_message(
                        connection,
                        {"ok": False, "code": exc.code, "error": exc.message},
                    )
                except (ValueError, TypeError, OSError) as exc:
                    send_message(
                        connection, {"ok": False, "code": "invalid", "error": str(exc)}
                    )
                finally:
                    if response_fd is not None:
                        os.close(response_fd)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            with self.condition:
                self.group_readers = {
                    token: group
                    for token, group in self.group_readers.items()
                    if group[0] != owner
                }
                for entry in list(self.entries.values()):
                    if entry.state == "building" and entry.owner == owner:
                        self._remove(entry)
                        continue
                    entry.readers = {
                        token: client
                        for token, client in entry.readers.items()
                        if client != owner
                    }
                    if entry.evict_pending and not entry.readers:
                        self._remove(entry)
                self.connections.discard(connection)
                self.owners.pop(owner, None)
                self.condition.notify_all()
            connection.close()

    def _reap_dead_owners(self) -> None:
        # A fork can inherit the socket after its original producer exits. PID
        # and /proc starttime, rather than socket EOF alone, bound partial builds.
        with self.condition:
            for pid, start, connection in list(self.owners.values()):
                try:
                    proc = (
                        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    )
                    alive = proc[0] not in ("Z", "X") and int(proc[19]) == start
                except (OSError, ValueError, IndexError):
                    alive = False
                if not alive:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

    def serve(self) -> None:
        parent = self.socket_path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError(
                "cache socket directory must be private, owned and mode 0700"
            )
        self.lock_fd = os.open(
            str(self.socket_path) + ".lock",
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            raise RuntimeError("a cache daemon already owns this socket")
        if self.socket_path.exists() or self.socket_path.is_symlink():
            if not stat.S_ISSOCK(self.socket_path.lstat().st_mode):
                raise RuntimeError("refusing to replace a non-socket cache path")
            self.socket_path.unlink()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.socket_path))
        self.socket_path.chmod(0o600)
        self.listener.listen(128)
        self.listener.settimeout(0.25)
        try:
            while not self.stop_event.is_set():
                self._reap_dead_owners()
                self.threads = [worker for worker in self.threads if worker.is_alive()]
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                with self.condition:
                    self.connections.add(connection)
                worker = threading.Thread(
                    target=self._connection, args=(connection,), daemon=True
                )
                self.threads.append(worker)
                worker.start()
        finally:
            self.stop()
            for worker in self.threads:
                worker.join(timeout=2)
            with self.condition:
                for entry in list(self.entries.values()):
                    self._remove(entry)
            self.listener.close()
            self.socket_path.unlink(missing_ok=True)
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            for connection in list(self.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.condition.notify_all()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=default_socket_path())
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**3)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--headroom-bytes", type=int, default=32 * 1024**3)
    args = parser.parse_args()
    daemon = CpuWeightCacheDaemon(
        args.socket, args.max_bytes, args.source_commit, args.headroom_bytes
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: daemon.stop())
    daemon.serve()


if __name__ == "__main__":
    main()
