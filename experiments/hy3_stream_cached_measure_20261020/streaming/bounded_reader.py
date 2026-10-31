"""CPU primitives for a reviewed, bounded Hy3 loader; not a Guard exemption.

No Torch/CUDA imports, file mmap, background prefetch, or canonical-cache claims.
The controller must separately authenticate the lease and admit existing weight
banks + node staging quota + 64 GiB workspace + at least 32 GiB headroom.
Use one StagingArena per rank across ALL shards. Consumers copy to owned GPU
parameters and must supply a synchronization callback before a window is reused.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import threading
import time
import uuid


class Cancelled(RuntimeError):
    pass


def _cancel(check):
    if check is not None and check():
        raise Cancelled("Bounded weight operation cancelled")


def _ticks(pid):
    # Fields after ')' start at proc field 3; starttime is field 22.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _private_fd(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Runtime evidence path must be absolute")
    info = path.parent.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Runtime evidence parent must be private and UID-owned")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise ValueError("Runtime evidence must be a private UID-owned regular file")
    return fd


class SharedCredits:
    """One fixed node quota shared by independent rank processes.

    No stale reservation is reclaimed automatically. Exact owner death and Guard
    cleanup must be verified externally before deleting an abandoned ledger.
    """

    def __init__(self, path, limit_bytes):
        if type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("Positive node staging quota required")
        self.path = Path(path)
        self.limit = limit_bytes
        self.pid, self.start_ticks = os.getpid(), _ticks(os.getpid())
        self.snapshot()

    @contextmanager
    def _locked(self):
        fd = _private_fd(self.path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.pread(fd, 4 * 1024 * 1024, 0)
            value = json.loads(raw) if raw else {"schema": 1, "limit_bytes": self.limit, "leases": {}}
            if value.get("schema") != 1 or value.get("limit_bytes") != self.limit:
                raise ValueError("Staging ledger identity/quota differs")
            leases = value.get("leases")
            if not isinstance(leases, dict) or any(
                not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] <= 0
                for row in leases.values()
            ) or sum(row["bytes"] for row in leases.values()) > self.limit:
                raise ValueError("Invalid/overcommitted staging ledger")
            yield value
            out = json.dumps(value, sort_keys=True).encode()
            os.pwrite(fd, out, 0)
            os.ftruncate(fd, len(out))
            os.fsync(fd)
        finally:
            os.close(fd)

    def snapshot(self):
        with self._locked() as value:
            return {**value, "used_bytes": sum(row["bytes"] for row in value["leases"].values())}

    def reserve(self, amount, *, cancel=None, timeout_s=30):
        if type(amount) is not int or not 0 < amount <= self.limit:
            raise ValueError("Reservation outside node quota")
        deadline = time.monotonic() + timeout_s
        while True:
            _cancel(cancel)
            with self._locked() as value:
                used = sum(row["bytes"] for row in value["leases"].values())
                if used + amount <= self.limit:
                    token = uuid.uuid4().hex
                    value["leases"][token] = {"pid": self.pid, "start_ticks": self.start_ticks, "bytes": amount}
                    return token
            if time.monotonic() >= deadline:
                raise TimeoutError("Node staging quota exhausted; no implicit eviction")
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def release(self, token):
        with self._locked() as value:
            row = value["leases"].get(token)
            if row is None or row.get("pid") != os.getpid() or row.get("start_ticks") != _ticks(os.getpid()):
                raise RuntimeError("Staging release requires exact live owner identity")
            del value["leases"][token]


def _identity(info):
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def verify_file(path, expected_sha256, *, cancel=None, chunk_bytes=1024 * 1024):
    """Hash a completed download once; freeze the returned proof before use."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Expected lowercase SHA256 required")
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 16 * 1024 * 1024:
        raise ValueError("Hash scratch must be bounded to at most 16 MiB")
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Absolute nonsymlink source required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = _identity(os.fstat(fd))
        digest = hashlib.sha256()
        offset = 0
        while offset < before["size"]:
            _cancel(cancel)
            data = os.pread(fd, min(chunk_bytes, before["size"] - offset), offset)
            if not data:
                raise RuntimeError("Source truncated during verification")
            digest.update(data)
            offset += len(data)
        if _identity(os.fstat(fd)) != before or digest.hexdigest() != expected_sha256:
            raise RuntimeError("Source changed or SHA256 verification failed")
        return {"schema": 1, "path": str(path), "sha256": expected_sha256,
                "sha256_verified": True, **before}
    finally:
        os.close(fd)


def _receipt(path, row):
    if path is None:
        return
    if isinstance(path, ReceiptJournal):
        path.append(row)
        return
    fd = _private_fd(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_END)
        data = (json.dumps(row, sort_keys=True) + "\n").encode()
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)


class ReceiptJournal:
    """One private descriptor per rank; sync at shard and iterator boundaries.

    Each record is written immediately, without an unbounded in-memory queue.
    Completion is attested only after the final fsync/close succeeded. An
    interrupted load may lack durable final records and cannot pass postcheck.
    """

    def __init__(self, path):
        self.fd = _private_fd(path)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(self.fd).st_size:
                raise ValueError("A fresh empty tensor receipt journal is required")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def append(self, row):
        if self.fd is None:
            raise RuntimeError("Tensor receipt journal is closed")
        data = (json.dumps(row, sort_keys=True) + "\n").encode()
        # Receipts contain scalar metadata only. Fail instead of retaining an
        # arbitrarily large metadata object or allowing a giant write.
        if len(data) > 64 * 1024:
            raise ValueError("Tensor receipt exceeds the bounded metadata limit")
        while data:
            written = os.write(self.fd, data)
            if written <= 0:
                raise OSError("Short tensor receipt write")
            data = data[written:]

    def flush(self):
        if self.fd is None:
            raise RuntimeError("Tensor receipt journal is closed")
        os.fsync(self.fd)

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


class StagingArena:
    """A single owned bytearray whose ENTIRE capacity stays charged until close.

    Retained CPU views cannot cause additional payload allocations: subsequent
    tensors reuse this same buffer after synchronization. Consumers must not use
    a previous view after the context exits. A sync failure poisons the arena.
    """

    def __init__(self, credits, capacity_bytes, *, rank_limit_bytes, cancel=None, timeout_s=30):
        if type(capacity_bytes) is not int or not 0 < capacity_bytes <= rank_limit_bytes:
            raise ValueError("Arena exceeds admitted per-rank staging limit")
        self.credits, self.capacity = credits, capacity_bytes
        self.token = credits.reserve(capacity_bytes, cancel=cancel, timeout_s=timeout_s)
        try:
            self.buffer = bytearray(capacity_bytes)
        except BaseException:
            credits.release(self.token)
            raise
        self.lock = threading.Lock()
        self.poisoned = self.closed = False

    def close(self):
        if self.closed:
            return
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Unsafe arena close: synchronization failed or transfer active")
        try:
            if self.poisoned:
                raise RuntimeError("Unsafe arena close: synchronization failed or transfer active")
            # Raises BufferError if an external Torch/CPU view still exports it.
            # Keep the reservation in that case; never undercount live owned bytes.
            self.buffer.clear()
            self.credits.release(self.token)
            self.closed = True
        finally:
            self.lock.release()


@dataclass
class TensorWindow:
    name: str
    dtype: str
    shape: tuple
    view: memoryview
    source_offset: int


_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


class SafetensorsReader:
    def __init__(self, proof, arena, *, receipts_path=None, max_header_bytes=16 * 1024 * 1024):
        if proof.get("schema") != 1 or proof.get("sha256_verified") is not True:
            raise ValueError("Frozen, once-hash-verified download proof required")
        self.proof, self.arena, self.receipts_path = dict(proof), arena, receipts_path
        self.fd = os.open(proof["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            self._check_identity()
            raw = os.pread(self.fd, 8, 0)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors header length")
            size = struct.unpack("<Q", raw)[0]
            if not 2 <= size <= max_header_bytes or size + 8 > proof["size"]:
                raise ValueError("Safetensors header outside bounded source")
            raw = os.pread(self.fd, size, 8)
            if len(raw) != size:
                raise ValueError("Truncated safetensors header")
            self.header_sha256 = hashlib.sha256(raw).hexdigest()
            self.data_start = size + 8
            self.tensors = json.loads(raw)
            if not isinstance(self.tensors, dict):
                raise ValueError("Safetensors header must be an object")
            intervals = []
            for name, desc in self.tensors.items():
                if name == "__metadata__":
                    continue
                shape, offsets, dtype = desc.get("shape"), desc.get("data_offsets"), desc.get("dtype")
                if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(n) is not int for n in offsets) or dtype not in _DTYPE_BYTES):
                    raise ValueError("Invalid safetensors shape/offset/dtype")
                a, b = offsets
                if not 0 <= a <= b <= proof["size"] - self.data_start or b - a != math.prod(shape) * _DTYPE_BYTES[dtype]:
                    raise ValueError("Safetensors tensor range/shape mismatch")
                intervals.append((a, b))
            intervals.sort()
            if any(a < previous_b for (_, previous_b), (a, _) in zip(intervals, intervals[1:])):
                raise ValueError("Overlapping safetensors tensor ranges")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def _check_identity(self):
        expected = {key: self.proof[key] for key in _identity(os.fstat(self.fd))}
        if _identity(os.fstat(self.fd)) != expected:
            raise RuntimeError("Source identity changed after verified download")

    @contextmanager
    def tensor(self, name, *, cancel=None, synchronize=None):
        if name == "__metadata__" or name not in self.tensors:
            raise KeyError(name)
        arena = self.arena
        if arena.closed or arena.poisoned:
            raise RuntimeError("Staging arena closed or poisoned")
        desc = self.tensors[name]
        start, end = desc["data_offsets"]
        size = end - start
        if size > arena.capacity:
            raise ValueError("Tensor exceeds admitted arena; no implicit clone/chunk fallback")
        with arena.lock:
            if arena.closed or arena.poisoned:
                raise RuntimeError("Staging arena closed or poisoned")
            _cancel(cancel)
            self._check_identity()
            view = memoryview(arena.buffer)[:size]
            loaded = False
            row = {"schema": 1, "source": "bounded_verified_safetensors", "tensor": name,
                   "file_sha256": self.proof["sha256"], "header_sha256": self.header_sha256,
                   "offset": self.data_start + start, "bytes": size,
                   "arena_capacity_bytes": arena.capacity, "credit_token": arena.token,
                   "pid": os.getpid(), "started_at": time.time(),
                   "read_elapsed_s": 0.0, "consumption_sync_elapsed_s": 0.0}
            consume_started = None
            try:
                offset = 0
                while offset < size:
                    _cancel(cancel)
                    part = view[offset:min(size, offset + 16 * 1024 * 1024)]
                    try:
                        read_started = time.perf_counter()
                        n = os.preadv(self.fd, [part], self.data_start + start + offset)
                        row["read_elapsed_s"] += time.perf_counter() - read_started
                    finally:
                        part.release()
                    if n <= 0:
                        raise RuntimeError("Source truncated during tensor transfer")
                    offset += n
                self._check_identity()
                loaded = True
                consume_started = time.perf_counter()
                try:
                    yield TensorWindow(name, desc["dtype"], tuple(desc["shape"]), view, self.data_start + start)
                except BaseException:
                    row["status"] = "consumer_failed"
                    raise
            finally:
                try:
                    if loaded and synchronize is not None:
                        try:
                            synchronize()
                        except BaseException:
                            arena.poisoned = True
                            row["status"] = "synchronization_failed_credit_retained"
                            raise
                    row.setdefault("status", "consumed" if loaded else "read_failed")
                finally:
                    if consume_started is not None:
                        row["consumption_sync_elapsed_s"] = time.perf_counter() - consume_started
                    view.release()
                    row["finished_at"] = time.time()
                    _receipt(self.receipts_path, row)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            if isinstance(self.receipts_path, ReceiptJournal):
                self.receipts_path.flush()
