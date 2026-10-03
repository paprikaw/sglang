"""Small authenticated Unix-socket protocol for the node-local weight cache.

Only JSON metadata travels over the socket. Weight bytes use memfd descriptors.
This file intentionally imports neither torch nor SGLang and can be imported
directly by operational tools on a CPU-only host.
"""

from __future__ import annotations

import array
import ctypes
import fcntl
import hashlib
import json
import os
import socket
import struct
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
# Some Spartan Python installations were built against headers predating memfd.
# These stable Linux UAPI values let the same stdlib-only code use the kernel API.
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
REQUIRED_SEALS = 0x0008 | 0x0004 | 0x0002 | 0x0001


def create_memfd(name: str) -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, 0x0001 | 0x0002)
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "memfd_create", None)
    if function is not None:
        function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        fd = function(name.encode(), 0x0001 | 0x0002)
    else:
        number = {"x86_64": 319, "aarch64": 279}.get(os.uname().machine)
        if number is None:
            raise RuntimeError("memfd_create is unavailable on this architecture")
        library.syscall.restype = ctypes.c_long
        fd = library.syscall(
            ctypes.c_long(number), ctypes.c_char_p(name.encode()), ctypes.c_uint(3)
        )
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def seal_memfd(fd: int) -> None:
    if fcntl.fcntl(fd, F_GET_SEALS) & REQUIRED_SEALS != REQUIRED_SEALS:
        fcntl.fcntl(fd, F_ADD_SEALS, REQUIRED_SEALS)


class CpuWeightCacheError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def fingerprint_key(fingerprint: Any) -> str:
    """Canonical identity; includes model, source, topology and tensor schema."""
    encoded = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_socket_path() -> str:
    return f"/tmp/sglang-cpu-weights-{os.getuid()}/cache.sock"


def send_message(sock: socket.socket, message: dict, fd: int | None = None) -> None:
    encoded = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError("cache protocol metadata is too large")
    header = struct.pack("!I", len(encoded))
    ancillary = []
    if fd is not None:
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    sent = sock.sendmsg([header], ancillary)
    if sent < len(header):
        sock.sendall(header[sent:])
    sock.sendall(encoded)


def receive_message(sock: socket.socket) -> tuple[dict, int | None]:
    header, ancillary, flags, _ = sock.recvmsg(
        4,
        socket.CMSG_SPACE(array.array("i").itemsize),
        getattr(socket, "MSG_CMSG_CLOEXEC", 0x40000000),
    )
    if not header:
        raise EOFError("cache peer disconnected")
    fds: list[int] = []
    try:
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array.array("i")
                received.frombytes(data[: len(data) - len(data) % received.itemsize])
                fds.extend(received)
        if flags & socket.MSG_CTRUNC or len(fds) > 1:
            raise ValueError("unexpected cache protocol descriptors")
        while len(header) < 4:
            more = sock.recv(4 - len(header))
            if not more:
                raise EOFError("incomplete cache protocol header")
            header += more
        length = struct.unpack("!I", header)[0]
        if not 0 < length <= MAX_MESSAGE_BYTES:
            raise ValueError("invalid cache protocol length")
        data = bytearray()
        while len(data) < length:
            part = sock.recv(min(65536, length - len(data)))
            if not part:
                raise EOFError("incomplete cache protocol payload")
            data.extend(part)
        message = json.loads(data)
        if not isinstance(message, dict):
            raise ValueError("cache protocol requires an object")
        return message, fds[0] if fds else None
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


class CpuWeightCacheClient:
    """A connection owns its build/read leases; close after mappings are closed.

    ``acquire`` allocates a writable memfd once per fingerprint. ``get`` never
    creates a cache. A publisher must close every writable mmap before sealing
    the file and publishing. A fetched FD can only be mmap'ed read-only.
    """

    def __init__(self, socket_path: str | None = None, timeout: float = 60):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        try:
            self.socket.connect(socket_path or default_socket_path())
            _, uid, _ = struct.unpack(
                "3i", self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid():
                raise CpuWeightCacheError("owner", "cache daemon UID differs")
        except BaseException:
            self.socket.close()
            raise
        self._fds: dict[str, int] = {}

    def _request(self, operation: str, **fields: Any) -> dict:
        send_message(
            self.socket,
            {"op": operation, "protocol_version": PROTOCOL_VERSION, **fields},
        )
        result, fd = receive_message(self.socket)
        if not result.get("ok"):
            if fd is not None:
                os.close(fd)
            raise CpuWeightCacheError(
                result.get("code", "error"), result.get("error", "")
            )
        if fd is not None:
            token = result.get("token") or result.get("reader_token")
            if not token:
                os.close(fd)
                raise CpuWeightCacheError("protocol", "descriptor without a lease")
            self._fds[token] = fd
            result["fd"] = fd
        return result

    def status(self) -> dict:
        return self._request("status")

    def acquire(
        self, key: str, expected_bytes: int, fingerprint: Any, wait_s: float = 0
    ) -> dict:
        return self._request(
            "acquire",
            key=key,
            expected_bytes=expected_bytes,
            fingerprint=fingerprint,
            wait_s=wait_s,
        )

    def get(self, key: str, fingerprint: Any, wait_s: float = 0) -> dict:
        return self._request("get", key=key, fingerprint=fingerprint, wait_s=wait_s)

    fetch = get

    def publish(
        self,
        key: str,
        token: str,
        fingerprint: Any,
        manifest: Any,
        used_bytes: int | None = None,
    ) -> dict:
        fd = self._fds.get(token)
        if fd is None:
            raise CpuWeightCacheError("lease", "no local build descriptor")
        if used_bytes is not None and used_bytes != os.fstat(fd).st_size:
            os.ftruncate(fd, used_bytes)
        # F_SEAL_WRITE rejects any surviving writable mmap. No mutable snapshot
        # becomes visible, even if its producer crashes midway through copying.
        seal_memfd(fd)
        result = self._request(
            "publish",
            key=key,
            token=token,
            fingerprint=fingerprint,
            manifest=manifest,
            used_bytes=os.fstat(fd).st_size,
        )
        os.close(self._fds.pop(token))
        return result

    def release(self, token: str, evict: bool = False) -> dict:
        fd = self._fds.pop(token, None)
        if fd is not None:
            os.close(fd)
        return self._request("release", token=token, evict=evict)

    def evict(self, key: str) -> dict:
        return self._request("evict", key=key)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
