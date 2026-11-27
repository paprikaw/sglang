"""Pin a Prefill registration while any queued or executing chunk can read it."""
from collections import Counter
from threading import Lock, local


class PrefillTransferFence:
    def __init__(self):
        self._lock = Lock()
        self._chunks = set()
        self._closed = False
        self.worker = local()
        self._io = Counter()

    def begin_io(self, room):
        with self._lock:
            self._io[room] += 1

    def end_io(self, room):
        with self._lock:
            self._io[room] -= 1
            if not self._io[room]:
                del self._io[room]

    def sending(self, room):
        with self._lock:
            return self._io[room] > 0

    def admit(self, chunk):
        with self._lock:
            if self._closed or id(chunk) in self._chunks:
                raise RuntimeError("Cannot queue this chunk on the current registration")
            self._chunks.add(id(chunk))

    def finish(self, chunk):
        with self._lock:
            self._chunks.remove(id(chunk))

    def pending(self):
        with self._lock:
            return len(self._chunks)

    def close(self):
        with self._lock:
            if self._chunks or self._closed:
                raise RuntimeError("Old Prefill registration is still in use")
            self._closed = True

    def reopen(self):
        with self._lock:
            if not self._closed:
                raise RuntimeError("Prefill registration was not suspended")
            self._closed = False
