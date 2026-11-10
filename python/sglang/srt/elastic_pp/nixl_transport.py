"""Bounded GPU staging transport independent of the inference NCCL sequence.

One agent and two send/receive slots per peer are registered at startup. Only
the progress thread calls NIXL after bootstrap. A transfer's DONE is insufficient:
the receiver's handler must finish its CUDA writes before returning an ACK.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future

import torch

logger = logging.getLogger(__name__)
BUFFER_BYTES = 8 << 20
MAX_HEADER_BYTES = 64 << 10
_QUARANTINED = []  # Keep registrations alive if native cancellation cannot drain.


class NixlCompletion:
    def __init__(self, future, timeout):
        self.future, self.timeout = future, timeout

    def query(self):
        if not self.future.done():
            return False
        self.future.result()
        return True

    def synchronize(self):
        return self.future.result(timeout=self.timeout)

    def drain(self):
        """Failed application ACKs do not imply a source gather is still live."""
        try:
            self.synchronize()
        except Exception:
            if not self.future.done() or not getattr(
                self.future, "_nixl_source_drained", False
            ):
                raise
            # Only application/transport errors after a successful source-stream
            # drain are safe to suppress during allocator rollback.


class NixlMigrationEndpoint:
    def __init__(
        self,
        *,
        rank,
        world_size,
        device,
        buffer_bytes=BUFFER_BYTES,
        timeout=30.0,
        agent=None,
    ):
        self.rank, self.world_size = rank, world_size
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.buffer_bytes, self.timeout = int(buffer_bytes), float(timeout)
        if not 0 <= rank < world_size or buffer_bytes < 8 or timeout <= 0:
            raise ValueError("invalid NIXL endpoint geometry/budget")
        self.agent = agent
        self._registrations = []
        self._thread = None
        self._stopping = threading.Event()
        self._jobs = queue.Queue(maxsize=4)
        self._ready_notifications = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._peers, self._names, self._ready = {}, {}, {}
        self._last_message = {}
        self._waiting = self._reply = None
        self._sequence = 0
        self._stranded = []
        self._closed = False
        self.handler = None
        self.metrics = dict(
            bytes_sent=0,
            bytes_received=0,
            applied_messages=0,
            warm_channels=0,
            registered_gpu_bytes=0,
            error=None,
        )
        self.stream = None
        try:
            if self.agent is None:
                from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

                if self.device.type != "cuda":
                    raise ValueError("real NIXL elastic PP requires a CUDA device")
                config = nixl_agent_config(
                    backends=["UCX"],
                    enable_prog_thread=True,
                    sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
                )
                self.agent = nixl_agent(f"elastic-pp-{rank}-{uuid.uuid4().hex}", config)
            self.send_buffer = torch.zeros(
                (2, self.buffer_bytes), dtype=torch.uint8, device=self.device
            )
            self.recv_buffer = torch.zeros(
                (world_size, 2, self.buffer_bytes),
                dtype=torch.uint8,
                device=self.device,
            )
            if self.device.type == "cuda":
                self.stream = torch.cuda.Stream(device=self.device)
                torch.cuda.current_stream(self.device).synchronize()
            kind = "VRAM" if self.device.type == "cuda" else "DRAM"
            for tensor in (self.send_buffer, self.recv_buffer):
                registration = self.agent.register_memory(
                    [(tensor.data_ptr(), tensor.nbytes, self.device_id, "")], kind
                )
                if not registration:
                    raise RuntimeError("NIXL fixed GPU buffer registration failed")
                self._registrations.append(registration)
            self.metrics["registered_gpu_bytes"] = (
                self.send_buffer.nbytes + self.recv_buffer.nbytes
                if self.device.type == "cuda"
                else 0
            )
        except BaseException:
            try:
                while self._registrations:
                    self.agent.deregister_memory(self._registrations[-1])
                    self._registrations.pop()
            except BaseException:
                _QUARANTINED.append(self)
                logger.exception("NIXL startup cleanup retained registered buffers")
            raise

    @property
    def device_id(self):
        return self.device.index or 0

    def metadata(self):
        return dict(
            rank=self.rank,
            metadata=self.agent.get_agent_metadata(),
            recv_ptr=self.recv_buffer.data_ptr(),
            device_id=self.device_id,
            buffer_bytes=self.buffer_bytes,
            world_size=self.world_size,
        )

    def connect(self, peers):
        if self._thread is not None or len(peers) != self.world_size:
            raise RuntimeError("NIXL endpoints must connect once, before serving")
        if {p["rank"] for p in peers} != set(range(self.world_size)):
            raise ValueError("incomplete/duplicate candidate GPU endpoint set")
        for peer in peers:
            if (
                peer["buffer_bytes"] != self.buffer_bytes
                or peer["world_size"] != self.world_size
            ):
                raise ValueError("NIXL candidate buffer geometry mismatch")
            if peer["rank"] == self.rank:
                continue
            name = self.agent.add_remote_agent(peer["metadata"])
            self._peers[peer["rank"]] = dict(peer, name=name)
            self._names[name] = peer["rank"]
        self._thread = threading.Thread(
            target=self._run, name=f"pp-nixl-{self.rank}", daemon=True
        )
        self._thread.start()

    def submit(self, function):
        with self._lock:
            if self._stopping.is_set() or self._thread is None:
                raise RuntimeError("NIXL progress endpoint is not running")
            future = Future()
            future._nixl_source_drained = True  # A queued job has not read KV yet.
            self._jobs.put_nowait((function, future))
        return NixlCompletion(future, self.timeout)

    def warmup(self):
        def run():
            for peer in self._peers:
                self.rpc(peer, {"op": "warm"}, pack=lambda b: self._warm_bytes(b))
                self.metrics["warm_channels"] += 1

        self.submit(run).synchronize()

    @staticmethod
    def _warm_bytes(buffer):
        buffer[:8].zero_()
        return 8

    def announce_ready(self, generation, source_ranks):
        # This control path must progress even while a copy job waits for a
        # neighbour's preparation. Putting it behind that job creates a cycle.
        self._ready_notifications.put((generation, tuple(source_ranks)))

    def peers_ready(self, generation, peers):
        with self._lock:
            return all(self._ready.get(peer) == generation for peer in peers)

    def _notify(self, peer, message):
        encoded = json.dumps(message, separators=(",", ":")).encode()
        if len(encoded) > MAX_HEADER_BYTES:
            raise ValueError("NIXL migration control header exceeds its bound")
        self.agent.send_notif(self._peers[peer]["name"], encoded)

    def _poll(self):
        while not self._ready_notifications.empty():
            generation, peers = self._ready_notifications.get_nowait()
            for peer in peers:
                self._notify(peer, dict(op="ready", generation=generation))
        for name, messages in self.agent.get_new_notifs().items():
            peer = self._names.get(name)
            if peer is None:
                raise ValueError("notification from an unregistered NIXL peer")
            for raw in messages:
                if len(raw) > MAX_HEADER_BYTES:
                    raise ValueError("oversized NIXL notification")
                message = json.loads(raw)
                op = message.get("op")
                if op == "ack":
                    if self._waiting == (peer, message.get("id")):
                        self._reply = message
                    continue
                if op == "ready":
                    with self._lock:
                        self._ready[peer] = max(
                            self._ready.get(peer, 0), message["generation"]
                        )
                    continue
                number = message.get("id")
                reply = dict(op="ack", id=number, ok=False)
                try:
                    if type(number) is not int or number <= self._last_message.get(
                        peer, 0
                    ):
                        raise ValueError("stale/duplicate NIXL buffer notification")
                    turn, nbytes = message["turn"], message["nbytes"]
                    if (
                        turn != number % 2
                        or type(nbytes) is not int
                        or not 0 <= nbytes <= self.buffer_bytes
                    ):
                        raise ValueError("invalid NIXL buffer turn/length")
                    if op != "warm":
                        if self.handler is None:
                            raise RuntimeError(
                                "NIXL generation receiver is not configured"
                            )
                        self.handler(
                            peer, message, self.recv_buffer[peer, turn, :nbytes]
                        )
                    self.metrics["bytes_received"] += nbytes
                    self.metrics["applied_messages"] += 1
                    self._last_message[peer] = number
                    reply["ok"] = True
                except Exception as exc:
                    reply["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                    self.metrics["error"] = reply["error"]
                self._notify(peer, reply)

    def rpc(self, peer, header, *, pack=None):
        """Progress-thread-only request; service inbound work while awaiting ACK."""
        if threading.current_thread() is not self._thread:
            raise RuntimeError("NIXL requests belong to the progress thread")
        self._sequence += 1
        number, turn = self._sequence, self._sequence % 2
        message = dict(header, id=number, turn=turn, nbytes=0)
        self._waiting, self._reply = (peer, number), None
        handle, done = None, False
        deadline = time.monotonic() + self.timeout
        try:
            if pack is None:
                self._notify(peer, message)
                done = True
            else:
                nbytes = int(pack(self.send_buffer[turn]))
                if not 0 < nbytes <= self.buffer_bytes:
                    raise ValueError("NIXL packed data exceeds fixed GPU buffer")
                if self.stream is not None:
                    self.stream.synchronize()  # Only this private gather stream.
                message["nbytes"] = nbytes
                encoded = json.dumps(message, separators=(",", ":")).encode()
                if len(encoded) > MAX_HEADER_BYTES:
                    raise ValueError("NIXL migration header exceeds its bound")
                target = self._peers[peer]
                ptr = target["recv_ptr"] + (self.rank * 2 + turn) * self.buffer_bytes
                kind = "VRAM" if self.device.type == "cuda" else "DRAM"
                src = self.agent.get_xfer_descs(
                    [(self.send_buffer[turn].data_ptr(), nbytes, self.device_id)], kind
                )
                dst = self.agent.get_xfer_descs(
                    [(ptr, nbytes, target["device_id"])], kind
                )
                handle = self.agent.initialize_xfer(
                    "WRITE", src, dst, target["name"], encoded
                )
                state = self.agent.transfer(handle)
                if hasattr(self.agent, "query_xfer_backend"):
                    self.metrics["backend"] = self.agent.query_xfer_backend(handle)
                if state == "ERR":
                    raise RuntimeError("NIXL transfer returned ERR")
                done = state == "DONE"
                self.metrics["bytes_sent"] += nbytes
            while not done or self._reply is None:
                self._poll()
                if handle is not None and not done:
                    state = self.agent.check_xfer_state(handle)
                    if state == "ERR":
                        raise RuntimeError("NIXL transfer returned ERR")
                    done = state == "DONE"
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "NIXL transfer/application ACK deadline exceeded"
                    )
                if not done or self._reply is None:
                    self._stopping.wait(0.001)
            if not self._reply.get("ok"):
                raise RuntimeError(
                    "NIXL receiver rejected patch: "
                    + self._reply.get("error", "unknown")
                )
        finally:
            self._waiting = self._reply = None
            if handle is not None:
                try:
                    self.agent.release_xfer_handle(handle)
                except Exception:
                    self._stranded.append(handle)
                    # Do not reuse any registered buffer after failed cancellation.
                    self._stopping.set()
                    raise

    def _run(self):
        from contextlib import nullcontext

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while not self._stopping.is_set():
            try:
                self._poll()
                function, future = self._jobs.get(timeout=0.001)
            except queue.Empty:
                continue
            except Exception as exc:
                self.metrics["error"] = f"{type(exc).__name__}: {exc}"
                self._stopping.set()
                break
            try:
                future._nixl_source_drained = False
                context = (
                    torch.cuda.stream(self.stream)
                    if self.stream is not None
                    else nullcontext()
                )
                with context:
                    result = function()
                if self.stream is not None:
                    self.stream.synchronize()
                future._nixl_source_drained = True
                future.set_result(result)
            except BaseException as error:
                outcome_error = error
                try:
                    if self.stream is not None:
                        self.stream.synchronize()
                    future._nixl_source_drained = True
                except BaseException as drain_error:
                    outcome_error = drain_error
                    self._stopping.set()
                self.metrics["error"] = (
                    f"{type(outcome_error).__name__}: {outcome_error}"
                )
                future.set_exception(outcome_error)
        with self._lock:
            self._stopping.set()
            while not self._jobs.empty():
                _, future = self._jobs.get_nowait()
                future.set_exception(RuntimeError("NIXL progress endpoint stopped"))

    def close(self):
        if self._closed:
            return
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1)
        if (self._thread is not None and self._thread.is_alive()) or self._stranded:
            _QUARANTINED.append(self)
            raise RuntimeError(
                "NIXL registration remains live; requires owned worker shutdown"
            )
        try:
            if self.stream is not None:
                self.stream.synchronize()
            for name in self._names:
                self.agent.remove_remote_agent(name)
            while self._registrations:
                self.agent.deregister_memory(self._registrations[-1])
                self._registrations.pop()
        except BaseException:
            _QUARANTINED.append(self)
            raise
        self._registrations.clear()
        self._closed = True
