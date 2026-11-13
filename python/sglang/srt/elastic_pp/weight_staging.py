"""Bounded pinned CPU transport for immutable, inactive destination weights.

No candidate GPU copy is cached. One host buffer is reused only after the
network and CUDA readers finish. Byte views preserve dense transposed layouts;
receives write directly into already-admitted destination parameter storage.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from sglang.srt.elastic_pp.layer_transport import flatten_layer_states

_HEADER = "__elastic_pp_pinned_weight_packet__"
_DATA = "weight_bytes"
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024


class WeightPacketPlan:
    def __init__(self, *, generation, transfer, layer_states):
        payload = flatten_layer_states(
            generation=generation,
            transfer=transfer,
            layer_states=layer_states,
            copy_noncontiguous=False,
        )
        self.tensors = []
        descriptors = []
        self.total_bytes = 0
        metadata = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                size = value.numel() * value.element_size()
                view = value.reshape(-1).view(torch.uint8)
                self.tensors.append((self.total_bytes, size, view))
                descriptors.append((key, value.dtype, tuple(value.shape), size))
                self.total_bytes += size
            else:
                metadata[key] = value
        self.header = {"version": 1, "layer": metadata, "tensors": descriptors}

    def segments(self, offset, size):
        if offset < 0 or size <= 0 or offset + size > self.total_bytes:
            raise ValueError("weight packet range is outside layer storage")
        for start, count, tensor in self.tensors:
            lo, hi = max(start, offset), min(start + count, offset + size)
            if lo < hi:
                yield tensor.narrow(0, lo - start, hi - lo), lo - offset, hi - lo


class PinnedWeightStager:
    def __init__(self, *, device, stream, chunk_bytes=DEFAULT_CHUNK_BYTES):
        if chunk_bytes <= 0:
            raise ValueError("weight staging chunk size must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and stream is None:
            raise ValueError("CUDA weight staging requires its own migration stream")
        self.stream = stream
        self.chunk_bytes = chunk_bytes
        started = time.monotonic()
        self.buffer = torch.empty(
            chunk_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        self.allocation_seconds = time.monotonic() - started
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = dict(
            kind="pinned_chunked_cpu",
            host_buffer_bytes=self.chunk_bytes,
            host_buffer_pinned=self.device.type == "cuda",
            buffer_allocation_seconds=self.allocation_seconds,
            d2h_seconds=0.0,
            send_seconds=0.0,
            recv_seconds=0.0,
            h2d_seconds=0.0,
            bytes_sent=0,
            bytes_received=0,
            packets_sent=0,
            packets_received=0,
            scope="rank-local wall time including waits; never sum parallel ranks",
        )

    def _context(self):
        return nullcontext() if self.stream is None else torch.cuda.stream(self.stream)

    def _drain(self):
        if self.stream is not None:
            self.stream.synchronize()

    @staticmethod
    def _header(plan, offset, size):
        return dict(plan.header, offset=offset, size=size, total=plan.total_bytes)

    def send(self, *, plan, group, destination, wait):
        for offset in range(0, plan.total_bytes, self.chunk_bytes):
            size = min(self.chunk_bytes, plan.total_bytes - offset)
            host = self.buffer[:size]
            started = time.monotonic()
            try:
                with self._context():
                    for tensor, start, count in plan.segments(offset, size):
                        host.narrow(0, start, count).copy_(tensor, non_blocking=True)
            finally:
                # D2H must finish before Gloo reads the pinned bytes. Also drain
                # on partial-copy failure before a future transition can reuse.
                self._drain()
            self.metrics["d2h_seconds"] += time.monotonic() - started
            started = time.monotonic()
            works = group.send_tensor_dict(
                {_HEADER: self._header(plan, offset, size), _DATA: host},
                dst=destination,
                async_send=True,
            )
            wait(works or ())
            self.metrics["send_seconds"] += time.monotonic() - started
            self.metrics["bytes_sent"] += size
            self.metrics["packets_sent"] += 1

    def receive(self, *, plan, group, source):
        for offset in range(0, plan.total_bytes, self.chunk_bytes):
            size = min(self.chunk_bytes, plan.total_bytes - offset)
            host = self.buffer[:size]
            expected = self._header(plan, offset, size)

            def validate(metadata):
                headers = [v for k, v in metadata if k == _HEADER]
                if len(headers) != 1 or headers[0] != expected:
                    raise ValueError("pinned weight packet layout/generation mismatch")

            started = time.monotonic()
            group.recv_tensor_dict_into(
                {_DATA: host}, src=source, metadata_validator=validate
            )
            self.metrics["recv_seconds"] += time.monotonic() - started
            started = time.monotonic()
            try:
                with self._context():
                    for tensor, start, count in plan.segments(offset, size):
                        tensor.copy_(host.narrow(0, start, count), non_blocking=True)
            finally:
                # Do not overwrite the pinned source until H2D is complete.
                # Marking the layer loaded happens only after all packets drain.
                self._drain()
            self.metrics["h2d_seconds"] += time.monotonic() - started
            self.metrics["bytes_received"] += size
            self.metrics["packets_received"] += 1
