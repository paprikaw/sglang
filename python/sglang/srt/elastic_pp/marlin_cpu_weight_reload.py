"""Startup-prepared FP8 Marlin runtime weights, backed only by bounded CPU RAM.

Marlin's GPU post-load repacking cannot use the BF16 CPU-only native loader.
The private startup Graph walk uses the regular runtime-tensor transport once,
then retains the processed incoming layers in pinned CPU RAM. Serving restores
those immutable bytes into evictable stable GPU addresses without repacking,
GPU export, weight collectives, or a second GPU copy. This is deliberately a
separate preparation path, not CPU-only packing of checkpoint tensors.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from sglang.srt.elastic_pp.graph_bank import binding_signature
from sglang.srt.elastic_pp.topology import PPTopology


def variable_layers(partitions, rank):
    intervals = [PPTopology.from_layer_counts(p).layer_range(rank) for p in partitions]
    layers = [set(range(r.start, r.end)) for r in intervals]
    return set.union(*layers) - set.intersection(*layers)


def state_layout(state):
    return tuple(
        (name, tuple(t.shape), tuple(t.stride()), str(t.dtype))
        for name, t in sorted(state.items())
    )


def host_storage_bytes(tensor):
    if tensor.layout != torch.strided or any(s < 0 for s in tensor.stride()):
        raise ValueError("processed CPU weights require nonnegative strided storage")
    if not tensor.numel():
        return 0
    return (
        1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    ) * tensor.element_size()


class MarlinCpuWeightReloader:
    def __init__(self, *, model_runner, stream):
        self.runner, self.stream = model_runner, stream
        args, model = model_runner.server_args, model_runner.model
        bank = getattr(model_runner, "_elastic_pp_prepared_graphs", None)
        if not (
            type(model).__name__.startswith("Qwen3_5")
            and args.quantization == "fp8"
            and args.moe_runner_backend == "marlin"
            and args.tp_size == args.ep_size == 1
            and args.elastic_pp_weight_staging == "cpu_prepacked"
            and args.elastic_pp_prepared_weight_policy == "offload"
            and bank is not None
            and bank.building
        ):
            raise ValueError(
                "processed CPU weights require startup Qwen3.5 FP8 Marlin TP1/EP1 Graphs/offload"
            )
        self.key = getattr(model, "_canonical_cpu_weight_cache_key", None)
        if not self.key:
            raise RuntimeError(
                "processed CPU weights require the admitted initial-load bank"
            )
        self.wanted = variable_layers(bank.partitions, model_runner.ps.pp_rank)
        self.cap = int(args.elastic_pp_cpu_weight_cache_gib) * 1024**3
        if self.cap <= 0:
            raise ValueError("processed CPU weights require a positive host-memory cap")
        self.entries, self.layouts = {}, {}
        self.nbytes = self.startup_export_bytes = 0
        self.startup_seconds = 0.0
        self.reset_metrics()

    def _building(self):
        return self.runner._elastic_pp_prepared_graphs.building

    def capture(self, layer_ids):
        if not self._building():
            raise RuntimeError("runtime GPU weight export is disabled")
        ids = (set(layer_ids) & self.wanted) - self.entries.keys()
        if not ids:
            return
        states = self.runner.export_elastic_pp_layer_state(sorted(ids))
        needed = sum(
            host_storage_bytes(t) for state in states.values() for t in state.values()
        )
        if self.nbytes + needed > self.cap:
            raise RuntimeError("processed CPU weight memory cap exceeded")
        started = time.monotonic()
        pending = {}
        context = (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )
        try:
            with torch.no_grad(), context:
                for layer, state in states.items():
                    if not state or any(
                        t.layout != torch.strided for t in state.values()
                    ):
                        raise ValueError(
                            "processed weights require nonempty strided layer states"
                        )
                    pending[layer] = {}
                    for name, tensor in state.items():
                        cpu = torch.empty_strided(
                            tensor.shape,
                            tensor.stride(),
                            dtype=tensor.dtype,
                            device="cpu",
                            pin_memory=self.stream is not None,
                        )
                        pending[layer][name] = cpu
                        cpu.copy_(tensor, non_blocking=self.stream is not None)
        finally:
            if self.stream is not None:
                self.stream.synchronize()
        self.entries.update(pending)
        self.layouts.update(
            {layer: state_layout(state) for layer, state in states.items()}
        )
        self.nbytes += needed
        self.startup_export_bytes += sum(
            t.numel() * t.element_size()
            for state in states.values()
            for t in state.values()
        )
        self.startup_seconds += time.monotonic() - started

    def reset_metrics(self):
        self.metrics = dict(
            kind="startup_processed_fp8_marlin_cpu",
            cache_key=self.key,
            target_weight_bytes=0,
            gpu_export_bytes=0,
            gloo_weight_bytes=0,
            runtime_pack_seconds=0.0,
            fallback_count=0,
            layers_loaded=[],
            cpu_prepare_seconds=0.0,
            canonical_checkpoint_read_bytes=0,
            checkpoint_bytes=0,
            checkpoint_tensors=0,
            processed_tensor_bytes=0,
            processed_tensor_count=0,
            storage_verified=False,
            cache_key_matches_initial_load=False,
        )

    def metrics_snapshot(self):
        return dict(
            self.metrics,
            prepared_cpu_cache_bytes=self.nbytes,
            prepared_cpu_cache_cap_bytes=self.cap,
            prepared_cpu_cache_entries=len(self.entries),
            startup_gpu_export_bytes=self.startup_export_bytes,
            startup_cpu_prepare_seconds=self.startup_seconds,
        )

    def reload(self, *, generation, layer_ids):
        if self._building():
            raise RuntimeError(
                "startup must populate processed weights through its private transport"
            )
        ids = set(layer_ids)
        if not ids:
            return
        if self.key != getattr(
            self.runner.model, "_canonical_cpu_weight_cache_key", None
        ):
            raise RuntimeError(
                "processed CPU weights differ from the admitted initial bank"
            )
        if not ids <= self.entries.keys():
            raise RuntimeError("startup did not prepare every incoming FP8 layer")
        targets = self.runner.export_elastic_pp_layer_state(sorted(ids))
        if any(
            state_layout(state) != self.layouts[layer]
            for layer, state in targets.items()
        ):
            raise RuntimeError("processed CPU weight layout changed")
        before = {layer: binding_signature(state) for layer, state in targets.items()}
        started = time.monotonic()
        context = (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )
        begin = (
            torch.cuda.Event(enable_timing=True) if self.stream is not None else None
        )
        end = torch.cuda.Event(enable_timing=True) if self.stream is not None else None
        try:
            with torch.no_grad(), context:
                if begin is not None:
                    begin.record(self.stream)
                for layer, state in targets.items():
                    for name, target in state.items():
                        target.copy_(
                            self.entries[layer][name],
                            non_blocking=self.stream is not None,
                        )
                if end is not None:
                    end.record(self.stream)
        finally:
            self._wait_for_restore()
        after = self.runner.export_elastic_pp_layer_state(sorted(ids))
        if before != {
            layer: binding_signature(state) for layer, state in after.items()
        }:
            raise RuntimeError("processed CPU restore changed prepared Graph bindings")
        self.runner.mark_elastic_pp_layers_loaded(generation=generation, layer_ids=ids)
        self.metrics.update(
            target_weight_bytes=sum(
                t.numel() * t.element_size()
                for state in targets.values()
                for t in state.values()
            ),
            checkpoint_load_seconds=time.monotonic() - started,
            layers_loaded=sorted(ids),
            storage_verified=True,
            cache_key_matches_initial_load=True,
            prepared_cpu_cache_hit=True,
        )
        self.metrics.update(
            h2d_bytes=self.metrics["target_weight_bytes"],
            processed_tensor_bytes=self.metrics["target_weight_bytes"],
            processed_tensor_count=sum(len(s) for s in targets.values()),
            h2d_wall_seconds=self.metrics["checkpoint_load_seconds"],
            h2d_stream_seconds=(
                begin.elapsed_time(end) / 1000 if begin is not None else None
            ),
        )

    def _wait_for_restore(self):
        if self.stream is None:
            return
        # A blocking CUDA wait on this background thread can contend with the
        # scheduler's CUDA/NCCL calls on the same device. Keep host-side waits
        # outside CUDA; readiness still requires the entire copy stream to be
        # complete, including any work enqueued before a copy raises.
        started = time.monotonic()
        polls, longest_query = 0, 0.0
        while True:
            queried = time.monotonic()
            complete = self.stream.query()
            longest_query = max(longest_query, time.monotonic() - queried)
            polls += 1
            if complete:
                break
            time.sleep(0.001)
        self.metrics.update(
            h2d_wait_kind="stream_query_poll",
            h2d_wait_polls=polls,
            h2d_wait_seconds=time.monotonic() - started,
            h2d_longest_query_seconds=longest_query,
        )
