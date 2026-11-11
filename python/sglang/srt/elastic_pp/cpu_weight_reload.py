"""Restore inactive native layers from the already admitted canonical RAM bank.

Checkpoint tensors remain CPU mmap views of authenticated daemon memfds. The
model's native loaders apply packing/sharding directly to existing target
storage; no source GPU export, weight collective or disk fallback is allowed.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch

from sglang.srt.elastic_pp.graph_bank import binding_signature
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.weight_cache.cpu_canonical import model_view


class CanonicalCpuWeightReloader:
    def __init__(self, *, model_runner, stream):
        self.runner = model_runner
        self.model = model_runner.model
        self.stream = stream
        self.device = torch.device(model_runner.device)
        args = model_runner.server_args
        # Other post-load formats may repack or replace parameters. Qualify
        # those separately instead of silently loading canonical values into
        # a backend-specific runtime layout captured by CUDA Graphs.
        if (
            not type(self.model).__name__.startswith("Qwen3_5")
            or getattr(model_runner.model_config, "quantization", None) is not None
            or getattr(args, "quantization", None) is not None
            or getattr(args, "tp_size", 1) != 1
            or getattr(args, "ep_size", 1) != 1
            or getattr(args, "moe_runner_backend", None) != "triton"
        ):
            raise ValueError(
                "CPU cache reload currently requires native Qwen3.5 TP1/EP1 Triton weights"
            )
        self.key = getattr(self.model, "_canonical_cpu_weight_cache_key", None)
        self.socket = os.environ.get("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET")
        self.source_commit = os.environ.get(
            "SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT"
        )
        if not self.key or not self.socket or not self.source_commit:
            raise RuntimeError(
                "CPU cache reload requires the existing authenticated initial-load bank"
            )
        if self.device.type == "cuda" and stream is None:
            raise ValueError(
                "CPU cache reload requires an independent migration stream"
            )
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = dict(
            kind="canonical_cpu_cache",
            cache_key=self.key,
            cache_key_matches_initial_load=False,
            checkpoint_bytes=0,
            checkpoint_tensors=0,
            postprocess_modules=0,
            target_weight_bytes=0,
            layers_loaded=[],
            cache_open_seconds=0.0,
            checkpoint_load_seconds=0.0,
            gpu_export_bytes=0,
            gloo_weight_bytes=0,
            fallback_count=0,
            storage_verified=False,
            scope="rank-local native checkpoint loading and CUDA completion; includes CPU packing, not isolated DMA bandwidth",
        )

    def _drain(self):
        if self.stream is not None:
            self.stream.synchronize()

    def reload(self, *, generation, layer_ids):
        """Publish readiness only after full checkpoint coverage and stream drain."""
        layer_ids = frozenset(layer_ids)
        if not layer_ids:
            return
        profile = self.model.get_elastic_pp_weight_profile()
        expected = {
            item.name: item
            for item in profile.checkpoint_tensors
            if item.layer_id in layer_ids
        }
        if {item.layer_id for item in expected.values()} != layer_ids:
            raise RuntimeError("initial checkpoint profile is missing target layers")

        parameters = {
            name: param
            for name, param in self.model.named_parameters(remove_duplicate=False)
            if get_layer_id(name) in layer_ids
        }
        buffers = {
            name: buf
            for name, buf in self.model.named_buffers(remove_duplicate=False)
            if get_layer_id(name) in layer_ids
        }
        if {get_layer_id(name) for name in parameters} != layer_ids:
            raise RuntimeError("target weight storage was not prepared")
        bindings = dict(parameters, **buffers)
        signature = binding_signature(bindings)
        identities = {name: id(param) for name, param in parameters.items()}
        seen = set()
        started = time.monotonic()
        # The pin must outlive all checkpoint mappings and outstanding H2D.
        # A failed lookup or fingerprint mismatch is an error, never disk I/O.
        with model_view(
            self.runner.model_config.model_path, self.socket, self.source_commit
        ) as (view, bank):
            if bank["group_key"] != self.key:
                raise RuntimeError(
                    "canonical CPU cache differs from the initially loaded model"
                )
            self.metrics["cache_key_matches_initial_load"] = True
            self.metrics["cache_open_seconds"] += time.monotonic() - started
            index = Path(view) / "model.safetensors.index.json"
            if index.is_file():
                weight_map = json.loads(index.read_text())["weight_map"]
                if not expected.keys() <= weight_map.keys():
                    raise RuntimeError(
                        "canonical checkpoint index is missing target weights"
                    )
                allowed = {item["name"] for item in bank["shards"]}
                if any(weight_map[name] not in allowed for name in expected):
                    raise RuntimeError(
                        "canonical checkpoint index references an unadmitted shard"
                    )
            elif len(bank["shards"]) == 1:
                weight_map = {name: bank["shards"][0]["name"] for name in expected}
            else:
                raise RuntimeError(
                    "canonical multi-shard reload requires its checkpoint index"
                )

            from safetensors import safe_open

            with ExitStack() as mappings:
                shards = {
                    name: mappings.enter_context(
                        safe_open(str(Path(view) / name), framework="pt", device="cpu")
                    )
                    for name in sorted({weight_map[name] for name in expected})
                }

                def weights():
                    for name, item in sorted(expected.items()):
                        tensor = shards[weight_map[name]].get_tensor(name)
                        if (
                            tensor.device.type != "cpu"
                            or tuple(tensor.shape) != item.shape
                            or str(tensor.dtype) != item.dtype
                            or tensor.numel() * tensor.element_size() != item.nbytes
                        ):
                            raise RuntimeError(
                                "canonical checkpoint tensor metadata changed"
                            )
                        yield name, tensor
                        seen.add(name)
                        self.metrics["checkpoint_bytes"] += item.nbytes
                        self.metrics["checkpoint_tensors"] += 1

                context = (
                    nullcontext()
                    if self.stream is None
                    else torch.cuda.stream(self.stream)
                )
                started = time.monotonic()
                with self.model.elastic_pp_weight_load(generation, layer_ids):
                    try:
                        with torch.no_grad(), context:
                            loaded = self.model.load_weights(weights())
                            if (
                                seen != expected.keys()
                                or not parameters.keys() <= loaded
                            ):
                                raise RuntimeError(
                                    "canonical CPU reload did not cover all target weights"
                                )
                            # Match ordinary loading, including optional
                            # in-place BF16 SwiGLU interleaving. Reprocess only
                            # incoming layers; active weights must not change.
                            for name, module in self.model.named_modules():
                                method = getattr(module, "quant_method", None)
                                if (
                                    get_layer_id(name + ".") in layer_ids
                                    and method is not None
                                ):
                                    method.process_weights_after_loading(module)
                                    self.metrics["postprocess_modules"] += 1
                    finally:
                        # Drain before closing mappings or releasing the bank,
                        # including partial-copy failures. Never mark ready on
                        # failure, partial coverage, or a changed Graph address.
                        self._drain()
                    self.metrics["checkpoint_load_seconds"] += (
                        time.monotonic() - started
                    )
                    current_parameters = {
                        name: param
                        for name, param in self.model.named_parameters(
                            remove_duplicate=False
                        )
                        if get_layer_id(name) in layer_ids
                    }
                    current_buffers = {
                        name: buf
                        for name, buf in self.model.named_buffers(
                            remove_duplicate=False
                        )
                        if get_layer_id(name) in layer_ids
                    }
                    if identities != {
                        name: id(param) for name, param in current_parameters.items()
                    } or signature != binding_signature(
                        dict(current_parameters, **current_buffers)
                    ):
                        raise RuntimeError(
                            "CPU cache reload changed prepared Graph weight storage"
                        )
                    self.metrics["storage_verified"] = True
                    self.metrics["target_weight_bytes"] = sum(
                        p.numel() * p.element_size() for p in parameters.values()
                    )
                    self.metrics["layers_loaded"] = sorted(layer_ids)
