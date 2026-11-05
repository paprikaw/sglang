# SPDX-License-Identifier: Apache-2.0
"""SGLang client for independent, node-local CPU model snapshots.

Snapshots are populated by an idle engine's explicit weight release, after
HiCache teardown. New engines consume them before allocating HiCache; the
daemon, not the old engine, owns the bytes between experiments.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time

import torch

from .cpu_protocol import CpuWeightCacheClient, fingerprint_key
from .cpu_snapshot import describe_model, restore_snapshot, write_snapshot

logger = logging.getLogger(__name__)
SOCKET_ENV = "SGLANG_CPU_WEIGHT_CACHE_SOCKET"
SOURCE_ENV = "SGLANG_CPU_WEIGHT_CACHE_SOURCE_COMMIT"


def enabled():
    return bool(os.environ.get(SOCKET_ENV))


def _identity(model_config, load_config):
    from sglang.srt.runtime_context import get_parallel, get_server_args

    source = os.environ.get(SOURCE_ENV, "")
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError(f"{SOURCE_ENV} must pin an immutable source commit")
    args = get_server_args()
    if args.weight_cache_mode != "off" or args.speculative_algorithm is not None:
        raise ValueError(
            "CPU daemon snapshots require a single static model with weight_cache_mode=off"
        )
    if args.enable_elastic_pp or args.enable_eplb or args.elastic_ep_backend:
        raise ValueError(
            "CPU daemon snapshot identity does not support changing model topology"
        )
    if args.enable_lora or args.is_startup_weight_load_overlap:
        raise ValueError(
            "CPU daemon snapshots currently require serial loading without LoRA"
        )
    if getattr(args, "enable_weights_cpu_backup", False):
        raise ValueError(
            "CPU daemon snapshots require --enable-weights-cpu-backup disabled "
            "to avoid a second CPU copy outside the daemon byte budget"
        )
    if getattr(args, "moe_runner_backend", None) != "triton":
        raise ValueError(
            "CPU daemon snapshots currently require --moe-runner-backend triton; "
            "other prepared MoE layouts need explicit state adapters"
        )
    arch = model_config.hf_config.architectures or []
    # GLM's MoE AWQ conversion happens inside load_weights, so a tensor snapshot
    # must restore the final uint8 layout, not rerun conversion on a cache hit.
    # Keep the first integration scoped to the experiment's constructor family.
    if arch != ["Glm4MoeForCausalLM"] or model_config.quantization not in (
        None,
        "moe_wna16",
        "awq",
    ):
        raise ValueError(
            "CPU daemon model integration currently supports Glm4MoeForCausalLM (AWQ/moe_wna16 or unquantized)"
        )
    root = Path(model_config.model_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("CPU daemon cache requires a local immutable model snapshot")
    files = []
    # Metadata, not hundreds of GB of model content, is read on every start.
    # Size/mtime/ctime/inode bind the prepared tensors to the local snapshot.
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".pt", ".json"):
            st = path.stat()
            row = [path.name, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
            if path.suffix == ".json":
                row.append(hashlib.sha256(path.read_bytes()).hexdigest())
            files.append(row)
    ps = get_parallel()
    replica_dp_rank = getattr(
        load_config, "_cpu_weight_cache_dp_rank", getattr(ps, "dp_rank", None)
    )
    return {
        "snapshot_version": 1,
        "source_commit": source,
        "model_path": str(root),
        "model_files": files,
        "hf_config": model_config.hf_config.to_dict(),
        "revision": model_config.revision,
        "dtype": str(model_config.dtype),
        "quantization": model_config.quantization,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": (
            list(torch.cuda.get_device_capability())
            if torch.cuda.is_available()
            else None
        ),
        "load_format": str(load_config.load_format),
        "loader_extra_config": load_config.model_loader_extra_config,
        "parallel": {
            name: replica_dp_rank if name == "dp_rank" else getattr(ps, name, None)
            for name in (
                "tp_size",
                "tp_rank",
                "pp_size",
                "pp_rank",
                "dp_size",
                "dp_rank",
                "moe_ep_size",
                "moe_ep_rank",
                "moe_dp_size",
                "moe_dp_rank",
                "moe_tp_size",
                "moe_tp_rank",
                "attn_tp_size",
                "attn_tp_rank",
                "attn_cp_size",
                "attn_cp_rank",
                "attn_dp_size",
                "attn_dp_rank",
                "attn_dcp_size",
                "attn_dcp_rank",
            )
        },
        "model_settings": {
            name: getattr(args, name, None)
            for name in (
                "context_length",
                "json_model_override_args",
                "model_impl",
                "moe_runner_backend",
                "moe_a2a_backend",
                "moe_dense_tp_size",
                "disable_shared_experts_fusion",
                "enforce_shared_experts_fusion",
                "enable_fp32_lm_head",
                "enable_tf32_matmul",
                "enable_dp_lm_head",
                "weight_version",
                "enable_elastic_expert_backup",
            )
        },
    }


def _client():
    return CpuWeightCacheClient(os.environ[SOCKET_ENV], timeout=1800)


def _check_daemon(client, identity):
    status = client.status()
    if status["source_commit"] != identity["source_commit"]:
        raise RuntimeError("CPU daemon source commit differs from the model loader")
    return status


def _restore_glm_aliases(model):
    """Rebind GLM's constructor-owned routing alias after replacing Parameters.

    TopKConfig is a dataclass, not a Module/buffer. Its correction bias initially
    refers to the skeleton gate Parameter, which remains on meta even after the
    gate's registered Parameter is restored. Keep its runtime alias explicit.
    """
    for module in model.modules():
        if type(module).__name__ == "Glm4MoeSparseMoeBlock":
            module.topk.topk_config.correction_bias = (
                module.gate.e_score_correction_bias
            )


class CpuDaemonModelLoader:
    """Wrap the ordinary loader for the cold miss; bypass disk on a hit."""

    def __init__(self, load_config, fallback):
        self.load_config = load_config
        self.fallback = fallback

    def download_model(self, model_config):
        return self.fallback.download_model(model_config)

    def load_model(self, *, model_config, device_config):
        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
            _initialize_model,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        identity = _identity(model_config, self.load_config)
        key = fingerprint_key(identity)
        started = time.perf_counter()
        with _client() as client:
            _check_daemon(client, identity)
            result = client.get(key, identity, wait_s=120)
            if result["state"] == "ready":
                quant_config = _get_quantization_config(model_config, self.load_config)
                with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
                    model = _initialize_model(
                        model_config, self.load_config, quant_config
                    )
                restore_snapshot(
                    model, result["fd"], result["manifest"], device_config.device
                )
                _restore_glm_aliases(model)
                # The GPU model owns independent storage. Closing and consuming
                # this FD makes the node's CPU budget available to HiCache.
                released = client.release(result["reader_token"], evict=True)
                if released["state"] != "evicted":
                    raise RuntimeError(
                        "CPU snapshot still has readers; cannot allocate HiCache yet"
                    )
                source = "cpu_daemon"
            elif result["state"] == "missing":
                if os.environ.get("SGLANG_CPU_WEIGHT_CACHE_REQUIRE_HIT") == "1":
                    raise RuntimeError("Required CPU daemon snapshot is absent")
                model = self.fallback.load_model(
                    model_config=model_config, device_config=device_config
                )
                source = "disk"
            else:
                raise RuntimeError(f"CPU snapshot is not consumable: {result['state']}")
        model._cpu_weight_cache_identity = identity
        model._cpu_weight_cache_key = key
        model._cpu_weight_cache_load_source = source
        logger.info(
            "CPU weight cache load: source=%s key=%s elapsed=%.3fs",
            source,
            key,
            time.perf_counter() - started,
        )
        return model.eval()


def publish_model(model):
    """Transfer the idle prepared model to the node daemon before GPU release."""
    if os.environ.get("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET") or hasattr(
        model, "_canonical_cpu_weight_cache_key"
    ):
        raise RuntimeError(
            "canonical CPU banks do not accept prepared snapshot publication"
        )
    identity = model._cpu_weight_cache_identity
    key = model._cpu_weight_cache_key
    started = time.perf_counter()
    manifest, storages = describe_model(model)
    with _client() as client:
        _check_daemon(client, identity)
        result = client.acquire(key, manifest["nbytes"], identity, wait_s=120)
        if result["state"] != "build":
            # A second model must not silently overwrite a snapshot with weights
            # whose data may have changed. Guard owns topology admission.
            if result["state"] == "ready":
                client.release(result["reader_token"])
            raise RuntimeError(
                f"CPU daemon cannot publish this model: {result['state']}"
            )
        write_snapshot(result["fd"], manifest, storages)
        result = client.publish(
            key, result["token"], identity, manifest, used_bytes=manifest["nbytes"]
        )
    logger.info(
        "CPU weight cache published: key=%s bytes=%d elapsed=%.3fs",
        key,
        manifest["nbytes"],
        time.perf_counter() - started,
    )
    return result


def resume_model(model):
    """Fill memory-saver's resumed allocations without reconstructing a model."""
    if os.environ.get("SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET") or hasattr(
        model, "_canonical_cpu_weight_cache_key"
    ):
        raise RuntimeError(
            "canonical CPU banks require a new normal-loader model process"
        )
    identity = model._cpu_weight_cache_identity
    key = model._cpu_weight_cache_key
    with _client() as client:
        _check_daemon(client, identity)
        result = client.get(key, identity, wait_s=120)
        if result["state"] != "ready":
            raise RuntimeError(f"CPU daemon snapshot cannot resume: {result['state']}")
        device = next(model.parameters()).device
        restore_snapshot(model, result["fd"], result["manifest"], device, reuse=True)
        _restore_glm_aliases(model)
        released = client.release(result["reader_token"], evict=True)
        if released["state"] != "evicted":
            raise RuntimeError("CPU snapshot still has readers after resume")
