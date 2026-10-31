"""Hy3-only integration of the bounded reader with a frozen SGLang checkout.

Not an alternative admission path: install() requires the new authenticated
streaming policy.  No mmap, checkpoint prefetch, secondary model, quantization,
or CPU weight offload is permitted.  All H2D copies finish before arena reuse.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

LAYER = re.compile(r"^model\.layers\.(\d+)\.")
EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def target_parameter(name):
    """Map only the exact BF16 checkpoint grammar; reject unknown tensors."""
    match = EXPERT.fullmatch(name)
    if match:
        base, expert, kind = match.groups()
        return base + (".w2_weight" if kind == "down_proj" else ".w13_weight")
    name = name.replace(".router.gate.", ".gate.")
    for original in ("q_proj", "k_proj", "v_proj"):
        name = name.replace(".self_attn." + original + ".", ".self_attn.qkv_proj.")
    for original in ("gate_proj", "up_proj"):
        name = name.replace("." + original + ".", ".gate_up_proj.")
    return name


def select_weights(index, *, start_layer, end_layer, first_stage, last_stage,
                   local_expert, parameter_names):
    """Select by actual stage/expert storage, then require parameter coverage."""
    if not 0 <= start_layer < end_layer <= 80:
        raise ValueError("Invalid explicit Hy3 main-model layer interval")
    selected = []
    coverage = Counter()
    for name, shard in sorted(index.items()):
        match = LAYER.match(name)
        if match:
            layer = int(match.group(1))
            if not start_layer <= layer < end_layer:
                continue
        elif name == "model.embed_tokens.weight":
            if not first_stage:
                continue
        elif name in ("lm_head.weight", "model.norm.weight"):
            if not last_stage:
                continue
        else:
            raise ValueError("Unknown non-layer Hy3 checkpoint tensor: " + name)
        expert = EXPERT.fullmatch(name)
        if expert and not local_expert(int(LAYER.match(name).group(1)), int(expert.group(2))):
            continue
        target = target_parameter(name)
        if target not in parameter_names:
            raise ValueError("Checkpoint maps to missing model parameter: " + target)
        selected.append((name, shard))
        coverage[target] += 1
    missing = set(parameter_names) - set(coverage)
    if missing:
        raise ValueError("Parameters missing checkpoint coverage: " + repr(sorted(missing)))
    if not selected:
        raise ValueError("Empty Hy3 weight selection")
    # File order amortizes opens. The original key remains the load_weights key.
    return sorted(selected, key=lambda row: (row[1], row[0])), dict(coverage)


def _read_frozen_manifest(binding):
    contract = binding["hy3_contract"]
    desc = contract["model_manifest"]
    raw = Path(desc["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != desc["sha256"]:
        raise RuntimeError("Hy3 frozen source manifest changed")
    return json.loads(raw)


def install():
    """Called in every spawned SGLang interpreter before model loading."""
    import torch
    import hy3_streaming_policy as policy
    from bounded_reader import SharedCredits, StagingArena, SafetensorsReader, ReceiptJournal
    from sglang.srt.model_loader.loader import DefaultModelLoader
    from sglang.srt.runtime_context import get_parallel

    if getattr(DefaultModelLoader, "_hy3_bounded_patch", False):
        return
    original_load = DefaultModelLoader.load_model

    def weights(loader, model_config, model):
        rank = get_parallel().world_rank
        binding, node, use = policy.validate_bound_model(model_config.model_path, rank)
        contract = binding["hy3_contract"]
        spec_desc = contract["node_specs"][node]
        raw = Path(spec_desc["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec_desc["sha256"]:
            raise RuntimeError("Hy3 node staging specification changed")
        spec = json.loads(raw)
        effective = model_config.hf_config.num_hidden_layers
        if effective != spec["effective_num_hidden_layers"] or effective not in (2, 80):
            raise RuntimeError("Actual model layers differ from admitted qualification slice")
        if model.__class__.__name__ != "HYV3ForCausalLM" or getattr(model, "secondary_weights", ()):
            raise RuntimeError("Bounded loader only admits a single Hy3 model")
        if getattr(model_config, "cpu_offload_gb", 0):
            raise RuntimeError("CPU weight offload is not admitted")
        manifest = policy.local_source_manifest(binding, node)
        proofs = {proof["relative_path"]: proof for proof in manifest["files"]}
        index_proof = proofs["model.safetensors.index.json"]
        raw = Path(index_proof["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != index_proof["sha256"]:
            raise RuntimeError("Hy3 weight index changed")
        index = json.loads(raw)["weight_map"]
        backbone = model.model

        def local_expert(layer, expert):
            module = backbone.layers[layer].mlp.experts
            if getattr(module, "_has_fused_shared", False):
                raise RuntimeError("Fused shared expert storage is not admitted")
            return module._map_global_expert_id_to_local_expert_id(expert) >= 0

        params = dict(model.named_parameters())
        selected, coverage = select_weights(
            index, start_layer=backbone.start_layer, end_layer=backbone.end_layer,
            first_stage=model.pp_group.is_first_rank,
            last_stage=model.pp_group.is_last_rank,
            local_expert=local_expert, parameter_names=set(params))
        if any(param.device.type != "cuda" for param in params.values()):
            raise RuntimeError("All parameter storage must be CUDA before streaming")
        credits = SharedCredits(spec["staging_ledger"], spec["staging_bytes"])
        arena = StagingArena(credits, spec["rank_staging_limit_bytes"],
                             rank_limit_bytes=spec["rank_staging_limit_bytes"])
        summary = {"rank": rank, "node": node, "effective_num_hidden_layers": effective,
                   "start_layer": backbone.start_layer, "end_layer": backbone.end_layer,
                   "arena_capacity_bytes": arena.capacity,
                   "selected_tensors": len(selected), "loaded_tensors": 0,
                   "selected_checkpoint_bytes": 0, "parameter_count": len(params),
                   "coverage_sha256": hashlib.sha256(json.dumps(coverage, sort_keys=True).encode()).hexdigest(),
                   "copy_synchronization": "cuda_synchronize_after_each_tensor",
                   "began_at": time.time(), "arena_released": False}
        loader._hy3_stream_summary = summary
        receipts_path = Path(use["path"]).with_name(Path(use["path"]).name + f".rank{rank}.tensors.jsonl")
        # No stale receipt or accidental append across separate server loads.
        receipt_fd = os.open(receipts_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(receipt_fd)
        journal = ReceiptJournal(receipts_path)
        summary["receipts_path"] = str(receipts_path)
        reader = None
        current_shard = None
        dtype_map = {"BF16": torch.bfloat16, "F32": torch.float32}
        tensor = None
        try:
            for name, shard in selected:
                if shard != current_shard:
                    if reader is not None:
                        reader.close()
                    reader = SafetensorsReader(proofs[shard], arena, receipts_path=journal)
                    current_shard = shard
                with reader.tensor(name, synchronize=torch.cuda.synchronize) as window:
                    if window.dtype not in dtype_map:
                        raise RuntimeError("Unexpected Hy3 checkpoint dtype")
                    tensor = torch.frombuffer(window.view, dtype=dtype_map[window.dtype]).reshape(window.shape)
                    summary["selected_checkpoint_bytes"] += window.view.nbytes
                    yield name, tensor
                    tensor = None
                summary["loaded_tensors"] += 1
            if summary["loaded_tensors"] != summary["selected_tensors"]:
                raise RuntimeError("Hy3 iterator did not consume its complete selection")
        finally:
            tensor = None
            try:
                if reader is not None:
                    reader.close()
            finally:
                journal.close()
            # Consumer may retain its last CPU view until load_weights returns.
            # Closing here is unsafe; the load_model wrapper closes afterward.
            loader._hy3_stream_arena = arena
            summary["iterator_finished_at"] = time.time()

    def load(loader, *, model_config, device_config):
        result = original_load(loader, model_config=model_config, device_config=device_config)
        summary = getattr(loader, "_hy3_stream_summary", None)
        arena = getattr(loader, "_hy3_stream_arena", None)
        if summary is None or arena is None:
            raise RuntimeError("Hy3 model load bypassed the admitted bounded iterator")
        torch.cuda.synchronize()
        arena.close()
        summary["arena_released"] = True
        receipt_raw = Path(summary["receipts_path"]).read_bytes()
        receipt_rows = [json.loads(line) for line in receipt_raw.splitlines()]
        if (len(receipt_rows) != summary["selected_tensors"]
                or any(row["status"] != "consumed" for row in receipt_rows)
                or sum(row["bytes"] for row in receipt_rows) != summary["selected_checkpoint_bytes"]):
            raise RuntimeError("Bounded tensor-copy receipts differ from actual selected weights")
        summary["receipts_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
        summary["completed_at"] = time.time()
        policy.attest_loaded_model(model_config.model_path, get_parallel().world_rank,
                                   weights_on_cuda=all(p.device.type == "cuda" for p in result.parameters()),
                                   stream_summary=summary)
        return result

    DefaultModelLoader._get_all_weights = weights
    DefaultModelLoader.load_model = load
    DefaultModelLoader._hy3_bounded_patch = True
