"""Stage-local SllmStore indices over an admitted Qwen3.5 CPU weight bank."""

import json
import logging
import math
import os
import struct
import time
from pathlib import Path

logger = logging.getLogger(__name__)
DTYPES = {
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "F8_E4M3": ("float8_e4m3fn", 1),
}
LAYER = "model.language_model.layers."
GLOBAL = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
}


def read_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if length == 0 or length > min(100_000_000, os.fstat(f.fileno()).st_size - 8):
            raise ValueError(f"Invalid safetensors header in {path}")
        return 8 + length, json.loads(f.read(length))


def stage_index(header, base, start, end):
    """Coalesce contiguous tensor ranges. The official Store moves opaque bytes.

    FP8 is reinterpreted on the GPU because the Store's native dtype table does
    not contain float8. Every tensor view retains its original shape and bits.
    """
    selected = []
    for name, info in header.items():
        if name == "__metadata__" or name.startswith(("mtp.", "model.visual.")):
            continue
        if name.startswith(LAYER):
            if not start <= int(name[len(LAYER) :].split(".", 1)[0]) < end:
                continue
        elif name not in GLOBAL:
            raise ValueError(f"Unexpected language model tensor {name}")
        elif name == "model.language_model.embed_tokens.weight" and start != 0:
            # The elastic model replicates norm and output head; only PP0 embeds.
            continue
        dtype, width = DTYPES[info["dtype"]]
        lo, hi = info["data_offsets"]
        if hi - lo != math.prod(info["shape"]) * width or lo < 0 or hi <= lo:
            raise ValueError(f"Invalid weight extent for {name}")
        selected.append((base + lo, base + hi, name, dtype, info["shape"]))

    ranges, views = [], []
    for lo, hi, name, dtype, shape in sorted(selected):
        if ranges and lo < ranges[-1][1]:
            raise ValueError(f"Overlapping safetensors at {name}")
        if ranges and lo == ranges[-1][1]:
            ranges[-1][1] = hi
        else:
            ranges.append([lo, hi])
        views.append(
            (name, str(len(ranges) - 1), lo - ranges[-1][0], hi - lo, dtype, shape)
        )
    index = {
        str(i): [lo, hi - lo, [hi - lo], [1], "torch.uint8"]
        for i, (lo, hi) in enumerate(ranges)
    }
    return index, views


def prepare_indices(root, view, bank, partitions):
    root, view = Path(root), Path(view)
    root.mkdir(mode=0o700, exist_ok=False)
    parts = []
    host_bytes = 0
    for shard in bank["shards"]:
        name = shard["name"]
        source = view / name
        base, header = read_header(source)
        for rank, (start, end) in enumerate(partitions):
            index, _ = stage_index(header, base, start, end)
            if not index:
                continue
            directory = root / f"pp{rank}" / name
            directory.mkdir(parents=True)
            (directory / "tensor_index.json").write_text(json.dumps(index))
            backing = root / "store" / name / "tensor.data_0"
            if not backing.exists():
                backing.parent.mkdir(parents=True)
                backing.symlink_to(source)
                parts.append(name)
                host_bytes += source.stat().st_size
    description = dict(
        version=1,
        group_key=bank["group_key"],
        parts=parts,
        partitions=partitions,
        host_bytes=host_bytes,
    )
    (root / "manifest.json").write_text(json.dumps(description))
    return description


def iter_stage_weights(model_config, model, load_config, parallel):
    import torch
    from sllm_store.client import SllmStoreClient
    from sllm_store.torch import load_dict_non_blocking

    if (
        model_config.hf_config.model_type != "qwen3_5_moe"
        or not getattr(model, "language_model_only", False)
        or parallel.pp_size != 2
        or parallel.tp_size != 1
        or parallel.moe_ep_size != 1
    ):
        raise ValueError(
            "This SllmStore adapter requires language-only Qwen3.5 MoE PP2/TP1/EP1"
        )
    view = Path(model_config.model_path)
    if str(view) != getattr(load_config, "_canonical_weight_cache_view", None):
        raise ValueError("SllmStore needs a pinned canonical weight view")
    root = Path(os.environ["SGLANG_SLLMSTORE_STAGE_ROOT"])
    manifest = json.loads((root / "manifest.json").read_text())
    rank = parallel.pp_rank
    start, end = model.start_layer, model.end_layer
    if manifest["version"] != 1 or manifest["partitions"][rank] != [start, end]:
        raise ValueError("SllmStore checkpoint and live PP layer ranges disagree")
    client = SllmStoreClient()
    idx_root = root / f"pp{rank}"
    num_bytes = count = parts = 0
    loaded_layers = set()
    load_s = 0.0
    for source in sorted(view.glob("*.safetensors")):
        name = source.name
        base, header = read_header(source)
        index, views = stage_index(header, base, start, end)
        if not index:
            continue
        # The resident store and this worker must see the same sealed memfd.
        # An equally named disk file or a stale store from another bank is invalid.
        backing = root / "store" / name / "tensor.data_0"
        if (
            not source.samefile(backing)
            or json.loads((idx_root / name / "tensor_index.json").read_text()) != index
        ):
            raise RuntimeError(f"SllmStore backing or stage index differs for {name}")
        before = time.perf_counter()
        replica, spans = load_dict_non_blocking(
            name, {"": torch.cuda.current_device()}, storage_path=str(idx_root)
        )
        if client.confirm_model_loaded(name, replica) is False:
            raise RuntimeError(f"SllmStore GPU transfer failed for {name}")
        load_s += time.perf_counter() - before
        if set(spans) != set(index):
            raise RuntimeError(f"Incomplete SllmStore GPU spans for {name}")
        for key, span, offset, length, dtype, shape in views:
            if key.startswith(LAYER):
                loaded_layers.add(int(key[len(LAYER) :].split(".", 1)[0]))
            tensor = spans[span].narrow(0, offset, length)
            yield key, tensor.view(getattr(torch, dtype)).reshape(shape)
            num_bytes += length
            count += 1
        # Only the offset-zero Store tensor owns the underlying CUDA allocation.
        # Keep every span alive until the native weight loaders have consumed it.
        torch.cuda.synchronize()
        del tensor, spans
        parts += 1
    if loaded_layers != set(range(start, end)):
        raise RuntimeError("Incomplete SllmStore PP layer coverage")
    logger.info(
        "SllmStore PP transfer %s",
        json.dumps(
            dict(
                pp=rank,
                layers=[start, end],
                tensors=count,
                bytes=num_bytes,
                parts=parts,
                group_key=manifest["group_key"],
                confirmed_gpu_load_s=load_s,
            )
        ),
    )
