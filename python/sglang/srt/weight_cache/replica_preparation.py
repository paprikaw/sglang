"""One-shot, node-local preparation for a fixed Qwen3.5/A100 PP2 replica.

The donor owns sealed expert memfds until the official Store has populated its
pinned pool. A separate empty HiCache bank belongs to the new replica. Neither
GPU pointers nor writable prefix slots are shared between serving instances.
"""

import fcntl
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from sglang.srt.weight_cache.cpu_protocol import (
    F_GET_SEALS,
    REQUIRED_SEALS,
    create_memfd,
    seal_memfd,
)


_active_prepared_weights = ContextVar("fixed_pp2_processed_experts", default=None)


def publish(path, value):
    """Publish complete immutable metadata. A second producer is an error."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x") as out:
            json.dump(value, out, separators=(",", ":"))
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def preparation_key(root, group_key):
    return dict(
        version=1,
        run=root.parent.name,
        bank=group_key,
        engine=os.environ["SGLANG_REPLICA_ENGINE_COMMIT"],
        adapter=os.environ["SGLANG_REPLICA_MARLIN_SHA256"],
        sm=[8, 0],
        pp_layers=[[0, 24], [24, 48]],
        tp=1,
        ep=1,
    )


def read_packet(root, filename, key):
    packet = json.loads((root / filename).read_text())
    if packet["key"] != key:
        raise ValueError(f"Foreign or stale replica preparation: {filename}")
    return packet


EXPERT_DTYPES = {
    "w13_weight": "int32",
    "w2_weight": "int32",
    "w13_weight_scale": "bfloat16",
    "w2_weight_scale": "bfloat16",
}


def export_layer(layer):
    import torch

    fd = create_memfd("sglang-marlin-layer")
    try:
        index = {}
        offset = 0
        digest = hashlib.sha256()
        # Stream one tensor at a time. All offsets have natural dtype alignment.
        with os.fdopen(os.dup(fd), "wb", buffering=0) as out:
            for name, expected_dtype in EXPERT_DTYPES.items():
                tensor = getattr(layer, name).detach()
                if (
                    tensor.dtype != getattr(torch, expected_dtype)
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"Unexpected packed Marlin layout for {name}")
                raw = memoryview(tensor.view(torch.uint8).cpu().numpy()).cast("B")
                size = len(raw)
                for start in range(0, size, 64 << 20):
                    part = raw[start : start + (64 << 20)]
                    if out.write(part) != len(part):
                        raise OSError("Incomplete processed weight export")
                digest.update(raw)
                index[name] = [offset, size, list(tensor.shape), expected_dtype]
                offset += size
                del raw
        seal_memfd(fd)
        st = os.fstat(fd)
        return fd, dict(
            pid=os.getpid(),
            fd=fd,
            device=st.st_dev,
            inode=st.st_ino,
            size=offset,
            sha256=digest.hexdigest(),
            tensors=index,
        )
    except BaseException:
        os.close(fd)
        raise


def export_stage(runner, modules, root):
    import torch

    rank = runner.ps.pp_rank
    model = runner.model
    key = preparation_key(root, model._canonical_cpu_weight_cache_key)
    start, end = key["pp_layers"][rank]
    if (
        torch.cuda.get_device_capability() != (8, 0)
        or [model.start_layer, model.end_layer] != [start, end]
        or sorted(int(m.layer_id) for _, m in modules) != list(range(start, end))
    ):
        raise ValueError("Prepared source must be a full A100 PP2 stage")
    fds, layers = [], {}
    started = time.monotonic()
    try:
        for _, layer in modules:
            previously_exported = sum(e["size"] for e in layers.values())
            layer_size = sum(
                getattr(layer, n).numel() * getattr(layer, n).element_size()
                for n in EXPERT_DTYPES
            )
            if previously_exported + layer_size > (64 << 30):
                raise ValueError("Marlin donor stage exceeds its 64-GiB staging budget")
            fd, entry = export_layer(layer)
            fds.append(fd)
            layers[int(layer.layer_id)] = entry
        publish(
            root / f"weights-{rank}.json",
            dict(
                key=key,
                layers=layers,
                seconds=time.monotonic() - started,
                bytes=sum(e["size"] for e in layers.values()),
            ),
        )
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise

    def release_after_store_load():
        while not (root / "weights-consumed.json").exists():
            time.sleep(0.2)
        for fd in fds:
            os.close(fd)
        publish(root / f"weights-released-{rank}.json", dict(key=key))

    threading.Thread(target=release_after_store_load, daemon=True).start()


def validate_weight_file(entry):
    path = Path(f"/proc/{int(entry['pid'])}/fd/{int(entry['fd'])}")
    with path.open("rb", buffering=0) as file:
        st = os.fstat(file.fileno())
        if (
            st.st_uid != os.getuid()
            or (st.st_dev, st.st_ino, st.st_size)
            != (entry["device"], entry["inode"], entry["size"])
            or fcntl.fcntl(file, F_GET_SEALS) & REQUIRED_SEALS != REQUIRED_SEALS
        ):
            raise ValueError("Stale or mutable Marlin backing")
    return path


def register_stage_files(root, preparation, key):
    """Build official Store byte indices, recording the exact immutable tensors."""
    parts, entries = [], {}
    for rank, (start, end) in enumerate(key["pp_layers"]):
        packet = read_packet(preparation, f"weights-{rank}.json", key)
        if set(packet["layers"]) != {str(i) for i in range(start, end)}:
            raise ValueError("Incomplete prepared layer coverage")
        for layer, entry in packet["layers"].items():
            name = f"marlin-pp{rank}-layer{layer}"
            backing = validate_weight_file(entry)
            store_dir = root / "store" / name
            store_dir.mkdir()
            (store_dir / "tensor.data_0").symlink_to(backing)
            size = entry["size"]
            idx = root / "packed" / f"pp{rank}" / name
            idx.mkdir(parents=True)
            publish(
                idx / "tensor_index.json",
                {"blob": [0, size, [size], [1], "torch.uint8"]},
            )
            entries[layer] = dict(
                name=name, size=size, sha256=entry["sha256"], tensors=entry["tensors"]
            )
            parts.append(name)
    publish(root / "packed.json", dict(key=key, layers=entries))
    return parts


def compact_index(index):
    offset = 0
    compact = {}
    for span, (_, length, *meta) in index.items():
        compact[span] = [offset, length, *meta]
        offset += length
    return compact


def prepare_dense_files(root, view, bank, partitions, owned_fds, cap):
    """Keep only ordinary checkpoint spans alongside the prepared experts."""
    from sglang.srt.weight_cache.sllmstore_stage import read_header, stage_index

    view = Path(view)
    parts, total = [], 0
    for shard in bank["shards"]:
        source = view / shard["name"]
        base, header = read_header(source)
        for rank, (start, end) in enumerate(partitions):
            spans, _ = stage_index(header, base, start, end, omit_packed_experts=True)
            if not spans:
                continue
            total += sum(r[1] for r in spans.values())
            if total > cap:
                raise ValueError(
                    "Dense checkpoint staging exceeds its admitted CPU budget"
                )
            name = f"dense-pp{rank}-{source.name}"
            fd = create_memfd(name)
            owned_fds.append(fd)
            with source.open("rb") as src, os.fdopen(os.dup(fd), "wb") as dest:
                for offset, length, *_ in spans.values():
                    src.seek(offset)
                    while length:
                        data = src.read(min(length, 8 << 20))
                        if not data:
                            raise EOFError("Short canonical weight span")
                        dest.write(data)
                        length -= len(data)
            seal_memfd(fd)
            directory = root / "store" / name
            directory.mkdir()
            (directory / "tensor.data_0").symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
            idx = root / "packed" / f"pp{rank}" / name
            idx.mkdir(parents=True)
            publish(idx / "tensor_index.json", compact_index(spans))
            parts.append(name)
    return parts


def packed_expert_views(entry, root, rank, expected_shapes):
    """Return typed views of one completed official Store GPU transfer."""
    import torch
    from sllm_store.client import SllmStoreClient
    from sllm_store.torch import load_dict_non_blocking

    name = entry["name"]
    if set(entry["tensors"]) != set(EXPERT_DTYPES):
        raise ValueError("Incomplete Marlin tensor set")
    replica, spans = load_dict_non_blocking(
        name,
        {"": torch.cuda.current_device()},
        storage_path=str(root / "packed" / f"pp{rank}"),
    )
    if SllmStoreClient().confirm_model_loaded(name, replica) is False:
        raise RuntimeError(f"Official Store did not complete {name}")
    blob = spans["blob"]
    if blob.dtype != torch.uint8 or blob.numel() != entry["size"]:
        raise ValueError("Invalid Store payload size")
    views = {}
    offset = 0
    for field, expected_dtype in EXPERT_DTYPES.items():
        begin, size, shape, dtype = entry["tensors"][field]
        actual_dtype = getattr(torch, expected_dtype)
        if (
            dtype != expected_dtype
            or tuple(shape) != expected_shapes[field]
            or offset != begin
            or math.prod(shape) * actual_dtype.itemsize != size
        ):
            raise ValueError(f"Invalid packed tensor extent for {field}")
        weight = blob.narrow(0, begin, size).view(actual_dtype).view(shape)
        views[field] = weight
        offset += size
    if offset != entry["size"]:
        raise ValueError("Trailing data in packed expert image")
    return views


def install_expert(layer, geometry, root):
    import torch
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.runtime_context import get_parallel

    # Set by the stage iterator only after checking the canonical bank and index.
    packet = _active_prepared_weights.get()
    if packet is None:
        raise RuntimeError("Prepared Marlin weights require the admitted Store loader")
    rank = get_parallel().pp_rank
    idx = int(layer.layer_id)
    lo, hi = packet["key"]["pp_layers"][rank]
    if not lo <= idx < hi or torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("Prepared Marlin layer on wrong stage or GPU")
    loaded = packet.setdefault("loaded_experts", {})
    if idx in loaded:
        raise ValueError("Duplicate prepared Marlin layer")
    before = time.perf_counter()
    entry = packet["layers"][str(idx)]
    device = layer.w13_weight.device
    views = packed_expert_views(entry, root, rank, geometry.packed_shapes)
    for field, weight in views.items():
        setattr(layer, field, torch.nn.Parameter(weight.clone(), requires_grad=False))
    del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    layer.workspace = marlin_make_workspace(device, 4)
    torch.cuda.synchronize()
    loaded[idx] = (entry["size"], time.perf_counter() - before)
