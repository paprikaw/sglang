# SPDX-License-Identifier: Apache-2.0
"""Bounded copies of prepared model tensors to/from a daemon-owned memfd.

The manifest is JSON, never pickle. Shared storages, strided views, tied
parameters, non-persistent buffers, and quantization metadata survive a round
trip. The caller constructs the module skeleton; no classes are imported from
the manifest and no post-load quantization is run on already prepared weights.
"""

from __future__ import annotations

import enum
import hashlib
import math
import mmap
import os
import warnings

import torch
from torch import nn

SNAPSHOT_VERSION = 1
COPY_CHUNK_BYTES = 64 * 1024**2
_NESTED_STATE = ("quant_method", "scheme", "kernel")
_INTERNAL = {
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
}


def _class_name(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if type(value) in (list, tuple):
        return any(_contains_tensor(item) for item in value)
    if type(value) is dict:
        return any(_contains_tensor(item) for item in value.values())
    return False


def _encode(value):
    if isinstance(value, enum.Enum):
        raise TypeError("enum state is reconstructed by the model constructor")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return {"type": type(value).__name__, "items": [_encode(v) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        return {"type": "dict", "items": {k: _encode(v) for k, v in value.items()}}
    raise TypeError("non-JSON runtime object")


def _decode(value):
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "dict":
        return {k: _decode(v) for k, v in value["items"].items()}
    items = [_decode(v) for v in value["items"]]
    if kind == "tuple":
        return tuple(items)
    if kind == "list":
        return items
    raise ValueError(f"Unknown snapshot metadata type: {kind}")


def _objects(model):
    for module_name, module in model.named_modules():
        yield [module_name], module
        # These are constructor-owned quantization collaborators. Only their
        # data/flags are copied, not Python callbacks, runners or config classes.
        seen = {id(module)}

        def walk(obj, path):
            for attr in _NESTED_STATE:
                child = getattr(obj, attr, None)
                if child is None or id(child) in seen or not hasattr(child, "__dict__"):
                    continue
                seen.add(id(child))
                child_path = path + [attr]
                yield child_path, child
                yield from walk(child, child_path)

        yield from walk(module, [module_name])


def describe_model(model):
    """Return JSON layout and raw storage views without allocating CPU weights."""
    manifest = {
        "version": SNAPSHOT_VERSION,
        "objects": [],
        "storages": [],
        "tensors": [],
    }
    raw_storages = []
    storage_ids = {}
    tensor_ids = {}
    total = 0

    def tensor_entry(path, attr, tensor, kind, persistent=True):
        nonlocal total
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError(
                f"Unsupported snapshot tensor at {path}.{attr}: {tensor.layout}/{tensor.device}"
            )
        if tensor.is_conj() or tensor.is_neg() or any(s < 0 for s in tensor.stride()):
            raise ValueError("Snapshot requires resolved conjugate/negative views")
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        if key not in storage_ids:
            storage_id = len(raw_storages)
            storage_ids[key] = storage_id
            total = (total + 63) // 64 * 64
            size = storage.nbytes()
            manifest["storages"].append(
                {
                    "offset": total,
                    "nbytes": size,
                    "device_type": tensor.device.type,
                }
            )
            raw_storages.append(
                torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (size,), (1,)
                )
            )
            total += size
        tensor_id = tensor_ids.setdefault(id(tensor), len(tensor_ids))
        metadata = {}
        for name, value in tensor.__dict__.items():
            try:
                metadata[name] = _encode(value)
            except TypeError:
                pass
        manifest["tensors"].append(
            {
                "path": path,
                "name": attr,
                "kind": kind,
                "persistent": persistent,
                "storage": storage_ids[key],
                "tensor_id": tensor_id,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "metadata": metadata,
            }
        )

    for path, obj in _objects(model):
        row = {"path": path, "class": _class_name(obj), "attrs": {}, "none_tensors": []}
        registered = set()
        if isinstance(obj, nn.Module):
            for kind, values in (
                ("parameter", obj._parameters),
                ("buffer", obj._buffers),
            ):
                for name, tensor in values.items():
                    registered.add(name)
                    if tensor is None:
                        row["none_tensors"].append({"name": name, "kind": kind})
                    else:
                        tensor_entry(
                            path,
                            name,
                            tensor,
                            kind,
                            name not in obj._non_persistent_buffers_set,
                        )
        for name, value in vars(obj).items():
            if (
                name in _INTERNAL
                or name in registered
                or name.startswith("_cpu_weight_cache")
            ):
                continue
            if isinstance(value, torch.Tensor):
                tensor_entry(path, name, value, "attribute")
            else:
                try:
                    row["attrs"][name] = _encode(value)
                except TypeError:
                    if _contains_tensor(value):
                        raise ValueError(
                            f"Tensor container state requires an explicit adapter: {path}.{name}"
                        )
        manifest["objects"].append(row)
    manifest["nbytes"] = total
    if total <= 0:
        raise ValueError("Cannot cache an empty model snapshot")
    return manifest, raw_storages


def write_snapshot(fd, manifest, raw_storages, chunk_bytes=COPY_CHUNK_BYTES):
    """Copy directly into mmap in bounded chunks, hashing the copied bytes."""
    if os.fstat(fd).st_size != manifest["nbytes"]:
        raise ValueError("Snapshot fd size does not match model layout")
    with mmap.mmap(fd, manifest["nbytes"], access=mmap.ACCESS_WRITE) as mapping:
        with torch.no_grad():
            for row, source in zip(manifest["storages"], raw_storages, strict=True):
                digest = hashlib.sha256()
                for start in range(0, row["nbytes"], chunk_bytes):
                    length = min(chunk_bytes, row["nbytes"] - start)
                    offset = row["offset"] + start
                    destination = torch.frombuffer(
                        mapping, dtype=torch.uint8, count=length, offset=offset
                    )
                    destination.copy_(
                        source[start : start + length], non_blocking=False
                    )
                    del destination
                    view = memoryview(mapping)[offset : offset + length]
                    digest.update(view)
                    view.release()
                row["sha256"] = digest.hexdigest()


def _resolve(model, path):
    obj = model.get_submodule(path[0]) if path[0] else model
    for attr in path[1:]:
        obj = getattr(obj, attr)
    return obj


def restore_snapshot(
    model, fd, manifest, device, chunk_bytes=COPY_CHUNK_BYTES, reuse=False
):
    """Restore prepared storage and views into an independently built skeleton.

    reuse=True fills already resumed memory-saver allocations in place; ordinary
    new-engine startup allocates the final GPU storages only once.
    """
    if manifest.get("version") != SNAPSHOT_VERSION or os.fstat(
        fd
    ).st_size != manifest.get("nbytes"):
        raise ValueError("Snapshot version/size mismatch")
    for row in manifest["objects"]:
        if _class_name(_resolve(model, row["path"])) != row["class"]:
            raise ValueError(f"Snapshot module class mismatch at {row['path']}")
    storages = []
    device = torch.device(device)
    with mmap.mmap(fd, manifest["nbytes"], access=mmap.ACCESS_READ) as mapping:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable"
            )
            for storage_id, row in enumerate(manifest["storages"]):
                size = row["nbytes"]
                storage_device = (
                    torch.device("cpu") if row["device_type"] == "cpu" else device
                )
                if row["offset"] < 0 or size < 0 or row["offset"] + size > len(mapping):
                    raise ValueError("Snapshot storage exceeds fd bounds")
                if reuse:
                    entry = next(
                        t for t in manifest["tensors"] if t["storage"] == storage_id
                    )
                    tensor = getattr(_resolve(model, entry["path"]), entry["name"])
                    storage = tensor.untyped_storage()
                    if storage.nbytes() != size or tensor.device != storage_device:
                        raise ValueError(
                            "Resumed model storage does not match cached allocation"
                        )
                    destination = torch.empty(
                        0, dtype=torch.uint8, device=storage_device
                    ).set_(storage, 0, (size,), (1,))
                else:
                    destination = torch.empty(
                        size, dtype=torch.uint8, device=storage_device
                    )
                digest = hashlib.sha256()
                for start in range(0, size, chunk_bytes):
                    length = min(chunk_bytes, size - start)
                    offset = row["offset"] + start
                    view = memoryview(mapping)[offset : offset + length]
                    digest.update(view)
                    view.release()
                    source = torch.frombuffer(
                        mapping, dtype=torch.uint8, count=length, offset=offset
                    )
                    destination[start : start + length].copy_(
                        source, non_blocking=False
                    )
                    del source
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError(
                        f"Snapshot checksum mismatch for storage {storage_id}"
                    )
                storages.append(destination)

    restored = {}
    for row in manifest["objects"]:
        obj = _resolve(model, row["path"])
        for name, value in row["attrs"].items():
            setattr(obj, name, _decode(value))
        for item in row["none_tensors"]:
            if item["kind"] == "parameter":
                obj.register_parameter(item["name"], None)
            else:
                obj.register_buffer(item["name"], None)

    for entry in manifest["tensors"]:
        obj = _resolve(model, entry["path"])
        dtype = getattr(torch, entry["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("Invalid snapshot dtype")
        storage = storages[entry["storage"]].untyped_storage()
        span = entry["storage_offset"]
        if (
            len(entry["shape"]) != len(entry["stride"])
            or span < 0
            or any(x < 0 for x in entry["shape"] + entry["stride"])
        ):
            raise ValueError("Invalid snapshot tensor view")
        if all(entry["shape"]):
            span += (
                sum(
                    (dim - 1) * stride
                    for dim, stride in zip(entry["shape"], entry["stride"])
                )
                + 1
            )
        if span * dtype.itemsize > storage.nbytes():
            raise ValueError("Snapshot tensor view exceeds storage")
        key = (entry["tensor_id"], entry["kind"] == "parameter")
        if key not in restored:
            tensor_device = storages[entry["storage"]].device
            tensor = torch.empty(0, dtype=dtype, device=tensor_device).set_(
                storage,
                entry["storage_offset"],
                tuple(entry["shape"]),
                tuple(entry["stride"]),
            )
            if entry["kind"] == "parameter":
                tensor = nn.Parameter(tensor, requires_grad=False)
            restored[key] = tensor
        tensor = restored[key]
        for name, value in entry["metadata"].items():
            setattr(tensor, name, _decode(value))
        name = entry["name"]
        if entry["kind"] == "parameter":
            obj._buffers.pop(name, None)
            obj.register_parameter(name, tensor)
        elif entry["kind"] == "buffer":
            obj._parameters.pop(name, None)
            obj.register_buffer(name, tensor, persistent=entry["persistent"])
        else:
            setattr(obj, name, tensor)

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == "meta":
            raise ValueError(f"Snapshot omitted tensor {name}")
    return model
