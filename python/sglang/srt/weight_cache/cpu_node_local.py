"""Hy3 layout-specific immutable CPU weight bank; stdlib only.

The bank contains one compact safetensors memfd holding only the union required
by the four ranks on this node. Construction reads selected source ranges once;
inference requires a prebuilt bank and never falls back to checkpoint payloads.
The existing authenticated daemon owns/seals/pins all bytes. Canonical complete
checkpoint banks and their identities are deliberately unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from .cpu_canonical import _check_daemon, _private_parent, _stat_identity
    from .cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key
except ImportError:  # Direct CPU-only operational entry point.
    from cpu_canonical import _check_daemon, _private_parent, _stat_identity
    from cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key

PROVIDER = "sglang_node_local_weight_v1"
PROFILE = "hy3-bf16-node-local-v1"
FORMAT = "sglang-node-local-safetensors-v1"
OUTPUT_NAME = "node-local-00001-of-00001.safetensors"
MAX_BANK_BYTES = 288 * 1024**3
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")
_DTYPE_BYTES = {"BF16": 2, "F32": 4}


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("lowercase SHA256 digest required")
    return value


def _descriptor(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("immutable descriptor must contain path and sha256")
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("immutable descriptor must be an absolute regular file")
    digest = _digest(value["sha256"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("immutable descriptor hash changed")
    return _json(raw)


def _portable_filesystem(root, portable):
    # st_dev is host-local on GPFS. Permit a different value only with the
    # independently pinned portable filesystem identity, never by dropping it.
    expected = portable.get("filesystem", {})
    result = subprocess.run(
        ["findmnt", "-J", "-T", str(root), "-o", "SOURCE,FSTYPE,UUID"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = _json(result.stdout)["filesystems"]
    if len(observed) != 1:
        raise ValueError("source filesystem identity is ambiguous")
    fs = observed[0]
    if (
        expected.get("fsid") != os.statvfs(root).f_fsid
        or expected.get("source") != fs.get("source")
        or expected.get("fstype") != fs.get("fstype")
        or expected.get("uuid") != fs.get("uuid")
    ):
        raise ValueError("portable source filesystem differs")


def _source_proof(root, source_descriptor, portable_descriptor):
    source = _descriptor(source_descriptor)
    if (
        source.get("schema") != 1
        or Path(source.get("model_path", "")).resolve() != root
        or source.get("repo") != "tencent/Hy3"
        or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))
    ):
        raise ValueError("verified Hy3 source identity required")
    portable = None
    if portable_descriptor is not None:
        portable = _descriptor(portable_descriptor)
        if (
            portable.get("source_manifest") != source_descriptor
            or Path(portable.get("model_path", "")).resolve() != root
        ):
            raise ValueError("portable identity is not bound to this source proof")
        _portable_filesystem(root, portable)
    files = {}
    for item in source.get("files", []):
        name = item.get("relative_path")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or name in files
        ):
            raise ValueError("invalid or duplicate verified source filename")
        if item.get("sha256_verified") is not True:
            raise ValueError("source must have a completed SHA256 verification receipt")
        _digest(item.get("sha256"))
        files[name] = item
    return source, files, portable


def _verify_source_file(root, name, proof, portable):
    path = root / name
    if path.is_symlink() or str(path) != proof.get("path"):
        raise ValueError("source path differs from verified original")
    identity = _stat_identity(path)
    for field in ("size", "inode", "mtime_ns", "ctime_ns"):
        if identity[field] != proof.get(field):
            raise ValueError(f"verified source {field} changed: {name}")
    if identity["device"] != proof.get("device") and portable is None:
        raise ValueError("source device differs without a portable identity")
    return identity


def _header(path, expected_identity):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if _fd_identity(fd) != expected_identity:
            raise ValueError("source changed before header read")
        prefix = os.pread(fd, 8, 0)
        if len(prefix) != 8:
            raise ValueError("short safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size <= 0 or size > 16 * 1024**2 or size + 8 > expected_identity["size"]:
            raise ValueError("invalid bounded safetensors header size")
        raw = os.pread(fd, size, 8)
        if len(raw) != size:
            raise ValueError("short safetensors header")
        header = _json(raw)
        ranges = []
        result = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape, dtype, offsets = (
                meta.get("shape"),
                meta.get("dtype"),
                meta.get("data_offsets"),
            )
            if (
                dtype not in _DTYPE_BYTES
                or not isinstance(shape, list)
                or not all(type(n) is int and n > 0 for n in shape)
            ):
                raise ValueError("unsupported Hy3 tensor dtype/shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(type(n) is int for n in offsets)
            ):
                raise ValueError("invalid safetensors offsets")
            begin, end = offsets
            if (
                begin < 0
                or end <= begin
                or end - begin != math.prod(shape) * _DTYPE_BYTES[dtype]
                or 8 + size + end > expected_identity["size"]
            ):
                raise ValueError("invalid safetensors tensor range")
            ranges.append((begin, end))
            result[name] = {
                "dtype": dtype,
                "shape": shape,
                "source_offset": 8 + size + begin,
                "bytes": end - begin,
            }
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor:
                raise ValueError("safetensors payload has overlap or gaps")
            cursor = end
        if (
            cursor + 8 + size != expected_identity["size"]
            or _fd_identity(fd) != expected_identity
        ):
            raise ValueError("safetensors payload size/source identity differs")
        return result, hashlib.sha256(prefix + raw).hexdigest()
    finally:
        os.close(fd)


def _fd_identity(fd):
    info = os.fstat(fd)
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def _selected(name, layout, node_rank, effective_layers):
    layer = _LAYER.match(name)
    if layer:
        index = int(layer.group(1))
        if index >= effective_layers:
            return False
        if layout == "pp2ep4" and not node_rank * (effective_layers // 2) <= index < (
            node_rank + 1
        ) * (effective_layers // 2):
            return False
        expert = _EXPERT.search(name)
        return (
            not (expert and layout == "ep8")
            or node_rank * 96 <= int(expert.group(1)) < (node_rank + 1) * 96
        )
    if name not in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}:
        raise ValueError(f"unsupported Hy3 checkpoint tensor: {name}")
    if layout == "ep8":
        return True
    return name == (
        "model.embed_tokens.weight" if node_rank == 0 else "lm_head.weight"
    ) or (node_rank == 1 and name == "model.norm.weight")


def _output_header(tensors, selection_sha256):
    header = {
        "__metadata__": {"provider": PROVIDER, "selection_sha256": selection_sha256}
    }
    for tensor in tensors:
        header[tensor["name"]] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": tensor["target_data_offsets"],
        }
    raw = _json_bytes(header)
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def make_selection_manifest(
    model_root,
    *,
    layout,
    node_rank,
    effective_layers,
    source_manifest,
    portable_source_identity=None,
):
    """Describe exact immutable source ranges without reading weight payloads.

    Write the returned JSON once and freeze its SHA256 before build/launch.
    Per-node manifests include host-local device IDs and must be made on the
    node that will build/consume the bank (portable GPFS proof permits this).
    """
    if (
        layout not in ("ep8", "pp2ep4")
        or type(node_rank) is not int
        or node_rank not in (0, 1)
        or effective_layers not in (2, 80)
    ):
        raise ValueError("only Hy3 EP8/PP2EP4, node0/1 and layers2/80 are supported")
    root = Path(model_root).resolve(strict=True)
    source, proofs, portable = _source_proof(
        root, source_manifest, portable_source_identity
    )
    metadata = {}
    for name in ("config.json", "model.safetensors.index.json"):
        if name not in proofs:
            raise ValueError("source proof lacks model metadata")
        _verify_source_file(root, name, proofs[name], portable)
        raw = (root / name).read_bytes()
        metadata[name] = hashlib.sha256(raw).hexdigest()
        if metadata[name] != proofs[name]["sha256"]:
            raise ValueError("model metadata content changed")
    config = _json((root / "config.json").read_bytes())
    if (
        any(
            config.get(key) != value
            for key, value in {
                "model_type": "hy_v3",
                "num_hidden_layers": 80,
                "num_experts": 192,
                "num_experts_per_tok": 8,
                "hidden_size": 4096,
                "num_key_value_heads": 8,
                "num_attention_heads": 64,
                "tie_word_embeddings": False,
            }.items()
        )
        or config.get("dtype", config.get("torch_dtype")) != "bfloat16"
    ):
        raise ValueError("source is outside the fixed Hy3 BF16 support profile")
    index = _json((root / "model.safetensors.index.json").read_bytes())
    weight_map = index.get("weight_map", {})
    if not weight_map or not isinstance(weight_map, dict):
        raise ValueError("nonempty complete safetensors index required")
    shard_names = sorted(set(weight_map.values()))
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise ValueError("index contains an invalid source shard name")
    tensors, source_shards, seen = [], [], set()
    all_payload = 0
    for name in shard_names:
        if name not in proofs:
            raise ValueError("source proof lacks a checkpoint shard")
        identity = _verify_source_file(root, name, proofs[name], portable)
        header, header_sha256 = _header(root / name, identity)
        for tensor_name, meta in header.items():
            if tensor_name in seen or weight_map.get(tensor_name) != name:
                raise ValueError("header/index tensor coverage differs")
            seen.add(tensor_name)
            all_payload += meta["bytes"]
            if _selected(tensor_name, layout, node_rank, effective_layers):
                tensors.append(dict(meta, name=tensor_name, source_shard=name))
        # Keep the verifier's original device in a portable manifest. The
        # current host device was checked above against the pinned filesystem;
        # an opened copy FD is later checked against a fresh local identity.
        bound_identity = dict(identity, device=proofs[name]["device"])
        source_shards.append(
            dict(
                bound_identity,
                name=name,
                sha256=proofs[name]["sha256"],
                header_sha256=header_sha256,
            )
        )
    if (
        seen != set(weight_map)
        or index.get("metadata", {}).get("total_size") != all_payload
    ):
        raise ValueError("complete checkpoint payload/index coverage differs")
    tensors.sort(key=lambda t: (t["source_shard"], t["source_offset"], t["name"]))
    cursor = 0
    for tensor in tensors:
        tensor["target_data_offsets"] = [cursor, cursor + tensor["bytes"]]
        cursor += tensor["bytes"]
    if not tensors:
        raise ValueError("selection must contain weights")
    basis = {
        "schema": 1,
        "provider": PROVIDER,
        "support_profile": PROFILE,
        "model_root": str(root),
        "repo": source["repo"],
        "revision": source["revision"],
        "source_manifest": source_manifest,
        "portable_source_identity": portable_source_identity,
        "device_binding": (
            {"kind": "portable_filesystem", "filesystem": portable["filesystem"]}
            if portable is not None
            else {"kind": "verified_local_device"}
        ),
        "metadata_sha256": metadata,
        "layout": layout,
        "node_rank": node_rank,
        "effective_layers": effective_layers,
        "ranks": list(range(node_rank * 4, node_rank * 4 + 4)),
        "source_shards": source_shards,
        "tensors": tensors,
        "selected_tensors_sha256": fingerprint_key(tensors),
        "selected_tensor_names_sha256": hashlib.sha256(
            ("\n".join(sorted(t["name"] for t in tensors)) + "\n").encode()
        ).hexdigest(),
        "selected_tensor_count": len(tensors),
        "selected_payload_bytes": cursor,
    }
    selection_sha256 = fingerprint_key(basis)
    header = _output_header(tensors, selection_sha256)
    total = len(header) + cursor
    if total > MAX_BANK_BYTES:
        raise ValueError(
            "selected compact bank exceeds the 288 GiB support-profile cap"
        )
    return dict(
        basis,
        selection_sha256=selection_sha256,
        output={
            "name": OUTPUT_NAME,
            "size": total,
            "header_bytes": len(header),
            "header_sha256": hashlib.sha256(header).hexdigest(),
        },
    )


def build_selection_manifest(
    model_root,
    *,
    layout,
    node_rank,
    effective_num_hidden_layers,
    source_manifest,
    portable_source_identity=None,
):
    """Named construction alias for experiment preparation (metadata only)."""
    return make_selection_manifest(
        model_root,
        layout=layout,
        node_rank=node_rank,
        effective_layers=effective_num_hidden_layers,
        source_manifest=source_manifest,
        portable_source_identity=portable_source_identity,
    )


def describe_model(model_root, selection_manifest_path):
    path = Path(selection_manifest_path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("selection manifest must be an absolute non-symlink path")
    raw = path.read_bytes()
    selection = _json(raw)
    expected = make_selection_manifest(
        model_root,
        layout=selection.get("layout"),
        node_rank=selection.get("node_rank"),
        effective_layers=selection.get("effective_layers"),
        source_manifest=selection.get("source_manifest"),
        portable_source_identity=selection.get("portable_source_identity"),
    )
    if selection != expected:
        raise ValueError(
            "selection manifest differs from exact source-derived tensor coverage"
        )
    fingerprint = {
        "format": FORMAT,
        "provider": PROVIDER,
        "support_profile": PROFILE,
        "model_root": selection["model_root"],
        "metadata_sha256": selection["metadata_sha256"],
        "source_manifest_sha256": selection["source_manifest"]["sha256"],
        "portable_source_identity_sha256": (
            selection["portable_source_identity"] or {}
        ).get("sha256"),
        "layout": selection["layout"],
        "node_rank": selection["node_rank"],
        "effective_layers": selection["effective_layers"],
        "ranks": selection["ranks"],
        "selection_sha256": selection["selection_sha256"],
    }
    key = fingerprint_key(fingerprint)
    shard = selection["output"]
    identity = {
        "format": "sglang-node-local-shard-v1",
        "group_key": key,
        "selection_sha256": selection["selection_sha256"],
        "shard": shard,
    }
    return {
        "model_root": selection["model_root"],
        "group_key": key,
        "fingerprint": fingerprint,
        "items": [{"key": fingerprint_key(identity), "fingerprint": identity}],
        "shards": [shard],
        "shard_count": 1,
        "total_bytes": shard["size"],
        "selection": selection,
        "selection_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_members(bank, fetched):
    if len(fetched.get("entries", [])) != 1:
        raise CpuWeightCacheError(
            "fingerprint", "node-local bank must have one compact shard"
        )
    member = fetched["entries"][0]
    manifest = member.get("manifest") or {}
    expected = {
        "kind": "node_local_safetensors",
        "group_key": bank["group_key"],
        "name": OUTPUT_NAME,
        "selection_sha256": bank["selection"]["selection_sha256"],
        "selected_tensors_sha256": bank["selection"]["selected_tensors_sha256"],
        "selected_tensor_count": bank["selection"]["selected_tensor_count"],
        "selected_payload_bytes": bank["selection"]["selected_payload_bytes"],
    }
    if (
        member.get("key") != bank["items"][0]["key"]
        or member.get("size") != bank["total_bytes"]
        or any(manifest.get(k) != v for k, v in expected.items())
    ):
        raise CpuWeightCacheError(
            "fingerprint", "node-local bank coverage metadata differs"
        )
    _digest(manifest.get("sha256"))


def bank_status(model_root, socket_path, source_commit, *, selection_manifest_path):
    bank = describe_model(model_root, selection_manifest_path)
    with CpuWeightCacheClient(socket_path) as client:
        status = _check_daemon(client, source_commit)
        fetched = client.get_group(bank["items"])
        ready = fetched["state"] == "ready"
        if ready:
            try:
                _validate_members(bank, fetched)
            finally:
                client.release(fetched["reader_token"])
    return {
        "passed": ready,
        "ready": ready,
        "group_key": bank["group_key"],
        "total_bytes": bank["total_bytes"],
        "shard_count": 1,
        "selection_sha256": bank["selection"]["selection_sha256"],
        "selected_tensor_count": bank["selection"]["selected_tensor_count"],
        "missing_keys": fetched.get("missing_keys", []),
        "daemon_pid": status["pid"],
        "daemon_used_bytes": status["used_bytes"],
    }


def _write_all(fd, data, offset):
    view = memoryview(data)
    done = 0
    while done < len(view):
        count = os.pwrite(fd, view[done:], offset + done)
        if count <= 0:
            raise OSError("short node-local memfd write")
        done += count


def build_bank(
    model_root,
    socket_path,
    source_commit,
    *,
    selection_manifest_path,
    chunk_bytes=8 * 1024**2,
    progress=None,
):
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 64 * 1024**2:
        raise ValueError("copy chunk must be within (0,64MiB]")
    started = time.monotonic()
    parent = _private_parent(socket_path)
    lock = os.open(
        parent / "node-local-build.lock",
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bank = describe_model(model_root, selection_manifest_path)
        selection = bank["selection"]
        with CpuWeightCacheClient(socket_path) as client:
            status = _check_daemon(client, source_commit)
            if "node_local_no_eviction_v1" not in status.get("features", []):
                raise CpuWeightCacheError(
                    "protocol", "daemon lacks atomic no-eviction admission"
                )
            if bank["total_bytes"] > status["max_bytes"]:
                raise CpuWeightCacheError(
                    "capacity", "node-local bank exceeds daemon cap"
                )
            item = bank["items"][0]
            acquired = client.acquire(
                item["key"],
                bank["total_bytes"],
                item["fingerprint"],
                allow_eviction=False,
            )
            if acquired["state"] not in ("build", "ready"):
                raise CpuWeightCacheError(
                    "busy", "another builder owns this node-local bank"
                )
            copied = 0
            if acquired["state"] == "build":
                header = _output_header(
                    selection["tensors"], selection["selection_sha256"]
                )
                _write_all(acquired["fd"], header, 0)
                digest = hashlib.sha256(header)
                source_identities = {}
                for shard in selection["source_shards"]:
                    observed = _stat_identity(Path(bank["model_root"]) / shard["name"])
                    fields = ("size", "mtime_ns", "ctime_ns", "inode")
                    if any(observed[k] != shard[k] for k in fields) or (
                        selection["device_binding"]["kind"] == "verified_local_device"
                        and observed["device"] != shard["device"]
                    ):
                        raise ValueError(
                            "source changed before opening selected shards"
                        )
                    source_identities[shard["name"]] = observed
                current_name, fd = None, None
                try:
                    for index, tensor in enumerate(selection["tensors"]):
                        if tensor["source_shard"] != current_name:
                            if fd is not None:
                                if _fd_identity(fd) != source_identities[current_name]:
                                    raise ValueError(
                                        "source changed during selected range copy"
                                    )
                                if hasattr(os, "posix_fadvise"):
                                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                                os.close(fd)
                                fd = None
                            current_name = tensor["source_shard"]
                            fd = os.open(
                                Path(bank["model_root"]) / current_name,
                                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                            )
                            if _fd_identity(fd) != source_identities[current_name]:
                                raise ValueError(
                                    "source changed before selected range copy"
                                )
                        offset = 0
                        while offset < tensor["bytes"]:
                            data = os.pread(
                                fd,
                                min(chunk_bytes, tensor["bytes"] - offset),
                                tensor["source_offset"] + offset,
                            )
                            if not data:
                                raise OSError("short selected source range")
                            _write_all(
                                acquired["fd"],
                                data,
                                len(header) + tensor["target_data_offsets"][0] + offset,
                            )
                            digest.update(data)
                            offset += len(data)
                            copied += len(data)
                        if progress and (
                            index % 100 == 0 or index + 1 == len(selection["tensors"])
                        ):
                            progress(
                                {
                                    "completed_tensors": index + 1,
                                    "tensor_count": len(selection["tensors"]),
                                    "copied_payload_bytes": copied,
                                    "group_key": bank["group_key"],
                                }
                            )
                    if _fd_identity(fd) != source_identities[current_name]:
                        raise ValueError(
                            "source changed during final selected range copy"
                        )
                finally:
                    if fd is not None:
                        if hasattr(os, "posix_fadvise"):
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                if copied != selection["selected_payload_bytes"]:
                    raise ValueError("selected range copy coverage is incomplete")
                # Revalidate metadata/stat bindings after all selected reads.
                if (
                    describe_model(model_root, selection_manifest_path)["group_key"]
                    != bank["group_key"]
                ):
                    raise ValueError("source binding changed during build")
                client.publish(
                    item["key"],
                    acquired["token"],
                    item["fingerprint"],
                    {
                        "kind": "node_local_safetensors",
                        "group_key": bank["group_key"],
                        "name": OUTPUT_NAME,
                        "selection_sha256": selection["selection_sha256"],
                        "selected_tensors_sha256": selection["selected_tensors_sha256"],
                        "selected_tensor_count": selection["selected_tensor_count"],
                        "selected_payload_bytes": selection["selected_payload_bytes"],
                        "sha256": digest.hexdigest(),
                    },
                    pin_after_publish=True,
                )
            fetched = client.get_group(bank["items"])
            if fetched["state"] != "ready":
                raise CpuWeightCacheError("incomplete", "node-local bank is incomplete")
            try:
                _validate_members(bank, fetched)
            finally:
                client.release(fetched["reader_token"])
        return {
            "passed": True,
            "ready": True,
            "group_key": bank["group_key"],
            "total_bytes": bank["total_bytes"],
            "shard_count": 1,
            "copied_shards": int(copied > 0),
            "reused_shards": int(copied == 0),
            "copied_payload_bytes": copied,
            "selection_sha256": selection["selection_sha256"],
            "selected_tensor_count": selection["selected_tensor_count"],
            "build_elapsed_s": time.monotonic() - started,
        }
    finally:
        os.close(lock)


@contextlib.contextmanager
def model_view(model_root, socket_path, source_commit, *, selection_manifest_path):
    bank = describe_model(model_root, selection_manifest_path)
    parent = _private_parent(socket_path)
    with CpuWeightCacheClient(socket_path) as client:
        _check_daemon(client, source_commit)
        fetched = client.get_group(bank["items"])
        if fetched["state"] != "ready":
            raise CpuWeightCacheError(
                "missing", "node-local bank must be prebuilt; disk fallback forbidden"
            )
        view = None
        try:
            _validate_members(bank, fetched)
            target = Path(fetched["entries"][0]["path"])
            if not target.is_file() or target.stat().st_size != bank["total_bytes"]:
                raise CpuWeightCacheError(
                    "missing", "node-local backing file disappeared"
                )
            view = Path(tempfile.mkdtemp(prefix="node-local-view-", dir=parent))
            # Only metadata/configuration links; never link original weight files
            # or the original full index into this explicitly incomplete view.
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            ):
                source = Path(bank["model_root"]) / name
                if source.is_file():
                    (view / name).symlink_to(source)
            (view / OUTPUT_NAME).symlink_to(target)
            index = {
                "metadata": {"total_size": bank["selection"]["selected_payload_bytes"]},
                "weight_map": {
                    tensor["name"]: OUTPUT_NAME
                    for tensor in bank["selection"]["tensors"]
                },
            }
            (view / "model.safetensors.index.json").write_bytes(_json_bytes(index))
            yield str(view), bank
        finally:
            if view is not None:
                shutil.rmtree(view)
            client.release(fetched["reader_token"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "build", "status"))
    parser.add_argument("--model-root", "--model", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--socket")
    parser.add_argument("--source-commit")
    parser.add_argument("--layout", choices=("ep8", "pp2ep4"))
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--effective-layers", type=int, choices=(2, 80))
    parser.add_argument("--source-manifest")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--portable-source-identity")
    parser.add_argument("--portable-source-identity-sha256")
    args = parser.parse_args()
    try:
        if args.operation == "plan":
            portable = None
            if args.portable_source_identity:
                portable = {
                    "path": args.portable_source_identity,
                    "sha256": args.portable_source_identity_sha256,
                }
            selection = make_selection_manifest(
                args.model_root,
                layout=args.layout,
                node_rank=args.node_rank,
                effective_layers=args.effective_layers,
                source_manifest={
                    "path": args.source_manifest,
                    "sha256": args.source_manifest_sha256,
                },
                portable_source_identity=portable,
            )
            path = Path(args.selection_manifest)
            if not path.is_absolute():
                raise ValueError("absolute output manifest path required")
            # Explicit immutable creation: never silently rewrite a frozen plan.
            with path.open("xb") as handle:
                handle.write(_json_bytes(selection) + b"\n")
            result = {
                "passed": True,
                "selection_manifest": str(path),
                "selection_sha256": selection["selection_sha256"],
                "total_bytes": selection["output"]["size"],
                "selected_tensor_count": selection["selected_tensor_count"],
            }
        else:
            if not args.socket or not args.source_commit:
                raise ValueError(
                    "socket and source-commit are required for build/status"
                )
            fn = build_bank if args.operation == "build" else bank_status
            extra = {}
            if args.operation == "build":
                extra["progress"] = lambda value: print(
                    "Node-local CPU bank copy: " + json.dumps(value),
                    file=sys.stderr,
                    flush=True,
                )
            result = fn(
                args.model_root,
                args.socket,
                args.source_commit,
                selection_manifest_path=args.selection_manifest,
                **extra,
            )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    except (
        CpuWeightCacheError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
