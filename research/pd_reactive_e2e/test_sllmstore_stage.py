"""CPU checks for real safetensors offsets passed to the native Store."""

import importlib.util
import json
from pathlib import Path
import struct

import pytest

file = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/weight_cache/sllmstore_stage.py"
)
spec = importlib.util.spec_from_file_location("sllmstore_stage", file)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_native_indices_partition_source_bytes(tmp_path):
    tensors = [
        ("lm_head.weight", "BF16", [2], b"head"),
        ("model.language_model.embed_tokens.weight", "BF16", [1], b"ef"),
        (stage.LAYER + "0.expert", "F8_E4M3", [2], bytes([255, 0])),
        (stage.LAYER + "24.expert", "F8_E4M3", [2], bytes([1, 127])),
        ("model.visual.weight", "BF16", [1], b"vv"),
        ("model.language_model.norm.weight", "BF16", [1], b"nn"),
    ]
    header, raw = {}, bytearray()
    for name, dtype, shape, payload in tensors:
        header[name] = dict(
            dtype=dtype, shape=shape, data_offsets=[len(raw), len(raw) + len(payload)]
        )
        raw.extend(payload)
    encoded = json.dumps(header).encode()
    src = tmp_path / "raw.safetensors"
    src.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    root = tmp_path / "indices"
    manifest = stage.prepare_indices(
        root,
        tmp_path,
        {"group_key": "test", "shards": [{"name": src.name}]},
        [[0, 24], [24, 48]],
    )
    assert manifest["parts"] == [src.name]
    assert (root / "store" / src.name / "tensor.data_0").samefile(src)
    base, decoded = stage.read_header(src)
    for rank, (lo, hi) in enumerate(manifest["partitions"]):
        index, views = stage.stage_index(decoded, base, lo, hi)
        assert index == json.loads(
            (root / f"pp{rank}" / src.name / "tensor_index.json").read_text()
        )
        payloads = {}
        for name, span, offset, nbytes, dtype, shape in views:
            position = index[span][0] + offset
            payloads[name] = src.read_bytes()[position : position + nbytes]
        assert payloads == {
            name: data
            for name, _, _, data in tensors
            if (
                name in {"lm_head.weight", "model.language_model.norm.weight"}
                or rank == 0
                and name == "model.language_model.embed_tokens.weight"
                or name == stage.LAYER + str(lo) + ".expert"
            )
        }
    bad = dict(header)
    bad["lm_head.weight"] = dict(header["lm_head.weight"], data_offsets=[0, 3])
    with pytest.raises(ValueError, match="Invalid weight extent"):
        stage.stage_index(bad, base, 0, 24)
    bad["lm_head.weight"] = dict(header["lm_head.weight"], data_offsets=[1, 5])
    with pytest.raises(ValueError, match="Overlapping"):
        stage.stage_index(bad, base, 0, 24)
