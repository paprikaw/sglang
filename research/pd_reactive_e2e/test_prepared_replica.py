"""CPU regression tests for processed weights and the replica's independent host state."""

from pathlib import Path
from types import SimpleNamespace as NS
import os

import pytest
import torch

from sglang.srt.weight_cache import replica_preparation as weights
from sglang.srt.elastic_pp import replica_host_cache as history
from sglang.srt.weight_cache.sllmstore_stage import LAYER, stage_index


def test_sealed_marlin_bytes_and_foreign_packet(tmp_path):
    layer = NS(
        **{
            name: torch.arange(12).to(getattr(torch, dtype)).reshape(3, 4)
            for name, dtype in weights.EXPERT_DTYPES.items()
        }
    )
    fd, entry = weights.export_layer(layer)
    try:
        data = weights.validate_weight_file(entry).read_bytes()
        for name, (offset, size, shape, dtype) in entry["tensors"].items():
            restored = torch.frombuffer(
                bytearray(data[offset : offset + size]), dtype=getattr(torch, dtype)
            )
            assert torch.equal(restored.reshape(shape), getattr(layer, name))
        with pytest.raises(OSError):
            os.pwrite(fd, b"other", 0)
        with pytest.raises(ValueError):
            weights.validate_weight_file(dict(entry, inode=entry["inode"] + 1))
    finally:
        os.close(fd)
    weights.publish(tmp_path / "key", {"key": {"engine": "source"}})
    with pytest.raises(FileExistsError):
        weights.publish(tmp_path / "key", {"key": {}})
    with pytest.raises(ValueError, match="stale"):
        weights.read_packet(tmp_path, "key", {"engine": "other"})


def test_packed_stage_omits_only_routed_experts():
    names = [
        "mlp.experts.5.down_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.gate.weight",
    ]
    header = {
        LAYER
        + "0."
        + n: {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
        for i, n in enumerate(names)
    }
    idx, rows = stage_index(header, 8, 0, 24, omit_packed_experts=True)
    assert [r[0] for r in rows] == [LAYER + "0." + n for n in names[1:]]
    assert idx["0"][1] == 4


def runner(rank):
    args = NS(
        elastic_pp_hicache_shared_host=True,
        elastic_pp_initial_partition="24,24",
        disaggregation_mode="decode",
        pp_size=2,
        tp_size=1,
        ep_size=1,
        page_size=1,
        max_total_tokens=32,
        max_mamba_cache_size=4,
        hicache_size=1,
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        hicache_write_policy="write_through",
        kv_cache_dtype="auto",
        elastic_pp_kv_capacities="32",
    )
    return NS(
        server_args=args,
        model=NS(_canonical_cpu_weight_cache_key="bank"),
        ps=NS(pp_rank=rank),
    )


def test_replica_snapshot_uses_native_radix_and_independent_rows(tmp_path, monkeypatch):
    from array import array
    from sglang.srt.elastic_pp import hicache_transition as native
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from sglang.srt.mem_cache.unified_cache.components.full_component import (
        FullComponent,
    )
    from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
        MambaComponent,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    class Slots:
        def __init__(self):
            self.used = 0

        def alloc(self, count):
            before = self.used
            self.used += count
            return torch.arange(before, self.used)

    def cache():
        comps = {
            CT.FULL: object.__new__(FullComponent),
            CT.MAMBA: object.__new__(MambaComponent),
        }
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=tuple(comps),
        )
        tree = UnifiedTreeCore(params, comps)
        for c in comps.values():
            c.tree_core = tree
        tree.set_hicache_enabled()
        return NS(
            tree_core=tree,
            writing_check=lambda **kwargs: None,
            page_size=1,
            full_kv_pool_host=Slots(),
            mamba_pool_host=Slots(),
            _apply_cache_actions=lambda actions: None,
        )

    monkeypatch.setenv("SGLANG_REPLICA_ENGINE_COMMIT", "engine")
    monkeypatch.setenv("SGLANG_REPLICA_MARLIN_SHA256", "adapter")
    monkeypatch.setenv("SGLANG_REPLICA_EXPORT_DIR", str(tmp_path))
    src = cache()
    key = RadixKey(array("q", [42, 13]), extra_key="tenant", cache_salt="session")
    res = src.tree_core.insert_host(
        src.tree_core.root_node.id, key, torch.tensor([6, 4]), []
    )
    src.tree_core.node_by_id(res.inserted_host_node).component_data[
        CT.MAMBA
    ].host_value = torch.tensor([2])
    fields = {
        "full.kv": ([2, 8, 2, 1], "float32"),
        "mamba.temporal": ([4, 2, 2], "int32"),
        "mamba.conv_0": ([4, 2, 1], "bfloat16"),
    }
    schema = dict(
        layers={"full": [1, 3], "mamba": [0, 2]},
        capacities={"full": 8, "mamba": 4},
        regions={
            n: dict(shape=s, dtype="torch." + dt) for n, (s, dt) in fields.items()
        },
    )
    weights.publish(
        tmp_path / "source-schema.json",
        dict(**history.contract(runner(0), tmp_path), schema=schema),
    )
    bank = history.prepare_spare(
        tmp_path, history.contract(runner(0), tmp_path)["key"], 4096
    )
    try:
        for r in (0, 1):
            donor = torch.arange(8).float().view(8, 1) + r * 10
            source = {
                int(CT.FULL): {2 * r + 1: {"k": donor, "v": donor + 20}},
                int(CT.MAMBA): {
                    2
                    * r: {
                        "temporal": torch.arange(8).int().reshape(4, 2) + r,
                        "conv_0": torch.arange(4).bfloat16().view(4, 1) + r,
                    }
                },
            }
            monkeypatch.setattr(native, "cache_host_views", lambda c: source)
            assert history.seed_history(
                NS(
                    is_fully_idle=lambda: True,
                    tree_cache=src,
                    tp_worker=NS(model_runner=runner(r)),
                )
            )
            donor.fill_(99)
            assert bank.regions["full.kv"].tensor[0, :2, r, 0].tolist() == [
                6 + 10 * r,
                4 + 10 * r,
            ]
            assert bank.regions["mamba.temporal"].tensor[0, r].tolist() == [
                4 + r,
                5 + r,
            ]
        dst = cache()
        out = history.restore_history(dst, runner(0), tmp_path)
        assert out["host_rows"] == {"full": 2, "mamba": 1}
        n = dst.tree_core.root_node.children[key.child_key(1)]
        assert (
            n.key.extra_key == "tenant" and n.key.cache_salt == "session" and n.evicted
        )
        assert n.component_data[CT.MAMBA].host_value.tolist() == [0]
        assert dst.tree_core.host_lru_lists[CT.MAMBA].in_list(n)
        assert src.tree_core.node_by_id(res.inserted_host_node).component_data[
            CT.FULL
        ].host_value.tolist() == [6, 4]
        with pytest.raises(RuntimeError, match="empty"):
            history.restore_history(dst, runner(0), tmp_path)
    finally:
        bank.close()


def test_official_compact_dense_indices_reconstruct_checkpoint_bytes(tmp_path):
    import json, struct
    from sglang.srt.weight_cache.sllmstore_stage import read_header

    source = tmp_path / "canonical"
    source.mkdir()
    root = tmp_path / "store-root"
    (root / "store").mkdir(parents=True)
    name = "part.safetensors"
    names = [
        LAYER + "0.mlp.gate.weight",
        LAYER + "0.mlp.experts.0.w",
        LAYER + "0.attn.weight",
        LAYER + "24.attn.weight",
    ]
    header = {
        n: {"dtype": "BF16", "shape": [2], "data_offsets": [4 * i, 4 * i + 4]}
        for i, n in enumerate(names)
    }
    encoded = json.dumps(header).encode()
    (source / name).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"gateSKIPatt0att1"
    )
    fds = []
    try:
        parts = weights.prepare_dense_files(
            root,
            str(source),
            {"shards": [{"name": name}]},
            [[0, 24], [24, 48]],
            fds,
            1024,
        )
        assert len(fds) == len(parts) == 2
        for rank, expected in enumerate((b"gateatt0", b"att1")):
            assert (
                root / "store" / parts[rank] / "tensor.data_0"
            ).read_bytes() == expected
            base, raw = read_header(source / name)
            idx, _ = stage_index(
                raw, base, rank * 24, 24 + rank * 24, omit_packed_experts=True
            )
            actual = json.loads(
                (
                    root / "packed" / f"pp{rank}" / parts[rank] / "tensor_index.json"
                ).read_text()
            )
            assert actual == weights.compact_index(idx)
    finally:
        for fd in fds:
            os.close(fd)


def test_install_expert_owns_packed_storage(monkeypatch, tmp_path):
    import sys, types
    from sglang.srt import runtime_context
    from sglang.srt.layers.quantization import marlin_utils

    layer = torch.nn.Module()
    layer.layer_id = 0
    for n in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
        setattr(layer, n, torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    expected = {
        n: torch.arange(8).to(getattr(torch, dtype)).view(2, 4)
        for n, dtype in weights.EXPERT_DTYPES.items()
    }
    fd, e = weights.export_layer(NS(**expected))
    try:
        blob = torch.frombuffer(
            bytearray(weights.validate_weight_file(e).read_bytes()), dtype=torch.uint8
        )
    finally:
        os.close(fd)
    client = types.ModuleType("sllm_store.client")
    client.SllmStoreClient = lambda: NS(confirm_model_loaded=lambda name, replica: True)
    api = types.ModuleType("sllm_store.torch")
    api.load_dict_non_blocking = lambda *a, **kw: ("id", {"blob": blob})
    for k, v in (
        ("sllm_store", types.ModuleType("sllm_store")),
        ("sllm_store.client", client),
        ("sllm_store.torch", api),
    ):
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(pp_rank=0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        marlin_utils, "marlin_make_workspace", lambda *args: torch.zeros(4)
    )
    expected_shapes = {k: tuple(v.shape) for k, v in expected.items()}
    token = weights._active_prepared_weights.set(
        {
            "key": {"pp_layers": [[0, 24], [24, 48]]},
            "layers": {"0": dict(e, name="layer0")},
        }
    )
    try:
        weights.install_expert(layer, NS(packed_shapes=expected_shapes), tmp_path)
        blob.fill_(0)
        assert all(torch.equal(getattr(layer, k), v) for k, v in expected.items())
        assert not hasattr(layer, "w13_weight_scale_inv")
        assert layer.workspace.numel() == 4
        assert (
            weights._active_prepared_weights.get()["loaded_experts"][0][0] == e["size"]
        )
        layer.layer_id = 24
        with pytest.raises(ValueError, match="wrong stage"):
            weights.install_expert(layer, NS(packed_shapes=expected_shapes), tmp_path)
    finally:
        weights._active_prepared_weights.reset(token)


def test_raw_weight_retirement_keeps_store_event_loop_responsive():
    import asyncio
    import threading
    from replica_startup import unload_store_weights

    async def probe():
        loop = asyncio.get_running_loop()
        retired = []

        def unload(name):
            serviced = threading.Event()
            loop.call_soon_threadsafe(serviced.set)
            assert serviced.wait(2)
            retired.append(name)
            return name != "missing"

        client = NS(unload_from_cpu=unload)
        await unload_store_weights(client, ["stage0", "stage1"])
        with pytest.raises(RuntimeError, match="missing"):
            await unload_store_weights(client, ["missing", "unused"])
        assert retired == ["stage0", "stage1", "missing"]

    asyncio.run(probe())


def test_one_http_response_needs_both_stage_receipts(tmp_path):
    from replica_startup import donor_published

    for kind in ("weights", "history"):
        (tmp_path / f"{kind}-0.json").touch()
        assert not donor_published([True], tmp_path, kind)
        (tmp_path / f"{kind}-1.json").touch()
        assert not donor_published([False], tmp_path, kind)
        assert donor_published([True], tmp_path, kind)
