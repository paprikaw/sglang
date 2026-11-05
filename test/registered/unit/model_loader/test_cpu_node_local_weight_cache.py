"""Node-local Hy3 bank: real CPU daemon/memfd, no torch or GPU imports."""

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/weight_cache"
sys.path.insert(0, str(SOURCE))
import cpu_node_local as bank_module
from cpu_node_local import (
    OUTPUT_NAME,
    bank_status,
    build_bank,
    build_selection_manifest,
    describe_model,
    make_selection_manifest,
    model_view,
)
from cpu_node_local_loader import NodeLocalCpuModelLoader
from cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError, fingerprint_key


class ScopedFlags(types.SimpleNamespace):
    @contextmanager
    def override(self, **values):
        old = {key: getattr(self, key) for key in values}
        self.__dict__.update(values)
        try:
            yield self
        finally:
            self.__dict__.update(old)


def safetensors(path, values):
    header, payload = {}, bytearray()
    for name, value in values.items():
        header[name] = {
            "dtype": "BF16",
            "shape": [len(value) // 2],
            "data_offsets": [len(payload), len(payload) + len(value)],
        }
        payload.extend(value)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def read_values(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    return {
        name: data[meta["data_offsets"][0] : meta["data_offsets"][1]]
        for name, meta in header.items()
        if name != "__metadata__"
    }


class TestNodeLocalCpuWeightBank(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="node-local-weights-")
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.sockdir = self.root / "daemon"
        self.sockdir.mkdir(mode=0o700)
        self.socket = str(self.sockdir / "cache.sock")
        config = {
            "model_type": "hy_v3",
            "dtype": "bfloat16",
            "num_hidden_layers": 80,
            "num_experts": 192,
            "num_experts_per_tok": 8,
            "hidden_size": 4096,
            "num_key_value_heads": 8,
            "num_attention_heads": 64,
            "tie_word_embeddings": False,
        }
        self.model_config = config
        (self.model / "config.json").write_text(json.dumps(config))
        (self.model / "tokenizer.json").write_text("{}")
        self.values = {
            "model.embed_tokens.weight": b"\x01\x02\x03\x04",
            "lm_head.weight": b"\x05\x06\x07\x08",
            "model.norm.weight": b"\x09\x0a",
        }
        # Boundary experts, stage0/1 and MTP use distinct source bytes. The
        # profile config remains real Hy3; fixture shapes stay tiny for CPU CI.
        for layer in (0, 1, 39, 40, 79, 80):
            for suffix in (
                "self_attn.q_proj.weight",
                "mlp.shared_mlp.gate_proj.weight",
                "input_layernorm.weight",
            ):
                self.values[f"model.layers.{layer}.{suffix}"] = bytes([layer, 99])
            for expert in (0, 23, 24, 95, 96, 119, 120, 191):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    self.values[
                        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    ] = bytes([layer, expert])
        # Mix stages and expert halves in each original file: compacting must
        # copy selected ranges, not retain entire touched source shards.
        maps = [{}, {}]
        for i, (name, value) in enumerate(self.values.items()):
            maps[i % 2][name] = value
        index = {}
        for i, values in enumerate(maps):
            name = f"model-{i+1:05d}-of-00002.safetensors"
            safetensors(self.model / name, values)
            index.update({key: name for key in values})
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(map(len, self.values.values()))},
                    "weight_map": index,
                }
            )
        )
        self.proof = self.root / "source.json"
        self.rewrite_proof()
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(SOURCE / "cpu_daemon.py"),
                "--socket",
                self.socket,
                "--max-bytes",
                "1048576",
                "--headroom-bytes",
                "0",
                "--source-commit",
                "test-source",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.fail(self.proc.communicate()[1].decode())
            try:
                with CpuWeightCacheClient(self.socket) as client:
                    client.status()
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.02)
        else:
            self.fail("daemon did not start")

    def rewrite_proof(self):
        files = []
        for path in self.model.iterdir():
            info = path.stat()
            files.append(
                dict(
                    path=str(path),
                    relative_path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sha256_verified=True,
                    device=info.st_dev,
                    inode=info.st_ino,
                    size=info.st_size,
                    mtime_ns=info.st_mtime_ns,
                    ctime_ns=info.st_ctime_ns,
                )
            )
        self.proof.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "repo": "tencent/Hy3",
                    "revision": "a" * 40,
                    "model_path": str(self.model),
                    "files": files,
                }
            )
        )
        self.descriptor = {
            "path": str(self.proof),
            "sha256": hashlib.sha256(self.proof.read_bytes()).hexdigest(),
        }

    def tearDown(self):
        self.proc.terminate()
        self.proc.communicate(timeout=10)
        self.temp.cleanup()

    def plan(self, layout="ep8", node=0, layers=2):
        manifest = make_selection_manifest(
            str(self.model),
            layout=layout,
            node_rank=node,
            effective_layers=layers,
            source_manifest=self.descriptor,
        )
        path = self.root / f"{layout}-{node}-{layers}.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def build(self, path):
        return build_bank(
            str(self.model),
            self.socket,
            "test-source",
            selection_manifest_path=str(path),
            chunk_bytes=3,
        )

    def test_exact_ep_partition_stage_filter_mtp_and_alias(self):
        for layout in ("ep8", "pp2ep4"):
            for layers in (2, 80):
                plans = [self.plan(layout, node, layers)[1] for node in (0, 1)]
                sets = [{t["name"] for t in m["tensors"]} for m in plans]
                for node, names in enumerate(sets):
                    self.assertFalse(
                        any(n.startswith("model.layers.80.") for n in names)
                    )
                    if layout == "ep8":
                        for n in names:
                            expert = bank_module._EXPERT.search(n)
                            if expert:
                                self.assertTrue(
                                    node * 96 <= int(expert.group(1)) < (node + 1) * 96
                                )
                    else:
                        for n in names:
                            layer = bank_module._LAYER.match(n)
                            if layer:
                                self.assertTrue(
                                    node * (layers // 2)
                                    <= int(layer.group(1))
                                    < (node + 1) * (layers // 2)
                                )
                expected = {
                    n
                    for n in self.values
                    if not bank_module._LAYER.match(n)
                    or int(bank_module._LAYER.match(n).group(1)) < layers
                }
                self.assertEqual(sets[0] | sets[1], expected)
                self.assertNotEqual(
                    plans[0]["selection_sha256"], plans[1]["selection_sha256"]
                )
                self.assertEqual(
                    plans[0],
                    build_selection_manifest(
                        str(self.model),
                        layout=layout,
                        node_rank=0,
                        effective_num_hidden_layers=layers,
                        source_manifest=self.descriptor,
                    ),
                )

    def test_real_daemon_selected_bytes_sealed_view_reuse_and_readers(self):
        path, selection = self.plan()
        result = self.build(path)
        self.assertEqual(
            result["copied_payload_bytes"], selection["selected_payload_bytes"]
        )
        self.assertLess(
            result["copied_payload_bytes"], sum(map(len, self.values.values()))
        )
        with model_view(
            str(self.model),
            self.socket,
            "test-source",
            selection_manifest_path=str(path),
        ) as (view, bank):
            values = read_values(Path(view) / OUTPUT_NAME)
            self.assertEqual(
                values,
                {t["name"]: self.values[t["name"]] for t in selection["tensors"]},
            )
            self.assertEqual(len(list(Path(view).glob("*.safetensors"))), 1)
            index = json.loads(
                (Path(view) / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(set(index["weight_map"]), set(values))
            self.assertTrue((Path(view) / "config.json").is_symlink())
            with self.assertRaises(OSError):
                with (Path(view) / OUTPUT_NAME).open("r+b") as f:
                    f.write(b"bad")
            with CpuWeightCacheClient(self.socket) as client:
                self.assertEqual(client.status()["entries"][0]["readers"], 1)
        self.assertFalse(list(self.sockdir.glob("node-local-view-*")))
        real_pread = os.pread

        def header_only(fd, size, offset):
            if offset not in (0, 8):
                self.fail("reuse reread a checkpoint payload range")
            return real_pread(fd, size, offset)

        with patch.object(bank_module.os, "pread", side_effect=header_only):
            reused = self.build(path)
        self.assertEqual(reused["copied_payload_bytes"], 0)
        self.assertTrue(
            bank_status(
                str(self.model),
                self.socket,
                "test-source",
                selection_manifest_path=str(path),
            )["ready"]
        )

    def test_real_parser_accepts_compact_bank(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors unavailable in stdlib-only interpreter")
        path, selection = self.plan()
        self.build(path)
        with model_view(
            str(self.model),
            self.socket,
            "test-source",
            selection_manifest_path=str(path),
        ) as (view, _):
            with safe_open(str(Path(view) / OUTPUT_NAME), framework="np") as handle:
                self.assertEqual(
                    set(handle.keys()), {t["name"] for t in selection["tensors"]}
                )
                for tensor in selection["tensors"]:
                    self.assertEqual(
                        handle.get_slice(tensor["name"]).get_shape(), tensor["shape"]
                    )

    def test_full_daemon_never_evicts_an_existing_layout(self):
        path0, _ = self.plan(node=0)
        path1, _ = self.plan(node=1)
        self.build(path0)
        with CpuWeightCacheClient(self.socket) as client:
            status = client.status()
            old_key = describe_model(str(self.model), str(path0))["items"][0]["key"]
            filler = {"fixture": "fill-existing-unborrowed-capacity"}
            key = fingerprint_key(filler)
            result = client.acquire(
                key, status["max_bytes"] - status["used_bytes"], filler
            )
            client.publish(key, result["token"], filler, {"fixture": True})
            before = {entry["key"] for entry in client.status()["entries"]}
        with self.assertRaisesRegex(CpuWeightCacheError, "eviction forbidden"):
            self.build(path1)
        with CpuWeightCacheClient(self.socket) as client:
            self.assertEqual(
                {entry["key"] for entry in client.status()["entries"]}, before
            )
            self.assertIn(old_key, before)
        self.assertTrue(
            bank_status(
                str(self.model),
                self.socket,
                "test-source",
                selection_manifest_path=str(path0),
            )["ready"]
        )

    def test_missing_source_changed_manifest_tampering_fail_closed(self):
        path, selection = self.plan()
        with self.assertRaisesRegex(CpuWeightCacheError, "prebuilt"):
            with model_view(
                str(self.model),
                self.socket,
                "test-source",
                selection_manifest_path=str(path),
            ):
                self.fail("disk fallback")
        modified = dict(selection)
        modified["tensors"] = selection["tensors"][:-1]
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, "exact source-derived"):
            describe_model(str(self.model), str(path))
        path.write_text(json.dumps(selection))
        shard = self.model / selection["tensors"][0]["source_shard"]
        os.utime(shard, ns=(shard.stat().st_atime_ns, shard.stat().st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, "mtime_ns changed"):
            self.build(path)
        with CpuWeightCacheClient(self.socket) as client:
            self.assertEqual(client.status()["used_bytes"], 0)

    def test_build_error_drops_unpublished_lease_and_view_error_unpins(self):
        path, _ = self.plan()
        real_pread = os.pread

        def fail_payload(fd, size, offset):
            if offset not in (0, 8):
                raise OSError("injected copy failure")
            return real_pread(fd, size, offset)

        with patch.object(bank_module.os, "pread", side_effect=fail_payload):
            with self.assertRaisesRegex(OSError, "injected"):
                self.build(path)
        deadline = time.monotonic() + 3
        while True:
            with CpuWeightCacheClient(self.socket) as client:
                status = client.status()
            if not status["entries"]:
                break
            if time.monotonic() > deadline:
                self.fail("failed build lease was retained")
            time.sleep(0.01)
        self.build(path)
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with model_view(
                str(self.model),
                self.socket,
                "test-source",
                selection_manifest_path=str(path),
            ):
                raise RuntimeError("consumer failed")
        with CpuWeightCacheClient(self.socket) as client:
            self.assertEqual(client.status()["entries"][0]["readers"], 0)

    def test_loader_binds_config_topology_hash_and_no_disk_fallback(self):
        path, selection = self.plan()
        self.build(path)
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_model = lambda: ScopedFlags(
            weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=True
        )
        topology = types.SimpleNamespace(
            tp_size=8,
            pp_size=1,
            attn_tp_size=4,
            moe_ep_size=8,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_dp_size=1,
            attn_dp_size=2,
        )
        runtime.get_parallel = lambda: topology
        moe = types.SimpleNamespace(
            enable_eplb=False,
            ep_join_mode=None,
            ep_num_redundant_experts=0,
            init_expert_location="trivial",
        )
        runtime.get_exec = lambda: types.SimpleNamespace(moe=moe)
        owner = self

        class Fallback:
            def load_model(self, *, model_config, device_config):
                owner.assertNotEqual(model_config.model_path, str(owner.model))
                owner.assertEqual(
                    read_values(Path(model_config.model_path) / OUTPUT_NAME),
                    {t["name"]: owner.values[t["name"]] for t in selection["tensors"]},
                )
                return types.SimpleNamespace(config=model_config)

        config = types.SimpleNamespace(
            model_path=str(self.model),
            dtype="torch.bfloat16",
            quantization=None,
            hf_config=types.SimpleNamespace(
                **dict(
                    self.model_config, num_hidden_layers=2, num_nextn_predict_layers=0
                )
            ),
        )
        load_config = types.SimpleNamespace(model_loader_extra_config=None)
        env = {
            "SGLANG_NODE_LOCAL_WEIGHT_CACHE_SOCKET": self.socket,
            "SGLANG_NODE_LOCAL_WEIGHT_CACHE_SOURCE_COMMIT": "test-source",
            "SGLANG_NODE_LOCAL_WEIGHT_CACHE_SELECTION_MANIFEST": str(path),
            "SGLANG_NODE_LOCAL_WEIGHT_CACHE_SELECTION_SHA256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        }
        with patch.dict(sys.modules, {runtime.__name__: runtime}), patch.dict(
            os.environ, env
        ):
            loader = NodeLocalCpuModelLoader(load_config, Fallback())
            loader.download_model(config)
            model = loader.load_model(model_config=config, device_config=None)
            self.assertEqual(model.config.model_path, str(self.model))
            self.assertIsNone(load_config.model_loader_extra_config)
            self.assertFalse(hasattr(load_config, "_node_local_weight_cache_view"))
            self.assertEqual(
                model._node_local_cpu_weight_cache_key,
                describe_model(str(self.model), str(path))["group_key"],
            )
            topology.moe_ep_rank = 4
            with self.assertRaisesRegex(ValueError, "topology"):
                loader.load_model(model_config=config, device_config=None)
            topology.moe_ep_rank = 0
            config.hf_config.num_nextn_predict_layers = 1
            with self.assertRaisesRegex(ValueError, "MTP"):
                loader.load_model(model_config=config, device_config=None)
            config.hf_config.num_nextn_predict_layers = 0
            config.hf_config.num_experts = 96
            with self.assertRaisesRegex(
                ValueError, "architecture override num_experts"
            ):
                loader.load_model(model_config=config, device_config=None)
            config.hf_config.num_experts = 192
            moe.enable_eplb = True
            with self.assertRaisesRegex(ValueError, "expert placement"):
                loader.load_model(model_config=config, device_config=None)
            moe.enable_eplb = False
            with patch.dict(
                os.environ,
                {"SGLANG_NODE_LOCAL_WEIGHT_CACHE_SELECTION_SHA256": "0" * 64},
            ):
                with self.assertRaisesRegex(ValueError, "launch binding"):
                    loader.load_model(model_config=config, device_config=None)

    def test_portable_source_device_is_explicitly_bound_not_ignored(self):
        source = json.loads(self.proof.read_text())
        for item in source["files"]:
            item["device"] += 99
        self.proof.write_text(json.dumps(source))
        self.descriptor["sha256"] = hashlib.sha256(self.proof.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "device differs"):
            self.plan()
        portable = {
            "model_path": str(self.model),
            "source_manifest": self.descriptor,
            "filesystem": {
                "fsid": os.statvfs(self.model).f_fsid,
                "source": "fixture",
                "fstype": "fixture",
                "uuid": None,
            },
        }
        p = self.root / "portable.json"
        p.write_text(json.dumps(portable))
        descriptor = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        with patch.object(bank_module, "_portable_filesystem") as check:
            selection = make_selection_manifest(
                str(self.model),
                layout="ep8",
                node_rank=0,
                effective_layers=2,
                source_manifest=self.descriptor,
                portable_source_identity=descriptor,
            )
            self.assertEqual(selection["device_binding"]["kind"], "portable_filesystem")
            check.assert_called_once()
            path = self.root / "portable-selection.json"
            path.write_text(json.dumps(selection))
            self.build(path)
        with patch.object(
            bank_module,
            "_portable_filesystem",
            side_effect=ValueError("wrong filesystem"),
        ):
            with self.assertRaisesRegex(ValueError, "wrong filesystem"):
                describe_model(str(self.model), str(path))

    def test_loader_registry_and_external_source_guard(self):
        tree = ast.parse((SOURCE.parent / "model_loader/loader.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_model_loader"
        )
        namespace = {
            "os": os,
            "LoadFormat": types.SimpleNamespace(AUTO="auto", SAFETENSORS="safe"),
            "DefaultModelLoader": lambda cfg: "normal-loader",
        }
        fake = types.ModuleType("sglang.srt.weight_cache.cpu_node_local_loader")
        fake.NodeLocalCpuModelLoader = lambda cfg, fallback: (cfg, fallback)
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "loader_registry", "exec"),
            dict(
                namespace,
                LoadConfig=object,
                ModelConfig=object,
                Optional=__import__("typing").Optional,
                BaseModelLoader=object,
            ),
            namespace,
        )
        with patch.dict(sys.modules, {fake.__name__: fake}), patch.dict(
            os.environ,
            {"SGLANG_NODE_LOCAL_WEIGHT_CACHE_SOCKET": self.socket},
            clear=True,
        ):
            cfg = types.SimpleNamespace(load_format="auto")
            self.assertEqual(namespace["get_model_loader"](cfg), (cfg, "normal-loader"))
            with patch.dict(
                os.environ, {"SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET": "bad"}
            ):
                with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                    namespace["get_model_loader"](cfg)


if __name__ == "__main__":
    unittest.main()
