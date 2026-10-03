"""Actual CPU daemon + immutable canonical shard view, without CUDA/torch."""

import concurrent.futures
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/weight_cache"
sys.path.insert(0, str(SOURCE))
from cpu_protocol import CpuWeightCacheClient, CpuWeightCacheError
from cpu_canonical import bank_status, build_bank, describe_model, model_view
from cpu_canonical_loader import CanonicalCpuModelLoader


class ScopedFlags(types.SimpleNamespace):
    @contextmanager
    def override(self, **values):
        saved = {key: getattr(self, key) for key in values}
        self.__dict__.update(values)
        try:
            yield self
        finally:
            self.__dict__.update(saved)


class TestCanonicalCpuWeightBank(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="canonical-weights-")
        self.root = Path(self.temp.name)
        self.socket_root = self.root / "daemon"
        self.socket_root.mkdir(mode=0o700)
        self.socket = str(self.socket_root / "cache.sock")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"tiny"}')
        self.expected = {}
        index = {}
        # Match the real GLM bank's shard count, using tiny valid safetensors.
        for i in range(65):
            name = f"model-{i:05d}-of-00065.safetensors"
            tensor_name = f"weight.{i}"
            header = json.dumps(
                {tensor_name: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}},
                separators=(",", ":"),
            ).encode()
            header += b" " * (-len(header) % 8)
            payload = struct.pack("<Q", len(header)) + header + bytes([i, 1, 2, 3])
            (self.model / name).write_bytes(payload)
            self.expected[name] = payload
            index[tensor_name] = name
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(SOURCE / "cpu_daemon.py"),
                "--socket",
                self.socket,
                "--max-bytes",
                "262144",
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
            self.fail("daemon startup timed out")

    def tearDown(self):
        self.proc.terminate()
        self.proc.communicate(timeout=10)
        self.temp.cleanup()

    def build(self):
        return build_bank(str(self.model), self.socket, "test-source", chunk_bytes=17)

    def status(self):
        with CpuWeightCacheClient(self.socket) as client:
            return client.status()

    def test_one_complete_bank_survives_builder_exit_and_reuses_across_layouts(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SOURCE / "cpu_canonical.py"),
                "build",
                "--model-root",
                str(self.model),
                "--socket",
                self.socket,
                "--source-commit",
                "test-source",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        build = json.loads(result.stdout)
        self.assertEqual(build["copied_shards"], 65)
        self.assertEqual(build["group_key"], describe_model(self.model)["group_key"])
        before = self.status()["used_bytes"]
        # Changing topology only changes the ordinary loader's slicing; it
        # neither changes canonical identity nor creates another bank.
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_model = lambda: ScopedFlags(
            weight_loader_disable_mmap=False,
            weight_loader_drop_cache_after_load=False,
            weight_loader_prefetch_checkpoints=False,
        )
        calls = []
        owner = self

        class Fallback:
            def load_model(self, *, model_config, device_config):
                calls.append((model_config.tp, model_config.pp))
                self_view = Path(model_config.model_path)
                owner.assertNotEqual(self_view, owner.model)
                for file in self_view.glob("*.safetensors"):
                    owner.assertTrue(file.is_symlink())
                    owner.assertEqual(file.read_bytes(), owner.expected[file.name])
                owner.assertTrue(
                    all(e["readers"] == 1 for e in owner.status()["entries"])
                )
                return types.SimpleNamespace(config=model_config)

        env = {
            "SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET": self.socket,
            "SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT": "test-source",
        }
        with patch.dict(sys.modules, {runtime.__name__: runtime}), patch.dict(
            os.environ, env
        ):
            for tp, pp in ((8, 1), (4, 2)):
                config = types.SimpleNamespace(model_path=str(self.model), tp=tp, pp=pp)
                load_config = types.SimpleNamespace(model_loader_extra_config={})
                loaded = CanonicalCpuModelLoader(load_config, Fallback()).load_model(
                    model_config=config, device_config=None
                )
                self.assertEqual(
                    loaded._canonical_cpu_weight_cache_key, build["group_key"]
                )
                self.assertFalse(hasattr(loaded, "_cpu_weight_cache_identity"))
                self.assertEqual(config.model_path, str(self.model))
                self.assertEqual(loaded.config.model_path, str(self.model))
                self.assertFalse(hasattr(load_config, "_canonical_weight_cache_view"))
        self.assertEqual(calls, [(8, 1), (4, 2)])
        self.assertEqual(self.status()["used_bytes"], before)
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))
        self.assertFalse(list(self.socket_root.glob("canonical-view-*")))
        original_open = Path.open

        def metadata_only_open(path, *args, **kwargs):
            if path.suffix == ".safetensors":
                raise AssertionError("disk weight reread")
            return original_open(path, *args, **kwargs)

        with patch("cpu_canonical.Path.open", new=metadata_only_open):
            reused = self.build()
        self.assertEqual(reused["copied_shards"], 0)
        self.assertEqual(reused["reused_shards"], 65)

    def test_incomplete_bank_never_calls_loader_and_view_exception_releases_group(self):
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_model = lambda: ScopedFlags(
            weight_loader_disable_mmap=False,
            weight_loader_drop_cache_after_load=False,
            weight_loader_prefetch_checkpoints=False,
        )
        fallback = types.SimpleNamespace(
            load_model=lambda **kwargs: self.fail("disk fallback")
        )
        env = {
            "SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET": self.socket,
            "SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT": "test-source",
        }
        with patch.dict(sys.modules, {runtime.__name__: runtime}), patch.dict(
            os.environ, env
        ):
            with self.assertRaises(CpuWeightCacheError) as caught:
                CanonicalCpuModelLoader(
                    types.SimpleNamespace(model_loader_extra_config={}), fallback
                ).load_model(
                    model_config=types.SimpleNamespace(model_path=str(self.model)),
                    device_config=None,
                )
        self.assertEqual(caught.exception.code, "missing")
        self.build()
        with self.assertRaisesRegex(RuntimeError, "loader failed"):
            with model_view(str(self.model), self.socket, "test-source"):
                raise RuntimeError("loader failed")
        self.assertFalse(list(self.socket_root.glob("canonical-view-*")))
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))
        bank = describe_model(self.model)
        with CpuWeightCacheClient(self.socket) as client:
            client.evict(bank["items"][0]["key"])
        self.assertFalse(
            bank_status(str(self.model), self.socket, "test-source")["ready"]
        )
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))
        self.assertEqual(self.build()["copied_shards"], 1)

    def test_memfd_prefetch_and_drop_cache_hints_preserve_complete_bank(self):
        self.build()
        with model_view(str(self.model), self.socket, "test-source") as (view, _):
            for name, expected in self.expected.items():
                path = Path(view) / name
                with path.open("rb", buffering=0) as handle:
                    before = os.fstat(handle.fileno()).st_blocks
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    self.assertEqual(handle.read(), expected)
                    self.assertEqual(os.fstat(handle.fileno()).st_blocks, before)
        self.assertTrue(
            bank_status(str(self.model), self.socket, "test-source")["ready"]
        )

    def test_real_safetensors_parser_reads_all_ram_files(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors unavailable in stdlib-only interpreter")
        self.build()
        with model_view(str(self.model), self.socket, "test-source") as (view, _):
            for i, name in enumerate(self.expected):
                with safe_open(str(Path(view) / name), framework="np") as handle:
                    self.assertEqual(
                        handle.get_tensor(f"weight.{i}").tolist(), [i, 1, 2, 3]
                    )

    def test_inconsistent_shard_manifest_fails_closed_and_releases_pins(self):
        self.build()
        item = describe_model(self.model)["items"][0]
        with CpuWeightCacheClient(self.socket) as client:
            client.evict(item["key"])
            built = client.acquire(item["key"], 128, item["fingerprint"])
            os.pwrite(built["fd"], b"incorrect fixture", 0)
            client.publish(
                item["key"], built["token"], item["fingerprint"], {"kind": "wrong"}
            )
        with self.assertRaises(CpuWeightCacheError) as caught:
            bank_status(str(self.model), self.socket, "test-source")
        self.assertEqual(caught.exception.code, "fingerprint")
        with self.assertRaises(CpuWeightCacheError):
            with model_view(str(self.model), self.socket, "test-source"):
                self.fail("invalid manifest exposed")
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))
        self.assertFalse(list(self.socket_root.glob("canonical-view-*")))

    def test_concurrent_build_is_single_copy_and_views_pin_all_shards(self):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.build(), range(2)))
        self.assertEqual(sum(r["copied_shards"] for r in results), 65)
        self.assertEqual(
            {r["group_key"] for r in results}, {describe_model(self.model)["group_key"]}
        )
        with model_view(str(self.model), self.socket, "test-source") as (first, _):
            with model_view(str(self.model), self.socket, "test-source") as (second, _):
                self.assertNotEqual(first, second)
                self.assertTrue(
                    all(e["readers"] == 2 for e in self.status()["entries"])
                )
                with CpuWeightCacheClient(self.socket) as client:
                    with self.assertRaises(CpuWeightCacheError):
                        client.evict(describe_model(self.model)["items"][0]["key"])
            self.assertTrue(all(e["readers"] == 1 for e in self.status()["entries"]))
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))

    def test_oversize_and_source_mismatch_fail_before_disk_copy(self):
        with patch("cpu_canonical._check_daemon", return_value={"max_bytes": 1}):
            with self.assertRaises(CpuWeightCacheError) as caught:
                self.build()
        self.assertEqual(caught.exception.code, "capacity")
        self.assertEqual(self.status()["current_bytes"], 0)
        with self.assertRaises(CpuWeightCacheError) as caught:
            build_bank(str(self.model), self.socket, "other-source")
        self.assertEqual(caught.exception.code, "source")
        self.assertEqual(self.status()["current_bytes"], 0)
        previous = describe_model(self.model)["group_key"]
        shard = self.model / next(iter(self.expected))
        shard.write_bytes(shard.read_bytes() + b"changed")
        self.assertNotEqual(describe_model(self.model)["group_key"], previous)

    def test_interrupted_bank_build_recovers_only_missing_shards(self):
        def interrupt(value):
            if value["completed_shards"] == 2:
                raise RuntimeError("producer interrupted")

        with self.assertRaisesRegex(RuntimeError, "producer interrupted"):
            build_bank(str(self.model), self.socket, "test-source", progress=interrupt)
        for _ in range(100):
            state = self.status()
            if state["reserved_bytes"] == 0 and all(
                e["readers"] == 0 for e in state["entries"]
            ):
                break
            time.sleep(0.01)
        self.assertEqual(state["reserved_bytes"], 0)
        self.assertEqual(len(state["entries"]), 2)
        self.assertFalse(
            bank_status(str(self.model), self.socket, "test-source")["ready"]
        )
        recovered = self.build()
        self.assertEqual(recovered["copied_shards"], 63)
        self.assertEqual(recovered["reused_shards"], 2)
        self.assertTrue(
            bank_status(str(self.model), self.socket, "test-source")["ready"]
        )

    def test_private_directory_rejects_permissions_without_partial_view(self):
        self.build()
        self.socket_root.chmod(0o755)
        try:
            with self.assertRaisesRegex(ValueError, "private owned"):
                with model_view(str(self.model), self.socket, "test-source"):
                    self.fail("unsafe view opened")
        finally:
            self.socket_root.chmod(0o700)
        self.assertFalse(list(self.socket_root.glob("canonical-view-*")))
        self.assertTrue(all(e["readers"] == 0 for e in self.status()["entries"]))


if __name__ == "__main__":
    unittest.main()
