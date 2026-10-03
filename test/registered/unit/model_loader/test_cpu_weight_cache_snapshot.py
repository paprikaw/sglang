"""Prepared tensor snapshots survive worker destruction and skip quantization."""

import copy
from contextlib import nullcontext
import gc
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/weight_cache"
sys.path.insert(0, str(SOURCE))
from cpu_protocol import CpuWeightCacheClient, fingerprint_key
from cpu_snapshot import describe_model, restore_snapshot, write_snapshot


class PreparedModel(nn.Module):
    def __init__(self, prepared=False):
        super().__init__()
        self.qweight = nn.Parameter(
            torch.empty((2, 3), dtype=torch.uint8), requires_grad=False
        )
        self.tied = self.qweight
        self.register_buffer("rotary", torch.empty(3), persistent=False)
        self.register_buffer("lazy", None, persistent=False)
        self.format_ready = False
        self.partition = (4, 1)
        if prepared:
            # Simulate a post-load repack. A warm skeleton initially has a
            # different shape; rerunning conversion would corrupt these bytes.
            self.qweight = nn.Parameter(
                torch.arange(12, dtype=torch.uint8).reshape(3, 4).t(),
                requires_grad=False,
            )
            self.tied = self.qweight
            self.plain_view = self.qweight[1:3, :]
            self.rotary = torch.tensor([1.0, 2.0, 3.0])
            self.format_ready = True

    def forward(self, x):
        if not self.format_ready:
            raise RuntimeError("post-load metadata was lost")
        return x @ self.qweight.float() + self.rotary


class Glm4MoeSparseMoeBlock(nn.Module):
    """Exercise the real GLM gate -> non-Module TopKConfig tensor reference."""

    def __init__(self, prepared=False):
        super().__init__()
        self.gate = nn.Module()
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(4), requires_grad=False
        )
        self.topk = nn.Module()
        self.topk.topk_config = types.SimpleNamespace(
            correction_bias=self.gate.e_score_correction_bias
        )
        if prepared:
            self.gate.e_score_correction_bias.data.copy_(
                torch.arange(4, dtype=torch.float32)
            )

    def forward(self, x):
        return x + self.topk.topk_config.correction_bias


class TestCpuModelSnapshots(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cpu-snapshot-")
        self.socket = str(Path(self.temp.name) / "cache.sock")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(SOURCE / "cpu_daemon.py"),
                "--socket",
                self.socket,
                "--max-bytes",
                str(1024**2),
                "--headroom-bytes",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(100):
            if Path(self.socket).exists():
                break
            if self.proc.poll() is not None:
                self.fail(self.proc.communicate()[1].decode())
            time.sleep(0.02)
        self.fingerprint = {"model": "fixture", "source": "immutable", "tp": 4, "pp": 1}
        self.key = fingerprint_key(self.fingerprint)

    def tearDown(self):
        self.proc.terminate()
        self.proc.communicate(timeout=10)
        self.temp.cleanup()

    def _publish(self, model):
        manifest, storages = describe_model(model)
        with CpuWeightCacheClient(self.socket) as client:
            build = client.acquire(self.key, manifest["nbytes"], self.fingerprint)
            write_snapshot(build["fd"], manifest, storages, chunk_bytes=7)
            client.publish(self.key, build["token"], self.fingerprint, manifest)
        return manifest

    def test_restore_new_meta_model_after_producer_destruction(self):
        producer = PreparedModel(prepared=True)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        expected = producer(x).detach().clone()
        self._publish(producer)
        ref = weakref.ref(producer)
        del producer
        gc.collect()
        self.assertIsNone(ref())

        with torch.device("meta"):
            consumer = PreparedModel()
        with CpuWeightCacheClient(self.socket) as client:
            hit = client.get(self.key, self.fingerprint)
            restore_snapshot(consumer, hit["fd"], hit["manifest"], "cpu", chunk_bytes=5)
            self.assertEqual(
                client.release(hit["reader_token"], evict=True)["state"], "evicted"
            )
            self.assertEqual(client.status()["current_bytes"], 0)

        self.assertTrue(torch.equal(consumer(x), expected))
        self.assertIs(consumer.tied, consumer.qweight)
        self.assertEqual(consumer.qweight.stride(), (1, 4))
        self.assertEqual(consumer.plain_view.storage_offset(), 1)
        self.assertEqual(
            consumer.plain_view.untyped_storage().data_ptr(),
            consumer.qweight.untyped_storage().data_ptr(),
        )
        self.assertNotIn("rotary", consumer.state_dict())
        self.assertEqual(consumer.partition, (4, 1))

    def test_resume_preserves_existing_device_allocation_addresses(self):
        model = PreparedModel(prepared=True)
        x = torch.ones(1, 4)
        expected = model(x).detach().clone()
        self._publish(model)
        before = model.qweight.untyped_storage().data_ptr()
        model.qweight.data.zero_()
        model.rotary.zero_()
        with CpuWeightCacheClient(self.socket) as client:
            hit = client.get(self.key, self.fingerprint)
            restore_snapshot(model, hit["fd"], hit["manifest"], "cpu", reuse=True)
            client.release(hit["reader_token"], evict=True)
        self.assertEqual(model.qweight.untyped_storage().data_ptr(), before)
        self.assertTrue(torch.equal(model(x), expected))

    def test_corruption_and_constructor_class_drift_fail_before_acceptance(self):
        self._publish(PreparedModel(prepared=True))
        with CpuWeightCacheClient(self.socket) as client:
            hit = client.get(self.key, self.fingerprint)
            changed = copy.deepcopy(hit["manifest"])
            changed["storages"][0]["sha256"] = "0" * 64
            with torch.device("meta"):
                consumer = PreparedModel()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                restore_snapshot(consumer, hit["fd"], changed, "cpu")
            changed = copy.deepcopy(hit["manifest"])
            changed["objects"][0]["class"] = "some.other.Class"
            with self.assertRaisesRegex(ValueError, "class mismatch"):
                restore_snapshot(consumer, hit["fd"], changed, "cpu")
            client.release(hit["reader_token"])

    def test_shared_storage_is_cached_once(self):
        manifest, storages = describe_model(PreparedModel(prepared=True))
        self.assertEqual(len(storages), 2)
        entries = {entry["name"]: entry for entry in manifest["tensors"]}
        self.assertEqual(entries["qweight"]["storage"], entries["tied"]["storage"])
        self.assertEqual(
            entries["qweight"]["storage"], entries["plain_view"]["storage"]
        )

    def test_loader_uses_disk_once_then_new_engine_consumes_daemon_snapshot(self):
        # Import the actual adapter without the heavyweight engine package.
        # Only model construction/runtime context are substituted; transport,
        # snapshot copying, daemon ownership and consumption remain real.
        package = types.ModuleType("cpu_snapshot_adapter_test")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        adapter = importlib.import_module(package.__name__ + ".cpu_loader")
        loader_module = types.ModuleType("sglang.srt.model_loader.loader")
        loader_module._get_quantization_config = lambda *_: None
        loader_module._initialize_model = lambda *_: PreparedModel()
        utils_module = types.ModuleType("sglang.srt.model_loader.utils")
        utils_module.set_default_torch_dtype = lambda *_: nullcontext()
        identity = {**self.fingerprint, "source_commit": ""}
        fallback = Mock()
        fallback.load_model.side_effect = lambda **_: PreparedModel(prepared=True)
        config = types.SimpleNamespace(dtype=torch.float32)
        device = types.SimpleNamespace(device="cpu")
        with (
            patch.dict(
                sys.modules,
                {
                    loader_module.__name__: loader_module,
                    utils_module.__name__: utils_module,
                },
            ),
            patch.dict(os.environ, {adapter.SOCKET_ENV: self.socket}),
            patch.object(adapter, "_identity", return_value=identity),
        ):
            cold = adapter.CpuDaemonModelLoader(object(), fallback).load_model(
                model_config=config, device_config=device
            )
            self.assertEqual(cold._cpu_weight_cache_load_source, "disk")
            x = torch.ones(1, 4)
            expected = cold(x).detach().clone()
            adapter.publish_model(cold)
            ref = weakref.ref(cold)
            del cold
            gc.collect()
            self.assertIsNone(ref())
            warm = adapter.CpuDaemonModelLoader(object(), fallback).load_model(
                model_config=config, device_config=device
            )
            self.assertEqual(warm._cpu_weight_cache_load_source, "cpu_daemon")
            self.assertTrue(torch.equal(warm(x), expected))
            fallback.load_model.assert_called_once()
            with CpuWeightCacheClient(self.socket) as client:
                self.assertEqual(client.status()["current_bytes"], 0)

    def test_glm_constructor_routing_alias_rebinds_for_new_engine_and_resume(self):
        package = types.ModuleType("cpu_snapshot_glm_alias_test")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        adapter = importlib.import_module(package.__name__ + ".cpu_loader")
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                producer = Glm4MoeSparseMoeBlock(prepared=True)
                x = torch.ones(4)
                expected = producer(x).detach().clone()
                self._publish(producer)
                if reuse:
                    consumer = producer
                    consumer.gate.e_score_correction_bias.data.zero_()
                else:
                    with torch.device("meta"):
                        consumer = Glm4MoeSparseMoeBlock()
                with CpuWeightCacheClient(self.socket) as client:
                    hit = client.get(self.key, self.fingerprint)
                    restore_snapshot(
                        consumer, hit["fd"], hit["manifest"], "cpu", reuse=reuse
                    )
                    adapter._restore_glm_aliases(consumer)
                    client.release(hit["reader_token"], evict=True)
                self.assertIs(
                    consumer.topk.topk_config.correction_bias,
                    consumer.gate.e_score_correction_bias,
                )
                self.assertTrue(torch.equal(consumer(x), expected))

    def test_identity_separates_dp_replicas_and_tensor_pipeline_layouts(self):
        package = types.ModuleType("cpu_snapshot_identity_test")
        package.__path__ = [str(SOURCE)]
        sys.modules[package.__name__] = package
        adapter = importlib.import_module(package.__name__ + ".cpu_loader")
        model_path = Path(self.temp.name) / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text('{"hidden_size":8}')
        (model_path / "weights.safetensors").write_bytes(b"immutable-tiny-fixture")
        hf_config = types.SimpleNamespace(
            architectures=["Glm4MoeForCausalLM"], to_dict=lambda: {"hidden_size": 8}
        )
        config = types.SimpleNamespace(
            model_path=str(model_path),
            hf_config=hf_config,
            revision="fixed",
            dtype=torch.bfloat16,
            quantization="moe_wna16",
        )
        load_config = types.SimpleNamespace(
            load_format="auto",
            model_loader_extra_config={},
            _cpu_weight_cache_dp_rank=0,
        )
        args = types.SimpleNamespace(
            weight_cache_mode="off",
            speculative_algorithm=None,
            enable_elastic_pp=False,
            enable_eplb=False,
            elastic_ep_backend=None,
            enable_lora=False,
            is_startup_weight_load_overlap=False,
            moe_runner_backend="triton",
        )
        parallel = types.SimpleNamespace(
            tp_size=4,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=2,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_dp_size=1,
            moe_dp_rank=0,
            moe_tp_size=4,
            moe_tp_rank=0,
            attn_tp_size=4,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        )
        runtime = types.ModuleType("sglang.srt.runtime_context")
        runtime.get_parallel = lambda: parallel
        runtime.get_server_args = lambda: args
        with (
            patch.dict(sys.modules, {runtime.__name__: runtime}),
            patch.dict(os.environ, {adapter.SOURCE_ENV: "a" * 40}),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            key = fingerprint_key(adapter._identity(config, load_config))
            self.assertEqual(
                key, fingerprint_key(adapter._identity(config, load_config))
            )
            load_config._cpu_weight_cache_dp_rank = 1
            self.assertNotEqual(
                key, fingerprint_key(adapter._identity(config, load_config))
            )
            load_config._cpu_weight_cache_dp_rank = 0
            for field, value in (
                ("tp_size", 8),
                ("pp_size", 2),
                ("moe_tp_rank", 1),
                ("attn_dp_rank", 1),
                ("attn_dcp_size", 2),
            ):
                with self.subTest(field=field):
                    previous = getattr(parallel, field)
                    setattr(parallel, field, value)
                    self.assertNotEqual(
                        key, fingerprint_key(adapter._identity(config, load_config))
                    )
                    setattr(parallel, field, previous)
            args.enable_weights_cpu_backup = True
            with self.assertRaisesRegex(ValueError, "second CPU copy"):
                adapter._identity(config, load_config)
            args.enable_weights_cpu_backup = False
            args.moe_runner_backend = "flashinfer_trtllm"
            with self.assertRaisesRegex(ValueError, "state adapters"):
                adapter._identity(config, load_config)


class TestCpuDaemonOnlineUpdates(unittest.TestCase):
    operations = {
        "update_weights_from_disk": "UpdateWeightFromDiskReqOutput",
        "update_weights_from_distributed": "UpdateWeightsFromDistributedReqOutput",
        "update_weights_from_tensor": "UpdateWeightsFromTensorReqOutput",
        "update_weights_from_ipc": "UpdateWeightsFromIPCReqOutput",
        "init_weights_update_group": "InitWeightsUpdateGroupReqOutput",
    }

    @classmethod
    def setUpClass(cls):
        cls.updater_module = importlib.import_module(
            "sglang.srt.managers.scheduler_components.weight_updater"
        )

    def manager(self, daemon_owned):
        model = nn.Module()
        if daemon_owned:
            model._cpu_weight_cache_identity = {"source": "immutable"}
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
        draft = types.SimpleNamespace()
        for name in self.operations:
            setattr(worker, name, Mock(return_value=(True, "updated")))
            setattr(draft, name, Mock(return_value=(True, "updated")))
        manager = self.updater_module.SchedulerWeightUpdaterManager(
            tp_worker=worker,
            draft_worker=draft,
            tp_cpu_group=object(),
            memory_saver_adapter=Mock(),
            flush_cache=Mock(return_value=True),
            is_fully_idle=Mock(return_value=True),
            metrics_collector=Mock(),
        )
        return manager

    def test_daemon_owned_model_rejects_all_online_updates_before_side_effects(self):
        for name, output_name in self.operations.items():
            with self.subTest(operation=name):
                manager = self.manager(daemon_owned=True)
                with patch("torch.distributed.barrier") as barrier:
                    result = getattr(manager, name)(types.SimpleNamespace())
                self.assertIsInstance(result, getattr(self.updater_module, output_name))
                self.assertFalse(result.success)
                self.assertIn("CPU daemon snapshots", result.message)
                if name == "update_weights_from_disk":
                    self.assertEqual(result.num_paused_requests, 0)
                for method in self.operations:
                    getattr(manager.tp_worker, method).assert_not_called()
                    getattr(manager.draft_worker, method).assert_not_called()
                manager.flush_cache.assert_not_called()
                manager.metrics_collector.observe_weight_load.assert_not_called()
                barrier.assert_not_called()

    def test_non_daemon_models_keep_existing_online_update_behavior(self):
        for name, output_name in self.operations.items():
            with self.subTest(operation=name):
                manager = self.manager(daemon_owned=False)
                request = types.SimpleNamespace(
                    flush_cache=True,
                    torch_empty_cache=False,
                    disable_draft_model=True,
                )
                with patch("torch.distributed.barrier") as barrier:
                    result = getattr(manager, name)(request)
                self.assertIsInstance(result, getattr(self.updater_module, output_name))
                self.assertTrue(result.success)
                getattr(manager.tp_worker, name).assert_called_once_with(request)
                if name in ("update_weights_from_disk", "update_weights_from_ipc"):
                    getattr(manager.draft_worker, name).assert_called_once_with(request)
                if name == "init_weights_update_group":
                    manager.flush_cache.assert_not_called()
                    manager.metrics_collector.observe_weight_load.assert_not_called()
                else:
                    manager.flush_cache.assert_called_once_with(empty_cache=False)
                    manager.metrics_collector.observe_weight_load.assert_called_once()
                if name in ("update_weights_from_tensor", "update_weights_from_ipc"):
                    barrier.assert_called_once_with(group=manager.tp_cpu_group)
                else:
                    barrier.assert_not_called()

    def test_non_daemon_tensor_update_still_selects_draft_worker(self):
        manager = self.manager(daemon_owned=False)
        request = types.SimpleNamespace(flush_cache=False, disable_draft_model=False)
        with patch("torch.distributed.barrier"):
            self.assertTrue(manager.update_weights_from_tensor(request).success)
        manager.draft_worker.update_weights_from_tensor.assert_called_once_with(request)
        manager.tp_worker.update_weights_from_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
