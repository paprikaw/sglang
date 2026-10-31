import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from loader_patch import install, select_weights, target_parameter
from bounded_reader import verify_file


class SelectionTests(unittest.TestCase):
    def test_exact_maps(self):
        self.assertEqual(target_parameter("model.layers.1.mlp.experts.42.up_proj.weight"), "model.layers.1.mlp.experts.w13_weight")
        self.assertEqual(target_parameter("model.layers.1.mlp.router.gate.weight"), "model.layers.1.mlp.gate.weight")
        self.assertEqual(target_parameter("model.layers.1.self_attn.k_proj.weight"), "model.layers.1.self_attn.qkv_proj.weight")

    def test_stage_and_expert_selection(self):
        index = {"model.embed_tokens.weight": "a", "model.norm.weight": "z", "lm_head.weight": "z",
                 "model.layers.0.mlp.gate_proj.weight": "b", "model.layers.1.mlp.experts.0.gate_proj.weight": "c",
                 "model.layers.1.mlp.experts.1.gate_proj.weight": "c", "model.layers.80.mlp.experts.0.gate_proj.weight": "d"}
        params = {"model.layers.1.mlp.experts.w13_weight", "model.norm.weight", "lm_head.weight"}
        selected, coverage = select_weights(index, start_layer=1, end_layer=2,
            first_stage=False, last_stage=True, local_expert=lambda layer, expert: expert == 1,
            parameter_names=params)
        self.assertEqual(len(selected), 3)
        self.assertEqual(set(coverage), params)
        self.assertIn(("model.layers.1.mlp.experts.1.gate_proj.weight", "c"), selected)

    def test_missing_or_unknown_parameters_fail(self):
        args = dict(start_layer=0, end_layer=2, first_stage=True, last_stage=True,
                    local_expert=lambda *_: True)
        with self.assertRaisesRegex(ValueError, "missing checkpoint coverage"):
            select_weights({"model.norm.weight": "z"}, parameter_names={"model.norm.weight", "lm_head.weight"}, **args)
        with self.assertRaisesRegex(ValueError, "missing model parameter"):
            select_weights({"model.norm.weight": "z"}, parameter_names=set(), **args)
        with self.assertRaisesRegex(ValueError, "Unknown non-layer"):
                select_weights({"secret.weight": "z"}, parameter_names={"secret.weight"}, **args)


class LoaderWiringTests(unittest.TestCase):
    def test_full_install_iteration_copy_journal_and_attestation_order(self):
        # This checks the actual patched generator/wrapper signature and arena
        # lifetime without importing Torch, CUDA, or the engine on a login node.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
            header = {name: {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
                      for i, name in enumerate(names)}
            encoded = json.dumps(header).encode()
            shard = root / "tiny.safetensors"
            shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"abcdef")
            proof = verify_file(shard, hashlib.sha256(shard.read_bytes()).hexdigest())
            proof["relative_path"] = shard.name
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {name: shard.name for name in names}}))
            index_proof = {"relative_path": index.name, "path": str(index),
                           "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
            node_spec = root / "node.json"
            node_spec.write_text(json.dumps({"effective_num_hidden_layers": 2,
                "staging_ledger": str(root / "credits.json"), "staging_bytes": 16,
                "rank_staging_limit_bytes": 8}))
            binding = {"hy3_contract": {"node_specs": {"node": {
                "path": str(node_spec), "sha256": hashlib.sha256(node_spec.read_bytes()).hexdigest()}}}}
            use = {"path": str(root / "usage.jsonl")}
            sync_calls, copied, attestations = [], {}, []
            cuda_parameter = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))

            class BorrowedTensor:
                def __init__(self, view):
                    self.view = view

                def reshape(self, shape):
                    return self

            class HYV3ForCausalLM:
                pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
                model = types.SimpleNamespace(start_layer=0, end_layer=2)

                def named_parameters(self):
                    return [(name, cuda_parameter) for name in names]

                def parameters(self):
                    return [cuda_parameter] * len(names)

            model = HYV3ForCausalLM()

            class DefaultModelLoader:
                def load_model(self, *, model_config, device_config):
                    for name, tensor in self._get_all_weights(model_config, model):
                        copied[name] = bytes(tensor.view)
                    return model

            def attest(model_path, rank, **kwargs):
                summary = kwargs["stream_summary"]
                self.assertTrue(summary["arena_released"])
                path = Path(summary["receipts_path"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), summary["receipts_sha256"])
                attestations.append(kwargs)

            fake_torch = types.SimpleNamespace(bfloat16="BF16", float32="F32",
                frombuffer=lambda view, dtype: BorrowedTensor(view),
                cuda=types.SimpleNamespace(synchronize=lambda: sync_calls.append(True)))
            fake_policy = types.SimpleNamespace(
                validate_bound_model=lambda model_path, rank: (binding, "node", use),
                local_source_manifest=lambda *args: {"files": [proof, index_proof]},
                attest_loaded_model=attest)
            modules = {"torch": fake_torch, "hy3_streaming_policy": fake_policy,
                "sglang.srt.model_loader.loader": types.SimpleNamespace(DefaultModelLoader=DefaultModelLoader),
                "sglang.srt.runtime_context": types.SimpleNamespace(
                    get_parallel=lambda: types.SimpleNamespace(world_rank=0))}
            config = types.SimpleNamespace(model_path=str(root),
                hf_config=types.SimpleNamespace(num_hidden_layers=2), cpu_offload_gb=0)
            with patch.dict(sys.modules, modules):
                install()
                loaded = DefaultModelLoader().load_model(model_config=config, device_config=None)
            self.assertIs(loaded, model)
            self.assertEqual(copied, dict(zip(names, [b"ab", b"cd", b"ef"])))
            self.assertEqual(len(sync_calls), len(names) + 1)
            self.assertEqual(len(attestations), 1)
            rows = [json.loads(line) for line in (root / "usage.jsonl.rank0.tensors.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(names))
            self.assertEqual(sum(row["bytes"] for row in rows), 6)
            self.assertTrue(all(row["status"] == "consumed" for row in rows))
            ledger = json.loads((root / "credits.json").read_text())
            self.assertEqual(ledger["leases"], {})


if __name__ == "__main__":
    unittest.main()
