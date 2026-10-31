"""CPU-only fairness checks for the full-model Decode Graph comparison.

These tests inspect configuration and exercise the real HTTP measurement
helper against an in-memory SSE response. They neither import a GPU framework
nor establish CUDA Graph correctness/performance; the untimed numerical and
all-rank Graph replay witnesses must supply that evidence on the real nodes.
"""
from __future__ import annotations

import ast
import builtins
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"


@contextmanager
def no_gpu_imports():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "sglang", "deep_ep", "deep_ep_cpp"}:
            raise AssertionError(f"CPU protocol tests imported GPU package {name}")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        yield


def load_runtime_modules():
    """Avoid accidentally testing another experiment's cached common module."""
    loaded = {}
    prior_path = list(sys.path)
    saved = {name: sys.modules.get(name) for name in ("common", "workload", "measure")}
    sys.path.insert(0, str(RUNTIME))
    try:
        with no_gpu_imports():
            for name in ("common", "workload", "measure"):
                spec = importlib.util.spec_from_file_location(name, RUNTIME / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                loaded[name] = module
    finally:
        sys.path[:] = prior_path
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return loaded


def option(argv, flag):
    if argv.count(flag) != 1:
        raise AssertionError(f"Expected exactly one {flag}, got {argv.count(flag)}")
    return argv[argv.index(flag) + 1]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_runtime_modules()
        cls.common, cls.workload, cls.measure = (modules[name] for name in ("common", "workload", "measure"))

    def configuration(self, layout):
        return dict(id="cpu-protocol-test", run_dir="/tmp/not-a-real-run", layout=layout,
                    nodes=["node0", "node1"], dist_port=10001, api_port=10002)

    def argv(self, layout):
        return self.common.server_args(self.configuration(layout), 1)

    def test_runtime_modules_import_without_gpu_frameworks(self):
        self.assertEqual(self.common.ROOT, ROOT)
        self.assertEqual(self.common.COMMIT, "870269702285d94bdd926a50cc29ad1a20bcf60c")
        self.assertEqual(self.common.NATIVE_SHA,
                         "9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581")

    def test_attention_expert_and_node_layouts_are_unchanged(self):
        for layout, tp, dp, ep, pp, chunk in (
            ("ep8", "8", "2", "8", "1", "2048"),
            ("pp2ep4", "4", "1", "4", "2", "1024"),
        ):
            with self.subTest(layout=layout):
                argv = self.argv(layout)
                expected = {"--tp-size": tp, "--dp-size": dp, "--ep-size": ep,
                            "--pp-size": pp, "--chunked-prefill-size": chunk,
                            "--nnodes": "2", "--node-rank": "1",
                            "--dist-init-addr": "node0:10001"}
                for flag, wanted in expected.items():
                    self.assertEqual(option(argv, flag), wanted)
                self.assertIn("--enable-dp-attention", argv)

    def test_both_arms_use_identical_full_b4_decode_graph(self):
        configurations = []
        for layout in ("ep8", "pp2ep4"):
            argv = self.argv(layout)
            self.assertNotIn("--disable-cuda-graph", argv)
            graph = json.loads(option(argv, "--cuda-graph-config"))
            self.assertEqual(graph["decode"]["backend"], "full")
            self.assertEqual(graph["decode"]["bs"], [4])
            self.assertEqual(graph["decode"]["max_bs"], 4)
            self.assertEqual(graph["prefill"]["backend"], "disabled")
            configurations.append(graph)
        self.assertEqual(*configurations)

    def test_pp_uses_two_b4_slots_without_extra_pipeline_depth(self):
        self.assertEqual(option(self.argv("pp2ep4"), "--pp-max-micro-batch-size"), "4")
        self.assertNotIn("--pp-max-micro-batch-size", self.argv("ep8"))
        for layout in ("ep8", "pp2ep4"):
            argv = self.argv(layout)
            self.assertEqual(option(argv, "--max-running-requests"), "8")
            self.assertNotIn("--pp-async-batch-depth", argv)
            self.assertIn("--disable-overlap-schedule", argv)

    def test_native_hicache_capacity_and_format_are_unchanged(self):
        expected = {"--hicache-size": "35", "--hicache-mem-layout": "page_first",
                    "--hicache-io-backend": "kernel", "--hicache-write-policy": "write_through",
                    "--page-size": "64", "--context-length": "131072",
                    "--mem-fraction-static": "0.90", "--max-prefill-tokens": "1024"}
        for layout in ("ep8", "pp2ep4"):
            argv = self.argv(layout)
            self.assertIn("--enable-hierarchical-cache", argv)
            self.assertIn("--skip-server-warmup", argv)
            for flag, wanted in expected.items():
                self.assertEqual(option(argv, flag), wanted)
            for forbidden in ("--max-total-tokens", "--hicache-storage-backend", "--disable-radix-cache"):
                self.assertNotIn(forbidden, argv)

    def test_model_compute_and_deepep_flags_are_unchanged(self):
        expected = {"--dtype": "bfloat16", "--quantization": "humming",
                    "--moe-runner-backend": "humming", "--attention-backend": "triton",
                    "--moe-a2a-backend": "deepep", "--deepep-mode": "low_latency",
                    "--deepep-dispatcher-output-dtype": "bf16", "--random-seed": "42"}
        for layout in ("ep8", "pp2ep4"):
            argv = self.argv(layout)
            for flag, wanted in expected.items():
                self.assertEqual(option(argv, flag), wanted)
            self.assertIn("--disable-custom-all-reduce", argv)
            self.assertIn("--disable-shared-experts-fusion", argv)
            self.assertEqual(option(argv, "--model-path"), str(self.common.MODEL))

    def test_transport_environment_and_guard_binding_survive(self):
        expected = dict(CUDA_VISIBLE_DEVICES="0,1,2,3", NCCL_IB_HCA="mlx5_bond_0",
                        NCCL_SOCKET_IFNAME="bond0.3027", GLOO_SOCKET_IFNAME="bond0.3027",
                        NCCL_CUMEM_HOST_ENABLE="0", NVSHMEM_HCA_LIST="mlx5_bond_0:1",
                        NVSHMEM_IBGDA_NIC_HANDLER="cpu_host_memory", NVSHMEM_QP_DEPTH="4096",
                        NVSHMEM_DISABLE_P2P="0", DEEPEP_LL_NVLINK_PAIR_ONLY="1",
                        DEEPEP_LL_PCIE_P2P_MAX_TOKENS="0",
                        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="256",
                        SGLANG_DEEPEP_DISABLE_NVLINK_LOW_LATENCY="0")
        guard = {"EXPERIMENT_GUARD_LEASE_RECEIPT": "/tmp/not-a-real-lease"}
        with patch.dict(os.environ, {"SGLANG_SHOULD_NOT_LEAK": "1", "DEEPEP_UNREVIEWED": "1"}):
            for layout in ("ep8", "pp2ep4"):
                env = self.common.environment(self.configuration(layout), 1, guard)
                for key, wanted in (expected | guard).items():
                    self.assertEqual(env[key], wanted)
                self.assertNotIn("SGLANG_SHOULD_NOT_LEAK", env)
                self.assertNotIn("DEEPEP_UNREVIEWED", env)
                self.assertEqual(env["GLM_LAYOUT"], layout)

    def test_usage_rank_partition(self):
        self.assertEqual(self.common.expected_ranks("ep8", 1), ["TP4", "TP5", "TP6", "TP7"])
        self.assertEqual(self.common.expected_ranks("pp2ep4", 1),
                         ["PP1 TP0", "PP1 TP1", "PP1 TP2", "PP1 TP3"])

    def test_fixed_b8_p81920_prefixes_are_disjoint_and_page_aligned(self):
        seqs = self.workload.prefixes(81920)
        self.assertEqual([len(seq) for seq in seqs], [81920] * 8)
        self.assertEqual(sum(map(len, seqs)), 655360)
        self.assertEqual(len({tuple(seq[:64]) for seq in seqs}), 8)
        self.assertEqual([len(self.workload.page_hashes(seq)) for seq in seqs], [1280] * 8)
        self.assertTrue(all(self.workload.query(i, 1, True) != self.workload.query(i, 1, False)
                            for i in range(8)))
        for invalid in (0, 81921, 131072):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.workload.prefixes(invalid)

    def streamed_response(self, *, host=81920, device=0, outputs=128, first=1):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                for count in range(first, outputs + 1):
                    obj = dict(meta_info=dict(prompt_tokens=81921, cached_tokens=81920,
                                              cached_tokens_details=dict(host=host, device=device),
                                              completion_tokens=count))
                    yield b"data: " + json.dumps(obj).encode() + b"\n"
                yield b"data: [DONE]\n"

        return Response()

    def request_with_response(self, layout, response):
        captured = []

        def urlopen(req, timeout):
            captured.append(json.loads(req.data))
            self.assertEqual(timeout, 1200)
            return response

        barrier = type("Barrier", (), {"wait": lambda self: None})()
        with patch.object(self.measure.urllib.request, "urlopen", side_effect=urlopen):
            row = self.measure.request("http://not-a-server", self.configuration(layout),
                                       [0] * 81920, 3, "decode", False, barrier)
        return row, captured[0]

    def test_actual_http_helper_preserves_128_output_decode_and_cpu_cold_boundary(self):
        for layout in ("ep8", "pp2ep4"):
            row, payload = self.request_with_response(layout, self.streamed_response())
            self.assertEqual(len(payload["input_ids"]), 81921)
            self.assertEqual(payload["sampling_params"],
                             dict(temperature=0, ignore_eos=True, max_new_tokens=128))
            self.assertTrue(payload["stream"])
            self.assertEqual(row["prefix_tokens"], 81920)
            self.assertEqual(row["query_tokens"], 1)
            self.assertEqual(row["completion_tokens"], 128)
            self.assertEqual(row["first_stream_chunk_tokens"], 1)
            self.assertEqual(len(row["token_events"]), 128)
            if layout == "ep8":
                self.assertEqual(payload["data_parallel_rank"], 1)
            else:
                self.assertNotIn("data_parallel_rank", payload)

    def test_http_helper_rejects_gpu_hot_partial_output_or_buffered_first_token(self):
        for kwargs in (dict(host=0, device=81920), dict(host=81919),
                       dict(outputs=127), dict(first=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.request_with_response("pp2ep4", self.streamed_response(**kwargs))

    def test_percentiles_remain_labeled_as_eight_request_samples(self):
        stats = self.measure.stats(list(range(8)))
        self.assertEqual(stats["mean"], 3.5)
        self.assertEqual(stats["median"], 3.5)
        self.assertEqual(stats["p99"], 7)
        self.assertEqual(stats["p99_estimator"], "nearest_rank_N8")

    def test_formal_measurement_retains_cold_command_and_h2d_timing(self):
        tree = ast.parse((RUNTIME / "measure.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "measure_all")
        cold = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "command"
                and len(node.args) > 3 and isinstance(node.args[3], ast.Constant)
                and node.args[3].value == "cold"]
        self.assertTrue(cold, "Formal comparison must restore a GPU-empty/CPU-ready start")
        keyword_values = {}
        for node in ast.walk(function):
            if isinstance(node, ast.keyword) and isinstance(node.value, ast.Constant):
                keyword_values.setdefault(node.arg, []).append(node.value.value)
        for name in ("hicache_h2d_timed", "p99_is_eight_request_sample"):
            self.assertIn(True, keyword_values.get(name, []))
        for name in ("weights_loading_timed", "old_context_constructed_by_model",
                     "persistent_source_kv_bank"):
            self.assertIn(False, keyword_values.get(name, []))

    def test_no_profiler_or_fixed_token_shortcuts_in_formal_runtime(self):
        forbidden = ("forced_token_id", "balanced_fixed", "attribution_probe", "torch.cuda._sleep")
        for path in RUNTIME.glob("*.py"):
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            for word in forbidden:
                self.assertNotIn(word, source, f"Unexpected timing/routing shortcut in {path.name}")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "torch.profiler", path.name)
                elif isinstance(node, ast.Import):
                    self.assertNotIn("torch.profiler", [alias.name for alias in node.names], path.name)
                elif isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                    self.assertNotIn(name, ("torch.profiler.profile", "torch.profiler.kineto"), path.name)


class GraphAcceptanceTests(unittest.TestCase):
    """Synthetic receipt fixtures exercise acceptance, not GPU performance."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_graph_opt_acceptance_under_test", RUNTIME / "graph_checks.py")
        module = importlib.util.module_from_spec(spec)
        with no_gpu_imports():
            spec.loader.exec_module(module)
        cls.validator = staticmethod(module.validate_graph_receipts)

    def receipts(self, layout, outputs=128):
        expected = (outputs - 1) * (2 if layout == "pp2ep4" else 1)
        receipts = []
        for rank in range(8):
            # EP8 uses DP2: ranks 0..3 own even requests, 4..7 odd.
            # PP2 ranks own the two identical, disjoint B4 slot groups.
            request_groups = ([list(range(4)), list(range(4, 8))] if layout == "pp2ep4"
                              else [list(range(rank // 4, 8, 2))])
            groups = [dict(request_ids=[f"timed-decode-81920-{index}" for index in group],
                           forwards=outputs - 1, verified_replays=outputs - 1)
                      for group in request_groups]
            cohort = dict(passed=True, decode_forwards=expected, model_graph_true=expected,
                          model_graph_false=0, execute_successes=expected,
                          backend_replay_successes=expected, verified_local_b4_replays=expected,
                          exceptions=0, violations={}, batch_sizes={"4": expected},
                          replay_key_sizes={"4": expected}, request_groups=groups)
            captures = [dict(capture_bs=[4], max_bs=4, captured_req_width=1,
                             backend_type="FullCudaGraphBackend", captured_graph_sizes=[4],
                             captured_graph_count=1)]
            receipts.append(dict(rank=rank, passed=True,
                                 graph=dict(installed=True, passed=True,
                                            capture_configurations=captures,
                                            cohorts={"timed-decode-81920": cohort})))
        return receipts

    def cohort(self, receipts, rank=7):
        return receipts[rank]["graph"]["cohorts"]["timed-decode-81920"]

    def test_all_eight_ep8_ranks_require_127_actual_local_b4_replays(self):
        with no_gpu_imports():
            report = self.validator(self.receipts("ep8"), "ep8")
        self.assertTrue(report["passed"])
        self.assertTrue(report["graph_measured_from_actual_replay"])
        self.assertFalse(report["profiler_enabled"])
        self.assertEqual(report["local_capture_bs"], 4)
        self.assertEqual([row["rank"] for row in report["rank_proofs"]], list(range(8)))
        self.assertEqual([row["actual_decode_replays"] for row in report["rank_proofs"]], [127] * 8)
        self.assertTrue(all(len(row["request_groups"]) == 1 for row in report["rank_proofs"]))

    def test_all_eight_pp_ranks_require_254_replays_and_two_stable_b4_slots(self):
        with no_gpu_imports():
            report = self.validator(self.receipts("pp2ep4"), "pp2ep4")
        self.assertTrue(report["passed"])
        self.assertEqual([row["actual_decode_replays"] for row in report["rank_proofs"]], [254] * 8)
        for row in report["rank_proofs"]:
            self.assertEqual(row["local_batch_size"], 4)
            self.assertEqual([group["forwards"] for group in row["request_groups"]], [127, 127])
            self.assertEqual([group["verified_replays"] for group in row["request_groups"]], [127, 127])
            ids = [rid for group in row["request_groups"] for rid in group["request_ids"]]
            self.assertEqual(len(set(ids)), 8)

    def test_missing_duplicate_misordered_or_failed_rank_cannot_pass(self):
        original = self.receipts("ep8")
        missing_first = copy.deepcopy(original[1:])
        missing_last = copy.deepcopy(original[:-1])
        duplicate = copy.deepcopy(original)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        misordered = copy.deepcopy(original)
        misordered[0], misordered[1] = misordered[1], misordered[0]
        failed = copy.deepcopy(original)
        failed[7]["passed"] = False
        for label, receipts in (("missing_first", missing_first), ("missing_last", missing_last),
                                ("duplicate", duplicate), ("misordered", misordered), ("failed", failed)):
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                self.validator(receipts, "ep8")

    def test_eager_fallback_or_only_model_graph_flag_is_not_replay_proof(self):
        for layout in ("ep8", "pp2ep4"):
            for field, replacement in (("model_graph_false", 1), ("execute_successes", 0),
                                       ("backend_replay_successes", 0),
                                       ("verified_local_b4_replays", 0)):
                receipts = self.receipts(layout)
                self.cohort(receipts)[field] = replacement
                with self.subTest(layout=layout, field=field), self.assertRaises(RuntimeError):
                    self.validator(receipts, layout)

    def test_wrong_empty_or_uninstalled_capture_identity_cannot_pass(self):
        for field, replacement in (("capture_bs", [8]), ("max_bs", 8),
                                   ("backend_type", "BreakableCudaGraphBackend"),
                                   ("captured_graph_sizes", [4, 8]), ("captured_graph_count", 0)):
            receipts = self.receipts("pp2ep4")
            receipts[7]["graph"]["capture_configurations"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validator(receipts, "pp2ep4")
        for field, replacement in (("capture_configurations", []), ("installed", False), ("passed", False)):
            receipts = self.receipts("ep8")
            receipts[7]["graph"][field] = replacement
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validator(receipts, "ep8")

    def test_padded_or_partial_local_batch_and_replay_keys_are_rejected(self):
        for field, replacement in (("batch_sizes", {"8": 127}),
                                   ("batch_sizes", {"4": 126, "1": 1}),
                                   ("replay_key_sizes", {"8": 127}),
                                   ("replay_key_sizes", {"4": 126})):
            receipts = self.receipts("ep8")
            self.cohort(receipts)[field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaises(RuntimeError):
                self.validator(receipts, "ep8")

    def test_missing_pp_slot_or_overlapping_request_groups_are_rejected(self):
        for mutation in ("missing", "overlap", "b3", "short_slot", "unverified_slot"):
            receipts = self.receipts("pp2ep4")
            groups = self.cohort(receipts)["request_groups"]
            if mutation == "missing":
                groups.pop()
            elif mutation == "overlap":
                groups[1]["request_ids"][0] = groups[0]["request_ids"][0]
            elif mutation == "b3":
                groups[1]["request_ids"].pop()
            elif mutation == "short_slot":
                groups[1]["forwards"] -= 1
            else:
                groups[1]["verified_replays"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.validator(receipts, "pp2ep4")

    def test_127_outputs_cannot_be_accepted_as_128_complete_tokens(self):
        for layout in ("ep8", "pp2ep4"):
            receipts = self.receipts(layout, outputs=127)
            with self.subTest(layout=layout), self.assertRaises(RuntimeError):
                self.validator(receipts, layout, outputs=128)

    def test_exception_violation_or_missing_formal_cohort_is_rejected(self):
        for field, replacement in (("exceptions", 1), ("violations", {"unexpected_padding": 1}),
                                   ("passed", False)):
            receipts = self.receipts("pp2ep4")
            self.cohort(receipts)[field] = replacement
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validator(receipts, "pp2ep4")
        receipts = self.receipts("ep8")
        cohorts = receipts[7]["graph"]["cohorts"]
        cohorts["warm-decode-81920"] = cohorts.pop("timed-decode-81920")
        with self.assertRaises(RuntimeError):
            self.validator(receipts, "ep8")


if __name__ == "__main__":
    unittest.main()
