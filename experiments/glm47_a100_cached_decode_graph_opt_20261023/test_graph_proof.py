"""Pure CPU behavioral tests: no torch/SGLang imports or synthetic GPU claims."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS

SOURCE = Path(__file__).parent / "runtime" / "graph_proof.py"


class Mode:
    def __init__(self, decode=True): self.decode = decode
    def is_decode(self): return self.decode


class Key:
    def __init__(self, size): self.size = size


class GraphProofTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("graph_proof_cpu_test", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

        class Backend:
            def __init__(self): self.key = Key(4); self._graphs = {self.key: object()}
            def replay(self, shape_key, static_forward_batch, **kwargs):
                if getattr(static_forward_batch, "fail", False): raise ValueError("original error")
                return static_forward_batch

        class GraphRunner:
            def __init__(self):
                self.backend = Backend(); self.capture_bs = [4]; self.max_bs = 4
                self.captured_req_width = 1; self.bs = 4; self._replay_graph_key = self.backend.key
            def execute(self, forward_batch, pp_proxy_tensors=None):
                return self.backend.replay(self._replay_graph_key, forward_batch)

        class ModelRunner:
            def __init__(self): self.graph = GraphRunner(); self.eager = False; self.flag = True
            def forward(self, forward_batch, skip_attn_backend_init=None, pp_proxy_tensors=None, **kwargs):
                if not self.eager: self.graph.execute(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
                return NS(logits_output=forward_batch, can_run_graph=self.flag)

        self.Backend, self.GraphRunner, self.ModelRunner = Backend, GraphRunner, ModelRunner
        self.install_args = dict(model_runner_cls=ModelRunner, decode_graph_runner_cls=GraphRunner,
                                 full_backend_cls=Backend, parallel_getter=lambda: NS(
                                     world_rank=4, pp_rank=1, tp_rank=0, attn_dp_rank=0))
        self.module.install(**self.install_args)
        self.runner = ModelRunner()

    def batch(self, cohort="timed-decode-81920", indices=(0,1,2,3), size=None):
        return NS(forward_mode=Mode(), batch_size=len(indices) if size is None else size,
                  rids=[f"{cohort}-{i}" for i in indices])

    def test_actual_full_replay_per_cohort_and_two_slots(self):
        for cohort in ("warm-decode-81920", "timed-decode-81920"):
            for indices in ((0,1,2,3), (4,5,6,7)):
                for _ in range(3):
                    batch = self.batch(cohort, indices)
                    self.assertIs(self.runner.forward(batch).logits_output, batch)
        snap = self.module.snapshot()
        self.assertTrue(snap["passed"])
        self.assertEqual(snap["rank"]["world_rank"], 4)
        self.assertEqual(snap["capture_configurations"][0]["captured_graph_keys"], [
            {"size": 4, "stream_idx": None, "variant_label": None, "dsa_variant": None}])
        for row in snap["cohorts"].values():
            self.assertEqual(row["decode_forwards"], 6)
            self.assertEqual(row["verified_local_b4_replays"], 6)
            self.assertEqual(len(row["request_groups"]), 2)
            self.assertEqual(row["replay_key_sizes"], {"4": 6})

    def test_true_flag_without_execute_is_not_replay_proof(self):
        self.runner.eager = True
        self.runner.forward(self.batch())
        row = self.module.snapshot()["cohorts"]["timed-decode-81920"]
        self.assertFalse(row["passed"])
        self.assertEqual(row["verified_local_b4_replays"], 0)
        self.assertIn("backend_replay_count_not_one", row["violations"])

    def test_padded_b3_cannot_claim_real_b4(self):
        self.runner.forward(self.batch(indices=(0,1,2)))
        row = self.module.snapshot()["cohorts"]["timed-decode-81920"]
        self.assertFalse(row["passed"])
        self.assertIn("real_local_batch_not_four", row["violations"])

    def test_wrong_capture_config_is_rejected(self):
        self.runner.graph.capture_bs = [4,8]
        self.runner.forward(self.batch())
        row = self.module.snapshot()["cohorts"]["timed-decode-81920"]
        self.assertFalse(row["passed"])
        self.assertIn("capture_configuration_not_b4_width_one", row["violations"])

    def test_backend_key_mismatch_and_output_flag_are_rejected(self):
        self.runner.graph._replay_graph_key = Key(8)
        self.runner.flag = False
        self.runner.forward(self.batch())
        row = self.module.snapshot()["cohorts"]["timed-decode-81920"]
        self.assertFalse(row["passed"])
        self.assertIn("actual_backend_replay_key_not_four", row["violations"])
        self.assertIn("model_can_run_graph_not_true", row["violations"])

    def test_missing_mixed_and_capture_dummy_ids_are_not_real_proof(self):
        batches = [self.batch(), self.batch(), self.batch(), self.batch()]
        batches[0].rids = None
        batches[1].rids[-1] = "timed-other-3"
        batches[2].rids = ["capture-0", "capture-1", "capture-2", "capture-3"]
        batches[3].rids = ["timed-decode-81920-0"] * 4
        for batch in batches: self.runner.forward(batch)
        snap = self.module.snapshot()
        self.assertFalse(snap["passed"])
        self.assertEqual(snap["cohorts"], {})
        self.assertEqual(snap["ignored_decode_forwards"], 4)

    def test_prefill_not_counted_and_no_imported_device_runtime(self):
        batch = self.batch(); batch.forward_mode = Mode(False)
        before = set(sys.modules)
        self.runner.forward(batch)
        self.assertEqual(self.module.snapshot()["cohorts"], {})
        self.assertFalse(any(name == "torch" or name.startswith("sglang")
                             for name in set(sys.modules) - before))

    def test_original_exception_propagates_and_context_clears(self):
        batch = self.batch(); batch.fail = True
        with self.assertRaisesRegex(ValueError, "original error"): self.runner.forward(batch)
        self.runner.forward(self.batch(cohort="warm-decode-81920"))
        snap = self.module.snapshot()
        self.assertEqual(snap["cohorts"]["timed-decode-81920"]["exceptions"], 1)
        self.assertTrue(snap["cohorts"]["warm-decode-81920"]["passed"])
        self.assertIsNone(self.module._FRAME.get())

    def test_snapshot_detached_reset_and_install_idempotent(self):
        self.runner.forward(self.batch())
        snap = self.module.snapshot(reset=True)
        snap["capture_configurations"][0]["capture_bs"].append(8)
        self.assertEqual(self.module.snapshot()["cohorts"], {})
        self.assertEqual(self.module.snapshot()["capture_configurations"][0]["capture_bs"], [4])
        self.assertTrue(self.module.install(**self.install_args)["already_installed"])
        self.runner.forward(self.batch())
        self.assertEqual(self.module.snapshot()["cohorts"]["timed-decode-81920"]["decode_forwards"], 1)

    def test_execute_and_backend_batch_identities_must_match(self):
        # Exercise the execute identity check by supplying another actual batch
        # from within the original forward implementation, not a tensor proxy.
        self.runner.graph.execute = lambda batch, **kwargs: self.GraphRunner.execute(
            self.runner.graph, self.batch())
        self.runner.forward(self.batch())
        row = self.module.snapshot()["cohorts"]["timed-decode-81920"]
        self.assertIn("execute_forward_batch_identity_mismatch", row["violations"])
        self.assertIn("backend_forward_batch_identity_mismatch", row["violations"])


if __name__ == "__main__": unittest.main()
