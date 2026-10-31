"""CPU-only checks of cohort provenance, filtering, chronology, and privacy."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location("idle_burst_dataset", Path(__file__).with_name("dataset.py"))
dataset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dataset)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "trace.csv"

    def write(self, rows):
        with self.source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"])
            for timestamp, model, inputs, outputs in rows:
                writer.writerow([timestamp, model, inputs, outputs, 99999, "Conversation log"])

    def prepare(self, **kwargs):
        return dataset.prepare_workload(self.source, None, **kwargs)

    def test_lengths_are_filtered_without_clipping(self):
        self.write([(1, "GPT-4", 25, 3), (2, "GPT-4", 51, 3),
                    (3, "GPT-4", 25, 8), (4, "GPT-4", 49, 7)])
        result = self.prepare(max_prompt_tokens=50, max_output_tokens=7, max_total_tokens=55)
        self.assertEqual([(r["prompt_tokens"], r["output_tokens"]) for r in result["requests"]], [(25, 3)])
        self.assertEqual(result["dataset"]["exclusion_counts"], {
            "above_max_prompt_tokens": 1, "above_max_output_tokens": 1,
            "above_max_total_tokens": 1})

    def test_contiguous_eligible_window_preserves_true_bursts_and_original_rows(self):
        self.write([(10, "GPT-4", 10, 2), (10, "GPT-4", 12, 3),
                    (11, "ChatGPT", 20, 4), (15, "GPT-4", 14, 4)])
        result = self.prepare()
        self.assertEqual([r["offset_s"] for r in result["requests"]], [0, 0, 5])
        self.assertEqual([r["source_index"] for r in result["requests"]], [0, 1, 3])
        self.assertEqual([r["sample_id"] for r in result["requests"]], [0, 1, 2])
        self.assertEqual(result["summary"]["equal_timestamp_adjacent_pairs"], 1)
        self.assertEqual(result["summary"]["prompt_tokens"]["p50"], 12)
        self.assertEqual(result["summary"]["output_tokens"]["p95"], 3.9)

    def test_same_seed_exact_reproducibility_and_bounded_count(self):
        self.write([(n, "GPT-4", n + 1, 2) for n in range(100)])
        a = self.prepare(max_requests=8, seed=12)
        b = self.prepare(max_requests=8, seed=12)
        self.assertEqual(a, b)
        indices = [r["source_index"] for r in a["requests"]]
        self.assertEqual(indices, list(range(indices[0], indices[0] + 8)))
        self.assertEqual(a["selection"]["candidate_windows"], 93)
        starts = {self.prepare(max_requests=8, seed=n)["selection"]["selected_eligible_start"] for n in range(10)}
        self.assertGreater(len(starts), 1)

    def test_all_explicit_limits_can_be_disabled_for_future_full_context(self):
        self.write([(1, "GPT-4", 1000000, 100000)])
        result = self.prepare(max_prompt_tokens=None, max_output_tokens=None, max_total_tokens=None)
        self.assertEqual(result["requests"][0]["prompt_tokens"], 1000000)
        self.assertEqual(result["requests"][0]["output_tokens"], 100000)
        self.assertIsNone(result["selection"]["max_total_tokens"])

    def test_output_one_can_be_explicitly_enabled_but_default_keeps_tpot_defined(self):
        self.write([(1, "GPT-4", 10, 1), (2, "GPT-4", 20, 2)])
        self.assertEqual(len(self.prepare()["requests"]), 1)
        self.assertEqual(len(self.prepare(min_output_tokens=1)["requests"]), 2)

    def test_global_failed_counts_and_exclusion_accounting_are_auditable(self):
        self.write([(1, "ChatGPT", 3, 0), (2, "GPT-4", 3, -1),
                    (3, "ChatGPT", 3, 2), (4, "GPT-4", 0, 2),
                    (5, "GPT-4", 3, 1), (6, "GPT-4", 3, 2),
                    (7, "GPT-4", "bad", 2)])
        result = self.prepare()
        info = result["dataset"]
        self.assertEqual(info["zero_response_rows"], 1)
        self.assertEqual(info["negative_response_rows"], 1)
        self.assertEqual(info["rows"], sum(info["exclusion_counts"].values()) + info["eligible_rows"])
        self.assertEqual(info["exclusion_counts"]["failed_or_nonpositive_output"], 2)
        self.assertEqual(info["exclusion_counts"]["malformed_numeric"], 1)

    def test_all_models_preserves_original_source_model_and_counts(self):
        self.write([(1, "ChatGPT", 3, 2), (2, "GPT-4", 4, 5)])
        result = self.prepare(model_filter="ALL")
        self.assertEqual(result["summary"]["models"], {"ChatGPT": 1, "GPT-4": 1})
        self.assertEqual(result["dataset"]["eligible_models"], result["summary"]["models"])

    def test_no_eligible_rows_does_not_create_output(self):
        self.write([(1, "GPT-4", 3, 0)])
        output = self.root / "missing" / "workload.json"
        with self.assertRaisesRegex(ValueError, "No eligible"):
            dataset.prepare_workload(self.source, output)
        self.assertFalse(output.exists())

    def test_source_hash_and_private_metadata_only_output(self):
        self.write([(10, "GPT-4", 10, 2), (12, "GPT-4", 20, 3)])
        output = self.root / "new" / "workload.json"
        result = dataset.prepare_workload(self.source, output)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(result["dataset"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(json.loads(output.read_text()), result)
        self.assertFalse(result["contains_original_prompt_text"])
        for record in result["requests"]:
            self.assertNotIn("input_ids", record)
            self.assertNotIn("text", record)
            self.assertNotIn("payload", record)
        self.assertEqual(result["summary"]["total_prompt_tokens"], 30)
        self.assertEqual(result["summary"]["total_output_tokens"], 5)

    def test_unsorted_eligible_rows_rejected_instead_of_silently_reordering(self):
        self.write([(20, "GPT-4", 10, 2), (10, "GPT-4", 10, 2)])
        with self.assertRaisesRegex(ValueError, "not chronological"):
            self.prepare()

    def test_nan_timestamps_excluded_and_finite_json_saved(self):
        self.write([("nan", "GPT-4", 10, 2), (10, "GPT-4", 10, 2)])
        result = self.prepare()
        self.assertEqual(result["dataset"]["exclusion_counts"]["malformed_numeric"], 1)
        self.assertEqual(result["summary"]["trace_span_s"], 0)
        self.assertIsNone(result["summary"]["interarrival_s"]["p95"])

    def test_missing_required_columns_fail_clearly(self):
        self.source.write_text("Timestamp,Model\n10,GPT-4\n")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.prepare()

    def test_invalid_limits_fail_before_source_scan(self):
        self.write([(1, "GPT-4", 10, 2)])
        for options in ({"max_requests": 0}, {"max_requests": None},
                        {"max_output_tokens": -1}, {"seed": True},
                        {"min_output_tokens": 8, "max_output_tokens": 2},
                        {"min_prompt_tokens": 8, "max_prompt_tokens": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)


if __name__ == "__main__":
    unittest.main()
