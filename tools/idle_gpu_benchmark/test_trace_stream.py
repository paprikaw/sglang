"""CPU-only cursor durability, retry, provenance, and whole-trace tests."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location("idle_trace_stream", Path(__file__).with_name("trace_stream.py"))
_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_module)
TraceStream = _module.TraceStream


class TraceStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "trace.csv"
        self.cursor = self.root / "state" / "cursor.json"
        self.handles = []

    def tearDown(self):
        for handle in self.handles:
            handle.close()

    def write(self, rows):
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Timestamp", "Model", "Request tokens", "Response tokens"])
            writer.writerows(rows)

    def open(self, **kwargs):
        handle = TraceStream(self.source, self.cursor, **kwargs)
        self.handles.append(handle)
        return handle

    def test_peek_retry_does_not_write_or_advance_counts(self):
        self.write([(1, "ChatGPT", 10, 2), (1, "GPT-4", 20, 3), (3, "ChatGPT", 30, 4)])
        stream = self.open(batch_size=2)
        first = stream.peek_batch()
        self.assertFalse(self.cursor.exists())
        self.assertEqual(stream.state["committed_requests"], 0)
        self.assertEqual(first, stream.peek_batch())
        first["requests"][0]["prompt_tokens"] = -1
        self.assertEqual(stream.peek_batch()["requests"][0]["prompt_tokens"], 10)
        stream.commit()
        self.assertEqual(stream.state["committed_requests"], 2)
        self.assertEqual(stream.peek_batch()["requests"][0]["source_index"], 2)

    def test_restart_uncommitted_batch_replays_exact_same_rows(self):
        self.write([(n, "GPT-4", n + 1, 3) for n in range(7)])
        first = self.open(batch_size=2)
        first.peek_batch()
        first.commit()
        pending = first.peek_batch()
        first.close()
        resumed = self.open(batch_size=2)
        self.assertEqual(pending, resumed.peek_batch())
        self.assertEqual(resumed.state["last_source_index"], 1)

    def test_commit_private_atomic_cursor_contains_only_counts_and_source(self):
        self.write([(10, "GPT-4", 100, 2), (12, "ChatGPT", 200, 3)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        result = stream.commit()
        saved = json.loads(self.cursor.read_text())
        self.assertEqual(result, saved)
        self.assertEqual(stat.S_IMODE(self.cursor.stat().st_mode), 0o600)
        self.assertEqual(saved["dataset_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        for key in ("input_ids", "prompt", "response", "payload", "requests"):
            self.assertNotIn(key, saved)
        self.assertEqual(saved["committed_batches"], 1)
        self.assertEqual(saved["committed_source_rows"], 1)
        self.assertEqual(saved["last_source_index"], 0)
        self.assertGreater(saved["source_offset"], 0)

    def test_partial_end_of_trace_is_explicit_then_wrap_is_durable(self):
        self.write([(1, "GPT-4", 10, 2), (2, "GPT-4", 20, 3), (3, "GPT-4", 30, 4)])
        stream = self.open(batch_size=2)
        stream.peek_batch()
        stream.commit()
        last = stream.peek_batch()
        self.assertEqual([r["source_index"] for r in last["requests"]], [2])
        self.assertTrue(last["stream"]["end_of_trace"])
        self.assertEqual(last["stream"]["cycle_after_commit"], 1)
        stream.commit()
        self.assertEqual(stream.state["completed_cycles"], 1)
        self.assertEqual(stream.state["last_source_index"], -1)
        stream.close()
        resumed = self.open(batch_size=2)
        wrapped = resumed.peek_batch()
        self.assertEqual([r["source_index"] for r in wrapped["requests"]], [0, 1])
        self.assertEqual([r["source_cycle"] for r in wrapped["requests"]], [1, 1])
        resumed.commit()
        self.assertEqual(resumed.state["committed_requests"], 5)

    def test_exact_batch_eof_wrap_is_explicit_without_replaying_old_cycle(self):
        self.write([(1, "GPT-4", 10, 2), (2, "GPT-4", 20, 3)])
        stream = self.open(batch_size=2)
        stream.peek_batch()
        stream.commit()
        wrapped = stream.peek_batch()
        self.assertTrue(wrapped["stream"]["end_of_trace"])
        self.assertEqual(wrapped["stream"]["wraps_observed"], 1)
        self.assertEqual(wrapped["requests"][0]["source_cycle"], 1)
        stream.commit()
        self.assertEqual(stream.state["completed_cycles"], 1)
        self.assertEqual(stream.state["committed_requests"], 4)

    def test_full_length_positive_rows_preserve_length_and_natural_model_mix(self):
        self.write([(1, "ChatGPT", 17, 24000), (1, "GPT-4", 30000, 2000),
                    (2, "ChatGPT", 32760, 8)])
        batch = self.open().peek_batch()
        self.assertEqual([(r["prompt_tokens"], r["output_tokens"]) for r in batch["requests"]],
                         [(17, 24000), (30000, 2000), (32760, 8)])
        self.assertEqual([r["offset_s"] for r in batch["requests"]], [0, 0, 1])
        self.assertEqual([r["sample_id"] for r in batch["requests"]], ["source-0", "source-1", "source-2"])
        self.assertEqual(batch["summary"]["models"], {"ChatGPT": 2, "GPT-4": 1})
        self.assertIsNone(batch["selection"]["max_output_tokens"])
        self.assertIsNone(batch["selection"]["max_prompt_tokens"])

    def test_exclusions_are_durable_only_after_successful_stage_commit(self):
        self.write([(1, "ChatGPT", 10, 0), (2, "GPT-4", "bad", 2),
                    (3, "GPT-4", 10, 1), (4, "GPT-4", 32768, 2),
                    (5, "GPT-4", 0, 2), (6, "GPT-4", 10, 2)])
        stream = self.open()
        batch = stream.peek_batch()
        expected = {"failed_or_nonpositive_output": 1, "malformed_numeric": 1,
                    "below_min_output_tokens": 1, "above_max_total_tokens": 1,
                    "nonpositive_prompt": 1}
        self.assertEqual(batch["stream"]["excluded_rows"], expected)
        self.assertEqual(stream.state["exclusion_counts"], {})
        self.assertEqual(stream.peek_batch(), batch)
        stream.commit()
        self.assertEqual(stream.state["exclusion_counts"], expected)
        self.assertEqual(stream.state["committed_source_rows"], 6)
        self.assertEqual(stream.state["committed_requests"], 1)

    def test_resume_handles_utf8_and_quoted_multiline_source_rows(self):
        self.write([(1, "GPT-4\n中文", 10, 2), (2, "ChatGPT", 20, 3)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        stream.commit()
        stream.close()
        resumed = self.open(batch_size=1)
        batch = resumed.peek_batch()
        self.assertEqual(batch["requests"][0]["source_index"], 1)
        self.assertEqual(batch["requests"][0]["source_model"], "ChatGPT")

    def test_backward_timestamps_raise_and_leave_durable_cursor_unchanged(self):
        self.write([(10, "GPT-4", 10, 2), (5, "GPT-4", 10, 2)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        stream.commit()
        before = self.cursor.read_bytes()
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "moved backward"):
                stream.peek_batch()
            self.assertEqual(self.cursor.read_bytes(), before)

    def test_backward_timestamp_in_filtered_row_is_not_silently_ignored(self):
        self.write([(10, "GPT-4", 10, 2), (5, "GPT-4", 10, 0)])
        with self.assertRaisesRegex(ValueError, "moved backward"):
            self.open().peek_batch()

    def test_expected_source_hash_checked_before_replay(self):
        self.write([(1, "GPT-4", 10, 2)])
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(self.open(expected_source_sha256=digest).sha256, digest)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.open(expected_source_sha256="0" * 64)

    def test_changed_source_and_changed_resume_filter_fail_closed(self):
        self.write([(1, "GPT-4", 10, 2), (2, "GPT-4", 20, 3)])
        stream = self.open(batch_size=1)
        stream.peek_batch()
        stream.commit()
        with self.assertRaisesRegex(ValueError, "filters differ"):
            self.open(min_output_tokens=3)
        self.source.write_text(self.source.read_text() + "3,GPT-4,30,4\n")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            stream.peek_batch()
        with self.assertRaisesRegex(ValueError, "different source SHA256"):
            self.open()

    def test_no_eligible_rows_raises_instead_of_infinite_wrap_or_consumption(self):
        self.write([(1, "GPT-4", 10, 0), (2, "GPT-4", 10, 1)])
        stream = self.open()
        with self.assertRaisesRegex(ValueError, "No eligible rows"):
            stream.peek_batch()
        self.assertFalse(self.cursor.exists())
        self.assertEqual(stream.state["committed_requests"], 0)

    def test_future_context_limit_can_be_disabled_explicitly(self):
        self.write([(1, "GPT-4", 100000, 90000)])
        batch = self.open(max_total_tokens=None).peek_batch()
        self.assertEqual(batch["requests"][0]["output_tokens"], 90000)

    def test_commit_without_pending_is_rejected(self):
        self.write([(1, "GPT-4", 10, 2)])
        with self.assertRaisesRegex(RuntimeError, "no pending"):
            self.open().commit()


if __name__ == "__main__":
    unittest.main()
