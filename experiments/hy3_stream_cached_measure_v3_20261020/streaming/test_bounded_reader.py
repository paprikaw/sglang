import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from bounded_reader import Cancelled, ReceiptJournal, SafetensorsReader, SharedCredits, StagingArena, verify_file


def _rank_reservation(path, limit, amount, ready, release):
    credits = SharedCredits(path, limit)
    arena = StagingArena(credits, amount, rank_limit_bytes=amount)
    ready.set()
    release.wait(5)
    arena.close()


class BoundedReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.credits = SharedCredits(self.root / "credits.json", 64)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, header=None, data=b"abcdefghijklmnop"):
        if header is None:
            header = {"first": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
                      "second": {"dtype": "I32", "shape": [2], "data_offsets": [8, 16]}}
        raw = json.dumps(header).encode()
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        return path, verify_file(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_offsets_byte_identity_reuse_and_receipts(self):
        path, proof = self.model()
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        reader = SafetensorsReader(proof, arena, receipts_path=self.root / "receipts.jsonl")
        buffer_id = id(arena.buffer)
        sync = []
        for name, expected in [("first", b"abcdefgh"), ("second", b"ijklmnop")]:
            with reader.tensor(name, synchronize=lambda: sync.append(name)) as item:
                self.assertEqual(bytes(item.view), expected)
                self.assertEqual(os.pread(reader.fd, 8, item.source_offset), expected)
                self.assertEqual(id(arena.buffer), buffer_id)
            self.assertEqual(self.credits.snapshot()["used_bytes"], 16)
        self.assertEqual(sync, ["first", "second"])
        rows = [json.loads(line) for line in (self.root / "receipts.jsonl").read_text().splitlines()]
        self.assertEqual([row["status"] for row in rows], ["consumed", "consumed"])
        self.assertNotIn("payload", rows[0])
        self.assertEqual(sum(row["bytes"] for row in rows), 16)
        for row in rows:
            self.assertGreaterEqual(row["read_elapsed_s"], 0)
            self.assertGreaterEqual(row["consumption_sync_elapsed_s"], 0)
        reader.close()
        arena.close()
        self.assertEqual(self.credits.snapshot()["used_bytes"], 0)

    def test_exception_syncs_before_reuse_and_cancel(self):
        _, proof = self.model()
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        reader = SafetensorsReader(proof, arena)
        sync = []
        with self.assertRaisesRegex(ValueError, "consumer"):
            with reader.tensor("first", synchronize=lambda: sync.append(True)):
                raise ValueError("consumer")
        self.assertEqual(sync, [True])
        with self.assertRaises(Cancelled):
            with reader.tensor("second", cancel=lambda: True):
                self.fail("Cancelled source read executed")
        reader.close()
        arena.close()

    def test_journal_writes_every_record_and_syncs_at_shard_boundary(self):
        _, proof = self.model()
        path = self.root / "journal.jsonl"
        journal = ReceiptJournal(path)
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        reader = SafetensorsReader(proof, arena, receipts_path=journal)
        real_sync = os.fsync
        with patch("bounded_reader.os.fsync", wraps=real_sync) as sync:
            for name in ("first", "second"):
                with reader.tensor(name, synchronize=lambda: None):
                    pass
            self.assertEqual(sync.call_count, 0)
            # No buffering of metadata until the end: both records are already
            # observable, even before their shard-boundary durability sync.
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["tensor"] for row in rows], ["first", "second"])
            self.assertTrue(all(row["status"] == "consumed" for row in rows))
            reader.close()
            self.assertEqual(sync.call_count, 1)
            journal.close()
            self.assertEqual(sync.call_count, 2)
            journal.close()
            self.assertEqual(sync.call_count, 2)
        arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            journal.append({"status": "late"})

    def test_journal_preserves_failed_consumer_and_rejects_reuse(self):
        _, proof = self.model()
        path = self.root / "journal.jsonl"
        journal = ReceiptJournal(path)
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        reader = SafetensorsReader(proof, arena, receipts_path=journal)
        with self.assertRaisesRegex(ValueError, "consumer"):
            with reader.tensor("first", synchronize=lambda: None):
                raise ValueError("consumer")
        reader.close()
        journal.close()
        self.assertEqual(json.loads(path.read_text())["status"], "consumer_failed")
        with self.assertRaisesRegex(ValueError, "fresh empty"):
            ReceiptJournal(path)
        arena.close()

    def test_journal_rejects_competing_writer_and_oversized_metadata(self):
        path = self.root / "journal.jsonl"
        journal = ReceiptJournal(path)
        try:
            with self.assertRaises(BlockingIOError):
                ReceiptJournal(path)
            with self.assertRaisesRegex(ValueError, "bounded metadata"):
                journal.append({"unexpected": "x" * (64 * 1024)})
            self.assertEqual(path.stat().st_size, 0)
        finally:
            journal.close()

    def test_sync_failure_poison_retains_credit(self):
        _, proof = self.model()
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        reader = SafetensorsReader(proof, arena)
        def fail():
            raise RuntimeError("CUDA synchronization failed")
        with self.assertRaisesRegex(RuntimeError, "synchronization"):
            with reader.tensor("first", synchronize=fail):
                pass
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            with reader.tensor("second"):
                pass
        with self.assertRaisesRegex(RuntimeError, "Unsafe arena"):
            arena.close()
        self.assertEqual(self.credits.snapshot()["used_bytes"], 16)
        reader.close()

    def test_header_bounds_overlap_and_mutation_rejected(self):
        for header in [
            {"x": {"dtype": "BF16", "shape": [20], "data_offsets": [0, 40]}},
            {"x": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
             "y": {"dtype": "BF16", "shape": [4], "data_offsets": [4, 12]}},
        ]:
            _, proof = self.model(header)
            arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
            with self.assertRaises(ValueError):
                SafetensorsReader(proof, arena)
            arena.close()
        path, proof = self.model()
        path.write_bytes(path.read_bytes() + b"x")
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            SafetensorsReader(proof, arena)
        arena.close()

    def test_rank_limit_exported_view_and_bad_hash(self):
        with self.assertRaises(ValueError):
            StagingArena(self.credits, 17, rank_limit_bytes=16)
        path, _ = self.model()
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            verify_file(path, "0" * 64)
        arena = StagingArena(self.credits, 16, rank_limit_bytes=16)
        view = memoryview(arena.buffer)
        with self.assertRaises(BufferError):
            arena.close()
        self.assertEqual(self.credits.snapshot()["used_bytes"], 16)
        view.release()
        arena.close()

    def test_two_processes_joint_quota_and_cancel(self):
        ctx = multiprocessing.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        child = ctx.Process(target=_rank_reservation,
                            args=(str(self.root / "credits.json"), 64, 40, ready, release))
        child.start()
        try:
            self.assertTrue(ready.wait(5))
            self.assertEqual(self.credits.snapshot()["used_bytes"], 40)
            with self.assertRaises(TimeoutError):
                StagingArena(self.credits, 32, rank_limit_bytes=32, timeout_s=0)
            token = self.credits.reserve(24, timeout_s=0)
            self.assertEqual(self.credits.snapshot()["used_bytes"], 64)
            self.credits.release(token)
            with self.assertRaises(Cancelled):
                self.credits.reserve(1, cancel=lambda: True)
        finally:
            release.set()
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join()
        self.assertEqual(child.exitcode, 0)
        self.assertEqual(self.credits.snapshot()["used_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
