"""CPU-only coverage of exact local timeline union/exclusive semantics."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("local_pp_timeline", Path(__file__).with_name("local_pp_timeline.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimelinePartitionTest(unittest.TestCase):
    def test_compute_and_pp_overlap_are_not_double_counted(self):
        events = [
            {"cat": "kernel", "name": "A", "ts": 0, "dur": 10},
            {"cat": "kernel", "name": "PP", "ts": 5, "dur": 10},
        ]
        result = module.overlap_partition(events, 0, 20, {"A": "attention", "PP": "pp_communication"}, "MARK")
        self.assertEqual(result["category_sets_ms"], {
            "no_traced_gpu_activity": .005,
            "attention": .005,
            "attention + pp_communication": .005,
            "pp_communication": .005,
        })
        self.assertAlmostEqual(sum(result["category_sets_ms"].values()), result["window_ms"])
        self.assertEqual(result["pp_kernel_union_ms"], .010)
        self.assertEqual(result["pp_kernel_exclusive_ms"], .005)
        self.assertEqual(result["pp_kernel_overlapping_compute_ms"], .005)

    def test_clip_events_to_the_window(self):
        result = module.overlap_partition([
            {"cat": "kernel", "name": "PP", "ts": 0, "dur": 50},
            {"cat": "cpu_op", "name": "A", "ts": 0, "dur": 50},
        ], 10, 30, {"PP": "pp_communication", "A": "attention"}, "MARK")
        self.assertEqual(result["window_ms"], .02)
        self.assertEqual(result["pp_kernel_union_pct"], 100)
        self.assertEqual(result["compute_kernel_union_pct"], 0)

    def test_no_activity_is_explicit(self):
        result = module.overlap_partition([], 0, 100, {}, "MARK")
        self.assertEqual(result["category_sets_ms"], {"no_traced_gpu_activity": .1})
        self.assertEqual(result["no_traced_gpu_activity_pct"], 100)


if __name__ == "__main__":
    unittest.main()
