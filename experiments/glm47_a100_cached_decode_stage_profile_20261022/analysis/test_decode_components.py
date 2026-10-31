"""Synthetic CPU fixtures only; these are not experiment measurements."""
import unittest
from decode_components import component, partition, summarize


def row(start, end, category, forward=1, device="0"):
    return dict(interval=(start, end), category=category, forward_index=forward,
                device=device, kind="kernel")


class DecodePartitionTest(unittest.TestCase):
    def test_overlap_is_separate_and_total_closes(self):
        rows = [row(0, 10, "attention"), row(5, 15, "deep_ep.dispatch"),
                row(20, 30, "expert.routed")]
        out = partition(rows, (0, 30))
        self.assertEqual(out["attention"], .005)
        self.assertEqual(out["communication"], .005)
        self.assertEqual(out["component_overlap"], .005)
        self.assertEqual(out["no_traced_gpu_activity"], .005)
        self.assertAlmostEqual(sum(out.values()), .03)

    def test_shared_and_dense_are_ffn_but_router_is_other(self):
        self.assertEqual(component("dense_mlp"), "expert_and_dense_mlp")
        self.assertEqual(component("expert.shared"), "expert_and_dense_mlp")
        self.assertEqual(component("router"), "other_gpu")

    def test_selection_and_foreign_overlap_are_explicit(self):
        rows = [row(0, 10, "attention"), row(5, 12, "pp_transport_api", forward=None),
                row(20, 30, "expert.routed", forward=2)]
        sample = dict(forwards=[dict(forward_index=1, extend_index=9, detailed=True,
                                    metadata=dict(batch_size=4)),
                                dict(forward_index=2, extend_index=10, detailed=False)])
        out = summarize(rows, sample)
        self.assertEqual(out["directly_mapped_forward_count"], 1)
        self.assertEqual(out["forwards"][0]["foreign_overlapping_gpu_events"], 1)
        self.assertEqual(out["pp_boundary_kernel_event_count"], 1)
        self.assertAlmostEqual(out["pp_boundary_kernel_duration_sum_ms"], .007)


if __name__ == "__main__":
    unittest.main()
