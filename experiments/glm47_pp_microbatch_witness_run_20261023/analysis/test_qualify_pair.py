"""Small fail-closed regression tests; no model, torch or witness payload reads."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("qualify_pair", Path(__file__).with_name("qualify_pair.py"))
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def outputs():
    return dict(status="valid", request_count=8, exact_token_id_requests=8,
                exact_text_requests=8, exact_ids_and_text=True,
                requests=[dict(request_index=i, token_ids_equal=True, text_equal=True,
                               lengths=[64, 64]) for i in range(8)])


def comparison():
    return dict(status="valid",
        batch_summary=dict(matched=32, unmatched_left=0, unmatched_right=0,
            classifications={"identical_logit_and_acceptance_witness": 32}),
        request_summary=dict(matched=128, unmatched_left=0, unmatched_right=0,
            classifications={"identical_request_logit_and_acceptance_witness": 128},
            first_nonidentical_observation_per_request={}))


def oracle():
    return dict(status="valid", snapshot_count=32, counts=dict(pass_=32),
        all_observed_snapshots_match_cpu_reference=True,
        snapshots=[dict(oracle_status="pass", oracle=dict(passed=True,
            reference_applicable=True, accepted_indices_equal=True)) for _ in range(32)],
        payload_proof=dict(rows=32, request_indices=list(range(8)), slots=[0, 1],
            synchronized_only_at_idle=True, parsed_as_weights_only_torch_payload=True,
            performance_measurement=False))


class QualificationTests(unittest.TestCase):
    def test_exact_outputs(self):
        q.exact_outputs(outputs(), "test")

    def test_text_difference_rejected(self):
        value = outputs()
        value["requests"][3]["text_equal"] = False
        with self.assertRaises(ValueError):
            q.exact_outputs(value, "test")

    def test_short_output_rejected(self):
        value = outputs()
        value["requests"][3]["lengths"] = [63, 63]
        with self.assertRaises(ValueError):
            q.exact_outputs(value, "test")

    def test_exact_comparison(self):
        q.exact_comparison(comparison(), "test")

    def test_padding_or_metadata_difference_not_qualified(self):
        value = comparison()
        value["batch_summary"]["classifications"] = {
            "identical_logit_and_acceptance_witness": 31,
            "identical_used_acceptance_unused_post_values_differ": 1}
        with self.assertRaises(ValueError):
            q.exact_comparison(value, "test")

    def test_unmatched_witness_rejected(self):
        value = comparison()
        value["request_summary"]["unmatched_right"] = 1
        with self.assertRaises(ValueError):
            q.exact_comparison(value, "test")

    def test_oracle_and_wrong_index_rejection(self):
        value = oracle()
        value["counts"] = {"pass": 32, "fail": 0, "inapplicable": 0}
        q.oracle(value, "test")
        value["snapshots"][0]["oracle"]["accepted_indices_equal"] = False
        with self.assertRaises(ValueError):
            q.oracle(value, "test")

    def test_inapplicable_oracle_rejected(self):
        value = oracle()
        value["counts"] = {"pass": 31, "fail": 0, "inapplicable": 1}
        with self.assertRaises(ValueError):
            q.oracle(value, "test")

    def test_duplicate_rank_rejected(self):
        value = dict(passed=True, ranks=[dict(rank=i) for i in range(8)])
        q.rank_proof(value, "test")
        value["ranks"][-1]["rank"] = 6
        with self.assertRaises(ValueError):
            q.rank_proof(value, "test")

    def test_n8_p99_is_maximum(self):
        value = q.stats(list(range(1, 9)))
        self.assertEqual(value["mean"], 4.5)
        self.assertEqual(value["median"], 4.5)
        self.assertEqual(value["p99"], 8)
        with self.assertRaises(ValueError):
            q.stats([1.0] * 7)


if __name__ == "__main__":
    unittest.main()
