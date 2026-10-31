import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import analyze_witness as analysis
import torch


def snapshot(rank, slot, cohort):
    candidates = torch.ones((4, 3), dtype=torch.int64)
    logits = torch.zeros((12, 16))
    logits[:, 1] = 2
    seq = torch.full((4,), 4097)
    return dict(rank=rank, slot=slot, microstep=100 + slot, verify_ordinal=0,
        cohort=cohort, tree_topk=1, draft_token_num=3,
        requests=[dict(request_index=i, committed_history_sha256=f"history-{i}") for i in range(slot * 4, slot * 4 + 4)],
        pre=dict(batch=dict(seq_lens=seq, seq_lens_cpu=seq.clone(), input_ids=candidates.flatten(),
                            out_cache_loc=torch.arange(12), req_pool_indices=torch.arange(4)),
                 verify=dict(draft_token=candidates.flatten(), positions=torch.arange(3).repeat(4),
                    custom_mask=torch.ones(12, dtype=torch.bool), retrieve_index=torch.arange(12).reshape(4, 3),
                    retrieve_next_token=torch.tensor([[1, 2, -1]]).repeat(4, 1),
                    retrieve_next_sibling=torch.full((4, 3), -1)), target_logits=logits),
        post=dict(predict=logits.argmax(-1).int(), accept_lens=torch.full((4,), 3),
                  accept_index=torch.arange(12).reshape(4, 3)))


def publish(run, rank, cohort, ordinal=0, drift=False):
    directory = run / "pp2ep4" / "witness"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{cohort}-decode-4096"
    rows = [snapshot(rank, slot, name) for slot in range(2)]
    if drift:
        rows[0]["pre"]["target_logits"][0, 0] = .001
    path = directory / f"rank{rank}_flush{ordinal:03d}.pt"
    torch.save(dict(schema=analysis.witness_gate.SCHEMA, rank=rank, rows=rows,
        synchronized_only_at_idle=True, performance_measurement=False), path)
    proof, _ = analysis.witness_gate.read_stable_payload(path)
    proof.update(passed=True, published_after_fsync_and_parse=True, flush_id=ordinal)
    path.with_suffix(".receipt.json").write_text(json.dumps(proof))
    return path


def run_fixture(root, name, drift=False):
    run = root / name
    (run / "task").mkdir(parents=True)
    (run / "pp2ep4").mkdir()
    (run / "config.json").write_text(json.dumps(dict(production_output_tokens=4, mode=name, prefix_lengths=[4096])))
    for path in (run / "task" / "results.json", run / "task" / "cleanup.json"):
        path.write_text(json.dumps(dict(passed=True)))
    zero = dict(installed=True, pending_snapshots=0, pending_gpu_bytes=0, total_selected_snapshots=0, flushed_cohorts=0)
    humming = dict(schema=analysis.humming_receipt.SCHEMA, kernels=[dict(kernel_id=i, gemm_type=kind,
        use_batch_invariant=True, use_stream_k=False, sm_version=80, block_shape=[16, 16, 16],
        warp_shape=[16, 16, 16], kernel_filename="fixture.cubin", kernel_name="fixture", source_sha256="0" * 64)
        for i, kind in enumerate(("dense", "grouped_masked"))])
    for cohort in analysis.COHORTS:
        receipts = []
        for rank in range(8):
            batches = [dict(pp_microstep=100 + slot, pp_slot_id=slot, mode="2", batch_size=4,
                request_ids=[f"{cohort}-decode-4096-{i}" for i in range(slot * 4, slot * 4 + 4)]) for slot in range(2)]
            receipts.append(dict(rank=rank, witness=copy.deepcopy(zero), humming=copy.deepcopy(humming), batches=batches))
        requests = [dict(request_index=i, output_ids=[1, 2, 3, 4], output_text="text") for i in range(8)]
        value = dict(requests=requests, native_cpu_ready_gpu_empty=receipts, all_rank_after=receipts)
        analysis.result_path(run, 4096, cohort).write_text(json.dumps(value))
    for cohort_index, cohort in enumerate(analysis.WITNESS_COHORTS):
        for rank in analysis.RANKS:
            publish(run, rank, cohort, cohort_index, drift=drift and rank == 5 and cohort_index == 0)
    return run


class AnalysisTests(unittest.TestCase):
    def test_missing_data_remains_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = analysis.analyze(Path(tmp) / "missing")
        self.assertEqual(value["baseline"]["config"]["status"], "missing")
        self.assertEqual(value["baseline"]["oracle_by_cohort_rank"]["witness"]["4"]["status"], "missing")
        self.assertEqual(value["baseline"]["output_exactness"]["timed_vs_repeat"]["status"], "missing")
        self.assertFalse(value["performance_comparison_claim_valid"])

    def test_complete_receipts_oracles_repeat_cross_tp_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = run_fixture(root, "baseline"), run_fixture(root, "candidate")
            value = analysis.analyze(a, b)
        for arm in ("baseline", "candidate"):
            for cohort in analysis.WITNESS_COHORTS:
                for rank in analysis.RANKS:
                    row = value[arm]["oracle_by_cohort_rank"][cohort][str(rank)]
                    self.assertEqual(row["counts"], {"pass": 2, "fail": 0, "inapplicable": 0})
                    self.assertTrue(row["payload_proof"]["parsed_as_weights_only_torch_payload"])
            self.assertEqual(value[arm]["within_repeat_by_rank"]["4"]["request_summary"]["matched"], 8)
            self.assertEqual(value[arm]["output_exactness"]["timed_vs_repeat"]["exact_token_id_requests"], 8)
            self.assertTrue(value[arm]["result_artifacts"]["witness"]["schedule_audit"]["passed"])
            self.assertTrue(value[arm]["result_artifacts"]["timed"]["humming_before"]["passed"])
            self.assertTrue(value[arm]["receipt_and_oracle_coverage"]["witness"]["all_four_tail_receipts_valid"])
            self.assertEqual(value[arm]["receipt_and_oracle_coverage"]["witness"]["observed_snapshot_counts"]["pass"], 8)
        cross = value["cross_arm"]["witness_by_cohort_rank"]["witness"]["4"]
        self.assertEqual(cross["batch_summary"]["classifications"], {"identical_logit_and_acceptance_witness": 2})
        self.assertEqual(value["cross_arm"]["output_exactness"]["timed"]["exact_token_id_requests"], 8)

    def test_tampered_payload_is_not_used_even_if_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = publish(run, 4, "witness")
            path.write_bytes(path.read_bytes() + b"tampered")
            value = analysis.Audit(4096).oracle(run, "witness", 4)
        self.assertEqual(value["status"], "invalid")
        self.assertIn("hash mismatch", value["error"])
        self.assertNotIn("snapshot_count", value)

    def test_cross_tp_logit_drift_is_reported_not_benign(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = run_fixture(Path(tmp), "baseline", drift=True)
            value = analysis.analyze(run)
        cross = value["baseline"]["cross_tp_by_cohort"]["witness"]["4_vs_5"]
        self.assertEqual(cross["batch_summary"]["classifications"]["target_logit_drift_same_argmax"], 1)
        self.assertFalse(cross["benign_numerical_branching_proven"])
        self.assertTrue(value["baseline"]["oracle_by_cohort_rank"]["witness"]["5"]["all_observed_snapshots_match_cpu_reference"])

    def test_output_exactness_reports_first_divergence_and_length(self):
        a = dict(status="valid", expected_tokens=4, value=dict(requests=[dict(request_index=i,
            output_ids=[1, 2, 3, 4], output_text="text") for i in range(8)]))
        b = copy.deepcopy(a)
        b["value"]["requests"][5]["output_ids"][2] = 7
        b["value"]["requests"][5]["output_text"] = "other"
        result = analysis.output_pair(a, b)
        self.assertEqual(result["exact_token_id_requests"], 7)
        self.assertEqual(result["requests"][5]["common_token_prefix"], 2)
        b["value"]["requests"][0]["output_ids"].pop()
        self.assertEqual(analysis.output_pair(a, b)["status"], "invalid")

    def test_humming_requires_all_eight_safe_loaded_kernel_receipts(self):
        with self.assertRaisesRegex(ValueError, "all eight"):
            analysis.humming_proof([dict(rank=rank) for rank in range(7)])
        with self.assertRaisesRegex(ValueError, "schema"):
            analysis.humming_proof([dict(rank=rank) for rank in range(8)])


if __name__ == "__main__":
    unittest.main()
