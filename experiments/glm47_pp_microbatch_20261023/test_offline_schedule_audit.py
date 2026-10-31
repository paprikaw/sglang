import copy
import unittest

from offline_schedule_audit import audit_point, output_parity


def point():
    groups = [[f"timed-decode-4096-{i}" for i in range(start, start + 4)] for start in (0, 4)]
    batches = []
    for step in range(100, 108):
        slot = step % 2
        rids = groups[slot][:3] if step >= 106 else groups[slot]
        batches.append(dict(pp_microstep=step, pp_slot_id=slot, mode="1" if step < 102 else "2",
                            batch_size=len(rids), request_ids=rids, extend_num_tokens=260))
    requests = [dict(request_index=i, output_ids=[1, 2, 3], output_text="abc") for i in range(8)]
    return dict(prefix_tokens_per_request=4096, output_tokens_per_request=3, requests=requests,
                all_rank_after=[dict(rank=rank, batches=copy.deepcopy(batches)) for rank in range(8)])


class OfflineAuditTests(unittest.TestCase):
    def test_exact_two_slots_and_monotonic_retirement(self):
        self.assertTrue(audit_point(point())["passed"])

    def test_rank_step_mismatch(self):
        value = point()
        value["all_rank_after"][7]["batches"].pop()
        with self.assertRaisesRegex(ValueError, "different logical"):
            audit_point(value)

    def test_slot_migration_rejected(self):
        value = point()
        value["all_rank_after"][0]["batches"][2]["request_ids"][0] = "timed-decode-4096-4"
        with self.assertRaisesRegex(ValueError, "migrated"):
            audit_point(value)

    def test_retired_rid_cannot_reappear(self):
        value = point()
        rows = value["all_rank_after"][0]["batches"]
        rows[4]["request_ids"] = rows[4]["request_ids"][:3]
        rows[4]["batch_size"] = 3
        rows[6]["request_ids"] = rows[2]["request_ids"][:]
        rows[6]["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "retired"):
            audit_point(value)

    def test_same_step_ordered_rid_mismatch(self):
        value = point()
        value["all_rank_after"][6]["batches"][2]["request_ids"].reverse()
        with self.assertRaisesRegex(ValueError, "composition mismatch"):
            audit_point(value)

    def test_output_parity_reports_first_difference(self):
        left, right = point(), point()
        self.assertTrue(output_parity(left, right)["passed"])
        right["requests"][3]["output_ids"][1] = 9
        result = output_parity(left, right)
        self.assertFalse(result["passed"])
        self.assertEqual(result["requests"][3]["common_token_prefix"], 1)
        self.assertEqual(result["requests"][3]["first_differing_token_ids"], [2, 9])


if __name__ == "__main__":
    unittest.main()
