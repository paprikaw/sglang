import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
import torch
import witness_gate


def publish(root,rank,cohort='witness-decode-4096'):
    path=root/f'rank{rank}_flush000.pt'
    rows=[dict(rank=rank,slot=slot,cohort=cohort,verify_ordinal=0,
        requests=[dict(request_index=i) for i in range(slot*4,(slot+1)*4)],
        pre=dict(target_logits=torch.zeros(1,2))) for slot in range(2)]
    torch.save(dict(schema=witness_gate.SCHEMA,rank=rank,rows=rows,
        synchronized_only_at_idle=True,performance_measurement=False),path)
    proof,_=witness_gate.read_stable_payload(path)
    proof.update(passed=True,published_after_fsync_and_parse=True,flush_id=0)
    receipt=path.with_suffix('.receipt.json');receipt.write_text(json.dumps(proof))
    return path,receipt


class WitnessGateTests(unittest.TestCase):
    def test_four_tail_hashes_and_cohorts_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in (4,5,6,7):publish(root,rank)
            value=witness_gate.wait_for_cohort(root,'witness-decode-4096',timeout=.1)
            self.assertTrue(value['passed']);self.assertEqual(value['tail_ranks'],[4,5,6,7])
            self.assertTrue(value['all_tail_flushes_stably_parsed_and_hashed'])
            self.assertFalse(value['performance_measurement'])

    def test_file_existence_or_json_alone_is_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in (4,5,6):publish(root,rank)
            path,receipt=publish(root,7)
            path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                witness_gate.validate_receipt(receipt,'witness-decode-4096',7)
            with self.assertRaises(TimeoutError):
                witness_gate.wait_for_cohort(root,'witness-decode-4096',timeout=.01,poll_s=.005)

    def test_wrong_rank_cohort_and_unparsed_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,receipt=publish(root,4)
            with self.assertRaisesRegex(RuntimeError,'rank/cohort'):
                witness_gate.validate_receipt(receipt,'witness-repeat-decode-4096',4)
            value=json.loads(receipt.read_text());value['published_after_fsync_and_parse']=False
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'rank/cohort'):
                witness_gate.validate_receipt(receipt,'witness-decode-4096',4)

    def test_production_requires_zero_snapshot_and_flush_counters(self):
        receipts=[dict(rank=rank,witness=dict(installed=True,pending_snapshots=0,
            pending_gpu_bytes=0,total_selected_snapshots=0,flushed_cohorts=0)) for rank in range(8)]
        self.assertTrue(witness_gate.validate_uninstrumented(receipts)['passed'])
        receipts[7]['witness']['total_selected_snapshots']=1
        with self.assertRaisesRegex(RuntimeError,'contaminated'):
            witness_gate.validate_uninstrumented(receipts)

    def test_witness_install_is_last_and_separate_cupti_never_called(self):
        root=Path(__file__).parent
        startup=ast.parse((root/'runtime/startup.py').read_text())
        self.assertNotIn('install_activity',ast.unparse(startup))
        install=next(n for n in startup.body if isinstance(n,ast.FunctionDef) and n.name=='install')
        self.assertIn('install_witness',ast.unparse(install.body[-2]))
        tree=ast.parse((root/'runtime/measure.py').read_text())
        measure=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure_all')
        calls=[n.func.id for n in ast.walk(measure) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertNotIn('profile_once',calls)
        self.assertIn('wait_for_cohort',calls)
        text=ast.unparse(measure)
        self.assertIn("performance_comparison_claim_valid=False",text)

if __name__=='__main__':unittest.main()
