"""CPU checks of role log separation and strict actual-MTP receipt validation."""
import copy
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import io
import json
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
import spec_proof
import load_logging
import workload
import ast
import measure

class HarnessTests(unittest.TestCase):
    def test_loader_filter_requires_actual_role(self):
        record = logging.LogRecord('loader', logging.INFO, '', 1, 'bank proof', (), None)
        target, draft = load_logging.RoleFilter('target'), load_logging.RoleFilter('draft')
        self.assertFalse(target.filter(record))
        token = load_logging._ROLE.set('draft')
        try:
            self.assertFalse(target.filter(record)); self.assertTrue(draft.filter(record))
        finally: load_logging._ROLE.reset(token)
        self.assertFalse(draft.filter(record))

    def receipts(self, layout):
        rows = []
        for rank in range(8):
            has_draft = layout == 'ep8' or rank >= 4
            row = dict(forwards={'target:TARGET_VERIFY': 42}, replays={'target': 42},
                       phases={'draft': 42} if has_draft else {}, batch_sizes={'4': 42}, widths={'target:12': 42})
            rows.append(dict(rank=rank, graph=dict(installed=True, cohorts={'timed-decode-81920': row})))
        return rows

    def test_proof_requires_real_replay_and_correct_draft_placement(self):
        for layout in ('ep8', 'pp2ep4'):
            receipts = self.receipts(layout)
            self.assertTrue(spec_proof.validate_receipts(receipts, layout)['passed'])
            broken = copy.deepcopy(receipts); broken[0]['graph']['cohorts']['timed-decode-81920']['replays'] = {}
            with self.assertRaises(RuntimeError): spec_proof.validate_receipts(broken, layout)
        broken = self.receipts('pp2ep4')
        broken[0]['graph']['cohorts']['timed-decode-81920']['phases']['draft'] = 1
        with self.assertRaises(RuntimeError): spec_proof.validate_receipts(broken, 'pp2ep4')

    def test_only_whole_cohort_is_recognized(self):
        make = lambda ids: SimpleNamespace(reqs=[SimpleNamespace(rid=x) for x in ids])
        self.assertEqual(spec_proof.cohort(make(['timed-decode-81920-0', 'timed-decode-81920-4'])), 'timed-decode-81920')
        self.assertIsNone(spec_proof.cohort(make(['timed-decode-81920-0', 'warm-decode-81920-4'])))
        self.assertIsNone(spec_proof.cohort(make(['capture_dummy'])))

    def test_semantic_inputs_exact_lengths_no_shared_page_and_same_query(self):
        seqs=workload.prefixes(81920)
        self.assertEqual(len(seqs),8)
        self.assertEqual(len({tuple(seq[:64]) for seq in seqs}),8)
        for index,seq in enumerate(seqs):
            self.assertEqual(len(seq),81920)
            self.assertEqual(seq+workload.query(index,1,False),workload.prompts()[index])
            self.assertEqual(workload.query(index,1,True),workload.query(index,1,False))
        self.assertEqual(workload.cached_prefix_length(81920),81856)
        with self.assertRaises(ValueError):workload.query(0,256,False)

    def test_cache_hook_does_not_construct_or_insert_host_kv(self):
        path=Path(__file__).parent/'runtime/scheduler_hooks.py'
        tree=ast.parse(path.read_text())
        forbidden={'fill_','insert_host'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):self.assertNotIn(node.attr,forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotEqual(node.func.id,'fill')

    def test_profile_uses_same_semantic_input_but_separate_request_ids_and_length(self):
        payloads=[]
        def response(req,timeout):
            payload=json.loads(req.data);payloads.append(payload)
            n=payload['sampling_params']['max_new_tokens']
            meta=dict(prompt_tokens=81921,cached_tokens=81856,
                      cached_tokens_details=dict(host=81856,device=0))
            lines=[]
            for count in [1,n]:
                lines.append(b'data: '+json.dumps(dict(meta_info=dict(meta,completion_tokens=count))).encode()+b'\n')
            return io.BytesIO(b''.join(lines)+b'data: [DONE]\n')
        prefix=workload.prefixes(81920)[0]
        with patch.object(measure.urllib.request,'urlopen',side_effect=response):
            production=measure.request('http://unused',dict(layout='ep8'),prefix,0,'decode',False,threading.Barrier(1))
            profile=measure.request('http://unused',dict(layout='ep8'),prefix,0,'decode',False,threading.Barrier(1),
                                    max_output_tokens=64,cohort_prefix='profile')
        self.assertEqual(payloads[0]['input_ids'],payloads[1]['input_ids'])
        self.assertEqual(payloads[0]['rid'],'timed-decode-81920-0')
        self.assertEqual(payloads[1]['rid'],'profile-decode-81920-0')
        self.assertEqual(production['completion_tokens'],128)
        self.assertEqual(profile['completion_tokens'],64)

    def test_profile_cannot_start_without_saved_production_result(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'cannot precede'):
                measure.profile_once('http://unused',Path(root),dict(layout='ep8'),81920,[],
                                     'p81920_decode')

    def test_profile_receipts_are_not_accepted_as_production_receipts(self):
        rows=self.receipts('ep8')
        for row in rows:
            value=row['graph']['cohorts'].pop('timed-decode-81920')
            row['graph']['cohorts']['profile-decode-81920']=value
        with self.assertRaises(RuntimeError):spec_proof.validate_receipts(rows,'ep8')
        self.assertTrue(spec_proof.validate_receipts(rows,'ep8',cohort_prefix='profile-')['passed'])

if __name__ == '__main__': unittest.main()
