"""CPU checks of role log separation and strict actual-MTP receipt validation."""
import copy
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
import spec_proof
import load_logging

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

if __name__ == '__main__': unittest.main()
