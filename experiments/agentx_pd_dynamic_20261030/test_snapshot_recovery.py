import json
import mmap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tempfile

import cluster_runner_v8 as runner
import snapshot_transport as transport


class SnapshotRecoveryTests(unittest.TestCase):
    def test_measurements_survive_a_post_replay_snapshot_failure(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td); result={'completed_requests':1,'requests':[{'output_ids':[11,12]}]}
            def save(p,d):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d))
            base=SimpleNamespace(h=SimpleNamespace(save=save))
            with patch.object(runner,'forward_snapshot',side_effect=TimeoutError('injected')):
                with self.assertRaises(TimeoutError):runner.finish_episode(base,out,{},'arm',out,0,{},result)
            self.assertEqual(json.loads((out/'replay_completed.r1.json').read_text()),result)
            self.assertFalse((out/'formal.r1.json').exists())
            self.assertFalse((out/'completed.r1.json').exists())

    def test_success_preserves_counts_and_marks_complete(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);result={'completed_requests':1,'requests':[{'output_ids':[11,12]}]}
            base=SimpleNamespace(h=SimpleNamespace(save=lambda p,d:p.write_text(json.dumps(d))))
            before={'rank':{'shape':{},'counts':{'DECODE:b1':10}}};after={'rank':{'shape':{},'counts':{'DECODE:b1':12}}}
            with patch.object(runner,'forward_snapshot',return_value=after):runner.finish_episode(base,out,{},'arm',out,0,before,result)
            self.assertEqual(json.loads((out/'formal.r1.json').read_text()),result)
            self.assertTrue(json.loads((out/'completed.r1.json').read_text())['passed'])
            self.assertEqual(json.loads((out/'forwards.r1.json').read_text())['delta']['rank']['counts']['DECODE:b1'],2)

    def test_local_control_rejects_foreign_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'control';target=Path(td)/'target';target.write_bytes(b'x');p.symlink_to(target)
            with patch.object(transport,'local_path',return_value=p):
                with self.assertRaises(OSError):transport.open_control('source')
            self.assertEqual(target.read_bytes(),b'x')


if __name__=='__main__':unittest.main()
