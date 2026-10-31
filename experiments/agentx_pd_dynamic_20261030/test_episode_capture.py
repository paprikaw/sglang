"""CPU checks against the exact newly frozen replay implementation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import test_agentx_replay as legacy
from episode_capture import replay_with_failure_capture

RUN = ROOT/'runs/eight_gpu_pd_ep4_e7_t300_b7200_r1'
spec = importlib.util.spec_from_file_location('new_episode_replay', RUN/'code/agentx_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
legacy.replay = replay


class EpisodeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'partial.r1.json'
        self.endpoint = dict(id='fake', ep=1, role='colocated', host='unused', port=0)

    def save(self, path, data):
        path.write_text(json.dumps(data))

    def serve(self, p, d, key, tokens, output, lane):
        start = time.perf_counter()
        time.sleep(.003)
        return dict(started=start, finished=time.perf_counter(), ttft_ms=1.,
            tpot_ms=2., e2e_ms=5., output_ids=[1]*output, output_tokens=output, meta_info={})

    def run_replay(self, **kwargs):
        return replay_with_failure_capture(replay.replay, self.save, self.path,
            [legacy.fixture()], [self.endpoint], 'sticky_hash', 1, '', [10],
            kwargs.pop('health', lambda: None), namespace='test', idle_gap_cap_s=0, **kwargs)

    def test_success_keeps_full_graph_and_has_no_extra_file_write(self):
        with patch.object(replay, 'request_one', self.serve):
            result = self.run_replay(episode_timeout_s=7200)
        self.assertEqual(result['completed_requests'], 3)
        self.assertEqual(result['episode_timeout_s'], 7200)
        self.assertFalse(self.path.exists())

    def test_request_failure_preserves_collected_rows_without_complete_marker(self):
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('controlled request failure')
            return self.serve(*args)
        with patch.object(replay, 'request_one', fail_second):
            with self.assertRaisesRegex(RuntimeError, 'controlled request failure'):
                self.run_replay()
        data = json.loads(self.path.read_text())
        self.assertEqual(data['collected_requests'], 1)
        self.assertEqual(data['partial_requests'][0]['turn'], 0)
        self.assertFalse(data['complete'])
        self.assertFalse(data['passed'])
        self.assertNotIn('means', data)
        self.assertEqual(list(self.path.parent.glob('completed.*')), [])

    def test_finite_deadline_still_aborts_and_records_diagnostic(self):
        with patch.object(replay, 'request_one', self.serve):
            with self.assertRaisesRegex(TimeoutError, '0.001 seconds'):
                self.run_replay(episode_timeout_s=.001, health=lambda: time.sleep(.01))
        data = json.loads(self.path.read_text())
        self.assertFalse(data['complete'])
        self.assertEqual(data['error_type'], 'TimeoutError')

    def test_diagnostic_write_failure_preserves_original_exception(self):
        def fail(*args, partial_rows, **kwargs):
            partial_rows.append({'diagnostic': True})
            raise ValueError('original failure')
        def bad_save(*args):
            raise OSError('controlled diagnostic write failure')
        with self.assertRaisesRegex(ValueError, 'original failure') as caught:
            replay_with_failure_capture(fail, bad_save, self.path)
        self.assertIn('diagnostic write failure', caught.exception.__notes__[0])


if __name__ == '__main__':
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(EpisodeCaptureTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(legacy.ReplayTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
