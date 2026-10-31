"""Ownership tests for composing local tail retirement with PP lookahead."""
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_five_layouts_hicache_20261028/runs/pp_171_r1/code')
ROOT = HERE.parent
BASE = HERE
sys.path.insert(0, str(BASE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load('gap_schedule', HERE / 'gap_schedule.py')
legacy = load('existing_lookahead_tests', Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/h100_pp_lookahead_workloads_20261027/test_gap.py'))


class LocalFeedback(unittest.TestCase):
    def fixture(self, policy, last):
        calls = []
        class Fake:
            ps = NS(pp_size=2)
            pp_loop_size = 2
            pp_group = NS(is_last_rank=last)
            device_module = NS(current_stream=lambda: NS(wait_event=lambda e: calls.append(('wait', e))))
            def _pp_topology(self): return None
            def _pp_requires_immediate_output_relay(self): return False
            def init_pp_loop_state(self): pass
            def _pp_send_output_to_next_stage(self, *args): calls.append(('native_send',)); return ['native-work']
            def _pp_recv_dict_from_prev_stage(self): calls.append(('native_recv',)); return 'native-output'
            def _pp_send_dict_to_next_stage(self, tensors, **kw):
                calls.append(('wire', tensors, kw)); return [('retained', tensors)]
            def _pp_send_recv_and_preprocess_output_tensors(self, first, nxt, batches, metadata, queue, previous):
                work = self._pp_send_output_to_next_stage(first, batches, queue, previous)
                output = self._pp_recv_dict_from_prev_stage() if batches[nxt] is not None else None
                return output, work
        m.install_local(Fake, lambda: policy, lambda batch: False)
        old = os.environ.pop('PP_GAP_SEPARATE_CHANNELS', None)
        try:
            obj = Fake()
            obj.init_pp_loop_state()
        finally:
            if old is not None: os.environ['PP_GAP_SEPARATE_CHANNELS'] = old
        return obj, calls

    @staticmethod
    def batch(rid):
        return NS(reqs=[NS(rid=rid)], forward_mode=NS(is_prebuilt=lambda: False))

    def test_tail_two_slots_drain_and_retain_native_payload(self):
        obj, calls = self.fixture(10, True)
        a, b = self.batch('a'), self.batch('b')
        initial, p0, p1 = (NS(tensors={'value': object()}) for _ in range(3))
        e_initial, e0, e1 = object(), object(), object()
        obj._gap_slots.put(1, e_initial, initial, ['b'])
        batches = [a, b]
        out, work = obj._pp_send_recv_and_preprocess_output_tensors(0, 1, batches, None, deque([(e0, p0)]), None)
        self.assertIs(out, initial.tensors)
        self.assertIs(work[0][1], p0.tensors)
        out, _ = obj._pp_send_recv_and_preprocess_output_tensors(1, 0, batches, None, deque([(e1, p1)]), None)
        self.assertIs(out, p0.tensors)
        batches[0] = None
        out, _ = obj._pp_send_recv_and_preprocess_output_tensors(0, 1, batches, None, deque(), None)
        self.assertIs(out, p1.tensors)
        self.assertEqual((obj._gap_slots.produced, obj._gap_slots.retired), (3, 3))
        self.assertFalse(obj._gap_slots.pending)
        self.assertFalse(any(c[0] == 'native_recv' for c in calls))
        self.assertIn(('wait', e0), calls)
        self.assertIn(('wait', e_initial), calls)
        self.assertIsNone(obj._gap_retiring)

    def test_head_skips_echo_but_still_receives_tail_output(self):
        obj, calls = self.fixture(10, False)
        out, work = obj._pp_send_recv_and_preprocess_output_tensors(0, 1, [self.batch('a'), self.batch('b')],
                                                                 None, deque(), object())
        self.assertEqual(out, 'native-output')
        self.assertEqual(work, [])
        self.assertEqual(obj._gap_echo_skipped, 1)
        self.assertEqual(calls, [('native_recv',)])

    def test_policy9_native_output_ring_stays_unchanged(self):
        obj, calls = self.fixture(9, True)
        out, work = obj._pp_send_recv_and_preprocess_output_tensors(0, 1, [self.batch('a'), self.batch('b')],
                                                                 None, deque(), object())
        self.assertEqual((out, work), ('native-output', ['native-work']))
        self.assertEqual(calls, [('native_send',), ('native_recv',)])

    def test_policy10_rejects_wrong_slot_request_ancestry(self):
        obj, _ = self.fixture(10, True)
        obj._gap_slots.put(1, None, NS(tensors={}), ['stale-request'])
        with self.assertRaisesRegex(RuntimeError, 'ancestry'):
            obj._pp_send_recv_and_preprocess_output_tensors(0, 1, [None, self.batch('new-request')], None, deque(), None)
        self.assertIsNone(obj._gap_retiring)

    def test_channel_warmup_retains_and_waits_before_synchronize(self):
        calls = []
        def submit(ops):
            calls.append(("submit", ops))
            return [NS(wait=lambda: calls.append(("wait",))) for _ in ops]
        distributed = NS(isend="isend", irecv="irecv", P2POp=lambda op, tensor, peer, **kw: (op, tensor, peer, kw), batch_isend_irecv=submit)
        fake_torch = NS(zeros=lambda *a, **kw: object(), empty_like=lambda t: object(),
                        distributed=distributed, cuda=NS(synchronize=lambda: calls.append(("sync",))))
        self.assertEqual(m.warm_pp_channels(fake_torch, ["activation", "tail_output", "head_echo"], NS(ranks=[0, 1], rank_in_group=1)), 3)
        self.assertEqual(sum(c[0] == "wait" for c in calls), 6)
        self.assertEqual(calls[-1], ("sync",))
        self.assertEqual([ops[0][3]["group"] for kind, *payload in calls if kind == "submit" for ops in payload], ["activation", "tail_output", "head_echo"])
        self.assertTrue(all(op[2] == 0 for c in calls if c[0] == "submit" for op in c[1]))

    def test_policy10_byte_is_explicit_opt_in(self):
        old = m._POLICY
        try:
            for value, expected in ((b'0', 0), (b'9', 9), (b'a', 10)):
                m._POLICY = value
                self.assertEqual(m.mode(), expected)
            m._POLICY = b'z'
            with self.assertRaises(RuntimeError): m.mode()
        finally:
            m._POLICY = old


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(legacy))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(LocalFeedback))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT / 'balance_warm_cpu_qualification.json').write_text(json.dumps(dict(
        passed=result.wasSuccessful(), tests=result.testsRun,
        scope='Lookahead and tail-local feedback ownership, exact send payload retention, both-slot drain and ancestry rejection'), indent=2) + '\n')
    if not result.wasSuccessful(): raise SystemExit(1)
