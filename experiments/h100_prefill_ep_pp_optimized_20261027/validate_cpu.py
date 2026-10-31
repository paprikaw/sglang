"""Run existing scheduling checks against the actual frozen adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    run = args.run.absolute()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / "code"))
    gap = load("gap_schedule", run / "code/gap_schedule.py")
    existing = load("prefill_existing_checks", ROOT.parent / "h100_pp_lookahead_workloads_20261027/test_gap.py")
    admission = load("prefill_admission", run / "code/prefill_admission.py")

    class PrefillSubmission(unittest.TestCase):
        def test_local_ep_cohort_waits_for_all_and_preserves_parity(self):
            requests = [SimpleNamespace(origin_input_ids=[i]) for i in (5006,5000,5004,5002)]
            self.assertIsNone(admission.ordered_local(requests[:3],8,0))
            self.assertEqual([r.origin_input_ids[0] for r in admission.ordered_local(requests,8,0)],
                             [5000,5002,5004,5006])
            with self.assertRaises(RuntimeError):admission.ordered_local(requests,8,1)
            with self.assertRaises(RuntimeError):admission.ordered_local([requests[0]]*2,8,0)

        def test_large_prefill_retains_producer_event_and_submits_exactly_once(self):
            calls = []
            event, result = object(), object()
            batch = SimpleNamespace(num_tokens=4096, forward_mode="EXTEND")
            policy = [0]

            class Scheduler:
                ps = SimpleNamespace(pp_size=2)
                def _pp_topology(self): return None
                def _pp_is_last_rank(self): return False
                def _pp_launch_or_skip_batch(self, *args): return result, event
                def _pp_send_launched_proxy(self, b, r):
                    calls.append((b, r, self.launch_event))

            gap.install_early(Scheduler, lambda: policy[0])
            scheduler = Scheduler()
            scheduler._pp_launch_or_skip_batch(0, batch, None, None, None)
            self.assertFalse(calls)
            policy[0] = 1
            scheduler._pp_launch_or_skip_batch(0, batch, None, None, None)
            scheduler._pp_send_launched_proxy(batch, result)
            self.assertEqual(calls, [(batch, result, event)])
            self.assertEqual(scheduler._gap_early_proxy_submissions, 1)
            self.assertFalse(scheduler._gap_proxy_sent)

    suite = unittest.defaultTestLoader.loadTestsFromModule(existing)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(PrefillSubmission))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    for path in (run / "code").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    receipt = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                   scope="frozen adapter: existing ownership/schema/visibility tests plus large Prefill event/send check")
    (run / "cpu_qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
