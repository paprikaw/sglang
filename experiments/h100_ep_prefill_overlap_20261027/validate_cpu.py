"""Check the actual event/handle schedule without importing the GPU runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1] / ".engine-worktrees/qwen35b-prefill-ep-overlap-r3-20261027"
SOURCE = ENGINE / "python/sglang/srt/layers/moe/prefill_overlap.py"
spec = importlib.util.spec_from_file_location("overlap_schedule", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Ownership(unittest.TestCase):
    def execute(self, overlap):
        events = []
        class Dispatcher:
            def __init__(self, name): self.name, self.stage = name, "idle"
            def dispatch_a(self, hidden, topk):
                assert self.stage == "idle"
                self.stage = "prepared"
                events.append(self.name + ".input_event")
            def dispatch_b(self):
                assert self.stage == "prepared"
                self.stage = "dispatched"
                events.append(self.name + ".dispatch")
                return self.name
            def combine_a(self, result):
                assert self.stage == "dispatched" and result == self.name + ".core"
                self.stage = "producer"
                events.append(self.name + ".producer_event")
            def combine_b(self):
                assert self.stage == "producer"
                self.stage = "idle"
                events.append(self.name + ".combine")
                return self.name + ".output"
        pair = [Dispatcher("A"), Dispatcher("B")]
        def core(name):
            events.append(name + ".core")
            return name + ".core"
        for _ in range(3):
            outputs = module.execute_two(pair, [(object(), object()), (object(), object())], core, overlap=overlap)
            self.assertEqual(outputs, ["A.output", "B.output"])
            self.assertTrue(all(d.stage == "idle" for d in pair))
        return events

    def test_overlap_producer_and_input_events(self):
        expected = ["A.input_event", "B.input_event", "A.dispatch", "A.core", "A.producer_event",
                    "B.dispatch", "B.core", "B.producer_event", "A.combine", "B.combine"]
        self.assertEqual(self.execute(True), expected * 3)

    def test_global_eligibility_and_forward_policy_latch(self):
        for rows in ([4096, 4096], [128, 512]): self.assertTrue(module.eligible_rows(rows))
        for rows in ([4096, 0], [0, 4096], [4096, 16], [16, 4096], []): self.assertFalse(module.eligible_rows(rows))
        selected = 2
        token = module._FORWARD_POLICY.set(selected)
        selected = 3
        self.assertEqual(module._FORWARD_POLICY.get(), 2)
        module._FORWARD_POLICY.reset(token)
        self.assertEqual(module._FORWARD_POLICY.get(), 0)

    def test_serial_control_has_same_exact_once_ownership(self):
        expected = [f"{name}.{op}" for name in ("A", "B") for op in
                    ("input_event", "dispatch", "core", "producer_event", "combine")]
        self.assertEqual(self.execute(False), expected * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    args = ap.parse_args()
    if args.run:
        subprocess.run([sys.executable, str(ROOT.parent / "h100_prefill_ep_pp_optimized_20261027/validate_cpu.py"),
                        "--run", str(args.run)], check=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Ownership))
    compile(SOURCE.read_text(), str(SOURCE), "exec")
    receipt = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                   scope="Two dispatch handles, early input events, producer capture, exactly-once combine, three reuse cycles")
    destination = args.run / "overlap_cpu_qualification.json" if args.run else ROOT / "source_cpu_qualification.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == "__main__": main()
