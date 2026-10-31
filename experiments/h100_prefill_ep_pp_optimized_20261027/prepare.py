"""Freeze a cold-Prefill comparison using existing phase-compatible optimizations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "h100_pp_lookahead_workloads_20261027"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected source in {path.name}: {old!r}")
    path.write_text(text.replace(old, new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--focus", action="store_true", help="Recheck the noisy long B8 point and two short-input numerical boundaries")
    ap.add_argument("--numerical-only", action="store_true", help="PP-only short-input logprob replay with additional GPU allocator headroom")
    args = ap.parse_args()
    run = args.run.absolute()
    subprocess.run([sys.executable, str(PREVIOUS / "prepare.py"), "--run", str(run)], check=True,
                   stdout=subprocess.PIPE, text=True)
    code = run / "code"
    for name in ("prefill_benchmark.py", "prefill_admission.py"):
        shutil.copyfile(ROOT / name, code / name)
    adapter = code / "runtime_adapter.py"
    adapter.write_text(adapter.read_text() + '''
_prefill_original_install = install
def install():
    _prefill_original_install()
    import prefill_admission
    prefill_admission.install()
''')
    replace_once(code / "runner.py", "if __name__ == '__main__':",
                 "import prefill_benchmark\nprefill_benchmark.install(sys.modules[__name__])\n\n"
                 "if __name__ == '__main__':")
    # Add only bounded counters; retain the already tested event/send ownership.
    replace_once(code / "gap_schedule.py", "            self._gap_proxy_sent.append((batch,result))",
                 "            self._gap_proxy_sent.append((batch,result))\n"
                 "            self._gap_early_proxy_submissions = getattr(self, '_gap_early_proxy_submissions', 0) + 1")
    replace_once(code / "gap_schedule.py", "            policy=mode(), produced=slots.produced if slots else 0,",
                 "            pp_rank=self.ps.pp_rank,\n"
                 "            early_proxy_submissions=getattr(self, '_gap_early_proxy_submissions', 0),\n"
                 "            early_proxy_pending=len(getattr(self, '_gap_proxy_sent', [])),\n"
                 "            policy=mode(), produced=slots.produced if slots else 0,")
    cfg = json.loads((run / "config.json").read_text())
    cases = [dict(id=f"b{b}_q{q}_repeated", requests=b, new_input=q, family="repeated")
             for b in (8, 32) for q in (128, 512, 4096)]
    cases.append(dict(id="b32_q4096_diverse", requests=32, new_input=4096, family="diverse"))
    if args.focus:
        cases = [c for c in cases if c["id"] in ("b8_q4096_repeated", "b32_q128_repeated", "b32_q512_repeated")]
    if args.numerical_only:
        cases = [c for c in cases if c["id"] in ("b32_q128_repeated", "b32_q512_repeated")]
        shutil.copyfile(ROOT / "numerical_replay.py",code / "numerical_replay.py")
        replace_once(code / "runner.py", "if __name__ == '__main__':",
            "import numerical_replay\nnumerical_replay.install(sys.modules[__name__])\n\nif __name__ == '__main__':")
    cfg.update(port=27121, dist_port=27122, workloads=cases, output_tokens=1,
               gap_policies=[0, 1], expected_points=len(cases) * 3,
               cached_context=0, prefill_only=True, prefill_warmups=3,
               protocol="Cold Prefill only: no cached context, new input128/512/4096 -> output1; "
                        "global B8/B32 repeated plus B32 Q4096 diverse; five profiler-off cohorts/config. "
                        "Same full BF16 Qwen3.6-35B-A3B and shared Expert selector on both layouts; "
                        "T>16 retains native large-batch GEMM. EP2 AttentionDP2/TP1, normal DeepEP, "
                        "native shared-expert overlap ON, TBO OFF. PP2 native two slots, B/2 request cap, "
                        "policy0 versus existing policy1 early activation submission using native serialization. "
                        "4096-token Prefill budget per Attention instance. Decode-only next-slot lookahead "
                        "and small fixed-shape transport are inapplicable and remain unused. "
                        "Both layouts wait for their declared request groups before native packing. "
                        "PP policies counterbalanced; exact zero-cache/input/output audit, PP same-input "
                        "output equality, EP native output variability recorded, live early-send counters "
                        "and independent eager Prefill profiles.")
    if args.focus:
        cfg.update(repeats=10, samples_per_policy=10, collect_prefill_logprobs=True,
                   followup_reason="B8 Q4096 variability and short-input EP/PP first-token numerical differences")
        cfg["protocol"] = cfg["protocol"].replace("five profiler-off", "ten profiler-off")
        cfg["protocol"] += " Focused independent follow-up: ten formal cohorts/config; separate cold first-token top10 logprob diagnostics."
    if args.numerical_only:
        cfg.update(arm_order=["pp2"],layouts=[r for r in cfg["layouts"] if r["id"]=="pp2"],
                   prefill_numerical_only=True,validation_mem_fraction=.60,expected_points=len(cases),
                   protocol="Numerical diagnostic only, no performance samples: PP2 cold P0, Q128/Q512, B32, output1; "
                            "native and early-send policies, two cold first-token top10 logprob probes each. "
                            "Same weights/precision/inputs/native kernels; static memory fraction0.60 reserves space "
                            "for native logprob temporaries. Main performance data retain0.94. Compare against "
                            "already saved native EP probes from the focused run.")
        node=cfg["node"]
        cfg["model_cache"]["usage_logs"][node]=cfg["layouts"][0]["usage_logs"]
    save(run / "config.json", cfg)
    for path in code.glob("*.py"):
        path.chmod(0o600)
    spec = json.loads((run / "dashboard_spec.json").read_text())
    inputs = [d for d in spec["frozen_inputs"]
              if not Path(d["path"]).is_relative_to(code) and d["path"] != str(run / "config.json")]
    inputs += [descriptor(p) for p in sorted(code.glob("*.py"))] + [descriptor(run / "config.json")]
    spec.update(title="H100 cold Prefill: optimized EP2 versus early-send PP2", port=27120,
                frozen_inputs=list({d["path"]: d for d in inputs}.values()))
    spec["tasks"][0].update(expected=cfg["expected_points"],
                           label="EP2 / PP2 native / PP2 early activation; cold Prefill length scan")
    if args.numerical_only:
        spec["model_cache"]=cfg["model_cache"]
        spec["tasks"][0]["label"]="PP2 first-token numerical replay; no timed performance data"
    save(run / "dashboard_spec.json", spec)
    save(run / "task/progress.json", dict(experiment_id=run.name, task_id="comparison",
         stage="queued", valid_points=0, expected=cfg["expected_points"]))
    manifest = json.loads((run / "launch_manifest.json").read_text())
    manifest.update(private_adapters=[descriptor(p) for p in sorted(code.glob("*.py"))],
                    workloads=cases, phase="cold_prefill", gap_policies=[0, 1],
                    formal_repeats_per_policy=cfg["repeats"],
                    decode_lookahead_used=False, fixed_shape_fast_transport_used=False,
                    early_activation_uses_original_native_send=True,
                    generic_expert_selector_same_on_both=True,
                    expert_small_kernel_eligibility="local source T<=16; larger Prefill uses native kernel",
                    protocol=cfg["protocol"])
    save(run / "launch_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), expected_points=cfg["expected_points"],
                          workloads=cases, policies=cfg["gap_policies"]), indent=2))


if __name__ == "__main__":
    main()
