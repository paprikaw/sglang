"""Freeze a fresh shared-GEMM and PP scheduling experiment; never edit a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FAIR = ROOT.parent / "h100_ep_pp_fair_expert_20261026"
SCHEDULE = ROOT.parent / "h100_decode_pp_opt_20261026"


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policies", type=int, nargs="+", default=[0,1,2,3])
    parser.add_argument("--arms", nargs="+", default=["ep2", "pp2"], choices=["ep2", "pp2"])
    parser.add_argument("--depth", type=int, choices=[0,1,2], default=0)
    parser.add_argument("--samples", type=int, choices=[3,5], default=3)
    args = parser.parse_args()
    run = args.run.absolute()
    if run.exists():
        raise RuntimeError("Run already exists; use a fresh immutable directory")
    if not args.policies or args.policies[0] != 0 or any(p not in (0,1,2,3,5,6,7,8,9) for p in args.policies):
        raise RuntimeError("Policies must start with baseline0; policy4 is rejected")
    if args.depth and any(p in (2,3) for p in args.policies):
        raise RuntimeError("Local feedback is qualified only for depth0")
    subprocess.run([sys.executable, str(FAIR / "prepare.py"), "--run", str(run),
                    "--mode", "triton_small", "--arms", *args.arms, "--repeats", str(args.samples)], check=True)
    cfg = json.loads((run / "config.json").read_text())
    spec = json.loads((run / "dashboard_spec.json").read_text())
    code = run / "code"
    for name in ("gap_schedule.py", "benchmark_gap.py"):
        shutil.copyfile(ROOT / name, code / name)
    for name in ("pp_schedule_opt.py", "pp_fixed_cohort.py", "pp_transfer_audit.py"):
        shutil.copyfile(SCHEDULE / name, code / name)
    adapter = code / "runtime_adapter.py"
    adapter.write_text(adapter.read_text() + '''
_shared_gemm_install = install
def install():
    _shared_gemm_install()
    if os.environ["EP2_PP2_ARM"] == "pp2":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
''')
    runner = code / "runner.py"
    text = runner.read_text()
    text = text.replace("expected=2 * len(cfg['layouts'])", "expected=cfg['expected_points']")
    marker = "if __name__ == '__main__':"
    assert text.count(marker) == 1
    runner.write_text(text.replace(marker, "import benchmark_gap\nbenchmark_gap.install(sys.modules[__name__])\n\n" + marker))
    for field, name, value in (("gap_policy_path", "gap_policy.bin", b"0"),
                               ("fixed_cohort_control", "fixed_cohort.bin", b"\x00"),
                               ("transfer_audit_control", "transfer_audit.bin", b"0")):
        cfg[field] = str(run / name)
        Path(cfg[field]).write_bytes(value)
        Path(cfg[field]).chmod(0o600)
    expected = sum((len(args.policies) if arm == "pp2" else 1) + 1 for arm in args.arms)
    cfg.update(port=26221, dist_port=26222, gap_policies=args.policies,
               pp_async_batch_depth=args.depth, pp_loop_slots=2+args.depth,
               samples_per_policy=args.samples,
               expected_points=expected, target_decode_tpot_ms=7.0,
               protocol=f"Same shared BF16 Triton Expert on EP2 and PP2, same GPU0/1; PP B4 even/odd groups; real GPUcached4096/new1/output128/B8. {args.samples} counterbalanced samples per cumulative scheduling policy; independent wire audits and actual Graph profiles.")
    for path in code.glob("*.py"):
        path.chmod(0o600)
    save(run / "config.json", cfg)
    inputs = [d for d in spec["frozen_inputs"] if not Path(d["path"]).is_relative_to(code) and d["path"] != str(run / "config.json")]
    inputs += [descriptor(path) for path in sorted(code.glob("*.py"))] + [descriptor(run / "config.json")]
    spec.update(title="H100 PP2 Graph-gap optimization with shared GEMM", port=26220,
                frozen_inputs=list({d["path"]: d for d in inputs}.values()))
    spec["tasks"][0].update(label="EP shared-GEMM reference; PP cumulative scheduling candidates", expected=expected)
    save(run / "dashboard_spec.json", spec)
    progress = json.loads((run / "task/progress.json").read_text())
    progress.update(expected=expected)
    save(run / "task/progress.json", progress)
    manifest = json.loads((run / "launch_manifest.json").read_text())
    manifest.update(private_adapters=[descriptor(path) for path in sorted(code.glob("*.py"))],
                    gap_policies=args.policies, pp_slots=2+args.depth, requests_per_slot=4,
                    generic_expert_optimization_applied_to_both=True,
                    model_mathematics_unchanged=True, target_decode_tpot_ms=7.0)
    save(run / "launch_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), expected=expected, policies=args.policies), indent=2))


if __name__ == "__main__":
    main()
