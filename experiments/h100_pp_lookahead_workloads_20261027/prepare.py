"""Freeze fair Expert kernels plus PP lookahead across four existing workloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKLOAD = ROOT.parent / "h100_expert_workload_validation_20261027"
GAP = ROOT.parent / "h100_pp_graph_gap_opt_20261027"
SCHEDULE = ROOT.parent / "h100_decode_pp_opt_20261026"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5, choices=(3, 5))
    args = parser.parse_args()
    run = args.run.absolute()
    if run.exists():
        raise RuntimeError("Choose a fresh immutable run directory")
    subprocess.run([sys.executable, str(WORKLOAD / "prepare.py"), "--run", str(run),
                    "--mode", "triton_small"], check=True)
    code = run / "code"
    for name in ("gap_schedule.py", "pp_fixed_cohort.py", "lookahead_workloads.py"):
        shutil.copyfile(ROOT / name, code / name)
    for name in ("pp_schedule_opt.py", "pp_transfer_audit.py"):
        shutil.copyfile(SCHEDULE / name, code / name)
    shutil.copyfile(Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py"),
                    code / "artifact_visibility.py")
    # Keep the exact prior input construction available to the new benchmark.
    shutil.copyfile(WORKLOAD / "workload_benchmark.py", code / "previous_workloads.py")
    adapter = code / "runtime_adapter.py"
    adapter.write_text(adapter.read_text() + '''
_shared_expert_install = install
def install():
    _shared_expert_install()
    if os.environ["EP2_PP2_ARM"] == "pp2":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
''')
    runner = code / "runner.py"
    text = runner.read_text()
    marker = "if __name__ == '__main__':"
    assert text.count(marker) == 1
    text = text.replace(marker, "import lookahead_workloads\nlookahead_workloads.install(sys.modules[__name__])\n\n" + marker)
    runner.write_text(text)
    cfg = json.loads((run / "config.json").read_text())
    for field, name, value in (("gap_policy_path", "gap_policy.bin", b"0"),
                               ("fixed_cohort_control", "fixed_cohort.bin", b"\x00"),
                               ("fixed_cohort_size", "fixed_cohort_size.bin", bytes([8])),
                               ("transfer_audit_control", "transfer_audit.bin", b"0")):
        cfg[field] = str(run / name)
        Path(cfg[field]).write_bytes(value)
        Path(cfg[field]).chmod(0o600)
    cfg.update(port=27021, dist_port=27022, repeats=args.samples,
               gap_policies=[0, 9], samples_per_policy=args.samples,
               pp_async_batch_depth=0, pp_loop_slots=2, pp_slots=2,
               profile_steps=56, selected_forwards=16, drop_initial_graphs=20,
               expected_points=len(cfg["workloads"]) * 5,
               protocol="Same full BF16 model and fair Expert Triton kernels, local T<=16; "
                        "EP2 DP2 balanced even/odd versus static PP2 with two B/2 slots and the same groups. "
                        "Global B8/B16/B32 repeated plus B32 diverse. Real GPUcached4096+new1->output128. "
                        "PP policy0 versus policy9 wide lookahead, counterbalanced profiler-off cohorts; "
                        "same-input output equality, actual wire/group audit and independent Decode profiles. "
                        "Cold4096->1 Prefill control. Native EP shared-expert overlap ON; TBO OFF.")
    # Preserve the inherited versioned Guard/cache contract, pinned source and
    # required daemon bindings. Only this new run's private inputs are changed.
    save(run / "config.json", cfg)
    for path in code.glob("*.py"):
        path.chmod(0o600)
    spec = json.loads((run / "dashboard_spec.json").read_text())
    inputs = [d for d in spec["frozen_inputs"]
              if not Path(d["path"]).is_relative_to(code) and d["path"] != str(run / "config.json")]
    inputs += [descriptor(p) for p in sorted(code.glob("*.py"))] + [descriptor(run / "config.json")]
    spec.update(title="H100 EP2 vs PP2 lookahead across workload sizes", port=27020,
                frozen_inputs=list({d["path"]: d for d in inputs}.values()))
    spec["tasks"][0].update(expected=cfg["expected_points"],
                           label="EP2 / PP2 baseline / PP2 lookahead; B8, B16, B32, diverse B32")
    save(run / "dashboard_spec.json", spec)
    save(run / "task/progress.json", dict(experiment_id=run.name, task_id="comparison",
         stage="queued", valid_points=0, expected=cfg["expected_points"]))
    manifest = json.loads((run / "launch_manifest.json").read_text())
    manifest.update(private_adapters=[descriptor(p) for p in sorted(code.glob("*.py"))],
                    pp_scheduling_optimization=True, gap_policies=[0, 9],
                    lookahead_size_extension=[4, 8, 16], fixed_cohort_sizes=[8, 16, 32],
                    generic_expert_optimization_applied_to_both=True,
                    shared_expert_overlap=True, two_batch_overlap=False,
                    protocol=cfg["protocol"])
    save(run / "launch_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), expected_points=cfg["expected_points"],
                          policies=cfg["gap_policies"], workloads=cfg["workloads"]), indent=2))


if __name__ == "__main__":
    main()
