"""Freeze the opt-in MoE-only overlap against native EP2 and unchanged PP2."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ENGINE = WORKSPACE / ".engine-worktrees/qwen35b-prefill-ep-overlap-r3-20261027"
TAG = "experiment/sglang/qwen35b-prefill-ep-overlap-20261027-r3-cpu"


def read(path): return json.loads(path.read_text())
def save(path, value): path.write_text(json.dumps(value, indent=2) + "\n")
def descriptor(path): return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--ep-budget", type=int, choices=(4096, 8192), default=4096)
    ap.add_argument("--pp-budget", type=int, choices=(4096, 8192), default=4096)
    ap.add_argument("--verify-diverse", action="store_true", help="Ten-round EP-native/reserved-SM timing-only recheck")
    args = ap.parse_args()
    run = args.run.absolute()
    subprocess.run([sys.executable, str(ROOT.parent / "h100_prefill_ep_pp_optimized_20261027/prepare.py"),
                    "--run", str(run)], check=True, stdout=subprocess.DEVNULL)
    cfg = read(run / "config.json")
    old_engine = Path(cfg["engine_root"])
    commit = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    tag_commit = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", TAG + "^{commit}"], text=True).strip()
    assert tag_commit == commit
    cases = [c for c in cfg["workloads"] if c["id"] in
             (("b32_q4096_repeated",) if args.focus else ("b8_q4096_repeated", "b32_q4096_repeated", "b32_q4096_diverse"))]
    if args.verify_diverse:
        cases = [c for c in cases if c["id"] == "b32_q4096_diverse"]
    cfg.update(engine_root=str(ENGINE), engine_commit=commit, engine_tag=TAG,
               ep_prefill_budget=args.ep_budget,
               pp_prefill_budget=args.pp_budget,
               ep_overlap_policies=[0, 1, 2, 3], ep_overlap_policy_path=str(run / "ep_overlap_policy.bin"),
               workloads=cases, expected_points=len(cases) * 6, port=27321, dist_port=27322,
               protocol="Cold P0 Prefill, same full BF16 Qwen3.6-35B-A3B on H100178 GPU0/1. "
                        "Same Attention/GDN whole-batch work, same once-computed routing. EP2 AttentionDP2/TP1 "
                        "with MoE-only two-chunk scheduling: p0 native, p1 serial split, p2 overlapped split, "
                        "p3 overlapped split with comm SM reservation. Shared Expert overlap remains ON. "
                        "PP2 p0 native and p1 early activation unchanged. Source budget4096/Attention, "
                        "output1, zero prefix cache, 3 cold warmups and 5 counterbalanced profiler-off cohorts/policy. "
                        "Real-weight per-layer serial/overlap numerical admission precedes timing; "
                        "independent GPU profiles and drained-handle receipts.")
    if args.verify_diverse:
        cfg.update(arm_order=["ep2"], layouts=[r for r in cfg["layouts"] if r["id"] == "ep2"],
                   ep_overlap_policies=[0, 3], repeats=10, samples_per_policy=10,
                   expected_points=2, timing_only=True)
        cfg["model_cache"]["usage_logs"][cfg["node"]] = cfg["layouts"][0]["usage_logs"]
    (run / "ep_overlap_policy.bin").write_bytes(b"0")
    code = run / "code"
    adapter = code / "runtime_adapter.py"
    adapter.write_text(adapter.read_text() + '''
_before_prefill_overlap = install
def install():
    _before_prefill_overlap()
    from sglang.srt.layers.moe import prefill_overlap
    prefill_overlap.install()
''')
    # Give the generic source-batch extension to both layouts. Every request's
    # actual input stays fixed; native/split EP controls share their budget.
    hopper = code / "hopper_reference.py"
    text = hopper.read_text()
    assert text.count("'8192' if arm == 'ep2' else '4096'") == 2
    hopper.write_text(text.replace("'8192' if arm == 'ep2' else '4096'",
                                   "str(2 * cfg['ep_prefill_budget']) if arm == 'ep2' else str(cfg['pp_prefill_budget'])"))
    benchmark = code / "prefill_benchmark.py"
    text = benchmark.read_text()
    text = text.replace('policies = [0, 1] if arm == "pp2" else [0]',
                        'policies = [0, 1] if arm == "pp2" else cfg["ep_overlap_policies"]')
    text = text.replace('byte(cfg["gap_policy_path"], ord(str(p)))',
                        'byte(cfg["gap_policy_path"] if arm == "pp2" else cfg["ep_overlap_policy_path"], ord(str(p)))')
    needle = '            measured = {p: [] for p in policies}'
    assert text.count(needle) == 1
    text = text.replace(needle, '''            if arm == "ep2":
                byte(cfg["ep_overlap_policy_path"], ord("8"))
                health(); base.flush_cache(cfg, key)
                qualification_group = base.cohort(current, key, packets)
                check(qualification_group)
                info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                receipts = [r["experiment_ep_overlap"] for r in info["internal_states"]]
                assert len(receipts) == 2 and all(r["qualification_layers"] == 40 and r["qualification_passed"] and r["drained"] for r in receipts)
                base.h.save(root / "numerical_admission.json", dict(performance_sample=False, receipts=receipts, result=qualification_group))
''' + needle)
    text = text.replace('                measured[p].append(group)', '''                if arm == "ep2":
                    info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                    receipts = [r["experiment_ep_overlap"] for r in info["internal_states"]]
                    assert len(receipts) == 2 and all(r["drained"] and r["policy"] == p and r["counts"].get(str(p), 0) > 0 for r in receipts)
                    group["overlap_receipts"] = receipts
                measured[p].append(group)''')
    text = text.replace('            byte(cfg["fixed_cohort_control"], 0)',
                        '            byte(cfg["ep_overlap_policy_path"], ord("0"))\n            byte(cfg["fixed_cohort_control"], 0)')
    text = text.replace('pp_policies=[0, 1], decode_lookahead_applicable=False)',
                        'pp_policies=[0, 1], ep_moe_overlap_policies=cfg["ep_overlap_policies"], decode_lookahead_applicable=False)')
    text = text.replace('actual_prefill_token_budget_per_attention=4096',
                        'actual_prefill_token_budget_per_attention=cfg["ep_prefill_budget"] if arm == "ep2" else cfg["pp_prefill_budget"]')
    text = text.replace('            for p in policies:\n                health(); byte(',
                        '            for p in ([] if cfg.get("timing_only") else policies):\n                health(); byte(')
    benchmark.write_text(text)
    cfg["protocol"] = cfg["protocol"].replace("Source budget4096/Attention", f"EP source budget{args.ep_budget}/Attention, PP{args.pp_budget}/Attention")
    if args.verify_diverse:
        cfg["protocol"] = ("Independent timing-only recheck: B32 diverse, cold context0 + input4096 -> output1; "
                           f"EP Attention source budget{args.ep_budget}; EP p0 native versus p3 reserved-SM overlap; "
                           "three cold warmups and ten counterbalanced profiler-off cohorts/policy. "
                           "Same complete model, source, routing, precision and numerical admission. "
                           "No PP run or new profiler cohort; main matrix retains independent two-rank profiles.")
    save(run / "config.json", cfg)
    spec = read(run / "dashboard_spec.json")
    inputs = []
    for d in spec["frozen_inputs"]:
        path = Path(d["path"])
        if path.is_relative_to(code) or path == run / "config.json": continue
        if path.is_relative_to(old_engine): path = ENGINE / path.relative_to(old_engine)
        inputs.append(descriptor(path))
    inputs += [descriptor(p) for p in sorted(code.glob("*.py"))]
    inputs += [descriptor(run / "config.json"), descriptor(ENGINE / "python/sglang/srt/layers/moe/prefill_overlap.py")]
    cache_sources = spec["sources"]
    spec.update(title="H100 MoE-only EP Prefill overlap: native / serial / overlap / reserved SM", port=27320,
                sources=cache_sources + [dict(worktree=str(ENGINE), commit=commit, tag=TAG)],
                model_cache=cfg["model_cache"],
                frozen_inputs=list({d["path"]: d for d in inputs}.values()))
    spec["tasks"][0].update(expected=cfg["expected_points"], label="Cold Prefill, EP overlap ablation and unchanged PP2 controls")
    save(run / "dashboard_spec.json", spec)
    manifest = read(run / "launch_manifest.json")
    manifest.update(engine_commit=commit, engine_source_tag=TAG, engine_root=str(ENGINE),
                    private_adapters=[descriptor(p) for p in sorted(code.glob("*.py"))],
                    ep_moe_overlap_source=descriptor(ENGINE / "python/sglang/srt/layers/moe/prefill_overlap.py"),
                    workloads=cases, protocol=cfg["protocol"], native_full_model_tbo=False,
                    ep_moe_overlap_policies=cfg["ep_overlap_policies"])
    save(run / "launch_manifest.json", manifest)
    save(run / "task/progress.json", dict(experiment_id=run.name, task_id="comparison", stage="queued", valid_points=0, expected=cfg["expected_points"]))
    print(json.dumps(dict(run=str(run), commit=commit, cases=cases, expected=cfg["expected_points"]), indent=2))


if __name__ == "__main__": main()
