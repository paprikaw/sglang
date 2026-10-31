"""Freeze workload validation of the already implemented EP/PP GEMM family."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FAIR = ROOT.parent / "h100_ep_pp_fair_expert_20261026"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected source for {path.name}: {old!r}")
    path.write_text(text.replace(old, new))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "triton_small"], required=True)
    args = parser.parse_args()
    run = args.run.absolute()
    if run.exists():
        raise RuntimeError("Choose a new immutable run directory")
    subprocess.run([sys.executable, str(FAIR / "prepare.py"), "--run", str(run),
                    "--mode", args.mode, "--arms", "ep2", "pp2", "--repeats", "5"],
                   check=True, stdout=subprocess.PIPE, text=True)
    code = run / "code"
    shutil.copyfile(ROOT / "workload_benchmark.py", code / "workload_benchmark.py")
    # Only extend the selection boundary. Both arms retain the same existing
    # shape-selected Triton kernel family, without a new GEMM implementation.
    replace_once(code / "fragmentation.py", "dispatch_output.hidden_states.shape[0] <= 4",
                 "dispatch_output.hidden_states.shape[0] <= 16")
    replace_once(code / "ep_fragmentation.py", "dispatch.topk_ids.shape[0] <= 4",
                 "dispatch.topk_ids.shape[0] <= 16")
    # Keep actual-route diagnostics separate for each prompt family and B.
    replace_once(code / "fragmentation.py", "key = (id(layer), phase, x.shape[0])",
                 "workload = Path(os.environ['EXPERT_WORKLOAD_PATH']).read_text().strip()\n"
                 "        key = (workload, id(layer), phase, x.shape[0])")
    replace_once(code / "fragmentation.py", " / 'pp2' / 'routing'",
                 " / 'pp2' / 'workloads' / workload / 'routing'")
    replace_once(code / "fragmentation.py", "bench_key = (ps.pp_rank, phase, x.shape[0])",
                 "bench_key = (workload, ps.pp_rank, phase, x.shape[0])")
    replace_once(code / "ep_fragmentation.py", "key = (rank, layer_id, t)",
                 "workload = Path(os.environ['EXPERT_WORKLOAD_PATH']).read_text().strip()\n"
                 "        key = (workload, rank, layer_id, t)")
    replace_once(code / "ep_fragmentation.py", "routing = root/'routing';routing.mkdir(exist_ok=True)",
                 "routing = root/'workloads'/workload/'routing';routing.mkdir(parents=True,exist_ok=True)")
    replace_once(code / "ep_fragmentation.py", "if (rank,t) not in _BENCHED:",
                 "if (workload,rank,t) not in _BENCHED:")
    replace_once(code / "ep_fragmentation.py", "_BENCHED.add((rank,t))",
                 "_BENCHED.add((workload,rank,t))")
    # Qualification covers every requested tile shape, including skew, empty,
    # all-active and block-boundary counts, using real first-layer weights.
    replace_once(code / "ep_fragmentation.py",
                 "tests=[compare(layer.quant_method,layer,dispatch,metadata=True)]",
                 "tests=[]\n        for logical_tokens in (4,8,16):\n"
                 "            probe=dispatch._replace(topk_ids=torch.zeros((logical_tokens,8),device=x.device,dtype=torch.int64),\n"
                 "                                    topk_weights=torch.ones((logical_tokens,8),device=x.device))\n"
                 "            tests.append(dict(logical_tokens=logical_tokens,**compare(layer.quant_method,layer,probe,metadata=True)))")
    replace_once(code / "runner.py", "if __name__ == '__main__':",
                 "import workload_benchmark\nworkload_benchmark.install(sys.modules[__name__])\n\n"
                 "if __name__ == '__main__':")
    cfg = json.loads((run / "config.json").read_text())
    workloads = [dict(id="b8_repeated", requests=8, family="repeated"),
                 dict(id="b16_repeated", requests=16, family="repeated"),
                 dict(id="b32_repeated", requests=32, family="repeated"),
                 dict(id="b32_diverse", requests=32, family="diverse")]
    cfg.update(port=26821, dist_port=26822, workloads=workloads,
               expected_points=2 * len(workloads) * len(cfg["arm_order"]),
               expert_workload_path=str(run / "active_workload.txt"),
               original_expert_cutoff=4, tested_expert_cutoff=16,
               pp_slots=2, pp_async_batch_depth=0,
               profile_steps=48, selected_forwards=16,
               protocol="Cached4096 + new1 -> output128; global B8/B16/B32, balanced EP DP2; "
                        "PP two slots with microbatch B/2 set through native idle runtime API. "
                        "Repeated prompts plus diverse B32. Same existing BF16 Triton family on both arms at local T<=16. "
                        "Five profiler-off cohorts; separate actual-route diagnostics and Graph profiles. No TBO.")
    Path(cfg["expert_workload_path"]).write_text("none\n")
    Path(cfg["expert_workload_path"]).chmod(0o600)
    # Pin native shared-expert overlap ON in both comparison runs. This is
    # the previously accepted EP baseline; two-batch overlap remains OFF.
    cfg["launch_env"].update(SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP="1")
    save(run / "config.json", cfg)
    spec = json.loads((run / "dashboard_spec.json").read_text())
    inputs = [d for d in spec["frozen_inputs"]
              if not Path(d["path"]).is_relative_to(code) and d["path"] != str(run / "config.json")]
    inputs += [descriptor(p) for p in sorted(code.glob("*.py"))] + [descriptor(run / "config.json")]
    spec.update(title="H100 Expert workload validation: " + args.mode, port=26820,
                frozen_inputs=list({d["path"]: d for d in inputs}.values()))
    spec["tasks"][0].update(expected=cfg["expected_points"],
                           label="EP2 / PP2; local 4, 8, 16 tokens; " + args.mode)
    save(run / "dashboard_spec.json", spec)
    save(run / "task/progress.json", dict(experiment_id=run.name, task_id="comparison",
         stage="queued", valid_points=0, expected=cfg["expected_points"]))
    manifest = json.loads((run / "launch_manifest.json").read_text())
    manifest.update(private_adapters=[descriptor(p) for p in sorted(code.glob("*.py"))],
                    workloads=workloads, original_expert_cutoff=4, tested_expert_cutoff=16,
                    kernel_implementation_changed=False, pp_scheduling_optimization=False,
                    two_batch_overlap=False, shared_expert_overlap=True,
                    protocol=cfg["protocol"])
    save(run / "launch_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), mode=args.mode, expected=cfg["expected_points"],
                          workloads=[w["id"] for w in workloads]), indent=2))


if __name__ == "__main__":
    main()
