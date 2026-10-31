"""Freeze representative native A100 EP2/PP2 comparisons on an NVLink pair."""
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OLD = WORKSPACE / "experiments/ep2_pp2_qwen35b_a100_20261025/runs/a100_qwen35b_ep2_pp2_r2"
LOOK = WORKSPACE / "experiments/h100_pp_lookahead_workloads_20261027/runs/lookahead_r2"
PREF = WORKSPACE / "experiments/h100_prefill_ep_pp_optimized_20261027/runs/qwen35b_h100_prefill_optimized_r2_20261027"
GUARD = WORKSPACE / "experiments/qwen_a100_capacity_20261019_r1/guard_bundle/skills/llm-experiment-guard"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    path.chmod(0o600)


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.absolute()
    if run.exists(): raise RuntimeError("Choose a fresh immutable directory")
    code = run / "code"
    code.mkdir(parents=True)
    (run/"task").mkdir()
    (run/"cache").mkdir()
    for arm in ("ep2","pp2"): (run/arm).mkdir()
    for name in ("runner.py", "process_helpers.py", "experiment_plan.py", "export_ready.py",
                 "a100_runtime_adapter.py", "sampling.py", "previous_workloads.py",
                 "gap_schedule.py", "pp_schedule_opt.py", "pp_fixed_cohort.py", "pp_transfer_audit.py"):
        shutil.copyfile(LOOK/"code"/name, code/name)
    shutil.copyfile(PREF/"code/prefill_admission.py",code/"prefill_admission.py")
    shutil.copyfile(GUARD/"scripts/artifact_visibility.py",code/"artifact_visibility.py")
    shutil.copyfile(ROOT/"benchmark.py", code/"a100_benchmark.py")
    shutil.copyfile(ROOT/"prepare.py", code/"prepare.py")
    (code/"runtime_adapter.py").write_text('''"""Native A100 Expert kernel shared by EP and PP; optional PP scheduling only."""
import os
def install():
    import a100_runtime_adapter
    a100_runtime_adapter.install()
    if os.environ["EP2_PP2_ARM"] == "pp2":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    else:
        import prefill_admission
        prefill_admission.install()
''')
    runner = code/"runner.py"
    text = runner.read_text()
    begin, tail = text.split("\nimport sampling\n",1)
    _, tail = tail.split("if __name__ == '__main__':",1)
    text = begin+"\nimport a100_benchmark\na100_benchmark.install(sys.modules[__name__])\n\nif __name__ == '__main__':"+tail
    anchor = "    (run / 'topology.txt').write_text(topo.stdout + topo.stderr)"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
    if topo.returncode:
        raise RuntimeError("Cannot verify physical GPU topology")
    clean = __import__("re").sub(chr(27)+r"\\[[0-9;]*[A-Za-z]", "", topo.stdout)
    lines = [line.split() for line in clean.splitlines() if line.strip()]
    header = next((r for r in lines if r[0] == "GPU0" and "GPU1" in r and "X" not in r), None)
    row = next((r for r in lines if r[0] == "GPU0" and len(r)>1 and r[1] == "X"), None)
    if header is None or row is None or not row[header.index("GPU1")+1].startswith("NV"):
        raise RuntimeError("Selected physical GPUs 0/1 are outside a shared NVLink domain")
    h.save(run / "topology_qualification.json", dict(passed=True, physical_gpus=[0,1],
        link=row[header.index("GPU1")+1], observed_under_guard=True))
''')
    runner.write_text(text)
    cfg=json.loads((OLD/"config.json").read_text())
    cfg.update(id=run.name, code_root=str(code), admin_key_file=str(run/"admin.key"),
        guard_scripts=str(GUARD/"scripts"), hardware="A100", port=27121, dist_port=27122,
        repeats=5, warmups=3, expected_points=15,
        precision="Native BF16 weights/activations/KV; FP32 GDN; identical Triton Expert in both layouts",
        profile_steps=56,selected_forwards=16,drop_initial_graphs=20,
        reference_run=str(LOOK),
        protocol="Qwen35B BF16; same A100 NVLink GPUs0/1. Attention DP2 x TP1 with EP2 versus PP2 x TP1 x EP1, "
                 "two B/2 slots and matched even/odd request groups. Same native Triton Expert family in both; "
                 "eager execution, overlap schedule off. Decode GPUhot4096+new1->output128; "
                 "Prefill cold P0/Q128 or Q4096->output1. Three warmups, five profiler-off cohorts per configuration. "
                 "PP0 baseline versus PP9 lookahead for Decode; independent component profiles; no CPU KV restore.")
    cfg["workloads"]=[
        dict(id="decode_b8_repeated",phase="decode",requests=8,family="repeated"),
        dict(id="decode_b32_repeated",phase="decode",requests=32,family="repeated"),
        dict(id="decode_b32_diverse",phase="decode",requests=32,family="diverse"),
        dict(id="prefill_b8_q128_repeated",phase="prefill",requests=8,family="repeated",new_input=128),
        dict(id="prefill_b8_q4096_repeated",phase="prefill",requests=8,family="repeated",new_input=4096),
        dict(id="prefill_b32_q4096_diverse",phase="prefill",requests=32,family="diverse",new_input=4096)]
    for field,name,value in (("gap_policy_path","gap_policy.bin",b"0"),
                              ("fixed_cohort_control","fixed_cohort.bin",b"\x00"),
                              ("fixed_cohort_size","fixed_cohort_size.bin",bytes([8])),
                              ("transfer_audit_control","transfer_audit.bin",b"0")):
        cfg[field]=str(run/name); (run/name).write_bytes(value); (run/name).chmod(0o600)
    (run/"admin.key").write_text(secrets.token_urlsafe(32)); (run/"admin.key").chmod(0o600)
    cache=cfg["model_cache"]
    shutil.copyfile(OLD/"cache/node.json",run/"cache/node.json")
    cache["node_specs"][cfg["node"]]=descriptor(run/"cache/node.json")
    uses=[]
    for row in cfg["layouts"]:
        use=dict(id=run.name+"."+row["id"],path=str(run/"cache"/(row["id"]+".log")),
                 expected_ranks=row["usage_logs"][0]["expected_ranks"])
        row["usage_logs"]=[use]; uses.append(use)
    cache["usage_logs"][cfg["node"]]=uses
    save(run/"config.json",cfg)
    spec=json.loads((OLD/"dashboard_spec.json").read_text())
    inputs=[d for d in spec["frozen_inputs"] if "/guard_bundle/" not in d["path"]
            and not Path(d["path"]).is_relative_to(OLD)
            and "/sglang-memory-bank/" not in d["path"]]
    # Reuse the already qualified Qwen-supporting Guard, with the current
    # separately owned CPU-budget dependency. The public GLM-only Guard cannot
    # admit this model; changing the declared profile would be incorrect.
    paths=list((GUARD/"scripts").glob("*.py"))
    bank=Path("/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank/src/sglang_memory_bank")
    paths += [bank/n for n in ("__init__.py","memory_budget.py","memory_bank.py")]
    paths += sorted(code.glob("*.py"))+[run/"config.json",run/"cache/node.json"]
    inputs += [descriptor(p) for p in paths]
    spec.update(id=run.name,title="A100 NVLink EP2 vs PP2: representative regression points",
        port=27120,runtime_dir=str(run/"dashboard_guard"),model_cache=cache,
        frozen_inputs=list({d["path"]:d for d in inputs}.values()))
    spec["tasks"][0].update(controller_script=str(runner),root=str(run/"task"),expected=15,
        label="35B; GPUs0/1 NVLink; Decode B8/B32/diverse, Prefill short/long")
    save(run/"dashboard_spec.json",spec)
    save(run/"task/progress.json",dict(experiment_id=run.name,task_id="comparison",stage="queued",valid_points=0,expected=15))
    save(run/"launch_manifest.json",dict(schema_version=1,id=run.name,source=spec["sources"],
        config=descriptor(run/"config.json"),protocol=cfg["protocol"],private_adapters=[descriptor(p) for p in sorted(code.glob("*.py"))],
        shared_expert_backend=True,generic_h100_masked_adapter_applied=False,pp_policies=[0,9],
        kv_checkpoint_lookup=dict(compatible_ready_node069_qwen35_entries=0,protocol="model-produced GPU-hot prefix"),
        topology_requirement="physical GPU0/1 must have a direct NVLink link; checked under lease before model launch"))
    for p in code.glob("*.py"): p.chmod(0o600)
    print(json.dumps(dict(run=str(run),workloads=cfg["workloads"],points=15),indent=2))


if __name__=="__main__": main()
