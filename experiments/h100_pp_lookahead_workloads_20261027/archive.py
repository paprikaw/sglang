"""Archive compact validated foreground results and update the shared indexes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
EXPERIMENT_ID = "qwen35b_h100_pp_lookahead_workloads_20261027"
ARCHIVE = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments") / EXPERIMENT_ID


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def register(path, content):
    begin, end = f"<!-- begin {EXPERIMENT_ID} -->", f"<!-- end {EXPERIMENT_ID} -->"
    block = begin + "\n" + content.strip() + "\n" + end
    text = path.read_text()
    if begin in text:
        first, last = text.index(begin), text.index(end) + len(end)
        updated = text[:first] + block + text[last:]
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    if updated != text:
        path.write_text(updated)


def main():
    result = read(ROOT / "report/results.json")
    assert not result["partial"] and not result["profiles_skipped"]
    assert result["run_audit"]["guard_postcheck"] == "passed"
    assert result["run_audit"]["all_frozen_hashes_verified"]
    run = Path(result["run"])
    cfg = result["config"]
    assert read(run / "cpu_qualification.json")["tests"] == 16
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for folder in (ROOT / "report", ROOT / "analysis" / run.name):
        for path in sorted(folder.iterdir()):
            if path.is_file():
                copy(path, ARCHIVE / ("report" if folder.name == "report" else "analysis") / path.name)
    for path in sorted(ROOT.glob("*.py")):
        copy(path, ARCHIVE / "scripts" / path.name)
    compact = [run / name for name in ("config.json", "dashboard_spec.json", "preparation.json",
        "launch_manifest.json", "cpu_qualification.json", "kv_checkpoint_discovery.json",
        "task/progress.json", "task/cleanup.json")]
    compact += list((run / "code").glob("*.py"))
    compact += list((run / "dashboard_guard/checks").glob("*.json"))
    raw = {d["path"]: d for d in result["source_descriptors"]}
    for arm in cfg["arm_order"]:
        compact += [run / arm / "completed.json", run / arm / "cleanup.json"]
        compact += list((run / arm / "qualification").glob("*.json"))
        compact += list((run / arm / "ep_kernel_qualification").glob("*.json"))
        for case in cfg["workloads"]:
            folder = run / arm / "workloads" / case["id"]
            compact += [folder / "protocol.json", folder / "completed.json", folder / "output_agreement.json"]
            compact += list(folder.glob("qualification.*.json"))
            compact += list((folder / "traces").glob("*/manifest.json"))
            for path in folder.glob("cohort.*.json"):
                raw[str(path)] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, kind="formal_cohort")
        for path in (run / arm / "transfer_audit").glob("*.json"):
            raw[str(path)] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, kind="wire_payload_hash_receipt")
    for path in compact:
        assert path.is_file() and path.name != "admin.key", path
        copy(path, ARCHIVE / "run" / path.relative_to(run))
    failed = ROOT / "runs/lookahead_r1"
    diagnostic_attempts = []
    if failed != run and (failed / "receipt_failure_diagnosis.json").exists():
        for name in ("launch_manifest.json", "task/progress.json", "task/cleanup.json", "receipt_failure_diagnosis.json"):
            copy(failed / name, ARCHIVE / "diagnostic_attempts/lookahead_r1" / name)
        for path in (failed / "dashboard_guard/checks").glob("*-post.json"):
            copy(path, ARCHIVE / "diagnostic_attempts/lookahead_r1" / path.name)
        path = failed / "controller.log"
        diagnostic_attempts.append(dict(run=str(failed), status="failed", performance_data_excluded=True,
            reason="Wire receipt was observed before JSON publication completed; final rerun uses stable-JSON barrier",
            cleanup_verified=read(failed / "task/cleanup.json")["passed"],
            controller_log=dict(path=str(path), sha256=sha(path))))
    (ARCHIVE / "raw_sources.json").write_text(json.dumps(list(raw.values()), indent=2) + "\n")
    headline = "; ".join(f"{r['workload']}: lookahead vs native PP {r['pp_lookahead_gain_pct']:+.2f}%, vs EP {r['pp_vs_ep_pct']:+.2f}%" for r in result["comparisons"])
    summary = f"""# PP lookahead across four H100 workloads

Full Qwen3.6-35B-A3B BF16 on the same H100178 NVLink GPUs0/1. Global B8/B16/B32 repeated prompts and B32 diverse prompts; EP2 AttentionDP2/TP1 versus PP2 two B/2 slots. Same prior inputs, shared generic Expert small-tile family at local T<=16, native EP shared-expert overlap ON, TBO OFF.

Every Decode request: real GPUcached4096+new1->output128. Five profiler-off cohorts/configuration; PP native and policy9 wide lookahead counterbalanced with separate warmups. Full output equality, actual even/odd request groups, PP payload hashes, live early-launch production/consumption counters, precise cache/output/SSE audits and independent late-Graph component/cycle profiles pass. Cold4096->1 Prefill controls retained. Source/native hashes, required CPU model-cache postchecks, owned cleanup and lease release verified.

{headline}.

[Readable report](report/REPORT.md), [throughput/TPOT plot](report/workload_comparison.png), [scheduling profile](report/scheduling_decomposition.png). Raw SSE and large traces remain at SHA256-referenced original paths. No credentials, prompt payloads, weights or KV bytes copied.
"""
    (ARCHIVE / "summary.md").write_text(summary)
    manifest = dict(id=EXPERIMENT_ID, title="Fair EP2 versus native and lookahead PP2 across workload sizes and prompt diversity",
        created_at=datetime.now(timezone.utc).isoformat(), status="complete", source_root=str(ROOT),
        engine=read(run / "launch_manifest.json"), protocol=cfg["protocol"], workloads=cfg["workloads"],
        comparisons=result["comparisons"], run_audit=result["run_audit"], raw_sources="raw_sources.json",
        diagnostic_attempts=diagnostic_attempts,
        scientific_checks=dict(same_generic_expert_optimization=True, same_prompt_fingerprints=True,
            pp_outputs_equal=True, pp_payloads_equal=True, lookahead_activated_every_formal_cohort=True,
            exact_cache_and_stream_audit=True, guard_postcheck=True, owned_cleanup_verified=True),
        artifacts=[dict(path=str(p.relative_to(ARCHIVE)), sha256=sha(p), bytes=p.stat().st_size)
                   for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "SHA256SUMS")])
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    hashes = [f"{sha(p)}  {p.relative_to(ARCHIVE)}" for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"), f"""### {EXPERIMENT_ID}

Snapshot: {ARCHIVE}

- Same full Qwen3.6-35B-A3B BF16, H100178 NVLink GPU0/1, frozen SG8702697022; global B8/B16/B32 repeated plus B32 diverse, local T4/T8/T16. EP2 AttentionDP2/TP1 and PP2 two native slots; identical prior prompt fingerprints and shared generic Expert kernels.
- PP policy0 versus policy9 wide next-slot lookahead, five counterbalanced profiler-off cached4096+new1->128 Decode cohorts/configuration; cold4096->1 Prefill controls. {headline}.
- Complete token equality, actual even/odd groups and wire hashes, per-formal-cohort live lookahead counters, unique SSE1..128, actual late-Graph Expert/Attention/communication and same-slot cycle attribution pass; 16CPU tests. Required canonical weight-cache/source/frozen-input/Guard postchecks, owned cleanup and lease release verified. TBO remains OFF. Failed initial receipt-publication race retained separately; all final performance rows come from the complete corrected rerun.
- Artifacts: report, PNG/PDF charts, formal request/cohort/metric and profile CSVs, small control/qualification evidence, frozen adapters, source descriptors and SHA256SUMS. Large raw SSE/traces remain at hashed original paths. Static two-slot fixed-cohort Decode scope.
""")
    register(ROOT.parents[1] / "experiments/experiment_summary/README.md", f"""## H100 PP 提前发射的 workload 扩展验证（2026-10-27）

[报告]({ARCHIVE}/report/REPORT.md) · [吞吐/TPOT图]({ARCHIVE}/report/workload_comparison.png) · [调度分解]({ARCHIVE}/report/scheduling_decomposition.png)

同一完整Qwen3.6-35B-A3B BF16、H100178 NVLink GPU0/1；B8/B16/B32及B32多主题，真实GPU缓存4096+new1→128。EP/PP均保留相同Expert内核；原生PP与提前发射PP每组5轮交替对照，实际分组、输出、wire hash及提前发射计数验证通过。{headline}。独立GPU分项及冷Prefill控制已保存，Guard与清理通过。
""")
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(hashes), headline=headline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
