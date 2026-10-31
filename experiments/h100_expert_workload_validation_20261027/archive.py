"""Archive compact validated results; retain large raw artifacts by hash."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ARCHIVE_ID = "qwen35b_h100_expert_workload_validation_20261027"
ARCHIVE = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments") / ARCHIVE_ID


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def register(path, text):
    begin, end = f"<!-- begin {ARCHIVE_ID} -->", f"<!-- end {ARCHIVE_ID} -->"
    block = f"{begin}\n{text.strip()}\n{end}"
    original = path.read_text()
    if begin in original:
        first = original.index(begin)
        last = original.index(end, first) + len(end)
        updated = original[:first] + block + original[last:]
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    if updated != original:
        path.write_text(updated)


def main():
    result = read(ROOT / "report/results.json")
    assert not result["partial"] and not result["profiles_skipped"]
    assert len(result["metrics"]) == 32 and len(result["components"]) == 64
    assert all(r["stage"] == "complete" and r["guard_postcheck"] == "passed" and r["all_frozen_hashes_verified"]
               for r in result["run_audits"])
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for directory in ("report", "analysis"):
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file():
                copy(path, ARCHIVE / directory / path.name)
    for path in sorted(ROOT.glob("*.py")):
        copy(path, ARCHIVE / "scripts" / path.name)
    raw_sources = {d["path"]: d for d in result["source_descriptors"]}
    for run in sorted((ROOT / "runs").iterdir()):
        if not (run / "config.json").exists():
            continue
        cfg = read(run / "config.json")
        for name in ("config.json", "dashboard_spec.json", "preparation.json", "launch_manifest.json",
                     "kv_checkpoint_discovery.json", "task/progress.json", "task/cleanup.json"):
            copy(run / name, ARCHIVE / "runs" / run.name / name)
        paths = list((run / "code").glob("*.py"))
        paths += list((run / "dashboard_guard/checks").glob("*.json"))
        for arm in cfg["arm_order"]:
            paths += [run / arm / "completed.json", run / arm / "cleanup.json"]
            paths += list((run / arm / "qualification").glob("*.json"))
            paths += list((run / arm / "ep_kernel_qualification").glob("*.json"))
            for case in cfg["workloads"]:
                root = run / arm / "workloads" / case["id"]
                paths += [root / "protocol.json", root / "completed.json", root / "prefix_preparation.json"]
                paths += list((root / "traces").glob("*/manifest.json"))
                for path in list((root / "routing").glob("*.json")) + [root / "validation.probabilities.json"]:
                    raw_sources[str(path)] = dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        for path in paths:
            assert path.is_file() and path.name != "admin.key"
            copy(path, ARCHIVE / "runs" / run.name / path.relative_to(run))
    (ARCHIVE / "raw_sources.json").write_text(json.dumps(list(raw_sources.values()),indent=2)+"\n")
    pp = [r for r in result["comparisons"] if r["arm"] == "pp2"]
    headline = "; ".join(f"{r['workload']} PP {r['throughput_gain_pct']:+.2f}%" for r in pp)
    summary = f"""# Shared Expert small-tile policy across larger H100 workloads

Same full Qwen3.6-35B-A3B BF16 and H100178 NVLink GPUs0/1. EP2 Attention DP2/TP1 versus two native PP2 slots, microbatch B/2.
Global B8/B16/B32 correspond to target local Decode T4/T8/T16; repeated prompts plus diverse B32. Every Decode request uses real GPU-cached4096+new1->128; five profiler-off cohorts/configuration. Cold Prefill4096->1 controls and independent real-routing/GPU profiles retained.

{headline}.

Same existing Triton GEMM family on EP and PP; only the experimental selector ceiling expands from T4 to T16. No new GEMM implementation, PP scheduling optimization or TBO. Native shared-expert overlap ON. Every formal request cache/count checked; source/native hashes, Guard/cache postchecks and owned cleanup verified.

Full tables and figures: [report/REPORT.md](report/REPORT.md). Original raw SSE and large traces remain at SHA256-referenced paths. No keys, prompt payloads, weights or KV bytes copied.
"""
    (ARCHIVE / "summary.md").write_text(summary)
    manifest = dict(id=ARCHIVE_ID,title="Fair EP2/PP2 Expert optimization across batch size and prompt diversity",
        created_at=datetime.now(timezone.utc).isoformat(),status="complete",source_root=str(ROOT),
        engine=read(ROOT / "runs/triton_r1/launch_manifest.json"),
        workloads=read(ROOT / "runs/triton_r1/config.json")["workloads"],
        protocol="Global B8/B16/B32; local T4/T8/T16; repeated plus diverse B32; cached4096+new1->128 Decode and cold4096->1 Prefill; five profiler-off cohorts; independent profiles",
        comparisons=result["comparisons"],run_audits=result["run_audits"],
        scientific_checks=dict(source_and_native_frozen=True,guard_postchecks_pass=True,
            exact_cache_and_output_protocol=True,same_kernel_opportunity_both_arms=True,
            actual_graph_attribution=True,numerical_and_index_checks=True,owned_cleanup_verified=True),
        raw_sources="raw_sources.json",
        artifacts=[dict(path=str(p.relative_to(ARCHIVE)),sha256=sha(p),bytes=p.stat().st_size)
                   for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name not in ("manifest.json","SHA256SUMS")])
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    hashes=[f"{sha(p)}  {p.relative_to(ARCHIVE)}" for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(hashes)+"\n")
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"),f"""### {ARCHIVE_ID}

Snapshot: {ARCHIVE}

- Same full Qwen3.6-35B-A3B BF16 on H100178 NVLink GPU0/1; EP2 AttentionDP2/TP1 vs PP2 with two native slots and microbatch B/2. Global B8/B16/B32, repeated and diverse prompts; cached4096+new1->128 Decode and cold4096->1 Prefill, five formal cohorts each.
- Same existing Expert Triton family on both layouts; experimental T<=16 cutoff extends earlier T<=4. {headline}. No PP scheduling changes or TBO; native shared overlap ON.
- Actual routed expert counts/assignments, same-input numerical controls, atomic probability probes, independent Graph GEMM/Attention/communication, full per-cohort/request metrics. Source/native hashes, required CPU weight-cache receipts, Guard postchecks, owned cleanup and lease release pass. Large raw artifacts retained at hashed original paths.

Artifacts: report, comparison/routing PNG+PDF, metrics/round/request/component/routing CSV, validation JSON, frozen private adapters and compact Guard evidence; manifest and SHA256SUMS.
""")
    register(ROOT.parents[1]/"experiments/experiment_summary/README.md",f"""## H100 Expert 优化的 workload 扩展验证（2026-10-27）

[报告]({ARCHIVE}/report/REPORT.md) · [图]({ARCHIVE}/report/workload_generalization.png) · [归档]({ARCHIVE}/manifest.json)

同一完整Qwen3.6-35B-A3B BF16、H100178 NVLink GPU0/1；EP2、PP2均比较native与相同small-tile内核。全局B8/B16/B32，对应每卡T4/T8/T16；另有B32多主题输入。每配置5轮真实GPU缓存4096+new1→128 Decode，含独立冷Prefill、路由与Graph分项。{headline}。本轮仅扩大小tile启用范围，不启用PP调度优化或TBO；完整检查及清理通过。
""")
    print(json.dumps(dict(archive=str(ARCHIVE),files=len(hashes),headline=headline),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
