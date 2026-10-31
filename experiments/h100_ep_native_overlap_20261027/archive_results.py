"""Archive the native EP2 overlap study without copying request payloads or keys."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ARCHIVE_ID = "qwen35b_h100_ep2_native_overlap_20261027"
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
    body = f"{begin}\n{text.strip()}\n{end}"
    original = path.read_text()
    if begin in original:
        first = original.index(begin)
        last = original.index(end, first) + len(end)
        updated = original[:first] + body + original[last:]
    else:
        updated = original.rstrip() + "\n\n" + body + "\n"
    if updated != original:
        path.write_text(updated)


def main():
    result = read(ROOT / "report/results.json")
    off, on = result["summary"][:2]
    assert not off["native_shared_overlap"] and on["native_shared_overlap"]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "report").iterdir()):
        if source.is_file():
            copy(source, ARCHIVE / source.name)
    for source in sorted(ROOT.glob("*.py")):
        copy(source, ARCHIVE / "scripts" / source.name)
    raw_sources, run_inventory, trace_sources = [], [], []
    for outcome in result["outcomes"]:
        run = Path(outcome["run"])
        progress = read(run / "task/progress.json")
        assert progress["stage"] in ("complete", "failed", "aborted", "interrupted")
        cleanup = read(run / "task/cleanup.json")
        assert cleanup["passed"], (run, cleanup)
        postchecks = [read(p) for p in (run / "dashboard_guard/checks").glob("*-post.json")]
        assert len(postchecks) == 1 and postchecks[0]["cleanup_verified"]
        qualified = progress["stage"] == "complete"
        if qualified:
            data = read(ROOT / "analysis" / run.name / "results.json")
            assert all(data["checks"].values())
            assert read(run / "ep2/native_overlap_qualification.json")["passed"]
            trace_sources += data["trace_sources"]
            for source in sorted((ROOT / "analysis" / run.name).iterdir()):
                if source.is_file():
                    copy(source, ARCHIVE / "analysis" / run.name / source.name)
        run_inventory.append(dict(run=str(run), stage=progress["stage"], included_in_performance=qualified,
                                  cleanup_verified=True, error=progress.get("error")))
        small = [run / n for n in ["config.json", "launch_manifest.json", "preparation.json", "dashboard_spec.json",
                                  "task/progress.json", "task/cleanup.json", "ep2/completed.json", "ep2/cleanup.json",
                                  "ep2/native_overlap_qualification.json"]]
        small += list((run / "code").glob("*.py"))
        small += list((run / "dashboard_guard/checks").glob("*.json"))
        small += list((run / "ep2").glob("qualification.p*.json"))
        small += list((run / "ep2/native_overlap_audit").glob("*.json"))
        small += list((run / "ep2/traces").glob("p*/manifest.json"))
        for source in small:
            if source.is_file():
                assert source.name != "admin.key"
                copy(source, ARCHIVE / "runs" / run.name / source.relative_to(run))
        for source in sorted((run / "ep2").glob("metrics.*.json")):
            raw_sources.append(dict(path=str(source), sha256=sha(source), bytes=source.stat().st_size,
                                    run=run.name, qualified_run=qualified))
    tbo_outcome = result["outcomes"][2]
    tbo_summary = next((r for r in result["summary"] if r["native_tbo_flag"]), None)
    tbo_note = (f"TBO flag completed; actual dispatcher subbatch1 seen={tbo_summary['actual_tbo_subbatch1_seen']}."
                if tbo_summary else f"TBO flag outcome={tbo_outcome['stage']}; no valid performance point.")
    description = f"""# SGLang native EP2 shared-expert / DeepEP overlap

Same H100178 NVLink GPU0/1, Qwen3.6-35B-A3B BF16, Attention DP2/TP1 and MoE EP2; same generic small-batch Triton GEMM.
Global B8, real GPU-cached4096 + new1 -> output128 per request; five formal profiler-off cohorts per configuration.

- Shared overlap OFF: {off['tpot_ms']:.3f} ms/token, {off['output_tok_s']:.1f} output token/s.
- Shared overlap ON: {on['tpot_ms']:.3f} ms/token, {on['output_tok_s']:.1f} output token/s.
- TPOT reduced {(1-on['tpot_ms']/off['tpot_ms'])*100:.1f}%; throughput increased {(on['output_tok_s']/off['output_tok_s']-1)*100:.1f}%.
- The frozen Qwen path defaults to ON: the earlier approximately7.66ms EP2 baseline already included this overlap.
- Actual B4 CUDA Graph capture receipts confirm shared expert on main stream when OFF, on alt stream when ON, on both ranks.
- {tbo_note} The frozen source explicitly rejects generic SBO on SM90; no check was bypassed.

Full analysis: [REPORT.md](REPORT.md). GPU concurrency profiles are separate from formal latency and throughput measurements.
Large original traces/raw SSE stay at source paths with SHA256 references. No credentials, model weights or KV payload copied.
"""
    (ARCHIVE / "summary.md").write_text(description)
    manifest = dict(id=ARCHIVE_ID, title="Native EP2 Qwen shared-expert/DeepEP overlap ON/OFF and TBO check",
        created_at=datetime.now(timezone.utc).isoformat(), status="complete", client_date="2026-10-27",
        source_root=str(ROOT), new_gpu_experiment=True,
        protocol="H100178 NVLink GPU0/1; Qwen3.6-35B-A3B BF16; Attention DP2/TP1, MoE EP2; B8 GPU-cached4096+new1->128; five profiler-off cohorts; separate steady GPU profiles",
        engine=read(ROOT / "runs/native_shared_on_r2/launch_manifest.json"),
        formal_and_profile_summary=result["summary"], outcomes=run_inventory,
        raw_metric_sources=raw_sources, trace_sources=trace_sources, source_audit=result["source_audit"],
        scientific_gate=dict(qualified_runs_frozen_and_guard_validated=True, actual_stream_branch_verified=True,
                             real_gpu_cache_checked=True, owned_cleanup_verified=True, figure_visually_inspected=True),
        implementation_scope="Explicit existing SGLang overlap env/flag plus one-time observational receipts; shared engine and GEMM unchanged",
        artifacts=[dict(path=str(p.relative_to(ARCHIVE)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "SHA256SUMS")])
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    sums = [f"{sha(p)}  {p.relative_to(ARCHIVE)}" for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(sums)+"\n")
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"), f"""### {ARCHIVE_ID}

Snapshot: {ARCHIVE}

- Same H100178 NVLink GPU0/1, Qwen3.6-35B-A3B BF16, AttentionDP2/TP1 and MoEEP2; identical small-batch Triton GEMM. B8 real GPU-cached4096+new1->128, five profiler-off cohorts per configuration.
- Native shared-expert overlap OFF {off['tpot_ms']:.3f}ms/{off['output_tok_s']:.1f}tok/s; ON {on['tpot_ms']:.3f}ms/{on['output_tok_s']:.1f}tok/s. Qwen source defaults ON, so previous approximately7.66ms EP2 already included it.
- Both ranks' actual B4 Graph capture stream branches verified; separate steady GPU D/C-vs-compute concurrency intersections measured. {tbo_note} Generic SBO explicitly rejected by frozen SM90 source, not bypassed.
- All outcomes and owned cleanup retained; qualified runs passed Guard/frozen/cache/output checks. Formal request/cohort CSVs, PNG/PDF, source excerpts and hashes, private observation code and exact configurations archived. Raw SSE and large traces remain original with hashes.
""")
    register(WORKSPACE / "experiments/experiment_summary/README.md", f"""## EP2 原生通信/计算 overlap（2026-10-27）

[报告]({ARCHIVE}/REPORT.md) · [图表]({ARCHIVE}/native_ep_overlap.png) · [归档]({ARCHIVE}/manifest.json)

同一 H100178 NVLink GPU0/1、Qwen3.6-35B-A3B BF16、B8真实GPU-cached4096+new1→128，同等小批次GEMM。五轮 profiler-off：原生shared-expert overlap关闭 {off['tpot_ms']:.3f}ms/token、{off['output_tok_s']:.1f}tok/s；开启 {on['tpot_ms']:.3f}ms/token、{on['output_tok_s']:.1f}tok/s。当前Qwen路径默认开启，因此此前约7.66ms的EP2已包含此优化。两rank实际Graph stream分支及独立GPU并行区间验证，所有结局与清理证据保留。{tbo_note}
""")
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(sums), outcomes=run_inventory)))


if __name__ == "__main__":
    main()
