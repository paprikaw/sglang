"""Keep a compact reproducible snapshot and idempotent linked index entries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ID = "qwen35b_h100_ep_prefill_overlap_20261027"
ARCHIVE = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments") / ID


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def register(path, body):
    start, end = f"<!-- begin {ID} -->", f"<!-- end {ID} -->"
    block = start + "\n" + body.strip() + "\n" + end
    text = path.read_text()
    updated = (text[:text.index(start)] + block + text[text.index(end)+len(end):]
               if start in text else text.rstrip() + "\n\n" + block + "\n")
    if updated != text: path.write_text(updated)


def main():
    value = read(ROOT / "report/results.json")
    assert not value["partial"] and value["run_audit"]["guard_and_cleanup_verified"]
    run, cfg = Path(value["run"]), value["config"]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "report").iterdir():
        if path.is_file(): copy(path, ARCHIVE / "report" / path.name)
    for path in ROOT.glob("*.py"): copy(path, ARCHIVE / "scripts" / path.name)
    for path in (ROOT / "analysis" / run.name).iterdir():
        if path.is_file(): copy(path, ARCHIVE / "analysis" / path.name)
    attempts = []
    rounds = [value] + [value[key] for key in ("round4096", "diverse_recheck") if key in value]
    raw = {d["path"]: d for current in rounds for d in current["source_descriptors"]}
    dependencies = [ROOT.parent / "h100_prefill_ep_pp_optimized_20261027/analyze.py",
                    ROOT.parent / "pp_ep_steady_decode_20261026/analyze.py",
                    ROOT.parent / "h100_pp_expert_fragmentation_20261026/launch.py"]
    for path in dependencies:
        copy(path, ARCHIVE / "analysis_dependencies" / path.parent.name / path.name)
        raw[str(path)] = dict(path=str(path), sha256=sha(path), kind="analysis_or_launch_dependency")
    for current in rounds[1:]:
        for path in (ROOT / "analysis" / Path(current["run"]).name).iterdir():
            if path.is_file(): copy(path, ARCHIVE / "analysis" / Path(current["run"]).name / path.name)
    patterns = ["config.json", "dashboard_spec.json", "launch_manifest.json", "preparation.json",
                "cpu_qualification.json", "overlap_cpu_qualification.json", "diagnosis.json", "gpu_qualification.json",
                "code/*.py", "task/*.json", "dashboard_guard/checks/*.json", "*/cleanup.json", "*/completed.json",
                "*/qualification/*.json", "*/overlap_qualification/*.json", "*/ep_kernel_qualification/*.json",
                "*/workloads/*/protocol.json", "*/workloads/*/numerical_admission.json",
                "*/workloads/*/metrics.prefill.p*.json", "*/workloads/*/output_agreement.json",
                "*/workloads/*/completed.json", "*/workloads/*/traces/*/manifest.json"]
    for attempt in sorted((ROOT / "runs").iterdir()):
        if not attempt.is_dir(): continue
        selected = set()
        for pattern in patterns: selected.update(attempt.glob(pattern))
        for path in sorted(selected):
            assert path.name != "admin.key"
            copy(path, ARCHIVE / "runs" / attempt.name / path.relative_to(attempt))
            raw[str(path)] = dict(path=str(path), sha256=sha(path), kind="run_evidence")
        config = read(attempt / "config.json")
        source = Path(config["engine_root"]) / "python/sglang/srt/layers/moe/prefill_overlap.py"
        if source.exists(): copy(source, ARCHIVE / "sources" / config["engine_commit"] / source.name)
        attempts.append(dict(run=str(attempt), progress=read(attempt / "task/progress.json"),
                             is_main=attempt == run, diagnosis=read(attempt / "diagnosis.json") if (attempt / "diagnosis.json").exists() else None))
    save(ARCHIVE / "raw_sources.json", list(raw.values()))
    main_rows = [r for r in value["metrics"] if r["workload"] == "b32_q4096_repeated"]
    native = next(r for r in main_rows if (r["arm"], r["policy"]) == ("ep2", 0))
    summary = "; ".join(f"{r['arm']} p{r['policy']}: {r['input_tok_s']:,.1f} input tok/s, {r['e2e_ms']:.2f}ms mean E2E"
                        for r in main_rows)
    opt = next(r for r in main_rows if (r["arm"], r["policy"]) == ("ep2", 3))
    gain = 100 * (opt["input_tok_s"] / native["input_tok_s"] - 1)
    description = ("Full Qwen3.6-35B-A3B BF16, H100178 NVLink GPU0/1; AttentionDP2/TP1 MoEEP2 versus "
                   "two-stage/two-slot PP2. Cold Prefill context0 + input4096 → output1; B8/B32 repeated and B32 diverse "
                   "in the main matrix, both layouts using an 8192-token per-Attention Prefill budget. "
                   "Three cold warmups and five counterbalanced profiler-off cohorts/policy. "
                   "EP p0 native, p1 serial two-chunk, p2 overlapped two-chunk, p3 overlapped with communication SM reservation. "
                   "Attention/GDN unchanged, shared-expert overlap ON, full-model TBO OFF; native path retained for mixed/idle tails. "
                   "Per-forward CPU policy synchronization included in all EP arms. PP native/early controls unchanged. "
                   "Actual same-GPU communication/GEMM intersections are measured in separate profiles. "
                   "Per-layer real-weight numerical tests, formal output agreement, exact input/cache/output and handle-drain receipts, "
                   "21 CPU tests, frozen source hashes, canonical weight-cache usage, Guard postcheck and cleanup/release audited. "
                   "An independent 4096-token-budget round and a ten-cohort diverse-input recheck remain separate. "
                   "Failed qualification attempts retain their status and contribute no performance samples. "
                   f"At the common 8192 budget, B32 repeated EP overlap/reserved-SM throughput improves {gain:+.2f}% over native EP; PP2 remains faster.")
    (ARCHIVE / "summary.md").write_text(f"# EP Prefill overlap on H100\n\n{description}\n\nB32 input4096: {summary}.\n\n[Report](report/REPORT.md) · [Figure](report/prefill_overlap.png)\n")
    previous = read(ARCHIVE / "manifest.json") if (ARCHIVE / "manifest.json").exists() else {}
    manifest = dict(id=ID, title="MoE-only DeepEP Prefill overlap on H100, with native EP2/PP2 controls",
                    created_at=previous.get("created_at", datetime.now(timezone.utc).isoformat()), status="complete", source_root=str(ROOT),
                    implementation=dict(engine="sglang", worktree=cfg["engine_root"], commit=cfg["engine_commit"],
                                        source_tag=cfg["engine_tag"], branch="codex/qwen35b-prefill-ep-overlap-r3-20261027", dirty=False),
                    deployment=dict(id=cfg["id"], node=cfg["node"], job_id=cfg["job_id"], endpoint_retired=True,
                                    launch_manifest=f"runs/{run.name}/launch_manifest.json"),
                    model=dict(path=cfg["model_path"], revision=cfg["model_revision"], precision=cfg["precision"]),
                    workloads=cfg["workloads"], layouts=cfg["layouts"], engine_parameters=cfg,
                    metrics=value["metrics"], components=value["components"], run_audit=value["run_audit"],
                    output_agreement=value["output_agreement"], attempts=attempts, raw_sources="raw_sources.json",
                    independent_rounds=[dict(run=current["run"], config=current["config"], metrics=current["metrics"],
                                             components=current["components"], run_audit=current["run_audit"],
                                             output_agreement=current["output_agreement"])
                                        for current in rounds[1:]],
                    gpu_qualification=read(run / "gpu_qualification.json"),
                    diverse_recheck_statistics=value.get("diverse_recheck_statistics"),
                    replay=dict(cwd=str(ROOT.parents[1]),
                                analyze=f"python3 {ROOT}/analyze.py --run {run}",
                                report=f"/home/bxb1/conda/bin/python {ROOT}/report.py --run {run}",
                                note="Raw traces and pinned runtime remain at hashed original GPFS paths; compact archive stores all analysis, metrics, code, and manifests."))
    save(ARCHIVE / "manifest.json", manifest)
    hashes = [f"{sha(path)}  {path.relative_to(ARCHIVE)}" for path in sorted(ARCHIVE.rglob("*"))
              if path.is_file() and path.name != "SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    body = f"### {ID}\n\nSnapshot: {ARCHIVE}\n\n{description}\n\nB32 input4096: {summary}.\n\n[Report]({ARCHIVE}/report/REPORT.md) · [Figure]({ARCHIVE}/report/prefill_overlap.png)"
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"), body)
    register(ROOT.parents[1] / "experiments/experiment_summary/README.md", body)
    for row in hashes:
        digest, name = row.split("  ", 1)
        assert sha(ARCHIVE / name) == digest
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(hashes), summary=summary), indent=2))


if __name__ == "__main__": main()
