"""Archive validated Prefill results, preserving failed attempts separately."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
EXPERIMENT_ID = "qwen35b_h100_prefill_optimized_20261027"
ARCHIVE = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments") / EXPERIMENT_ID


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def register(path, content):
    first, last = f"<!-- begin {EXPERIMENT_ID} -->", f"<!-- end {EXPERIMENT_ID} -->"
    block = first + "\n" + content.strip() + "\n" + last
    text = path.read_text()
    if first in text:
        updated = text[:text.index(first)] + block + text[text.index(last)+len(last):]
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    if updated != text: path.write_text(updated)


def main():
    result = read(ROOT / "report/results.json")
    assert not result["partial"] and not result["profiles_skipped"]
    assert result["run_audit"]["guard_postcheck"] == "passed"
    assert result["run_audit"]["all_frozen_hashes_verified"]
    run, cfg = Path(result["run"]), result["config"]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for folder in (ROOT / "report", ROOT / "analysis" / run.name):
        for path in sorted(folder.iterdir()):
            if path.is_file(): copy(path, ARCHIVE / ("report" if folder.name == "report" else "analysis") / path.name)
    report_path=ARCHIVE/'report/REPORT.md'
    portable=report_path.read_text().replace(str(ROOT/'analysis'/run.name)+'/', '../analysis/')
    portable=portable.replace(str(ROOT/'report')+'/', './')
    portable=portable.replace(str(run/'launch_manifest.json'),'../run/launch_manifest.json')
    report_path.write_text(portable)
    for path in sorted(ROOT.glob("*.py")): copy(path, ARCHIVE / "scripts" / path.name)
    compact = [run / p for p in ("config.json", "dashboard_spec.json", "preparation.json",
        "launch_manifest.json", "cpu_qualification.json", "kv_checkpoint_discovery.json", "task/progress.json", "task/cleanup.json")]
    compact += list((run / "code").glob("*.py"))
    compact += list((run / "dashboard_guard/checks").glob("*.json"))
    sources = {r["path"]:r for r in result["source_descriptors"]}
    for d in result.get('overlap_inspection',{}).get('inspected_sources',[]):
        sources[d['path']]=dict(**d,kind='overlap_inspected_source')
    followups=result['followups']
    assert followups['numerical_validation']['audit']['status']=='complete'
    assert followups['focused_recheck']['audit']['status']=='failed'
    native_agreement = []
    for arm in cfg["arm_order"]:
        compact += [run / arm / p for p in ("completed.json", "cleanup.json")]
        compact += list((run / arm / "qualification").glob("*.json"))
        compact += list((run / arm / "ep_kernel_qualification").glob("*.json"))
        for case in cfg["workloads"]:
            folder = run / arm / "workloads" / case["id"]
            compact += [folder / p for p in ("protocol.json", "completed.json", "output_agreement.json")]
            compact += list((folder / "traces").glob("*/manifest.json"))
            native_agreement.append(dict(arm=arm, workload=case["id"], **read(folder / "output_agreement.json")))
            for path in folder.glob("cohort.*.json"):
                sources[str(path)] = dict(path=str(path), sha256=sha(path), kind="formal_cohort")
    for path in compact:
        assert path.is_file() and path.name != "admin.key", path
        copy(path, ARCHIVE / "run" / path.relative_to(run))
    for item in (followups['focused_recheck'],followups['numerical_validation']):
        attempt=Path(item['audit']['run'])
        target=ARCHIVE/'followups'/attempt.name
        selected=[attempt/p for p in ('config.json','dashboard_spec.json','preparation.json',
            'launch_manifest.json','cpu_qualification.json','prelaunch_metadata_finalization.json',
            'task/progress.json','task/cleanup.json') if (attempt/p).exists()]
        selected+=list((attempt/'code').glob('*.py'))
        selected+=list((attempt/'dashboard_guard/checks').glob('*.json'))
        selected+=list(attempt.glob('*/cleanup.json'))+list(attempt.glob('*/completed.json'))
        for pattern in ('*/qualification/*.json','*/ep_kernel_qualification/*.json',
                        '*/workloads/*/protocol.json','*/workloads/*/output_agreement.json',
                        '*/workloads/*/validation.prefill_logprobs.json','*/workloads/*/metrics.prefill.*.json'):
            selected+=list(attempt.glob(pattern))
        if (attempt/'diagnosis.json').exists():selected.append(attempt/'diagnosis.json')
        for path in selected:
            assert path.name!='admin.key'
            copy(path,target/path.relative_to(attempt))
            sources[str(path)]=dict(path=str(path),sha256=sha(path),kind='followup_evidence')
        folder=ROOT/'analysis'/attempt.name
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file():copy(path,target/'analysis'/path.name)
    failed = []
    for attempt in sorted((ROOT / "runs").iterdir()):
        if attempt == run or not (attempt / "diagnosis.json").exists(): continue
        for name in ("launch_manifest.json", "task/progress.json", "task/cleanup.json", "diagnosis.json"):
            copy(attempt / name, ARCHIVE / "diagnostic_attempts" / attempt.name / name)
        independently_used=str(attempt)==followups['focused_recheck']['audit']['run']
        failed.append(dict(run=str(attempt), diagnosis=read(attempt / "diagnosis.json"),
            performance_excluded=not independently_used,
            independently_used_workloads=['b8_q4096_repeated'] if independently_used else []))
    (ARCHIVE / "raw_sources.json").write_text(json.dumps(list(sources.values()),indent=2)+"\n")
    headline = "; ".join(f"{c['workload']}: PP early vs EP {c['pp_early_vs_ep_pct']:+.2f}%, vs PP native {c['early_gain_pct']:+.2f}%"
                         for c in result["comparisons"])
    summary = f"""# Cold Prefill: phase-compatible optimized EP2 versus PP2

Full Qwen3.6-35B-A3B BF16, SG8702697022, H100178 NVLink GPU0/1. EP2 AttentionDP2/TP1 and PP2 with two stages/slots; 4096-token Prefill budget per Attention instance. Cached context0, new input128/512/4096 -> output1, B8/B32 repeated and B32 Q4096 diverse. Five profiler-off cohorts/configuration after three cold warmups; PP policies counterbalanced. Shared Expert selector on both layouts, native large-shape GEMM retained; EP native shared-expert overlap switch ON, TBO OFF. PP policy1 submits the original activation send early. Decode-only next-slot lookahead and fixed small-shape transport are unused.

{headline}.

PP native/early first-token equality, live per-cohort early-send counters, identical input hashes, exact zero-cache/input/output checks, independent Prefill-only GPU GEMM and dispatch/combine attribution,18CPU tests, canonical model-cache usage, frozen hashes, Guard postcheck and owned cleanup/lease release pass for the complete main matrix. Short-input native EP/PP numerical differences are explicitly retained; cross-layout quality equivalence is not established for those points. Q4096 repeated/diverse main first tokens match across layouts.

An independent ten-round B8/Q4096 recheck shows early PP versus native PP {followups['focused_recheck']['comparison']['early_gain_pct']:+.3f}%, with no reproducible early-send Prefill gain. Its wrapper remains failed after a later untimed native PP logprob OOM; only independently audited completed B8/Q4096 formal timings are used. A separate PP-only short-input numerical replay uses static memory fraction0.60 for logprob allocator headroom and contributes no performance data. The main/focused performance budget is0.94. Initial r1 is excluded. Full failure, numerical and provenance boundaries are in the report and followups.json.

[Report](report/REPORT.md), [throughput figure](report/prefill_throughput.png), [Expert/communication figure](report/prefill_components.png). Large original traces/responses remain at SHA256-referenced source paths.
"""
    (ARCHIVE / "summary.md").write_text(summary)
    manifest = dict(id=EXPERIMENT_ID, title="Cold Prefill EP2/PP2 input-length comparison with phase-compatible optimizations",
        created_at=datetime.now(timezone.utc).isoformat(), status="complete", source_root=str(ROOT),
        engine=read(run / "launch_manifest.json"), protocol=cfg["protocol"], workloads=cfg["workloads"],
        comparisons=result["comparisons"], run_audit=result["run_audit"], native_output_agreement=native_agreement,
        diagnostic_attempts=failed, followups=followups,
        overlap_inspection=result.get('overlap_inspection'),raw_sources="raw_sources.json",
        artifacts=[dict(path=str(p.relative_to(ARCHIVE)), sha256=sha(p), bytes=p.stat().st_size)
                   for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name not in ("manifest.json","SHA256SUMS")])
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    hashes=[f"{sha(p)}  {p.relative_to(ARCHIVE)}" for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(hashes)+"\n")
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"), f"""### {EXPERIMENT_ID}

Snapshot: {ARCHIVE}

- Full BF16 Qwen3.6-35B-A3B, same H100178 NVLink GPU0/1; EP2 AttentionDP2/TP1 versus PP2 two native stages/slots. Cold Prefill cache0 + new128/512/4096 -> output1, B8/B32 repeated plus B32 Q4096 diverse; same input hashes, native chunk4096/Attention instance. Both layouts gate declared request cohorts; three cold warmups, five profiler-off repetitions, PP policies counterbalanced.
- Same Expert selector on both layouts; large Prefill retains native GEMM. EP native shared overlap switch ON, TBO OFF; PP policy0 versus phase-compatible policy1 early native activation send. No Decode-specific next-slot lookahead or fixed small-shape fast transport. {headline}.
- PP native/early output equality/live early-send counters/exact input-cache-output checks, independent Prefill-only grouped-contiguous GEMM and dispatch/combine profiles,18CPU tests, canonical cache usage, frozen hashes, Guard postcheck, owned cleanup and lease release pass for the main matrix. Short-input native EP/PP numerical differences remain; do not infer cross-layout quality equivalence. Q4096 repeated/diverse main first tokens match.
- Independent B8/Q4096 ten-round recheck: early PP vs native PP {followups['focused_recheck']['comparison']['early_gain_pct']:+.3f}%; no stable early-send gain in Prefill. The recheck wrapper failed during a later untimed PP logprob OOM; completed timing points independently audited, failed status retained. Separate PP-only numerical replay at memory fraction0.60 adds no performance samples; timed runs retain0.94. Initial r1 excluded. See followups.json and numerical CSVs for measured cross-layout differences and replay-versus-formal agreement.
- Report, PNG/PDF figures, metric/request/cohort/profile CSVs, frozen adapters, small validation receipts and SHA256SUMS; large original traces/responses remain referenced by SHA256.
- Saved B32/Q4096 two-rank trace overlap inspection: dispatch324.82 + combine259.22 µs/card/actual layer, routed GEMM484.73; communication/routed-GEMM and communication/Attention intersections both0, communication/shared-expert intersection3.98 µs. No new GPU run or TBO implementation; see overlap.md/overlap_results.json. These are temporal kernel overlaps, not measured critical-path savings.
""")
    register(ROOT.parents[1] / "experiments/experiment_summary/README.md", f"""## H100 冷 Prefill 的 EP2 / PP2 优化对照（2026-10-27）

[报告]({ARCHIVE}/report/REPORT.md) · [吞吐图]({ARCHIVE}/report/prefill_throughput.png) · [Expert/通信分解]({ARCHIVE}/report/prefill_components.png)

完整Qwen3.6-35B-A3B BF16、同一H100178 NVLink GPU0/1。缓存0，输入128/512/4096→输出1，B8/B32及B32长文本多主题；双方相同Expert选择器，PP开启适用于Prefill的提前提交activation。主矩阵每点5轮，独立GPU GEMM/dispatch/combine归因；18项CPU及Guard/清理检查通过。{headline}。

B8/Q4096独立10轮复测：提前发送相对原生PP仅 {followups['focused_recheck']['comparison']['early_gain_pct']:+.3f}%，无稳定Prefill收益。该复测wrapper在后续非计时logprob探针OOM而保持failed，仅使用完整审计的已完成计时。短输入原生EP/PP有数值差异，未认定跨布局质量等价；Q4096主实验首token一致。单独降低静态显存预算的PP数值诊断不进入性能比较。详见报告的边界说明与followups.json。

两rank现有trace复查：B32/Q4096每卡每层dispatch324.82、combine259.22、GEMM484.73 µs；通信与routed GEMM/Attention的时间交集均0，与shared expert交集3.98 µs。未重启TBO或新GPU实验；详见[重叠分析]({ARCHIVE}/report/overlap.md)。
""")
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(hashes), headline=headline),ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
