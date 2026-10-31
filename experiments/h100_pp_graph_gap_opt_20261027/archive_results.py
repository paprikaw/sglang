"""Archive small qualified artifacts and register this PP scheduling study."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ARCHIVE_ID = "qwen35b_h100_pp_graph_gap_optimization_20261027"
ARCHIVE = Path("/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments") / ARCHIVE_ID
FINAL = "shared_gemm_final_r8"
QUALIFIED = ["shared_gemm_gap_r2", "shared_gemm_stream_r5", "shared_gemm_prep_r6", "shared_gemm_lookahead_r7", FINAL]


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


def archive():
    final = ROOT / "runs" / FINAL
    analysis = ROOT / "analysis" / f"{FINAL}_steady"
    data = read(analysis / "results.json")
    assert all(data["checks"].values())
    assert read(final / "task/progress.json")["stage"] == "complete"
    assert read(final / "task/cleanup.json")["passed"]
    metrics = {(r["arm"], r["policy"]): r for r in data["formal_metrics"]}
    ep, pp, best = [metrics[k] for k in [("ep2", 0), ("pp2", 0), ("pp2", 9)]]
    assert best["tpot_ms"] < 7.02
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "report").iterdir()):
        if source.is_file():
            copy(source, ARCHIVE / source.name)
    for source in sorted(ROOT.glob("*.py")):
        copy(source, ARCHIVE / "scripts" / source.name)
    copy(ROOT / "analysis/analyze_initial_window.py", ARCHIVE / "scripts/analyze_initial_window.py")
    for name in QUALIFIED + [f"{FINAL}_steady"]:
        for source in sorted((ROOT / "analysis" / name).iterdir()):
            if source.is_file():
                copy(source, ARCHIVE / "analysis" / name / source.name)

    inventory, raw = [], []
    for run in sorted((ROOT / "runs").iterdir()):
        if not run.is_dir():
            continue
        qualified = run.name in QUALIFIED
        if qualified:
            status = "complete_qualified"
        elif run.name.endswith("r1"):
            status = "failed_before_gpu_lease_no_measurement"
        elif run.name.endswith("r3"):
            status = "aborted_shared_channel_ordering_partial_timings_excluded"
        else:
            status = "prepared_only_never_launched"
        inventory.append(dict(run=str(run), status=status, included_in_results=qualified))
        small = [run / n for n in ["config.json", "launch_manifest.json", "preparation.json", "dashboard_spec.json", "REJECTED_CANDIDATE.json", "task/progress.json", "task/cleanup.json"]]
        small += list((run / "code").glob("*.py"))
        small += list((run / "dashboard_guard/checks").glob("*.json"))
        cfg = read(run / "config.json")
        for arm in cfg["arm_order"]:
            for name in ["completed.json", "cleanup.json", "output_agreement.json"]:
                small.append(run / arm / name)
            small += list((run / arm).glob("qualification.p*.json"))
            small += list((run / arm / "gap_receipts").glob("*.json"))
            small += list((run / arm / "traces").glob("p*/manifest.json"))
            for source in sorted((run / arm).glob("metrics.*.json")):
                raw.append(dict(path=str(source), sha256=sha(source), run=run.name,
                                qualified_run=qualified, bytes=source.stat().st_size))
        for source in small:
            if source.is_file():
                assert source.name != "admin.key"
                copy(source, ARCHIVE / "runs" / run.name / source.relative_to(run))
    summary = f"""# PP2 Decode Graph 外调度优化

原 PP2 的 Expert GEMM 已接近通用小批次内核的水平，但 CPU 输出退休和阶段反馈路径仍把下一槽位的真实 GPU 计算推迟。

同一 H100178 GPU0/1 NVLink 对，Qwen3.6-35B-A3B BF16，全局8请求、固定两组B4，真实 GPU-cached4096 + new1 -> output128；五个正式 profiler-off cohort/配置。

- 相同通用 GEMM：EP2 {ep['tpot_ms']:.3f} ms/token、{ep['output_tok_s']:.1f} output tok/s；PP2 {pp['tpot_ms']:.3f} ms/token、{pp['output_tok_s']:.1f} tok/s。
- PP 专属 lookahead（宽保护窗口）：{best['tpot_ms']:.3f} ms/token、{best['output_tok_s']:.1f} tok/s。TPOT 降低 {(1-best['tpot_ms']/pp['tpot_ms'])*100:.1f}%，吞吐提高 {(best['output_tok_s']/pp['output_tok_s']-1)*100:.1f}%；五轮 TPOT 均值范围 {best['cohort_tpot_min_ms']:.3f}–{best['cohort_tpot_max_ms']:.3f} ms，结论为约7ms。
- 原生 forward、sampling、GPU token relay、GEMM 数学不变；提前 metadata 与下一槽位 GPU 发射，隐藏旧输出 CPU 处理。最终20,480个PP正式token逐位置一致；wire/event/slot-drain验证、12项CPU测试、Guard清理通过。
- 独立稳态 profile：所有rank丢弃前20个Graph、选择后16个，保守/宽窗口lookahead都已启用；Graph与正式TPOT分属不同执行。原始早期窗口分析一并保留。
- 静态PP2、固定B4、ignoreEOS、无spec/stop/penalty/logprob的原型验证。三轮冷Prefill4096->1为基线控制，本次lookahead性能结论仅针对Decode。共享无类型channel的depth1失败候选及仅准备目录不计入成功结果。

完整分析：[REPORT.md](REPORT.md)。正式逐请求/逐cohort表、组件表、PNG/PDF图、私有实现、冻结配置和来源哈希均在此归档；大日志与trace保留原路径，不复制授权文件、模型权重或KV payload。
"""
    (ARCHIVE / "summary.md").write_text(summary)
    manifest = dict(
        id=ARCHIVE_ID, title="PP2 Decode lookahead: hide CPU output retirement behind the next slot",
        status="complete", created_at=datetime.now(timezone.utc).isoformat(), client_date="2026-10-27",
        parent_experiment="qwen35b_h100_ep_pp_tpot_diagnosis_20261027", new_gpu_experiment=True,
        protocol="H100178 NVLink GPU0/1; Qwen3.6-35B-A3B BF16; B8=fixed B4+B4; real GPU cached4096+new1->128; 5 profiler-off cohorts/config; separate CPU/GPU profiles",
        source_root=str(ROOT), final_run=str(final), final_analysis=str(analysis),
        engine=read(final / "launch_manifest.json"), formal=data["formal_metrics"],
        profile_window=data["profile_window"], independent_profile=data["independent_profile"],
        run_inventory=inventory, raw_metric_sources=raw, trace_sources=data["trace_sources"],
        scientific_gate=dict(**data["checks"], cpu_tests=12, pp_formal_output_tokens_equal=20480,
            formal_metrics_identical_across_profile_window_reanalysis=True,
            figure_visually_inspected=True, frozen_engine_clean=True),
        implementation_scope="Private per-run adapters; shared generic Expert GEMM in EP and PP; no shared engine/default changes",
        artifacts=[dict(path=str(p.relative_to(ARCHIVE)), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name not in ["manifest.json", "SHA256SUMS"]],
    )
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checks = [f"{sha(p)}  {p.relative_to(ARCHIVE)}" for p in sorted(ARCHIVE.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (ARCHIVE / "SHA256SUMS").write_text("\n".join(checks) + "\n")
    register(Path("/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md"), f"""### {ARCHIVE_ID}

Snapshot: {ARCHIVE}

- Same H100178 NVLink GPU0/1, full Qwen3.6-35B-A3B BF16; EP2 AttentionDP2/TP1 and PP2 AttentionTP1/20layers per stage/two B4 slots; B8, real GPU-cached4096+new1->128, five profiler-off cohorts/configuration.
- Shared generic Expert GEMM remains enabled in both. EP {ep['tpot_ms']:.3f}ms/{ep['output_tok_s']:.1f}tok/s; PP baseline {pp['tpot_ms']:.3f}ms/{pp['output_tok_s']:.1f}tok/s; PP-specific next-slot metadata/real-forward lookahead {best['tpot_ms']:.3f}ms/{best['output_tok_s']:.1f}tok/s. No token prediction or GEMM changes in scheduling comparison.
- Independent common late-Graph profile and complete raw SSE audit retained. Final20,480 PP output tokens match; wire/event/ownership/drain/12CPU tests pass; all qualified runs cleaned and leases released. Rejected header-free shared-channel depth1 and prepared-only attempts explicitly excluded.
- Static PP2 fixed-B4/no-spec/ignoreEOS prototype; cold Prefill is baseline-only. Formal vs profile scopes kept separate. Artifacts include report, PNG/PDF, request/cohort/history tables, code, configurations, all outcomes and SHA256s; no credentials/weights/KV payload copied.
""")
    register(WORKSPACE / "experiments/experiment_summary/README.md", f"""## PP2 Decode 的 Graph 外调度优化（2026-10-27）

[简短报告]({ARCHIVE}/REPORT.md) · [图表]({ARCHIVE}/pp_decode_scheduling.png) · [完整归档]({ARCHIVE}/manifest.json)

同一 H100178 NVLink GPU0/1、Qwen3.6-35B-A3B BF16、B8（固定 B4+B4）、真实 GPU-cached4096+new1→128。五轮 profiler-off：同等 GEMM 的 EP2 为 {ep['tpot_ms']:.3f} ms/token、{ep['output_tok_s']:.1f} tok/s；PP2 基线 {pp['tpot_ms']:.3f} ms/token、{pp['output_tok_s']:.1f} tok/s；PP 下一槽位提前 metadata/真实 GPU forward 为 {best['tpot_ms']:.3f} ms/token、{best['output_tok_s']:.1f} tok/s。GEMM 未再改，完整20,480个PP正式token逐位置一致。Graph外间隔包括对端依赖/CPU调度/其他槽位工作，不能当作纯NVLink传输。静态PP2固定B4原型；profile、启动/验证和正式计时分开。失败候选、逐轮数据、源哈希与清理证明均保留。
""")
    print(json.dumps(dict(archive=str(ARCHIVE), files=len(checks), qualified_runs=len(QUALIFIED), total_runs=len(inventory))))


if __name__ == "__main__":
    archive()
