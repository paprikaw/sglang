"""Compare native SGLang overlap flags, including observed GPU concurrency."""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
CASES = [("native_shared_off_r1", "Shared OFF"), ("native_shared_on_r2", "Shared ON"),
         ("native_tbo_on_r3", "Shared ON + TBO flag")]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def concurrency(analysis):
    timing = load("native_overlap_timing", ROOT.parent / "h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/analyze.py")
    parser = load("native_overlap_components", ROOT.parent / "h100_pp_expert_fragmentation_20261026/analyze.py")
    tools = parser.load_analysis()
    rows = []
    for rank, entry in enumerate(sorted(analysis["trace_sources"], key=lambda r:r["path"])):
        trace = timing.read_trace(Path(entry["path"]), entry["sha256"])
        by_correlation = defaultdict(list)
        for e in trace["events"]:
            if e.get("cat") == "kernel":
                by_correlation[e["args"].get("correlation")].append(e)
        launches = {e["ts"]: e for e in trace["events"] if e["name"] == "cudaGraphLaunch"}
        for g in trace["graphs"][20:36]:
            kernels = by_correlation[launches[g["launch"]]["args"]["correlation"]]
            comm, compute = [], []
            for e in kernels:
                span = (e["ts"], e["ts"] + e["dur"])
                bucket = tools.graph_bucket(e["name"])
                if bucket in ("communication.dispatch", "communication.combine"):
                    comm.append(span)
                elif not bucket.startswith("communication."):
                    compute.append(span)
            assert comm and compute
            overlap = tools.union_us(comm) + tools.union_us(compute) - tools.union_us(comm + compute)
            assert -1e-6 <= overlap <= tools.union_us(comm) + 1e-6
            rows.append(dict(rank=rank, sequence_sum=g["key"][0], occurrence=g["key"][1],
                graph_ms=(g["end"]-g["start"])/1000,
                dispatch_combine_union_ms=tools.union_us(comm)/1000,
                dispatch_combine_concurrent_compute_ms=overlap/1000,
                concurrent_compute_us_per_layer=overlap/40,
                graph_kernel_streams=len({e["args"].get("stream") for e in kernels})))
    return rows


def analyze():
    ROOT.joinpath("report").mkdir(exist_ok=True)
    report = ROOT / "report"
    summary, outcomes, all_overlap, sources = [], [], [], []
    for name, label in CASES:
        run = ROOT / "runs" / name
        progress = read(run / "task/progress.json")
        outcome = dict(run=str(run), label=label, stage=progress["stage"])
        if progress["stage"] != "complete":
            assert progress["stage"] in ("failed", "aborted", "interrupted"), outcome
            outcome["error"] = progress.get("error")
            log = run / "cache/ep2.log"
            if log.exists():
                fields = sorted(set(re.findall(r"Field (\w+) has value, but is not yet supported", log.read_text(errors="replace"))))
                outcome["failure_evidence"] = dict(log=str(log), sha256=sha(log),
                    phase="CUDA Graph capture / TboForwardBatchPreparer.filter_batch",
                    unsupported_batch_fields=fields)
            outcomes.append(outcome)
            continue
        out = ROOT / "analysis" / name
        if not (out / "results.json").exists():
            subprocess.run([sys.executable, str(ROOT.parent / "h100_pp_graph_gap_opt_20261027/analyze_gap.py"),
                "--run", str(run), "--out", str(out), "--drop-graphs", "20", "--forwards", "16"],
                check=True, stdout=subprocess.DEVNULL)
        data = read(out / "results.json")
        assert all(data["checks"].values())
        qualification = read(run / "ep2/native_overlap_qualification.json")
        assert qualification["passed"]
        cfg = read(run / "config.json")
        rows = concurrency(data)
        all_overlap += [dict(run=name, **r) for r in rows]
        formal = data["formal_metrics"][0]
        components = data["components"]
        cmean = lambda key:1000*st.mean(c.get(key,0) for c in components)
        tbo_seen = qualification["actual_tbo_subbatch1_seen"]
        summary.append(dict(run=name, label=label, native_shared_overlap=cfg["native_shared_overlap"],
            native_tbo_flag=cfg["native_two_batch_overlap"], actual_tbo_subbatch1_seen=tbo_seen,
            tpot_ms=formal["tpot_ms"], cohort_tpot_min_ms=formal["cohort_tpot_min_ms"],
            cohort_tpot_max_ms=formal["cohort_tpot_max_ms"], output_tok_s=formal["output_tok_s"],
            ttft_ms=formal["ttft_ms"], e2e_ms=formal["e2e_ms"],
            attention_core_us=cmean("attention.core"), expert_gemm_us=cmean("expert.routed_gemm"),
            dispatch_us=cmean("communication.dispatch"), combine_us=cmean("communication.combine"),
            graph_ms=data["independent_profile"][0]["model_graphs_ms"],
            dc_concurrent_compute_us_per_layer=st.mean(r["concurrent_compute_us_per_layer"] for r in rows)))
        outcome.update(qualified=True, qualification=qualification)
        outcomes.append(outcome)
        sources += [dict(path=str(out / "results.json"), sha256=sha(out / "results.json")),
                    dict(path=str(run / "ep2/native_overlap_qualification.json"), sha256=sha(run / "ep2/native_overlap_qualification.json"))]
    assert len(summary) >= 2
    off, on = summary[:2]
    table(report / "metrics.csv", summary)
    table(report / "profile_overlap_cycles.csv", all_overlap)
    request_rows, cohort_rows = [], []
    for item in summary:
        raw = read(ROOT / "runs" / item["run"] / "ep2/metrics.decode.p0.json")
        for i, group in enumerate(raw["rounds"]):
            cohort_rows.append(dict(run=item["run"], cohort=i, output_tok_s=1024/group["makespan_s"],
                tpot_ms=st.mean(r["tpot_ms"] for r in group["requests"])))
            for j, row in enumerate(group["requests"]):
                request_rows.append(dict(run=item["run"], cohort=i, request_index=j,
                    cached_tokens=row["cached_tokens"], output_tokens=row["output_tokens"],
                    tpot_ms=row["tpot_ms"], ttft_ms=row["ttft_s"]*1000, e2e_ms=row["e2e_s"]*1000))
    table(report / "formal_requests.csv", request_rows)
    table(report / "formal_cohorts.csv", cohort_rows)
    source_audit = []
    engine = Path(read(ROOT / "runs" / CASES[0][0] / "config.json")["engine_root"])
    for file, lines in [("environ.py", (1044,1046)), ("layers/moe/utils.py", (327,333)),
                        ("models/qwen2_moe.py", (533,588)), ("models/qwen3_5.py", (1510,1550))]:
        path = engine / "python/sglang/srt" / file
        text = path.read_text().splitlines()
        source_audit.append(dict(path=str(path), sha256=sha(path), start_line=lines[0], end_line=lines[1],
            excerpt="\n".join(text[lines[0]-1:lines[1]])))
    (report / "results.json").write_text(json.dumps(dict(summary=summary, outcomes=outcomes,
        sources=sources, source_audit=source_audit,
        profile_scope="Separate CPU/GPU profile; discard20 then16 Graphs/rank; union intersection of DeepEP D/C and noncommunication GPU kernels, not an additive formal TPOT attribution"), indent=2)+"\n")
    make_plot(summary, report)
    lines = ["# EP2：SGLang 原生通信/计算 overlap 复测", "",
        "当前冻结版本的 Qwen 专用 shared-expert/DeepEP overlap 默认开启。此前约7.66ms的 EP2 基线已包含这一路；本次显式关闭/开启并核验实际 stream 分支。", "",
        "同一 H100178 NVLink GPU0/1；Qwen3.6-35B-A3B BF16；AttentionDP2/TP1，MoEEP2；同一通用小批次 Triton GEMM。全局B8，每请求真实GPU-cached4096+new1→output128；五个正式 profiler-off cohort/配置。前缀构建、warmup、stream审计和独立profile不计入正式计时。", "",
        "| 配置 | TPOT ms/token ↓ | 五轮均值范围 ms | 输出 tok/s ↑ | 实际 TBO subbatch1 |",
        "|---|---:|---:|---:|---|",
    ]
    for r in summary:
        lines.append(f'| {r["label"]} | {r["tpot_ms"]:.3f} | {r["cohort_tpot_min_ms"]:.3f}–{r["cohort_tpot_max_ms"]:.3f} | {r["output_tok_s"]:.1f} | {r["actual_tbo_subbatch1_seen"]} |')
    lines += ["", f'原生 shared overlap 开启后，TPOT 降低 **{(1-on["tpot_ms"]/off["tpot_ms"])*100:.1f}%**，吞吐提高 **{(on["output_tok_s"]/off["output_tok_s"]-1)*100:.1f}%**。', "",
        "![Native overlap measurements](native_ep_overlap.png)", "",
        "## 实际重叠及组件", "",
        "原生 Qwen 路径将独立的 shared-expert MLP 放到 alt stream，主 stream 执行 routed experts 的 dispatch、GEMM、combine；最后通过事件依赖合并。相同 token 的 dispatch→routed GEMM→combine 依赖仍保留。两 rank 的 B4 Graph 捕获记录分别确认：OFF 的 shared expert 在主 stream，ON 在独立 stream。审计只记录首次 warmup/capture，不增加正式 Graph replay 的 Python 操作。", "",
        "| 配置 | Expert GEMM µs/层 | Dispatch µs/层 | Combine µs/层 | Graph ms | D/C与其他计算并行 µs/层 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f'| {r["label"]} | {r["expert_gemm_us"]:.2f} | {r["dispatch_us"]:.2f} | {r["combine_us"]:.2f} | {r["graph_ms"]:.3f} | {r["dc_concurrent_compute_us_per_layer"]:.2f} |')
    lines += ["", "组件与并行区间来自独立 profile：每 rank 丢弃前20个Graph，选后16个。并行时间是 DeepEP dispatch/combine GPU kernel 区间并集与其他计算 kernel 区间并集的交集，包含 shared expert 及其他独立计算，不等于净节省时间；不可直接从正式 TPOT 中相减。", "",
        "## 通用 SBO / TBO 开关", "",
        "- `SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP=1`：当前模型可用且默认开启，本次已显式测试0/1。",
        "- `--enable-single-batch-overlap`：当前源码在SM90初始化时明确抛出不支持错误，原因指向当前 sgl-deep-gemm wheel；本次没有绕过该检查。",
        "- `--enable-two-batch-overlap`：单独配置测试，是否实际拆分以 dispatcher subbatch0/1 的运行记录为准。当前Qwen3.5/GDN的完整模型forward是逐层循环，没有调用标准TBO模型调度入口。",
    ]
    for r in outcomes:
        if r["stage"] != "complete":
            lines.append(f'- {r["label"]}：未形成有效性能点，阶段 `{r["stage"]}`，错误 `{r.get("error")}`；保留启动和清理证据。')
            if r.get("failure_evidence"):
                evidence = r["failure_evidence"]
                fields = ", ".join(f'`{f}`' for f in evidence["unsupported_batch_fields"])
                lines.append(f'  首个原生异常发生在 CUDA Graph 初始化的 `TboForwardBatchPreparer.filter_batch`：不支持拆分字段 {fields}。日志原路径及SHA256记录在 `results.json`；本次没有修改模型或绕过字段检查。')
        elif "TBO" in r["label"]:
            lines.append(f'- TBO flag 开启后的实际 subbatch1：`{r["qualification"]["actual_tbo_subbatch1_seen"]}`；完整 dispatcher 与 server flag 记录在该实验的 `native_overlap_qualification.json`。')
    lines += ["", "## 来源与复现", "",
        "[正式指标](metrics.csv) · [逐请求](formal_requests.csv) · [逐cohort](formal_cohorts.csv) · [逐Graph并行区间](profile_overlap_cycles.csv) · [来源、开关和stream证据](results.json)", "",
        "所有成功实验均通过冻结输入哈希、真实GPU缓存/输出长度、完整SSE计数、Guard postcheck及owned cleanup校验。模型数学与GEMM未改；EP原生异步调度的输出可能有A/A差异，不将本实验当作任务质量评估。冷Prefill4096→1为每配置三轮基线控制，当前性能结论针对Decode。", "",
    ]
    for name, label in CASES:
        run = ROOT / "runs" / name
        lines.append(f'- [{label}]({run})：[冻结配置]({run/"config.json"})；[启动来源]({run/"launch_manifest.json"})。')
    (report / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(summary=summary, report=str(report / "REPORT.md")), indent=2))


def make_plot(rows, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1,3,figsize=(14.5,4.9))
    labels = [r["label"].replace(" + ","\n+ ") for r in rows]
    colors = ["#B1B7BE", "#159A8C", "#7960AB"][:len(rows)]
    for ax,key,title,unit in zip(axes,["tpot_ms","output_tok_s","dc_concurrent_compute_us_per_layer"],
            ["Formal TPOT ↓", "Formal output throughput ↑", "Profile: D/C concurrent with compute"],
            ["ms/token", "output token/s", "µs per GPU per layer"]):
        values = [r[key] for r in rows]
        ax.bar(np.arange(len(rows)),values,color=colors,width=.62)
        for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f"{v:.3f}" if key=="tpot_ms" else f"{v:.1f}",ha="center")
        ax.set_xticks(np.arange(len(rows)),labels)
        ax.set_title(title,fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0,max(values)*1.18 if max(values)>0 else 1)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("EP2: existing SGLang Qwen shared-expert / DeepEP overlap",fontsize=16,weight="bold")
    fig.text(.5,.015,"Same H100 NVLink pair | B8 | GPU-cached 4096 + new 1 → output 128 | 5 formal cohorts/config; GPU profile is a separate execution",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.90])
    for suffix in ["png","pdf"]:fig.savefig(report/f"native_ep_overlap.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analyze()
