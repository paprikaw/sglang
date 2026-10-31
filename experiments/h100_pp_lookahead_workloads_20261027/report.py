"""Readable tables and publication-style static figures from validated results."""
import argparse
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="lookahead_r2")
    args = parser.parse_args()
    source = ROOT / "analysis" / args.run / "results.json"
    data = json.loads(source.read_text())
    assert not data["partial"] and not data["profiles_skipped"]
    assert data["run_audit"]["guard_postcheck"] == "passed"
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    rows, cfg = data["comparisons"], data["config"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    labels = ["B8\nRepeated", "B16\nRepeated", "B32\nRepeated", "B32\nDiverse"]
    colors = ["#148f82", "#bdc7d3", "#7158b5"]
    names = ["EP2: shared Expert kernel", "PP2: native schedule", "PP2: lookahead"]
    x, width = np.arange(len(rows)), .24
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), constrained_layout=True)
    for j, (arm, policy) in enumerate((("ep2", 0), ("pp2", 0), ("pp2", 9))):
        values = [next(m for m in data["metrics"] if m["workload"] == r["workload"] and
                       m["phase"] == "decode" and m["arm"] == arm and m["policy"] == policy) for r in rows]
        axes[0].bar(x + (j - 1) * width, [v["throughput_tok_s"] for v in values], width,
                    yerr=[v["throughput_stdev"] for v in values], capsize=3, color=colors[j], label=names[j])
        axes[1].bar(x + (j - 1) * width, [v["tpot_ms"] for v in values], width, color=colors[j], label=names[j])
    axes[0].set_ylabel("Output throughput (token/s), higher is better")
    axes[1].set_ylabel("Mean TPOT (ms), lower is better")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("PP lookahead across batch sizes and prompt diversity\n"
                 "Qwen3.6-35B-A3B BF16 · same 2 H100 GPUs · cached4096 + new1 → output128", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"workload_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for i, policy in enumerate((0, 9)):
        values = [next(p for p in data["profiles"] if p["workload"] == r["workload"] and
                       p["arm"] == "pp2" and p["policy"] == policy) for r in rows]
        xx = x + (i - .5) * .34
        graph = [v["model_graphs_ms"] for v in values]
        gaps = [v["between_graphs_ms"] for v in values]
        axes[0].bar(xx, graph, .32, color=colors[1 if i == 0 else 2], label="Model Graphs" if i == 0 else None)
        axes[0].bar(xx, gaps, .32, bottom=graph, color="#df9851", alpha=1 if i == 0 else .65,
                    label="Between-Graph waits" if i == 0 else None)
        for position, bottom, gap in zip(xx, graph, gaps):
            axes[0].text(position, bottom + gap + .12, "Native" if i == 0 else "Ahead", ha="center", fontsize=8)
        # Average the two rank-local per-layer measurements; this is not a
        # critical-path sum or request-level TPOT.
        gemm = [1000 * st.mean(c["expert.routed_gemm"] for c in data["components"]
                    if c["workload"] == r["workload"] and c["arm"] == "pp2" and c["policy"] == policy) for r in rows]
        axes[1].bar(xx, gemm, .32, color=colors[1 if i == 0 else 2], label="Native" if i == 0 else "Lookahead")
    axes[0].set_ylabel("Profiled same-slot token cycle (ms)")
    axes[1].set_ylabel("PP Expert GEMM (µs / layer / rank)")
    for ax in axes:
        ax.set_xticks(x, labels); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True); ax.legend(fontsize=9)
    fig.suptitle("Scheduling benefit versus unchanged Expert work\nIndependent GPU profiles; formal throughput measured with profiler OFF", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"scheduling_decomposition.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# PP lookahead 在更大 workload 上的 EP / PP 对照", "",
             "每组请求总数为 B8、B16、B32；PP 有两个流水槽位，每槽分别为4、8、16个请求。另有B32多主题输入。"
             "完整 Qwen3.6-35B-A3B BF16，在同一 H100178 的NVLink GPU0/1顺序测量。"
             "EP2为Attention DP2/TP1、MoE EP2；PP2为两级、每级20层、TP1/EP1。", "",
             "Decode每个请求为4096个真实GPU缓存token，加1个新输入token，生成128个token。"
             f"每配置{cfg['repeats']}个正式批次，关闭profiler。PP原生调度和lookahead按正反顺序交替，切换后先做不计时预热。"
             "EP和两种PP均使用相同的Expert小tile内核家族（本地T≤16）；EP原生shared-expert overlap开启，TBO关闭。", "",
             "lookahead使用此前最佳policy9组合：提前提交activation、按消息类型分开的NCCL通道、提前准备metadata，以及在旧输出CPU处理前发射下一槽位的真实forward。"
             "本轮扩展固定batch大小，并与相同分组的原生PP调度对照。", "",
             "吞吐为整批输出token数除以首请求开始至末请求最后token的墙钟时间；"
             "TPOT为每请求首token至最后token的时间除以127，再对请求平均。"
             "TTFT包含请求发送与排队；E2E为请求开始至最后token。固定PP分组的等待包含在计时内。", "",
             "## Decode 性能", "",
             "| Workload | EP2 tok/s ↑ | PP2 native tok/s ↑ | PP2 lookahead tok/s ↑ | Lookahead vs native PP | Lookahead PP vs EP |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['workload']} | {r['ep_tok_s']:.1f} | {r['pp_baseline_tok_s']:.1f} | {r['pp_lookahead_tok_s']:.1f} | {r['pp_lookahead_gain_pct']:+.2f}% | {r['pp_vs_ep_pct']:+.2f}% |")
    lines += ["", "| Workload | EP2 TPOT ms ↓ | PP2 native TPOT ms ↓ | PP2 lookahead TPOT ms ↓ |", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['workload']} | {r['ep_tpot_ms']:.3f} | {r['pp_baseline_tpot_ms']:.3f} | {r['pp_lookahead_tpot_ms']:.3f} |")
    gains = [r["pp_lookahead_gain_pct"] for r in rows]
    winners = [r["workload"] for r in rows if r["pp_vs_ep_pct"] > 0]
    lines += ["", f"提前发射相对同一内核、同一分组的原生PP调度，吞吐变化为{min(gains):+.2f}%至{max(gains):+.2f}%。"
              f"优化PP在{len(winners)}/4组的吞吐高于EP。"
              "这些是本轮重新测量的对照；此前仅Expert内核的结果保留为独立历史实验。"]
    for r in rows:
        if r["requests"] == 32 and r["pp_lookahead_tpot_ms"] > r["ep_tpot_ms"]:
            ep = next(m for m in data["metrics"] if m["workload"] == r["workload"] and m["phase"] == "decode" and m["arm"] == "ep2")
            pp = next(m for m in data["metrics"] if m["workload"] == r["workload"] and m["phase"] == "decode" and m["arm"] == "pp2" and m["policy"] == 9)
            lines += ["", f"{r['workload']}：优化PP整批吞吐比EP高{r['pp_vs_ep_pct']:.2f}%，"
                      f"但TPOT高{(pp['tpot_ms']/ep['tpot_ms']-1)*100:.2f}%。"
                      f"PP的平均TTFT比EP短{ep['ttft_ms']-pp['ttft_ms']:.2f}ms，解释了这两个指标的不同方向。"]
    lines += ["", "| Workload | Layout | Mean TTFT ms | Mean request E2E ms |",
              "|---|---|---:|---:|"]
    for r in rows:
        for arm, p, name in (("ep2", 0, "EP2"), ("pp2", 0, "PP2 native"), ("pp2", 9, "PP2 lookahead")):
            m = next(m for m in data["metrics"] if m["workload"] == r["workload"] and m["phase"] == "decode"
                     and m["arm"] == arm and m["policy"] == p)
            lines.append(f"| {r['workload']} | {name} | {m['ttft_ms']:.2f} | {m['e2e_ms']:.2f} |")
    lines += ["", "整批128-token吞吐同时受到首token成本和最晚完成请求影响，不能直接用1/平均TPOT替代。"]
    lines += ["", "![Decode comparison](workload_comparison.png)", "", "## 原因：Graph之间的调度等待", "",
              "独立profile选择每rank第21–36个匹配本地batch大小的真实Decode Graph。"
              "PP通过序列长度总和与出现次序匹配两个stage，检查因果顺序，再回到同一流水槽位。"
              "between-Graph包含调度、反馈与通信等待，不能解释为纯链路传输时间；组件核函数累计时间不等于TPOT。", "",
              "| Workload | PP policy | Model Graphs ms | Between-Graph ms | Same-slot cycle ms |",
              "|---|---|---:|---:|---:|"]
    for r in data["profiles"]:
        if r["arm"] == "pp2":
            lines.append(f"| {r['workload']} | {'native' if r['policy']==0 else 'lookahead'} | {r['model_graphs_ms']:.3f} | {r['between_graphs_ms']:.3f} | {r['cycle_ms']:.3f} |")
    lines += ["", "![Scheduling decomposition](scheduling_decomposition.png)", "", "## Expert / Attention / Dispatch + Combine", "",
              "均为两个rank的每实际层调用GPU时间平均，单位µs；Expert只计routed GEMM，Attention只计core kernel，不含projection。"
              "EP/PP分别每rank40/20层。EP通信与共享计算可重叠，不将这些列相加为TPOT。", "",
              "| Workload | Layout | Expert GEMM µs | Attention µs | Dispatch + Combine µs |", "|---|---|---:|---:|---:|"]
    for case in cfg["workloads"]:
        for arm, p, name in (("ep2", 0, "EP2"), ("pp2", 0, "PP2 native"), ("pp2", 9, "PP2 lookahead")):
            values = [c for c in data["components"] if c["workload"] == case["id"] and c["arm"] == arm and c["policy"] == p]
            gemm = 1000 * st.mean(c["expert.routed_gemm"] for c in values)
            attn = 1000 * st.mean(c["attention.core"] for c in values)
            comm = 1000 * st.mean(c.get("communication.dispatch", 0) + c.get("communication.combine", 0) for c in values)
            lines.append(f"| {case['id']} | {name} | {gemm:.2f} | {attn:.2f} | {comm:.2f} |")
    lines += ["", "## 冷 Prefill 控制", "",
              "每请求4096个新输入token→1个输出，无缓存。lookahead只用于Decode，Prefill保持原生调度。", "",
              "| Workload | Layout | Input tok/s ↑ | TTFT ms | E2E ms |", "|---|---|---:|---:|---:|"]
    for m in data["metrics"]:
        if m["phase"] == "prefill":
            lines.append(f"| {m['workload']} | {m['arm']} | {m['throughput_tok_s']:.1f} | {m['ttft_ms']:.2f} | {m['e2e_ms']:.2f} |")
    lines += ["", "## 生效与正确性", "",
              "- 每个正式lookahead批次，两rank实际提前发射计数均增加，生产/消费匹配，pending为空。",
              "- PP同一workload所有正式批次的完整128-token输出逐位置一致；两stage真实payload的有序hash匹配，实际分组与EP偶/奇请求分配一致。",
              "- 每请求精确检查GPU缓存4096、输出128，以及SSE事件1–128；16项CPU调度、边界与回执读取检查通过。",
              "- 固定batch、两槽、ignoreEOS的Decode原型；这些测量支持四个所测workload，不代表动态到达场景。",
              "- 所有冻结输入、Guard模型缓存postcheck、owned cleanup与lease release检查通过。", "",
              "前次r1在读取尚未写完的wire回执JSON时中止，已验证清理并保留原始证据。"
              "本报告正式指标全部来自补上文件稳定性屏障后的完整重跑，不拼接r1的性能数据。", "",
              "## 可复现材料", "",
              f"- [正式指标]({source.parent / 'metrics.csv'})；[逐批次]({source.parent / 'cohorts.csv'})；[逐请求]({source.parent / 'requests.csv'})。",
              f"- [分项]({source.parent / 'components.csv'})；[profile cycles]({source.parent / 'profile_cycles.csv'})；[完整分析]({source})。",
              f"- [冻结运行]({data['run']})；[启动manifest]({Path(data['run']) / 'launch_manifest.json'})。",
              "- 图中误差线为5个整批吞吐样本的标准差；所有指标的样本范围和原始来源SHA256保留在分析中。", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(report=str(out / "REPORT.md"), comparisons=rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
