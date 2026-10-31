"""Build a compact, source-linked report from qualified scheduling experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
HISTORY = [
    "shared_gemm_gap_r2", "shared_gemm_stream_r5", "shared_gemm_prep_r6",
    "shared_gemm_lookahead_r7", "shared_gemm_final_r8",
]
LABELS = {
    ("ep2", 0): "EP2 / shared GEMM",
    ("pp2", 0): "PP2 / shared GEMM",
    ("pp2", 1): "PP2 / early activation",
    ("pp2", 8): "PP2 / lookahead",
    ("pp2", 9): "PP2 / wider lookahead",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def component(rows, arm, policy, key):
    selected = [r.get(key, 0) for r in rows if r["arm"] == arm and r["policy"] == policy]
    assert len(selected) == 2
    return 1000 * st.mean(selected)


def make_figure(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = data["formal_metrics"]
    profiles = {(r["arm"], r["policy"]): r for r in data["independent_profile"]}
    labels = [LABELS[r["arm"], r["policy"]] for r in metrics]
    colors = ["#7151A8", "#A9AFB7", "#D3A34B", "#45A69A", "#177E72"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15.6, 10.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.13], hspace=0.48, wspace=0.50)
    ax, bx, cx = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    ys = np.arange(len(metrics))
    tpots = np.array([r["tpot_ms"] for r in metrics])
    lows = tpots - np.array([r["cohort_tpot_min_ms"] for r in metrics])
    highs = np.array([r["cohort_tpot_max_ms"] for r in metrics]) - tpots
    ax.barh(ys, tpots, color=colors, xerr=[lows, highs], capsize=3, height=0.63)
    bx.barh(ys, [r["output_tok_s"] for r in metrics], color=colors, height=0.63)
    for i, r in enumerate(metrics):
        ax.text(r["cohort_tpot_max_ms"] + 0.12, i, f'{r["tpot_ms"]:.3f}', va="center")
        bx.text(r["output_tok_s"] + 12, i, f'{r["output_tok_s"]:.1f}', va="center")
    ax.set_yticks(ys, labels)
    bx.set_yticks(ys, [])
    for a in (ax, bx):
        a.invert_yaxis()
        a.set_axisbelow(True)
        a.grid(axis="x", alpha=0.20)
    ax.set_xlim(0, max(tpots) * 1.18)
    bx.set_xlim(0, max(r["output_tok_s"] for r in metrics) * 1.18)
    ax.set_xlabel("TPOT (ms/token), lower is better")
    bx.set_xlabel("Output throughput (token/s), higher is better")
    ax.set_title("A. Formal Decode: profiler off", loc="left", weight="bold")
    bx.set_title("B. Same 8-request cohort", loc="left", weight="bold")

    graph, handoff, feedback, epgap = [], [], [], []
    for r in metrics:
        p = profiles[r["arm"], r["policy"]]
        graph.append(p["model_graphs_ms"])
        handoff.append(p.get("forward_handoff_ms", 0))
        feedback.append(p.get("feedback_ms", 0))
        epgap.append(p["between_graphs_ms"] if r["arm"] == "ep2" else 0)
    left = np.zeros(len(metrics))
    for values, color, label in [
        (graph, "#476E98", "Model CUDA Graphs (includes EP collectives)"),
        (handoff, "#DFAB51", "PP stage-0 end to stage-1 start"),
        (feedback, "#CA7368", "PP stage-1 end to next same-slot stage-0"),
        (epgap, "#A9AFB7", "EP between-Graph interval"),
    ]:
        cx.barh(ys, values, left=left, color=color, label=label, height=0.63)
        for i, value in enumerate(values):
            if value > 0.4:
                cx.text(left[i] + value / 2, i, f"{value:.2f}", ha="center", va="center", color="white", weight="bold")
        left += np.array(values)
    for i, total in enumerate(left):
        cx.text(total + 0.10, i, f"{total:.2f}", va="center")
    cx.set_yticks(ys, labels)
    cx.invert_yaxis()
    cx.set_axisbelow(True)
    cx.grid(axis="x", alpha=0.20)
    cx.set_xlim(0, max(left) * 1.12)
    cx.set_xlabel("Separately profiled same-token cycle (ms); gaps include dependencies and other-slot work")
    cx.set_title("C. Model Graph duration stays stable; scheduling intervals shrink", loc="left", weight="bold", pad=14)
    cx.legend(loc="upper left", bbox_to_anchor=(0, -0.23), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("PP2 Decode: hide CPU retirement behind the next slot's real GPU work", x=0.04, ha="left", fontsize=19, weight="bold")
    fig.text(0.04, 0.935, "Qwen3.6-35B-A3B BF16 | same H100 NVLink pair | B8 = two fixed B4 groups | GPU-cached 4096 + new 1 → output 128", fontsize=11, color="#444444")
    fig.text(0.04, 0.018, "Five profiler-off cohorts per configuration. TPOT whiskers: observed range of cohort means. Panel C uses a separate, perturbed profiler execution; it is not an additive attribution of Panel A.", fontsize=9.1, color="#555555")
    fig.subplots_adjust(left=0.235, right=0.95, top=0.87, bottom=0.15)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"pp_decode_scheduling.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build(analysis):
    data = read(analysis / "results.json")
    assert all(data["checks"].values())
    assert all(r["cohorts"] == 5 for r in data["formal_metrics"])
    run = Path(data["run"])
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    metrics = {(r["arm"], r["policy"]): r for r in data["formal_metrics"]}
    ep, pp = metrics["ep2", 0], metrics["pp2", 0]
    best = min((r for r in metrics.values() if r["arm"] == "pp2"), key=lambda r: r["tpot_ms"])
    frozen = read(run / "config.json")
    launch = read(run / "launch_manifest.json")
    history = []
    for name in HISTORY:
        path = ROOT / "analysis" / name / "results.json"
        x = read(path)
        for r in x["formal_metrics"]:
            history.append({"run": name, **r})
    table(out / "optimization_history.csv", history)
    requests, rounds = [], []
    for metric in data["formal_metrics"]:
        raw = read(Path(metric["source"]))
        for cohort, group in enumerate(raw["rounds"]):
            rounds.append(dict(arm=metric["arm"], policy=metric["policy"], cohort=cohort,
                requests=len(group["requests"]), makespan_s=group["makespan_s"],
                output_tok_s=1024/group["makespan_s"],
                tpot_ms=st.mean(r["tpot_ms"] for r in group["requests"])))
            for index, r in enumerate(group["requests"]):
                requests.append(dict(arm=metric["arm"], policy=metric["policy"], cohort=cohort,
                    request_index=index, cached_tokens=r["cached_tokens"], output_tokens=r["output_tokens"],
                    tpot_ms=r["tpot_ms"], ttft_ms=r["ttft_s"]*1000, e2e_ms=r["e2e_s"]*1000,
                    output_ids_sha256=hashlib.sha256(json.dumps(r["output_ids"],separators=(",",":")).encode()).hexdigest()))
    table(out / "formal_requests.csv", requests)
    table(out / "formal_cohorts.csv", rounds)
    make_figure(data, out)
    comp = data["components"]
    profile = data["independent_profile"]
    lines = [
        "# PP2 Decode：瓶颈定位与调度优化", "",
        "当前 PP2 慢于 EP2 的主要可消除部分在模型 Graph 之间的依赖与 CPU 退休路径。保持相同 Expert GEMM，提前准备并发射下一槽位的真实 GPU 计算，可以把旧输出的 CPU 处理移出关键路径。", "",
        f'最后五轮对照：PP2 TPOT **{pp["tpot_ms"]:.3f} → {best["tpot_ms"]:.3f} ms/token**（降低 {(1-best["tpot_ms"]/pp["tpot_ms"])*100:.1f}%）；吞吐 **{pp["output_tok_s"]:.1f} → {best["output_tok_s"]:.1f} output tok/s**（提高 {(best["output_tok_s"]/pp["output_tok_s"]-1)*100:.1f}%）。相同通用 GEMM 优化的 EP2 为 **{ep["tpot_ms"]:.3f} ms/token、{ep["output_tok_s"]:.1f} tok/s**。', "",
        "![Formal measurements and independent profile](pp_decode_scheduling.png)", "",
        "## 正式测量", "",
        "同一 H100178 GPU0/1 NVLink 对、同一 BF16 模型；全局 8 请求，固定两组 B4。每请求已有真实 GPU 驻留的 4096-token 前缀，新增 1 token，生成 128 token；temperature=0，ignoreEOS，无推测解码。每配置五个正式 cohort。TPOT 为首末输出间隔 / 127；吞吐为 1024 / cohort makespan，包含首 token 的影响。", "",
        "| 配置 | TPOT ms/token | 五轮均值范围 ms | Output tok/s | TTFT ms | 请求 E2E ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["formal_metrics"]:
        lines.append(f'| {LABELS[r["arm"], r["policy"]]} | {r["tpot_ms"]:.3f} | {r["cohort_tpot_min_ms"]:.3f}–{r["cohort_tpot_max_ms"]:.3f} | {r["output_tok_s"]:.1f} | {r["ttft_ms"]:.2f} | {r["e2e_ms"]:.2f} |')
    lines += ["", "EP2 为 Attention DP2/TP1、40 层与 MoE EP2；PP2 为 Attention TP1、每阶段 20 层与 MoE EP1，两个流水槽位。固定 B4 成员在策略之间保持不变；等待八请求到齐的实验 admission 时间包含在客户端指标内。", "",
        "## 为什么 GEMM 差距解释不了原来的 TPOT 差距", "",
        "下面是独立 profile 中每 GPU、每层、B4 的均值，单位 µs；两张卡取均值。Expert GEMM 不含 EP 的布局适配和 dispatch/combine。", "",
        "| 配置 | Expert GEMM µs | Attention core µs | 跨 GPU dispatch + combine µs |",
        "|---|---:|---:|---:|",
    ]
    for r in data["formal_metrics"]:
        arm, policy = r["arm"], r["policy"]
        communication = sum(component(comp, arm, policy, k) for k in ("communication.dispatch", "communication.combine"))
        lines.append(f'| {LABELS[arm,policy]} | {component(comp,arm,policy,"expert.routed_gemm"):.2f} | {component(comp,arm,policy,"attention.core"):.2f} | {communication:.2f} |')
    ep_gemm, pp_gemm = [component(comp, a, 0, "expert.routed_gemm") for a in ("ep2", "pp2")]
    lines += ["", f'PP 单层 Expert 比 EP 多约 **{pp_gemm-ep_gemm:.2f} µs**；沿 40 层串行相加仅约 **{(pp_gemm-ep_gemm)*40/1000:.3f} ms**。这是量级比较，不是正式 TPOT 的精确归因。PP 整个模型 Graph 还省去了 EP 的逐层通信，所以判断端到端速度必须继续查看 Graph 外的反馈闭环。', "",
        "| 配置 | 模型 Graph 合计 ms | 前向交接间隔 ms | 反馈至下一同槽位间隔 ms | 独立 profile token cycle ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in profile:
        forward = f'{p["forward_handoff_ms"]:.3f}' if "forward_handoff_ms" in p else "—"
        feedback = f'{p["feedback_ms"]:.3f}' if "feedback_ms" in p else f'{p["between_graphs_ms"]:.3f}（EP Graph 间隔）'
        lines.append(f'| {LABELS[p["arm"],p["policy"]]} | {p["model_graphs_ms"]:.3f} | {forward} | {feedback} | {p["cycle_ms"]:.3f} |')
    window=data.get("profile_window",{})
    lines += ["", f'本表每 rank 丢弃前 {window.get("drop_initial_graphs",4)} 个 Graph，再选 {window.get("selected_forwards_per_rank",16)} 个 Graph；保守与宽窗口 lookahead 均已进入启用区间。原始早期窗口解析仍保留在 `analysis/shared_gemm_final_r8`，早期窗口中保守版本尚未完全启用，不能作为两种 lookahead 的稳态对比。', "",
        "这些区间由同一请求组的 stage0 Graph 结束、stage1 Graph 开始/结束及下一同组 stage0 Graph 开始配对而成，包含 CPU 调度、等待对端、通信与另一 microbatch 的工作。NCCL kernel 的驻留时间也包含等待，不能据此认定 NVLink 本身很慢。Profiler 会改变 CPU 路径，表内 cycle 与上面的正式 TPOT 属于不同执行；正式均值还覆盖启动和结束保护区间，不能与 profile 相减作精确分解。", "",
        "## 已实现的 PP 专属优化", "",
        "1. **提前提交 activation。** 原生 forward 返回后即用同一个 producer event 提交原有 payload；不等旧 microbatch 的 CPU 结果退休。保留发送完成与 Graph buffer 复用的依赖。", 
        "2. **提前准备下一步 metadata。** 已收到的真实 GPU token 继续通过原生 future_map 输入；KV 位置和长度提前更新，旧结果处理保留独立 snapshot，避免重复分配与重复递增。", 
        "3. **下一槽位提前发射。** 在 CPU 等待旧输出 D2H 前执行下个槽位的真实 forward。正常调度走到该槽位时复用同一 result/event，避免第二次计算、发送和 FIFO 入队。按消息类型隔离三个 NCCL channel；保留事件与 buffer 生命周期依赖。", "",
        "lookahead 使用原生 sampling、token relay、CUDA Graph 和通用 Expert GEMM；不预测 token 值。保守版本在稳定输出区间启用，宽窗口版本提前到已生成 2 token 后，并保留结束前与 Mamba/GDN 快照边界的保护。", "",
        "共同稳态窗口中，保守与宽窗口版本的 Graph 和反馈周期接近；结合其唯一实现差别，这支持宽窗口的额外正式收益主要来自让更多启动/结束附近 token 使用同一优化路径，而不是让单次 Expert 计算变快。", "",
        "## 优化过程与验证", "",
        "[逐轮数据](optimization_history.csv) 保留每轮自己的 EP/PP 对照和计时范围。提前发送有效；单独取消 tail echo、拆分 channel、增加 activation stream 没有稳定收益；提前 metadata 有小幅收益；进一步把真实 forward 放在 CPU 退休前带来主要改善。不同轮的均值不拼接成同一实验。", "",
        "异步 depth1 + 无类型共享 channel 的候选因消息顺序歧义发生停顿，已终止并清理，没有有效结果；另一个 typed-depth1 目录仅准备未运行。失败和未运行目录保留，不计入成功测量。", "",
        "最终对照中所有正式 PP 配置的完整 128-token 序列逐位置相同；另有 untimed payload hash、事件/消息通道检查以及 prepared/lookahead 生产与消费计数一致、pending 为空的检查。12 项 CPU 调度/边界测试通过。正式计时关闭 profiler，输出/通信验证与 profile 单独执行。EP 的相同 GEMM 优化保持开启；本次没有为 PP 再改 GEMM。", "",
        "该原型目前针对静态 PP2、固定 B4、无 stop/penalty/logprob/speculation 的稳定 Decode 做过完整验证；没有据此宣称动态到达与任意 batch 均可启用。冷 Prefill4096→1 为每布局三轮基线控制，本轮 lookahead 的性能结论仅针对 Decode。", "",
        "## 数据和复现入口", "",
        f'- [最终冻结实验]({run})；[冻结配置]({run/"config.json"})；[私有调度实现]({run/"code/gap_schedule.py"})。',
        f'- [正式指标]({analysis/"metrics.csv"})；[组件]({analysis/"components.csv"})；[逐 token profile]({analysis/"profile_cycles.csv"})；[完整解析及来源哈希]({analysis/"results.json"})。',
        f'- 源码 commit `{frozen["engine_commit"]}`；模型 revision `{launch["model_revision"]}`；调度改变只在实验私有 adapter，冻结 engine 未改。',
        f'- 分析：`python3 {ROOT/"analyze_gap.py"} --run {run} --out {analysis} --drop-graphs {window.get("drop_initial_graphs",4)} --forwards {window.get("selected_forwards_per_rank",16)}`；绘图/报告：`/home/bxb1/conda/bin/python {Path(__file__).resolve()} --analysis {analysis}`。',
        "- 每个已完成实验均有 Guard postcheck、owned cleanup 和冻结输入哈希验证；大日志与原始 trace 在原实验目录保留，归档保存引用和 SHA256。", "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "report_provenance.json").write_text(json.dumps({
        "analysis": str(analysis), "analysis_sha256": sha(analysis / "results.json"),
        "builder": str(Path(__file__).resolve()), "builder_sha256": sha(Path(__file__)),
        "selected_by": "lowest mean TPOT among all final PP policies; all policies shown",
        "best_policy": best["policy"], "history_analyses": [str(ROOT / "analysis" / n / "results.json") for n in HISTORY],
    }, indent=2) + "\n")
    print(json.dumps({"report": str(out / "REPORT.md"), "best_policy": best["policy"], "best_tpot_ms": best["tpot_ms"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    build(parser.parse_args().analysis.resolve())
