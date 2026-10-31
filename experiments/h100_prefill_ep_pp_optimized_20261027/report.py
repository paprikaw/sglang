"""Readable Prefill comparison and scientific figures from validated results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = ["#159b8b", "#8d9da9", "#7454b8"]
LABELS = ["EP2", "PP2 native", "PP2 early activation"]
KEYS = [("ep2", 0), ("pp2", 0), ("pp2", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    source = ROOT / "analysis" / args.run.name / "results.json"
    result = json.loads(source.read_text())
    assert not result["partial"] and not result["profiles_skipped"]
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(source.read_text())
    metrics = {(r["workload"], r["arm"], r["policy"]):r for r in result["metrics"]}
    components = {(r["workload"], r["arm"], r["policy"]):r for r in result["components"]}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, batch in zip(axes, (8, 32)):
        for index, ((arm, policy), color, label) in enumerate(zip(KEYS, COLORS, LABELS)):
            rows = [metrics[(f"b{batch}_q{q}_repeated",arm,policy)] for q in (128,512,4096)]
            ax.bar(np.arange(3) + (index-1)*.25, [r["throughput_tok_s"]/1000 for r in rows],
                width=.24, color=color, label=label,
                yerr=[r["throughput_stdev"]/1000 for r in rows], capsize=3)
        ax.set_xticks(np.arange(3), ["128*", "512*", "4096"])
        ax.set_title(f"B{batch}: {batch} concurrent requests")
        ax.set_xlabel("New input tokens per request; cached context = 0")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel("Input throughput (thousand tokens/s)")
    axes[0].legend(frameon=False)
    axes[0].text(.98,.02,"Q4096: see independent 10-round recheck",transform=axes[0].transAxes,
                 ha="right",va="bottom",fontsize=8,color="#555555")
    fig.suptitle("Cold Prefill: EP2 versus PP2 with early activation submission")
    fig.text(.5,-.015,"* Native EP/PP first-token outputs differ; cross-layout numerical equivalence remains unresolved.",
             ha="center",fontsize=9,color="#7b3838")
    fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(out / f"prefill_throughput.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    cases = ["b32_q128_repeated", "b32_q512_repeated", "b32_q4096_repeated", "b32_q4096_diverse"]
    for ax, field, title in zip(axes,
            ("expert_gemm_us_per_layer", "dispatch_combine_us_per_layer"),
            ("Routed Expert GEMM", "Expert dispatch + combine")):
        for i, ((arm,p), color, label) in enumerate(zip(KEYS,COLORS,LABELS)):
            ax.bar(np.arange(4)+(i-1)*.25, [components[(c,arm,p)][field] for c in cases],
                   width=.24, color=color, label=label)
        ax.set_xticks(np.arange(4), ["Q128", "Q512", "Q4096", "Q4096\ndiverse"])
        ax.set_title(title); ax.set_ylabel("GPU time (µs/card/actual layer call)")
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.legend(LABELS,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.025))
    fig.suptitle("B32 Prefill GPU profiles; actual token chunks reported in the table")
    fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(out / f"prefill_components.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cfg = result["config"]
    lines = ["# H100 Prefill: EP2 and PP2 after existing optimizations", "",
        "完整 Qwen3.6-35B-A3B BF16，固定 SG8702697022，H100178 NVLink GPU0/1，顺序运行。"
        "EP2 的 Attention 为 DP2×TP1、MoE 为 EP2，每卡40层；PP2 为两级、TP1/EP1，每级20层、两个流水槽。", "",
        "每请求缓存上下文 P=0，新输入 Q=128、512、4096，输出1个token。B表示同时提交的请求数，"
        "所测 B8/B32；总新输入为 B×Q。重复文档以及B32/Q4096的32主题输入分别报告。"
        "两种布局按相同偶/奇文档分组等待请求到齐，然后进入原生调度；等待计入客户端延迟。"
        "EP每个DP副本拿B/2个请求；PP微批次请求上限B/2。每个Attention实例的Prefill token预算4096，"
        "实际每次forward的token数量从独立trace核对。", "",
        "双方保留此前相同的Expert内核选择器：只有本地输入T≤16会选小tile内核，"
        "本轮Prefill大形状使用原生DeepGEMM；因此不能把Decode的小tile加速写成本轮Prefill加速。"
        "EP保留原生shared-expert overlap开关，DeepEP auto在大Prefill上采用normal路径，TBO关闭。"
        "PP对照policy0与既有policy1：提前提交原始activation发送，保留原生序列化、producer event和消息内容。"
        "Decode专用固定小形状传输与下一槽位提前forward未使用。", "",
        "每配置先3轮冷Prefill预热，再5轮profiler关闭的正式计时；PP两种policy交替、正反排序。"
        "每轮先清缓存，逐请求验证cached_tokens=0、输入Q、输出1。输入吞吐=整批B×Q/首请求开始至最后请求首token的墙钟时间；"
        "TTFT和E2E为逐请求均值。本实验只有1个输出token，TTFT=E2E，没有TPOT。", "",
        "## 吞吐与延迟", "",
        "| Workload | EP2 input tok/s ↑ | PP2 native input tok/s ↑ | PP2 early input tok/s ↑ | Early vs native PP | Early PP vs EP | EP TTFT/E2E ms ↓ | Early PP TTFT/E2E ms ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in result["comparisons"]:
        lines.append(f"| {c['workload']} | {c['ep_tok_s']:,.1f} | {c['pp_native_tok_s']:,.1f} | {c['pp_early_tok_s']:,.1f} | {c['early_gain_pct']:+.2f}% | {c['pp_early_vs_ep_pct']:+.2f}% | {c['ep_ttft_ms']:.2f} | {c['pp_early_ttft_ms']:.2f} |")
    lines += ["", "![Prefill throughput](prefill_throughput.png)", "",
        "图中误差线为5个整批吞吐样本的标准差；逐轮吞吐和makespan见cohorts.csv。", "",
        "## Expert GEMM 与通信", "",
        "以下来自另一次冷Prefill GPU profile，只统计host launch归属于真实Prefill层的GPU kernel，"
        "排除Decode调用。Expert GEMM只计routed Expert中的实际grouped GEMM kernel，"
        "包含本轮Prefill的grouped-contiguous GemmType1；不把完整MoE模块时间当成GEMM。"
        "Dispatch/combine计对应操作内GPU kernel，PP的Expert在本地，列为0，流水边界通信另属PP调度。"
        "每层时间先按各rank实际层调用数归一化，再对两个rank平均。GPU分项可能重叠，不能直接相加替代E2E。", "",
        "| Workload | Layout | Local source tokens/forward mean | Actual layer calls/card | Expert GEMM µs/card/layer | Dispatch+combine µs/card/layer | Expert GEMM ms/card/cohort | Attention module ms/card/cohort |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in cfg["workloads"]:
        for (arm,p), label in zip(KEYS,LABELS):
            c=components[(case["id"],arm,p)]
            lines.append(f"| {case['id']} | {label} | {c['local_source_tokens_mean']:.1f} | {c['actual_layer_calls']:.0f} | {c['expert_gemm_us_per_layer']:.2f} | {c['dispatch_combine_us_per_layer']:.2f} | {c['expert_gemm_ms_card']:.2f} | {c['attention_module_ms_card']:.2f} |")
    lines += ["", "![Prefill components](prefill_components.png)", "",
        "Attention module包含projection及其他Attention模块内kernel，口径与此前Decode的Attention core不同；"
        "不能直接跨阶段相减。每rank的实际chunk范围、调用数量和kernel清单在分析文件中。", "",
        "这些分项支持Attention相近、Expert执行与通信存在取舍，但不能用GEMM与通信kernel的和解释全部吞吐差距。"
        "例如B32/Q128的EP GEMM+dispatch/combine之和小于PP GEMM，PP的端到端吞吐仍更高；"
        "还需考虑完整执行顺序、调度、CPU开销与重叠。已检查的trace没有证据支持将差距直接归因于显式CUDA同步。", "",
        "## 生效与正确性", "",
        "- 每个正式PP policy1批次的stage0提前提交计数增加，stage1无提前提交；所有pending均清空。"
        "下一槽位lookahead、固定形状fast transport计数保持0，符合Prefill适用边界。",
        "- PP两种policy的首token输出逐请求一致，EP原生重复轮输出一致性单独保留；双方输入SHA256一致。",
        "- 原生GPU Prefill、客户端性能计时、独立profile各自保存；18项CPU检查通过。"
        "冻结输入、必需CPU权重缓存使用记录、Guard postcheck、owned cleanup和lease release通过。", "",
        "初始r1在EP原生跨轮输出一致性检查时中止，PP尚未运行，清理通过。"
        "已保留诊断，r2补上EP完整本地请求分组和额外预热。本报告所有性能数据来自完整r2，未拼接r1。", "",
        "## 可复现材料", ""]
    analysis=source.parent
    for label,name in (("正式指标","metrics.csv"),("逐批次","cohorts.csv"),("逐请求","requests.csv"),
                       ("GPU分项","components.csv"),("每rank分项","components_per_rank.csv"),("完整分析","results.json")):
        lines.append(f"- [{label}]({analysis/name})")
    lines += [f"- [冻结启动manifest]({Path(result['run'])/'launch_manifest.json'})", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(report=str(out/"REPORT.md"), comparisons=result["comparisons"]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
