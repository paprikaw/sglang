"""Render the offline timing diagnosis as a scientific figure and report."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUN = "fair_triton_20261026_v1"


def link(label, path):
    return f"[{label}]({path})"


def main():
    data = json.loads((OUT / "timing.json").read_text())
    timeline = json.loads((OUT / "timeline.json").read_text())
    ep, pp = (next(r for r in data["profile"] if r["run"] == RUN and r["arm"] == arm) for arm in ("ep2", "pp2"))
    formal = {r["arm"]: r for r in data["formal"] if r["run"] == RUN}
    components = list(csv.DictReader((ROOT / "analysis/components.csv").open()))
    costs = {}
    for arm in ("ep2", "pp2"):
        rows = [r for r in components if r["run"] == RUN and r["arm"] == arm and r["phase"] == "decode"]
        costs[arm] = {k: st.mean(float(r[k] or 0) for r in rows) for k in
                      ("expert.routed_gemm", "attention.core", "communication.dispatch", "communication.combine")}
    gemm_delta = 40 * (costs["pp2"]["expert.routed_gemm"] - costs["ep2"]["expert.routed_gemm"])
    formal_delta = formal["pp2"]["tpot_ms"] - formal["ep2"]["tpot_ms"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13.8, 8.1), gridspec_kw={"height_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.13, right=.98, top=.83, bottom=.13, hspace=.63)
    fig.suptitle("Why PP2 remains slower after shared Expert GEMM optimization", fontsize=19, y=.975)
    fig.text(.5, .925, f"Formal TPOT, profiler off: EP2 {formal['ep2']['tpot_ms']:.2f} ms  |  PP2 {formal['pp2']['tpot_ms']:.2f} ms", ha="center", fontsize=14)
    model, handoff, feedback = "#198F9B", "#E3A34E", "#D9DEE5"
    top.set_title("Profiled token cycle: model execution and gaps between model Graphs", loc="left", pad=15, fontsize=13)
    top.barh(1, ep["model_graphs_ms"], height=.48, color=model)
    top.barh(1, ep["between_graphs_ms"], left=ep["model_graphs_ms"], height=.48, color=feedback)
    top.text(ep["model_graphs_ms"] / 2, 1, f"40 layers: {ep['model_graphs_ms']:.2f} ms", color="white", va="center", ha="center")
    top.text(ep["token_cycle_ms"] + .15, 1, f"{ep['token_cycle_ms']:.2f} ms total", va="center")
    offset = 0
    for key, color, label in [("stage0_graph_ms", model, "Stage 0"),
                               ("forward_handoff_ms", handoff, "Handoff"),
                               ("stage1_graph_ms", model, "Stage 1"),
                               ("return_and_next_input_ms", feedback, "Feedback / next input")]:
        value = pp[key]
        top.barh(0, value, left=offset, height=.48, color=color, edgecolor="white", linewidth=1.2)
        top.text(offset + value / 2, 0, f"{label}\n{value:.2f} ms", ha="center", va="center", fontsize=10,
                 color="white" if color == model else "#182231")
        offset += value
    top.text(offset + .15, 0, f"{offset:.2f} ms total", va="center")
    top.set(yticks=[1, 0], yticklabels=["EP2", "PP2"], xlim=(0, 12.5), ylim=(-.6, 1.6), xlabel="Time per same-microbatch token cycle (ms, profiler enabled)")
    top.grid(axis="x", alpha=.2)
    top.set_axisbelow(True)
    t = timeline["pp2"]
    anchor = tuple(t["anchor_key"])
    slot_colors = ("#2878B5", "#36A98D")
    xmax = 14.4
    for row in t["rows"]:
        rank, y = row["rank"], 1 - row["rank"]
        bottom.barh(y, xmax, height=.34, color="#F0F2F4")
        for bar in row["graphs"]:
            start, end = max(0, bar["start_ms"]), min(xmax, bar["end_ms"])
            if start >= end:
                continue
            seq, occurrence = bar["key"]
            relative_token = (seq - anchor[0]) // 4
            name = "A" if occurrence == 0 else "B"
            suffix = "t" if relative_token == 0 else f"t{relative_token:+d}"
            bottom.barh(y, end - start, left=start, height=.34, color=slot_colors[occurrence])
            bottom.text((start + end) / 2, y, f"{name} ({suffix})", ha="center", va="center", color="white", fontsize=10)
    a = next(b for b in t["rows"][0]["graphs"] if tuple(b["key"]) == anchor)
    b = next(b for b in t["rows"][1]["graphs"] if tuple(b["key"]) == anchor)
    nxt = next(b for b in t["rows"][0]["graphs"] if tuple(b["key"]) == (anchor[0] + 4, anchor[1]))
    bottom.annotate("", xy=(b["start_ms"], .2), xytext=(a["end_ms"], .8), arrowprops={"arrowstyle": "->", "color": "#995F1A", "lw": 1.6})
    bottom.text((a["end_ms"] + b["start_ms"]) / 2 - .22, .38, "Handoff", fontsize=10, color="#995F1A", ha="center")
    bottom.annotate("", xy=(nxt["start_ms"], .8), xytext=(b["end_ms"], .2), arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.6})
    bottom.text((b["end_ms"] + nxt["start_ms"]) / 2 + .12, .38, "Feedback", fontsize=10, color="#6B7280", ha="center")
    bottom.annotate("", xy=(nxt["start_ms"], -.43), xytext=(0, -.43), arrowprops={"arrowstyle": "|-|", "color": "#28313B"})
    bottom.text(nxt["start_ms"] / 2, -.62, f"Slot A returns for its next token after {nxt['start_ms']:.2f} ms (one observed cycle)", ha="center", fontsize=10)
    bottom.set_title("PP2 trace: two B4 microbatches; only the model Graphs are colored", loc="left", fontsize=13, pad=14)
    bottom.set(yticks=[1, 0], yticklabels=["GPU0: layers 1–20", "GPU1: layers 21–40"], xlim=(0, xmax), ylim=(-.85, 1.5), xlabel="Time from the first Stage-0 Graph in this window (ms, profiler enabled)")
    bottom.grid(axis="x", alpha=.2)
    bottom.set_axisbelow(True)
    bottom.legend(handles=[Patch(color=slot_colors[0], label="Microbatch A (4 requests)"),
                           Patch(color=slot_colors[1], label="Microbatch B (4 requests)"),
                           Patch(color="#F0F2F4", label="Outside model Graph")],
                  loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False, fontsize=9)
    fig.text(.13, .045, "Profiled intervals explain the execution pattern. They are not an exact breakdown of profiler-off TPOT.", fontsize=10, color="#525B66")
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"tpot_timeline.{extension}", dpi=200)
    plt.close(fig)

    scheduler = Path(data["scheduler_source"]["path"])
    report = f"""# PP2 / EP2 Decode TPOT：离线时序诊断（2026-10-27）

结论：当前优化后PP2的剩余劣势，主要应从**模型Graph之间的stage交接、输出反馈与CPU调度开销**解释。GEMM差额不足以独立解释端到端差距。现有trace同时显示PP计算被压缩后，调度空档更加突出。

本次仅分析已保存结果，没有新增GPU测量。固定Qwen3.6-35B-A3B BF16，H100178同一NVLink GPU0/1，全局8请求；每请求缓存4096、新输入1、生成128。正式性能为5轮，profiler关闭；下述细分为另一次profiler采集，各rank丢弃前4个稳定B4 Graph后选16个，与原报告分项使用完全相同的Graph。

| 优化后配置 | Expert GEMM，µs/GPU/层 | 正式TPOT，ms | 正式Decode，tok/s |
|---|---:|---:|---:|
| EP2 | {costs['ep2']['expert.routed_gemm']*1000:.2f} | {formal['ep2']['tpot_ms']:.3f} | {formal['ep2']['throughput_tok_s']:.1f} |
| PP2 | {costs['pp2']['expert.routed_gemm']*1000:.2f} | {formal['pp2']['tpot_ms']:.3f} | {formal['pp2']['throughput_tok_s']:.1f} |

每层GEMM差约{(costs['pp2']['expert.routed_gemm']-costs['ep2']['expert.routed_gemm'])*1000:.2f} µs。按40层折算为约{gemm_delta:.3f} ms，而正式TPOT差为{formal_delta:.3f} ms（PP高{formal_delta/formal['ep2']['tpot_ms']*100:.1f}%）。这是量级核对，不能当成从正式TPOT扣除的一项精确归因。EP每层dispatch+combine GPU事件累计约{(costs['ep2']['communication.dispatch']+costs['ep2']['communication.combine'])*1000:.2f} µs；PP省去逐层DeepEP交换，其整体模型Graph反而更短。

**同一profile内部的逐token闭环。** EP两rank共32个同批次相邻token周期；PP16个同批次周期，跨卡按`g_sk`和同值出现序号匹配，验证下一token回到相同槽位。每一个PP样本都满足stage0结束≤stage1开始<stage1结束≤同槽位下一token的stage0开始，四段严格相加。

| Profile中的同一批请求 | GPU模型Graph | 前向stage交接 | 回传/下一输入准备 | 完整周期 |
|---|---:|---:|---:|---:|
| EP2 | {ep['model_graphs_ms']:.3f} ms（40层） | — | {ep['between_graphs_ms']:.3f} ms（Graph之外） | {ep['token_cycle_ms']:.3f} ms |
| PP2 | {pp['stage0_graph_ms']:.3f}+{pp['stage1_graph_ms']:.3f}={pp['model_graphs_ms']:.3f} ms（20+20层） | {pp['forward_handoff_ms']:.3f} ms | {pp['return_and_next_input_ms']:.3f} ms | {pp['token_cycle_ms']:.3f} ms |

PP模型Graph总计约5.77 ms，但同一组4请求必须跨两个stage，再等输出返回和下一输入就绪；两组4请求交替，一个请求的token间隔覆盖两个微步。EP每GPU同时推进本地4请求的40层，单个微步即覆盖全局8请求。**不能将PP一个20层forward的2.7–3.1 ms直接与EP的40层forward或请求TPOT比较。**

Profile里EP模型Graph覆盖各卡观察窗口约{ep['rank0_model_graph_fraction']*100:.1f}%/{ep['rank1_model_graph_fraction']*100:.1f}%；PP约{pp['rank0_model_graph_fraction']*100:.1f}%/{pp['rank1_model_graph_fraction']*100:.1f}%。这是Graph时间覆盖率，不是SM利用率；DeepEP等待已包含在EP的Graph之内，NCCL等待可能落在PP的Graph之外。

**源码与开关解释了这个模式。** 启动日志确认EP `disable_overlap_schedule=False`，PP为`True`，并明确记录“Pipeline parallelism is incompatible with overlap schedule.”。PP `pp_async_batch_depth=0`，`pp_loop_size=pp_size+depth=2`，实际两个B4槽位。{link('PP调度源码', str(scheduler)+':556')}依次发射当前batch，收取/处理待退休batch的输出，`d2h_event.synchronize()`，然后才提交当前stage激活；{link('结果退休及发送位置',str(scheduler)+':648')}。同步本身可与当前GPU计算重叠，不能把CPU同步时长再次加到Graph上；关键是未被覆盖的stage交接和下一轮准备进入了请求关键路径。

**PP通信时间也需区分等待。** Trace中的激活消息为两个8192元素的BF16数组（合计32 KiB），回传token为4个int64（32 B）。rank0的NCCL kernel可驻留数ms，而rank1对应kernel仅约数µs，符合接收端等待结果就绪的模式。这不是对NVLink纯带宽的测量，也不能把整个NCCL驻留时间算成搬运这些字节的成本。原始消息shape与事件驻留统计保存在timing.json。

**正式TPOT不是流式计数误差。** 四个配置各40请求、每请求128个事件，合计20480个事件均每次增加一个completion token，没有重复计数；根据唯一token重算TPOT与原值完全相同。TPOT仅计算首token到末token的127个间隔，不含TTFT。

Profiler确实改变了时间：PP的profile周期为{pp['token_cycle_ms']:.2f} ms，正式TPOT为{formal['pp2']['tpot_ms']:.2f} ms；EP分别为{ep['token_cycle_ms']:.2f}/{formal['ep2']['tpot_ms']:.2f} ms。因此，profile能证明等待位置和规模，不能声称正式9.28 ms里精确有4.91 ms调度开销。精确的profiler-off因果占比仍需要低干扰、按槽位记录GPU事件的受控测量。

下一步应优先优化PP的stage交接和autoregressive反馈闭环：让已就绪激活更早发送，将输出回传、D2H与CPU结果处理从下一批发射的关键路径移开，并联合选择微批次大小与在途槽位数。通用GEMM改进仍同时应用双方。两stage Graph时间也存在2.69/3.07 ms的不均衡；即便理想调度，单纯均分层数也不完全等于均分时间。假设保持这些stage执行时间、完全隐藏调度且有两个活跃B4槽位，模型给出的计算周期下界为`2×max(2.69,3.07)≈{pp['compute_only_pipeline_bound_ms']:.2f} ms`；这只是条件模型，尚未实测达到。

{link('时间轴PNG',OUT/'tpot_timeline.png')} · {link('PDF',OUT/'tpot_timeline.pdf')} · {link('全部逐周期数据',OUT/'profile_cycles.csv')} · {link('计时、开关及来源哈希',OUT/'timing.json')} · {link('原公平对比报告',ROOT/'deliverables/REPORT.md')}

![Profiled model execution and PP token feedback]({OUT/'tpot_timeline.png'})
"""
    (OUT / "REPORT.md").write_text(report)
    print(f"Report: {OUT / 'REPORT.md'}")
    print(f"Figure: {OUT / 'tpot_timeline.png'}")


if __name__ == "__main__":
    main()
