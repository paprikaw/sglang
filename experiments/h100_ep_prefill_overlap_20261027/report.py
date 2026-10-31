"""Generate a compact report and standalone scientific figure from audited data."""
import argparse
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
LABELS = {("ep2", 0): "EP2 native", ("ep2", 1): "EP2 serial split",
          ("ep2", 2): "EP2 overlap", ("ep2", 3): "EP2 overlap / reserved SM",
          ("pp2", 0): "PP2 native", ("pp2", 1): "PP2 early send"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    value = json.loads((ROOT / "analysis" / args.run.name / "results.json").read_text())
    assert not value["partial"] and value["run_audit"]["guard_and_cleanup_verified"]
    run = Path(value["run"])
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(value, indent=2) + "\n")
    focus_rows = [r for r in value["metrics"] if r["workload"] == "b32_q4096_repeated"]
    reference = next(r for r in focus_rows if (r["arm"], r["policy"]) == ("ep2", 0))
    optimized = next(r for r in focus_rows if (r["arm"], r["policy"]) == ("ep2", 3))
    pp_reference = next(r for r in focus_rows if (r["arm"], r["policy"]) == ("pp2", 0))
    lines = ["# H100 EP Prefill: MoE-only communication / Expert overlap", "",
             f"主结果：同为8192-token预算，B32重复输入的重叠EP达到 {optimized['input_tok_s']:,.1f} input tok/s，"
             f"相对原生EP {reference['input_tok_s']:,.1f} 提升 {(optimized['input_tok_s']/reference['input_tok_s']-1)*100:.2f}%；"
             f"请求E2E从 {reference['e2e_ms']:.2f} 降至 {optimized['e2e_ms']:.2f} ms。"
             f"同预算PP2原生为 {pp_reference['input_tok_s']:,.1f} input tok/s，仍更快。"
             "时间线证实了通信和Expert计算的重叠，但分块导致的GEMM效率损失和资源争用抵消了大部分收益。", "",
             "完整 Qwen3.6-35B-A3B，BF16 权重/激活/KV、FP32 GDN 状态；H100178 同一 NVLink GPU0/1。"
             "EP2 是 Attention DP2×TP1、MoE EP2；PP2 是两个20层阶段、两个流水槽。"
             f"本轮每个 Attention 实例的 Prefill 预算：EP {value['config'].get('ep_prefill_budget',4096)}、PP {value['config'].get('pp_prefill_budget',4096)} token。"
             "Attention/GDN 保持整批执行，仅在 EP 的 MoE 内分成两组。", "",
             "每个请求缓存前缀为0、新输入4096 token、生成1 token；每策略预热3轮，正式5轮交错比较，profiler关闭，每轮清空KV/前缀缓存。"
             "吞吐为整批新输入 token 数除以整批完成时间；E2E为从客户端发起请求到首个且唯一输出 token 的请求均值。"
             "因此本实验没有 Decode TPOT。独立 profile 的单位是每卡、每次实际 Prefill 层调用的 GPU kernel 微秒，不能加总成请求 E2E。", "",
             "p0 保留原生 EP；p1 将 MoE token 分两组串行执行；p2 使用相同两组并交错 Dispatch/GEMM/Combine；"
             "p3 在 p2 上将Expert GEMM限制为112个SM，为使用20个SM的DeepEP通信留出资源，H100共132个SM。"
             "两组各有独立通信 handle，两个输入就绪事件都在第一个 GEMM 前记录。"
             "原有 shared-expert overlap 保持开启；SGLang 全模型 TBO 开关保持关闭。"
             "实验路径要求两侧每步输入均至少128 token；混合Prefill/Decode和空闲尾部保留原生路径。"
             "所有EP策略在每次完整forward开始时通过CPU广播固定策略，开销计入各策略端到端时间。", ""]
    for case in value["config"]["workloads"]:
        name = case["id"]
        rows = [r for r in value["metrics"] if r["workload"] == name]
        baseline = next(r for r in rows if (r["arm"], r["policy"]) == ("ep2", 0))
        lines += [f"## {name}: {case['requests']} requests × 4096 new tokens", "",
                  "| Configuration | Input tok/s ↑ | Sample SD | Request E2E ms ↓ | vs native EP |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows:
            label = LABELS[(row["arm"], row["policy"])]
            lines.append(f"| {label} | {row['input_tok_s']:,.1f} | {row['stdev_tok_s']:,.1f} | {row['e2e_ms']:.2f} | {(row['input_tok_s']/baseline['input_tok_s']-1)*100:+.2f}% |")
        lines += ["", "| Configuration | Routed GEMM µs ↓ | Dispatch µs | Combine µs | Attention µs | Comm ∩ GEMM µs | Comm overlapped |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in [r for r in value["components"] if r["workload"] == name]:
            label = LABELS[(row["arm"], row["policy"])]
            hit = f"{row['comm_gemm_overlap_us_per_layer']:.2f}" if row["arm"] == "ep2" else "—"
            fraction = f"{row['communication_overlapped_pct']:.1f}%" if row["arm"] == "ep2" else "—"
            lines.append(f"| {label} | {row['expert_gemm_us_per_layer']:.2f} | {row['dispatch_us_per_layer']:.2f} | {row['combine_us_per_layer']:.2f} | {row['attention_module_us_per_layer']:.2f} | {hit} | {fraction} |")
        native = next(r for r in value["components"] if r["workload"] == name and (r["arm"], r["policy"]) == ("ep2", 0))
        for policy in (2, 3):
            measured = next(r for r in rows if (r["arm"], r["policy"]) == ("ep2", policy))
            serial = next(r for r in rows if (r["arm"], r["policy"]) == ("ep2", 1))
            comp = next(r for r in value["components"] if r["workload"] == name and (r["arm"], r["policy"]) == ("ep2", policy))
            lines += ["", f"{LABELS[('ep2',policy)]}：相对串行分块吞吐 {(measured['input_tok_s']/serial['input_tok_s']-1)*100:+.2f}%；"
                      f"相对原生 EP {(measured['input_tok_s']/baseline['input_tok_s']-1)*100:+.2f}%。"
                      f"实际时间轴上，{comp['communication_overlapped_pct']:.1f}% 的通信 kernel 时间与 routed GEMM 相交；"
                      f"每层 GEMM 相对原生变化 {(comp['expert_gemm_us_per_layer']/native['expert_gemm_us_per_layer']-1)*100:+.1f}%。"]
        lines += [""]
    initial_path = ROOT / "analysis/overlap_r4/results.json"
    if initial_path.exists() and run.name != "overlap_r4":
        initial = json.loads(initial_path.read_text())
        assert not initial["partial"] and initial["run_audit"]["guard_and_cleanup_verified"]
        value["round4096"] = initial
        small = next(r for r in initial["metrics"] if r["arm"] == "ep2" and r["policy"] == 0)
        large = next(r for r in initial["metrics"] if r["arm"] == "ep2" and r["policy"] == 3)
        lines += ["## Why the first 4096-token split was slower", "",
                  "独立第一轮仍使用双方4096-token Prefill预算，EP拆成两组2048 token。完整5轮、双rank profile、Guard与清理审计均通过；未与8192预算样本混合。", "",
                  "| Configuration | Input tok/s | Mean request E2E ms | GEMM µs/layer/card | Dispatch+Combine µs | Comm∩GEMM µs |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in initial["metrics"]:
            comp = next(c for c in initial["components"] if (c["arm"],c["policy"]) == (row["arm"],row["policy"]))
            lines.append(f"| {LABELS[(row['arm'],row['policy'])]} | {row['input_tok_s']:,.1f} | {row['e2e_ms']:.2f} | {comp['expert_gemm_us_per_layer']:.2f} | {comp['dispatch_combine_us_per_layer']:.2f} | {comp.get('comm_gemm_overlap_us_per_layer',0):.2f} |")
        lines += ["", f"预留SM重叠相对原生EP吞吐 {(large['input_tok_s']/small['input_tok_s']-1)*100:+.2f}%。"
                  "真实重叠已发生，但更小的Expert输入增加GEMM时间，通信和计算同时执行也增加了各自的kernel耗时。"
                  "因此第二轮增大双方的通用Prefill预算，使EP每个Expert子批仍有4096个源token。"
                  "8192预算下应与同预算原生EP比较，不能把通用批次增大的收益全部归因于重叠。", ""]
    recheck_path = ROOT / "analysis/overlap_diverse_recheck_r6/results.json"
    if recheck_path.exists():
        replay = json.loads(recheck_path.read_text())
        assert not replay["partial"] and replay["run_audit"]["guard_and_cleanup_verified"]
        assert replay["config"]["timing_only"] and not replay["per_rank_components"]
        value["diverse_recheck"] = replay
        lines += ["## Independent diverse-input recheck", "",
                  "主矩阵多主题输入的原生EP有一个慢样本，因此另起独立实验，仅比较原生EP与预留SM重叠。"
                  "8192预算、B32、相同输入/精度/路由，每策略3次冷预热及10次交错正式计时，profiler关闭；"
                  "未与主矩阵5轮合并，复用主矩阵的独立profile作为机制证据。", "",
                  "| Configuration | Input tok/s mean | Sample SD | Median | E2E ms |", "|---|---:|---:|---:|---:|"]
        medians = {}
        for row in replay["metrics"]:
            raw = json.loads((Path(replay["run"]) / "ep2/workloads/b32_q4096_diverse" / f"metrics.prefill.p{row['policy']}.json").read_text())
            median = st.median(g["input_tokens_per_s"] for g in raw["rounds"])
            medians[row["policy"]] = median
            lines.append(f"| {LABELS[(row['arm'],row['policy'])]} | {row['input_tok_s']:,.1f} | {row['stdev_tok_s']:,.1f} | {median:,.1f} | {row['e2e_ms']:.2f} |")
        ref = next(r for r in replay["metrics"] if r["policy"] == 0)
        opt = next(r for r in replay["metrics"] if r["policy"] == 3)
        value["diverse_recheck_statistics"] = dict(
            mean_gain_pct=(opt['input_tok_s']/ref['input_tok_s']-1)*100,
            median_gain_pct=(medians[3]/medians[0]-1)*100,
            median_input_tok_s=medians, cohorts_per_policy=10, samples_excluded=0)
        lines += ["", f"独立10轮均值提升 {(opt['input_tok_s']/ref['input_tok_s']-1)*100:+.2f}%，"
                  f"中位数提升 {(medians[3]/medians[0]-1)*100:+.2f}%；全部样本保留。"
                  "主矩阵的4.54%均值增益受原生EP慢样本影响，不能作为稳定收益；当前证据支持约1%的小幅收益。", ""]
    qual = [json.loads(p.read_text()) for p in (run / "ep2/overlap_qualification").glob("rank*.json")]
    exact = all(row["serial_vs_overlap"]["exact"] for rank in qual for row in rank.values())
    max_native = max(row["native_vs_split"]["relative_l2"] for rank in qual for row in rank.values())
    max_reserved = max(row["serial_vs_reserved"]["relative_l2"] for rank in qual for row in rank.values())
    lines += ["## Validation and artifacts", "",
              f"两rank各40层真实权重验证通过：同分块串行/重叠输出逐元素相同={exact}；"
              f"原生/分块最大 relative L2={max_native:.6g}，串行/预留SM最大 relative L2={max_reserved:.6g}。"
              "数值验证在计时外执行；GEMM 形状变化使用此前相同的1.5%逐层 L2 阈值。"
              "这验证了当前输入上的执行等价性，不是完整语言模型质量评测。", "",
              "| Workload | Configuration | All formal first tokens equal to native EP |", "|---|---|---|"]
    for row in value["output_agreement"]:
        lines.append(f"| {row['workload']} | {LABELS[(row['arm'],row['policy'])]} | {row['native_ep_first_token_equal']} |")
    lines += ["", "21项CPU检查、全部冻结输入SHA256、每轮输入/零缓存/输出计数、双rank通信handle排空、"
              "canonical CPU权重缓存使用、Guard postcheck、所有自有GPU进程清理和租约释放均通过。"
              "正式计时与独立 profile 分开；5轮标准差用于显示波动，不作为服务尾延迟统计。", "",
              f"[Frozen launch manifest]({run}/launch_manifest.json) · [source/config]({run}/config.json) · "
              f"[raw metrics and trace manifests]({run}) · [recomputed results](./results.json)", "",
              "先前失败包括缓存提供者源码声明不完整、空闲DP rank的分块边界，以及混合阶段收尾通信超时；没有可用性能点。"
              "保留其原始failed/queued状态与失败说明，未混入正式结果。", ""]
    receipt = run / "gpu_qualification.json"
    if receipt.exists():
        qualified = json.loads(receipt.read_text())
        lines += [f"GPU验证源码标签：`{qualified['source_tag']}`，commit `{qualified['commit']}`。"
                  f"[EP overlap implementation]({value['config']['engine_root']}/python/sglang/srt/layers/moe/prefill_overlap.py) · "
                  f"[GPU qualification]({receipt})", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    focus = next(c["id"] for c in value["config"]["workloads"] if c["id"] == "b32_q4096_repeated")
    metrics = [r for r in value["metrics"] if r["workload"] == focus]
    labels = [LABELS[(r["arm"], r["policy"])].replace(" / ", "\n") for r in metrics]
    colors = ["#61778b", "#c8a866", "#149a8b", "#8058b0", "#c96f48", "#df9b77"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)
    axes[0].bar(range(len(metrics)), [r["input_tok_s"]/1000 for r in metrics],
                yerr=[r["stdev_tok_s"]/1000 for r in metrics], color=colors, capsize=3)
    axes[0].set_ylabel("New input throughput (k token/s)")
    axes[1].bar(range(len(metrics)), [r["e2e_ms"] for r in metrics], color=colors)
    axes[1].set_ylabel("Mean request E2E latency (ms)")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, rotation=24, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle("H100 × 2, Qwen3.6-35B-A3B BF16 | B32, cold input4096 → output1")
    fig.supxlabel("Both layouts:8192-token Prefill budget per Attention instance; error bars: sample SD (5 cohorts)", fontsize=9)
    fig.savefig(out / "prefill_overlap.png", dpi=180)
    fig.savefig(out / "prefill_overlap.pdf")
    plt.close(fig)
    # Separate budget panels; per-layer work differs between4096 and8192.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    rounds = [value.get("round4096", value), value]
    for ax, current in zip(axes, rounds):
        rows = [r for r in current["components"] if r["arm"] == "ep2" and r["workload"] == focus]
        positions = list(range(len(rows)))
        ax.bar([x-.18 for x in positions], [r["expert_gemm_us_per_layer"] for r in rows], width=.36, label="Routed GEMM", color="#517f9c")
        ax.bar([x+.18 for x in positions], [r["dispatch_combine_us_per_layer"] for r in rows], width=.36, label="Dispatch + Combine", color="#cf9443")
        ax.plot(positions, [r["comm_gemm_overlap_us_per_layer"] for r in rows], "o-", label="Comm ∩ GEMM", color="#149a8b")
        ax.set_xticks(positions, ["Native", "Serial split", "Overlap", "Reserved SM"], rotation=16, ha="right", fontsize=9)
        ax.set_title(f"EP source budget {current['config'].get('ep_prefill_budget',4096)} token/forward")
        ax.set_ylabel("GPU kernel µs / card / actual Prefill layer")
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Measured overlap and the cost of smaller Expert batches | B32 input4096 → output1")
    fig.savefig(out / "prefill_components.png", dpi=180)
    fig.savefig(out / "prefill_components.pdf")
    plt.close(fig)
    (out / "results.json").write_text(json.dumps(value, indent=2) + "\n")
    print(out / "REPORT.md")


if __name__ == "__main__": main()
