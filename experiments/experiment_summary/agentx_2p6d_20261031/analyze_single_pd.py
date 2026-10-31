"""CPU-only decomposition of the fourteen source-verified C8 episodes."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def average(values):
    return statistics.fmean(values)


def key(row):
    return (row["conversation_id"].split(":copy0:", 1)[-1], row["turn"])


def main():
    comparison = json.loads((HERE / "c8_comparison_evidence.json").read_text())
    episodes = collections.defaultdict(list)
    for source in comparison["sources"]:
        path = Path(source["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        raw = json.loads(path.read_text())
        assert len(raw["requests"]) == raw["completed_requests"] == 171
        rows = {key(r): r for r in raw["requests"]}
        assert len(rows) == 171
        assert all(isinstance(r["output_ids"], list)
                   and len(r["output_ids"]) == r["output_tokens"] for r in rows.values())
        episodes[source["arm"]].append((path.stem.split(".")[-1], rows))
    baseline_ids = set(episodes["col_single"][0][1])
    assert all(set(rows) == baseline_ids
               for group in episodes.values() for _, rows in group)
    by_arm = {}
    bins = []
    for item in comparison["measurements"]:
        arm = item["arm"]
        rows = [r for _, ep in episodes[arm] for r in ep.values()]
        meta = [r["prefill_meta"] if arm.startswith("pd_") else r["meta_info"]
                for r in rows]
        for entry in meta:
            detail = entry.get("cached_tokens_details")
            if detail is None:
                assert entry["cached_tokens"] == 0
            else:
                assert detail["host"] == 0
                assert detail["device"] + detail["host"] == entry["cached_tokens"]
        span = average(1000 * (m["prefill_finished_time"] - m["forward_entry_time"])
                       for m in meta)
        queue = average(1000 * m["queue_time"] for m in meta)
        pre = average(r["ttft_ms"] for r in rows)
        post = average(r["e2e_ms"] - r["ttft_ms"] for r in rows)
        total = average(r["e2e_ms"] for r in rows)
        assert abs(pre + post - total) < 1e-6
        by_arm[arm] = dict(
            arm=arm, layout=item["layout"], cohort=item["cohort"],
            episodes=len(episodes[arm]), ttft_ms=pre, post_first_ms=post,
            e2e_ms=total, p_queue_ms=queue, p_forward_to_prefill_finish_ms=span,
            native_e2e_ms=average(1000 * r["meta_info"]["e2e_latency"] for r in rows),
            client_minus_native_e2e_ms=total - average(
                1000 * r["meta_info"]["e2e_latency"] for r in rows),
            ttft_minus_p_queue_and_span_ms=pre - queue - span,
            interval_weighted_tpot_ms=item["interval_weighted_tpot_ms"],
            p_prefix_hit_pct=item["p_prefix_hit_pct"],
            mean_new_prefill_tokens=item["mean_new_prefill_tokens"],
        )
        for lower, upper in [(0, 64), (64, 256), (256, 1024), (1024, 100000)]:
            sub = [r for r in rows if lower < r["output_tokens"] <= upper]
            bins.append(dict(
                arm=arm, output_lower_exclusive=lower, output_upper_inclusive=upper,
                requests=len(sub), mean_output_tokens=average(r["output_tokens"] for r in sub),
                post_first_ms=average(r["e2e_ms"] - r["ttft_ms"] for r in sub),
                interval_weighted_tpot_ms=sum(r["e2e_ms"] - r["ttft_ms"] for r in sub)
                / sum(r["output_tokens"] - 1 for r in sub),
            ))
    baseline = by_arm["col_single"]
    for arm, row in by_arm.items():
        delta = row["e2e_ms"] - baseline["e2e_ms"]
        row["extra_e2e_vs_single_ms"] = delta
        row["extra_ttft_vs_single_ms"] = row["ttft_ms"] - baseline["ttft_ms"]
        row["extra_post_first_vs_single_ms"] = row["post_first_ms"] - baseline["post_first_ms"]
        row["post_first_share_of_extra_pct"] = (
            100 * row["extra_post_first_vs_single_ms"] / delta if delta > 0 else None)
    singles = dict(episodes["col_single"])
    same_output = []
    for arm, group in episodes.items():
        if arm == "col_single":
            continue
        pairs = []
        total = 0
        for repeat, candidate in group:
            base = singles[repeat]
            assert candidate.keys() == base.keys()
            total += len(base)
            pairs.extend((base[k], candidate[k]) for k in base
                         if base[k]["output_ids"] == candidate[k]["output_ids"])
        result = dict(arm=arm, paired_requests=total, whole_output_equal=len(pairs))
        for index, label in enumerate(("single", "candidate")):
            sub = [pair[index] for pair in pairs]
            result[label + "_mean_output_tokens"] = average(r["output_tokens"] for r in sub)
            result[label + "_interval_tpot_ms"] = (
                sum(r["e2e_ms"] - r["ttft_ms"] for r in sub)
                / sum(r["output_tokens"] - 1 for r in sub))
        same_output.append(result)
    with (HERE / "single_pd_decomposition.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(baseline))
        writer.writeheader()
        writer.writerows(by_arm.values())
    payload = dict(
        source_comparison=str(HERE / "c8_comparison_evidence.json"),
        decomposition=list(by_arm.values()), output_length_bins=bins,
        identical_output_subsets=same_output,
        same_version_configuration_path_difference_ms=dict(
            single_to_col_ep4=by_arm["col_ep4"]["e2e_ms"] - baseline["e2e_ms"],
            col_ep4_to_pd_ep4=by_arm["pd_ep4"]["e2e_ms"] - by_arm["col_ep4"]["e2e_ms"]),
        limitations=[
            "Post-first latency includes GPU kernels, communication, scheduling and streaming.",
            "P forward-to-finish span is not exclusive CUDA execution time.",
            "P queue excludes time outside the native wait queue, including some bootstrap.",
            "Residual TTFT is not a standalone KV transfer measurement; phases may overlap.",
            "Identical-output subset is selected and is not a whole-population estimate.",
            "Mixed EP2/EP4 versus single still confounds Decode layout and P/D separation.",
            "No causal GPU component profile or time-weighted utilization measurement.",
        ],
    )
    (HERE / "single_pd_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# C8 单卡混部为什么比当前 PD 快",
        "",
        "当前最有支持的解释：低 Decode batch 下，单卡副本的执行路径更快；"
        "固定 P 池又增加了首 token 前等待。P 排队只解释部分总差距。"
        "当前 PD 的 D 使用 EP，尚不能把这组差距归因为 P/D 分离本身。",
        "",
        "这里的单卡是 8 个独立单卡实例，总预算仍为 8 张 H100。"
        "所有结果均为当前 AgentX 派生 C8 的完整 episode，HiCache 开启。"
        "旧单卡、EP4混部、4P4D均为 E5/V3 三轮，新2P6D为 E8/V4各一轮。",
        "",
        "## 先定位差距",
        "",
        "| 部署 | 首 token 前 ms | 首 token 后至结束 ms | 请求 E2E ms |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("col_single", "col_ep4", "pd_ep4",
                "pd_2p6d_pp2_ep4ep2", "pd_2p6d_pp2_3ep2"):
        row = by_arm[arm]
        lines.append(f'| {row["layout"]} | {row["ttft_ms"]:.2f} | '
                     f'{row["post_first_ms"]:.2f} | {row["e2e_ms"]:.2f} |')
    lines += [
        "",
        "对新的 EP4＋EP2 D：比单卡多 977.92ms，首 token 前增加267.93ms，"
        "首 token 后增加709.99ms，后者约占73%。"
        "旧同版本4P4D的额外1032.10ms中，首 token后约占78%。"
        "首 token后包含通信、GPU执行、调度和流式返回，不能直接称为纯Expert计算。",
        "",
        "同一E5/V3的配置路径为：单卡混部2558.71ms → EP4混部3542.97ms"
        "（增加984.27ms）→ EP4的4P4D 3590.81ms（再增加47.84ms）。"
        "这是配置均值差分；不能将其中任何百分比解释成纯通信占比。",
        "",
        "## 已有证据支持什么",
        "",
        "- 实际执行粒度小：旧单卡本地Decode batch步均值约1.04，"
        "旧EP4混部约1.06，新PD约1.12；这些计数以发生forward的步为权重，不能当GPU利用率。"
        "EP带来逐层专家交换和rank协调，其权重分片优势未必转化为低batch延迟优势。"
        "- 启动代码开启DP Attention，EP4的tp=dp=ep=4。"
        "Attention在每个DP rank独立处理本地请求；EP4不会把单个请求的长context Attention均摊到四卡。"
        "每卡只持有四分之一专家，也不等于每步只执行单卡的四分之一工作。",
        "- 一个均衡路由的简化例子：四卡各有1个Decode token、top-8。"
        "四个单卡副本各处理8行token-expert工作；EP4全组32行平均分到四卡，"
        "每卡仍是8行，却需要交换。此例不表示8个唯一专家，实际专家复用仍取决于路由。"
        "较大batch下权重复用与分片容量才可能抵消这些额外成本。",
        "- 独立纯Decode实验也有相同方向：四卡节点、热缓存P4096/N8时，"
        "单卡副本4.752ms、EP4 6.122ms；P131072/N8时为6.550/7.970ms。"
        "它们是不同协议的旁证，不与AgentX数值合并。",
        "- 缓存命中几乎一致：P侧约91.46%，本组host命中均为0；"
        "新增Prefill约7245与7248token/请求。没有证据把C8差距主要归为缓存未命中。"
        "权重已在GPU中，CPU权重加载不在本轮请求计时内。",
        "",
        "## P 排队与交接",
        "",
        "| 部署 | 原生P排队 ms | P首次forward至Prefill结束 ms |",
        "|---|---:|---:|",
    ]
    for arm in ("col_single", "pd_ep4", "pd_2p6d_pp2_ep4ep2", "pd_2p6d_pp2_3ep2"):
        row = by_arm[arm]
        lines.append(f'| {row["layout"]} | {row["p_queue_ms"]:.2f} | '
                     f'{row["p_forward_to_prefill_finish_ms"]:.2f} |')
    lines += [
        "",
        "P阶段的跨度包括chunk之间的等待，不是纯GPU kernel时间。"
        "PP2 P的这一跨度反而较短，但它集中承接全部请求，排队显著增多。"
        "原生queue_time是进入wait queue到首次forward，不包含所有bootstrap等待。"
        "TTFT还包含握手、接收预分配、缓存状态交接、D开始服务与API开销；"
        "不能用TTFT减P计算就得到纯传输耗时。",
        "客户端E2E与联合／D服务端原生E2E的差约为11–15ms"
        "（EP4混部约22ms）。这没有显示HTTP/客户端线程开销能解释约1秒的总差距；"
        "服务端CPU调度仍包含在原生计时中。",
        "",
        "8个单卡混部实例可随本地会话阶段执行P或D；"
        "固定2P6D把P工作汇聚到一个PP2实例，D也仅有六卡。"
        "这种资源分工可能在有限并发与工具等待时留下闲置容量，"
        "目前缺少时间加权利用率，不能定量归因。",
        "",
        "若保持当前PD首token后的执行轨迹，即使假设TTFT为0，平均E2E仍约2920ms，"
        "高于单卡完整请求的2559ms。该代数假设说明单独压缩首token等待不够，"
        "不是对改变合批或增加D资源后的性能预测。",
        "",
        "## 输出一致性与结论边界",
        "",
        "另外筛选同版本单卡与4P4D的逐token输出完全相同请求：185/513对，"
        "其间隔加权TPOT仍为5.312/7.114ms。"
        "新EP4＋EP2 D与单卡第一轮的40/171个同输出对子集为5.468/6.902ms。"
        "这个有选择的子集不是全量性能估计，但说明生成序列变化不能单独解释全部慢速。"
        "其余请求并未达到完整输出一致，不能称为全并发数值等价验证。",
        "",
        "尚缺的最关键对照是在同引擎、同回放下让PD的D端也使用单卡副本，"
        "并保持相同输入和可驻留的缓存工作集。单卡必须使用容量可行的GDN/KV预算；"
        "不能直接照搬EP的256个GDN槽。"
        "该对照有助于区分D布局成本与P/D分离成本；目前还没有这组结果。",
        "",
        f'[延迟分解CSV]({HERE / "single_pd_decomposition.csv"}) · '
        f'[输出分组及同输出子集]({HERE / "single_pd_diagnosis.json"}) · '
        f'[复算脚本]({Path(__file__).resolve()}) · '
        f'[C8总表与原始来源]({HERE / "C8_COLOCATED_COMPARISON.md"})',
        "",
        f'[独立纯Decode对照]({HERE.parent / "decode_pool_necessity_20261030/REPORT.md"}) · '
        f'[本轮DP Attention启动代码]({HERE.parents[2] / "experiments/agentx_pd_dynamic_20261030/runs/two_p_six_d_ep4ep2_owner_balanced_e8_v4_gdn256_20261031_control/code/runner.py"}) · '
        f'[原生计时定义]({HERE.parents[2] / ".engine-worktrees/agentx-mha-pp-kv-layer-map-r8-20261031/python/sglang/srt/observability/req_time_stats.py"})',
    ]
    (HERE / "SINGLE_VS_PD_DIAGNOSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(report=str(HERE / "SINGLE_VS_PD_DIAGNOSIS.md"),
                          decomposition=list(by_arm.values()),
                          identical_output_subsets=same_output), ensure_ascii=False))


if __name__ == "__main__":
    main()
