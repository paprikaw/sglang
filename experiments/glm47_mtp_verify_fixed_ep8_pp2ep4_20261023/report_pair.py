"""Read-only analysis of the fixed-Verify EP8 / PP2xEP4 paired experiment.

Production cohorts and fenced diagnostic profiles remain separate. This script
does not launch workers, alter the semantic bank, or edit frozen runtime code.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ARMS = ("ep8_mtp_fixed", "pp2ep4_mtp_fixed")
OLD_ARMS = ("ep8_mtp_on", "pp2ep4_mtp_on")
OLD = WORKSPACE / "experiments/glm47_semantic_bank_fourway_20261023_v5/runs/fourway_reuse_r3"
OFF = ("ep8_mtp_off", "pp2ep4_mtp_off")
OLD_PROFILE = WORKSPACE / "experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1"
SPEC = importlib.util.spec_from_file_location("saved_cohort_analysis", ROOT / "analysis/summarize.py")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def read(path):
    return json.loads(Path(path).read_text())


def mean(rows):
    return statistics.fmean(rows)


def spread(rows):
    return {"mean": mean(rows), "half_range": (max(rows) - min(rows)) / 2, "values": rows}


def clean_cohort(path):
    row = HELPERS.cohort(path)
    output_rows = row.pop("output_rows")
    cadence = []
    for request in output_rows:
        points = request["token_events"]
        gaps = [1000 * (b[1] - a[1]) for a, b in zip(points, points[1:]) if b[0] > a[0]]
        increments = [b[0] - a[0] for a, b in zip(points, points[1:]) if b[0] > a[0]]
        cadence.append({"request_index": request["request_index"],
                        "client_update_gap_ms": HELPERS.stats(gaps),
                        "client_update_tokens": HELPERS.stats(increments)})
    row["client_cadence"] = cadence
    row["client_cadence_boundary"] = "SSE update intervals, not GPU kernel cycles or pure communication time"
    return row, output_rows


def aggregate_activity(path):
    value = read(path)
    if not value["passed"] or value["production_tpot_partition"] is not False:
        raise ValueError("Unqualified profile or incompatible measurement boundary")
    ranks = sorted(value["ranks"], key=lambda r: r["rank"])
    if len(ranks) != 8 or not all(r["passed"] for r in ranks):
        raise ValueError("Eight passing diagnostic rank records required")
    groups = {"all_ranks": ranks} if ranks[0]["layout"] == "ep8" else {"stage0": ranks[:4], "stage1": ranks[4:]}
    result = {"source": HELPERS.provenance(path), "measurement": value["measurement"],
              "production_tpot_partition": False, "groups": {}}
    for name, rows in groups.items():
        group = {"ranks": [r["rank"] for r in rows], "roles": {}}
        for role in ("target", "draft"):
            selected = [r[role] for r in rows if r.get(role) and r[role].get("count", 0)]
            if not selected:
                continue
            metrics = {k: mean([r[k] for r in selected]) for k in
                       ("graph_span_ms", "attention_ms", "expert_ms", "communication_ms", "other_graph_ms", "overlap_ms")}
            exclusive = {k: mean([r["exclusive_ms"].get(k, 0) for r in selected])
                         for k in sorted(set().union(*(r["exclusive_ms"] for r in selected)))}
            metrics.update(exclusive_ms=exclusive, replay_count_per_rank=[r["count"] for r in selected])
            if role == "target":
                layers = 89 if name == "all_ranks" else 43 if name == "stage0" else 46
                metrics["moe_layers"] = layers
                metrics["dispatch_combine_ms"] = exclusive["deepep_dispatch"] + exclusive["deepep_combine"]
                metrics["dispatch_combine_per_moe_layer_ms"] = metrics["dispatch_combine_ms"] / layers
            group["roles"][role] = metrics
        group["phases"] = {}
        for phase in sorted(set().union(*(r.get("phases", {}) for r in rows))):
            phase_rows = [r["phases"][phase] for r in rows if phase in r.get("phases", {})]
            group["phases"][phase] = {k: mean([r[k] for r in phase_rows]) for k in
                                      ("graph_span_ms", "attention_ms", "expert_ms", "communication_ms", "other_graph_ms")}
        result["groups"][name] = group
    return result


def diagnostic_cycles(path):
    value = read(path)
    targets = [r for r in value["segments"] if r["phase"] == "target_verify"]
    grouped = {}
    for row in targets:
        grouped.setdefault(tuple(row["request_ids"]), []).append(row)
    cycles = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r["cpu_start_ns"])
        cycles.extend((b["cpu_start_ns"] - a["cpu_start_ns"]) / 1e6 for a, b in zip(rows, rows[1:]))
    return {"rank": value["rank"], "same_request_sets": [list(k) for k in grouped],
            "target_start_cycle_ms": HELPERS.stats(cycles),
            "target_host_call_ms": HELPERS.stats((r["cpu_end_ns"] - r["cpu_start_ns"]) / 1e6 for r in targets),
            "gap_to_next_local_target_ms": HELPERS.stats((b["cpu_start_ns"] - a["cpu_end_ns"]) / 1e6 for a, b in zip(targets, targets[1:])),
            "boundary_host_api_total_ms": sum((r["cpu_end_ns"] - r["cpu_start_ns"]) / 1e6 for r in value["pp_boundaries"]),
            "caveat": "Diagnostic completion fences change cadence. These gaps are not production bubble percentages, GPU idle, or pure NIC time.",
            "source": HELPERS.provenance(path)}


def summarize(run):
    task = read(run / "task/results.json")
    cleanup = read(run / "task/cleanup.json")
    if not task.get("passed") or not cleanup.get("passed"):
        raise ValueError("Both real-model arms and cleanup must complete before publishing this report")
    result = {"schema_version": 1, "run": str(run), "workload": {}, "arms": {}, "historical_same_protocol": {},
              "cleanup_passed": cleanup["passed"], "evidence": [HELPERS.provenance(run / "task/cleanup.json")],
              "correctness_status": "Experimental opt-in Verify implementation; throughput does not establish lossless speculative decoding"}
    cfg = read(run / "config.json")
    result["workload"] = {"model": "GLM-4.7-AWQ", "nodes": cfg["nodes"], "gpu": "8x A100 80GB",
                          "request_count": 8, "logical_prefix_per_request": 4096, "cpu_cached_per_request": 4032,
                          "recomputed_tail_per_request": 64, "query_per_request": 1, "first_forward_tokens_per_request": 65,
                          "output_per_request": 64, "mtp_draft_tokens": 3, "local_attention_graph_batch": 4,
                          "semantic_bank": cfg["semantic_bank_dir"], "synthetic_kv": False,
                          "engine_commit": cfg["engine_commit"], "production_cohorts_per_arm": 2,
                          "transport": "Pair NVLink; RDMA outside pairs; PCIe threshold0, no CE; unchanged between arms",
                          "timing_boundary": "Source-bank import and weight loading excluded; native CPU HiCache to GPU included"}
    outputs = {}
    for arm, old_arm, off_arm in zip(ARMS, OLD_ARMS, OFF):
        owned = read(run / arm / "arm_result.json")
        if not all(owned.get(k) for k in ("passed", "performance_valid", "profile_valid", "cleanup_verified")):
            raise ValueError(f"Incomplete paired arm {arm}")
        entry = {"cohorts": {}, "profile": aggregate_activity(run / arm / "stage_activity_summary.json")}
        arm_outputs = {}
        for name, filename in (("timed", "p4096_decode_results.json"), ("repeat", "p4096_decode_repeat_results.json")):
            entry["cohorts"][name], arm_outputs[name] = clean_cohort(run / arm / filename)
        entry["self_repeat_parity"] = HELPERS.output_parity(arm_outputs["timed"], arm_outputs["repeat"])
        entry["production_aggregate"] = {"throughput_tok_s": spread([r["output_throughput_tok_s"] for r in entry["cohorts"].values()])}
        for metric in ("ttft_ms", "tpot_ms", "end_to_end_ms", "spec_accept_length", "spec_accept_rate", "spec_verify_count_per_request"):
            entry["production_aggregate"][metric] = spread([r[metric]["mean"] for r in entry["cohorts"].values()])
        first = read(run / arm / "p4096_decode_results.json")
        entry["scheduler"] = {k: first["actual_scheduler_proof"].get(k) for k in
                              ("passed", "mode", "actual_slots", "max_requests_per_slot", "feedback_group", "actual_slot_request_composition")}
        entry["verify_fix"] = [read(run / arm / f"verify_fix_rank{i}_b4.json") for i in range(8)]
        if not all(r["qualified"] for r in entry["verify_fix"]):
            raise ValueError("Missing all-rank Verify fix qualification")
        routes = [read(run / arm / f"route_rank{i}.json") for i in range(8)]
        if not all(r["passed"] and r["pcie_threshold"] == 0 for r in routes):
            raise ValueError("Transport changed or route not qualified")
        entry["transport_proof"] = {"all_ranks_passed": True, "native_hashes": sorted({r["sha256"] for r in routes}),
                                    "policy": routes[0]["route_policy"], "pcie_threshold": 0}
        entry["diagnostic_host_cycles"] = [diagnostic_cycles(run / arm / f"event_spans_rank{i}.json")
                                           for i in ((0,) if arm == ARMS[0] else (0, 4))]
        if (OLD / off_arm / "p4096_decode_results.json").exists():
            baseline = read(OLD / off_arm / "p4096_decode_results.json")
            entry["versus_prior_same_layout_mtp_off"] = {"same_bank": baseline["bank_identity_sha256"] == first["bank_identity_sha256"],
                                                        "same_engine_commit": baseline["engine_commit"] == first["engine_commit"],
                                                        "parity": HELPERS.output_parity(arm_outputs["timed"], baseline["requests"]),
                                                        "source": HELPERS.provenance(OLD / off_arm / "p4096_decode_results.json")}
        outputs[arm] = arm_outputs["timed"]
        result["arms"][arm] = entry
        old = {"cohorts": {}}
        for name, filename in (("timed", "p4096_decode_results.json"), ("repeat", "p4096_decode_repeat_results.json")):
            old["cohorts"][name], _ = clean_cohort(OLD / old_arm / filename)
        old["profile"] = aggregate_activity(OLD_PROFILE / old_arm / "stage_activity_summary.json")
        old_first = read(OLD / old_arm / "p4096_decode_results.json")
        old["same_bank"] = old_first["bank_identity_sha256"] == first["bank_identity_sha256"]
        old["same_engine_commit"] = old_first["engine_commit"] == first["engine_commit"]
        old["fixed_vs_original_output_parity"] = HELPERS.output_parity(arm_outputs["timed"], old_first["requests"])
        old["original_vs_same_layout_mtp_off_parity"] = HELPERS.output_parity(old_first["requests"], baseline["requests"])
        result["historical_same_protocol"][old_arm] = old
    result["cross_topology_output_parity"] = HELPERS.output_parity(outputs[ARMS[0]], outputs[ARMS[1]])
    a, b = [result["arms"][arm]["production_aggregate"] for arm in ARMS]
    result["pp_relative_to_ep"] = {"throughput_ratio": b["throughput_tok_s"]["mean"] / a["throughput_tok_s"]["mean"],
                                   "tpot_ratio": b["tpot_ms"]["mean"] / a["tpot_ms"]["mean"]}
    result["lossless_speculative_comparison_qualified"] = False
    result["performance_measurement_passed_does_not_imply_numerical_correctness"] = True
    ep = result["arms"][ARMS[0]]["profile"]["groups"]["all_ranks"]["roles"]["target"]
    pp = [g["roles"]["target"] for g in result["arms"][ARMS[1]]["profile"]["groups"].values()]
    result["attribution_observations"] = {
        "dispatch_combine_per_moe_layer_ms": {"ep8": ep["dispatch_combine_per_moe_layer_ms"],
                                               "pp_stage0": pp[0]["dispatch_combine_per_moe_layer_ms"],
                                               "pp_stage1": pp[1]["dispatch_combine_per_moe_layer_ms"]},
        "logical_serial_stage_work_not_tpot_ms": {
            "ep8": {k: ep[k] for k in ("attention_ms", "expert_ms", "dispatch_combine_ms", "graph_span_ms")},
            "pp_sum_of_two_stage_means": {k: sum(r[k] for r in pp) for k in
                                           ("attention_ms", "expert_ms", "dispatch_combine_ms", "graph_span_ms")}},
        "production_mean_of_per_request_median_stream_update_gap_ms": {
            arm: mean([r["client_update_gap_ms"]["median"]
                       for c in result["arms"][arm]["cohorts"].values() for r in c["client_cadence"]])
            for arm in ARMS},
        "acceptance_explanation": "PP acceptance is marginally higher in this pair, so lower acceptance does not explain its longer TPOT here.",
        "residual_inference": "Reduced local MoE exchange and similar logical full-layer Attention work coexist with slower client cadence. PP stage handoff / closed-loop speculative feedback / scheduler work are leading suspects; no production critical-path partition or bubble percentage was measured.",
        "numerical_caveat": "Identical semantic inputs but divergent generated token IDs may also change later routing. These are experimental path measurements, not an exact-output-equivalent production ranking."}
    result["limitations"] = ["Two consecutive unprofiled cohorts are not a broad statistical performance study.",
                              "P99 is the maximum of only eight requests per cohort.",
                              "Graph component spans exclude PP boundary / CPU feedback and cannot be added into production TPOT.",
                              "Per-layer DeepEP comparison also differs in peer count and expert / token distribution; not pure link causality.",
                              "Closed-loop speculative scheduling may wait even with two equal slots; no production bubble percentage was directly measured."]
    return result


def markdown(x):
    lines = ["# GLM-4.7：修复 Verify Attention 后 EP8 与 PP2×EP4 对比", "",
             "065／167，8×A100 80GB，同型号高速 NIC。B8、逻辑 P4096、Q1、输出64。每条实际从原生 CPU HiCache 恢复4032 token，剩余64 token 与新增Q1重算。两组均开启 MTP、修复版 Verify Attention 和 CUDA Graph。权重／Bank 导入不计时，HiCache→GPU 计时。", "",
             "EP8：Attention TP4×DP2，Expert EP8。PP2×EP4：Attention TP4、Expert EP4／stage，两个 B4 slot。DeepEP 均为 pair 内 NVLink、pair 外 RDMA，未启用 PCIe CE。", "",
             "## 实测性能（无 profiler）", "",
             "| 配置 | cohort | 输出 tok/s | TTFT mean / median / P99 ms | TPOT mean / median / P99 ms | E2E mean / median / P99 ms |",
             "|---|---|---:|---:|---:|---:|"]
    def triple(row):
        return " / ".join(f"{row[k]:.3f}" for k in ("mean", "median", "p99"))
    for arm in ARMS:
        for name, row in x["arms"][arm]["cohorts"].items():
            lines.append(f"| {arm} | {name} | {row['output_throughput_tok_s']:.3f} | {triple(row['ttft_ms'])} | {triple(row['tpot_ms'])} | {triple(row['end_to_end_ms'])} |")
    lines += ["", "P99 仅为8条请求的样本最大值。两 cohort 的差异不是置信区间。", "",
              "## MTP 推进与额外验证次数", "",
              "| 配置 | 每轮推进 token（接口 spec_accept_length） | draft 接受率 | 每请求 Verify 次数 |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        a = x["arms"][arm]["production_aggregate"]
        lines.append(f"| {arm} | {a['spec_accept_length']['mean']:.4f} | {a['spec_accept_rate']['mean']:.4f} | {a['spec_verify_count_per_request']['mean']:.3f} |")
    lines += ["", "spec_accept_length 是每轮输出推进量，不能全部解释成被接受的 draft token。", "",
              "## 目标验证分段（独立诊断）", "",
              "单位 ms／每卡／一次实际目标 Graph 执行；EP 是8卡均值，PP 分 stage 取4卡均值。", "",
              "| 执行范围 | Attention | Routed Expert | Shared Expert | Dense FFN | Dispatch+Combine | TP/NCCL | 其他 | 重叠 | Graph |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for group, data in x["arms"][arm]["profile"]["groups"].items():
            r = data["roles"]["target"]
            e = r["exclusive_ms"]
            values = [r["attention_ms"], e["expert_routed"], e["expert_shared"], e["dense_ffn"],
                      r["dispatch_combine_ms"], e["nccl_tp"], r["other_graph_ms"], r["overlap_ms"], r["graph_span_ms"]]
            lines.append(f"| {arm} {group} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "**不能把 stage 均值相加当作 TPOT**：Graph 不包含完整的跨 stage 激活交接、草稿／接受结果反馈和主机调度；这些执行会与另一 slot 部分重叠。profile 的完成 fence 也改变了节奏，因此诊断 host gap 不能标成生产流水线气泡。", "",
              "## 与原 Verify 实现的历史同口径数据", "",
              "| 配置 | 原 Verify 输出 tok/s（两cohort均值） | 修复后 | 原 Verify TPOT ms | 修复后 |",
              "|---|---:|---:|---:|---:|"]
    for arm, old_arm in zip(ARMS, OLD_ARMS):
        old = x["historical_same_protocol"][old_arm]["cohorts"]
        a = x["arms"][arm]["production_aggregate"]
        lines.append(f"| {arm} | {mean([r['output_throughput_tok_s'] for r in old.values()]):.3f} | {a['throughput_tok_s']['mean']:.3f} | {mean([r['tpot_ms']['mean'] for r in old.values()]):.3f} | {a['tpot_ms']['mean']:.3f} |")
    lines += ["", "旧数据仅作同 Bank／同 workload 历史参照，不是本轮重跑的原实现控制组。", "",
              "## 差距的归因", ""]
    a = x["attribution_observations"]
    serial = a["logical_serial_stage_work_not_tpot_ms"]
    ep, pp = serial["ep8"], serial["pp_sum_of_two_stage_means"]
    d = a["dispatch_combine_per_moe_layer_ms"]
    cadence = a["production_mean_of_per_request_median_stream_update_gap_ms"]
    lines += [
        f"1. **节点内 Expert 交换优势仍存在。** Dispatch＋Combine 约为 EP8 {d['ep8']:.3f} ms／MoE层，PP stage0／1 {d['pp_stage0']:.3f}／{d['pp_stage1']:.3f} ms。后者低约三分之一。这是整体布局优势的证据，并非只改变链路、保持 peer 数和路由域不变的纯网络因果实验。",
        f"2. **Attention 修复对两边均生效。** 用两段均值相加仅观察逻辑全层工作量，PP Attention {pp['attention_ms']:.2f} ms，与 EP {ep['attention_ms']:.2f} ms 接近；Expert 合计 {pp['expert_ms']:.2f} 对 {ep['expert_ms']:.2f} ms，PP 略高。这个相加值不是实际 PP cycle 或 TPOT。",
        "3. **本轮排除了较低接受率这一解释。** PP 每轮推进2.465 token，EP为2.458；PP平均每请求26次Verify，EP26.25次。PP并没有因为接受率低而执行更多轮。旧版中存在的接受率差异不能直接沿用于本轮。",
        f"4. **差异更多出现在输出推进节奏。** 无profiler生产测量中，各请求输出更新间隔的中位数再取均值，EP约{cadence[ARMS[0]]:.2f} ms，PP约{cadence[ARMS[1]]:.2f} ms。独立诊断中同一slot的目标调用起点间隔中位数也为EP约119.9 ms、PP约132.4–132.6 ms；但该诊断有完成fence，不能换算成生产气泡比例。结合源码，stage激活交接、slot复用等待、接受结果／draft反馈和CPU调度是优先检查的环节；当前证据没有把这些项逐个定量拆开。",
        "5. **吞吐差2.3%和平均TPOT差13%并不矛盾。** 吞吐按整批512个输出token除以最晚完成时间；TPOT是每条请求首尾token间隔除以63后取均值。PP两slot首token错开，EP各请求的后续推进也不完全一样。",
        "6. **这不是无损MTP的最终公平排名。** 两边的语义输入和缓存相同，但输出逐token等价未通过，后续路由也可能随输出改变。现有证据支持路径分析；需先修复输出一致性，再作严格生产性能定论。", "",
              "## 数值与实验边界", ""]
    for arm in ARMS:
        e = x["arms"][arm]
        parity = e["versus_prior_same_layout_mtp_off"]["parity"]
        equal = sum(r["token_ids_equal"] for r in parity["requests"])
        lines.append(f"- {arm}：同配置两cohort逐token可重复={e['self_repeat_parity']['passed']}；与历史同布局 MTP-off 的输出完全一致 {equal}/8。")
    lines.append(f"- 本轮 EP／PP 输出完全相同：{x['cross_topology_output_parity']['passed']}。吞吐提升不证明 speculative decoding 无损。")
    for arm in OLD_ARMS:
        parity = x["historical_same_protocol"][arm]["original_vs_same_layout_mtp_off_parity"]
        equal = sum(r["token_ids_equal"] for r in parity["requests"])
        lines.append(f"- 历史原Verify的 {arm} 与同布局MTP-off逐token一致为 {equal}/8；原路径也未达到严格等价，不能把全部差异归为本次Attention优化新引入。")
    lines += ["- Verify 优化保持默认关闭，属于受限形状的实验实现；保留所有不一致证据。",
              "- 本 workload 远未用满 KV 容量，不能据此否定 PP 在容量受限时的收益。",
              "- 两组性能、profile、进程退出与 GPU 清理均通过。", "",
              "## 原始证据", "",
              f"- 本轮：`{x['run']}`", f"- 历史性能：`{OLD}`", f"- 历史分段：`{OLD_PROFILE}`", "",
              "`summary.json` 保存全部来源路径／SHA256、逐cohort mean/median/P99、分段、MTP接受率、输出一致性与限定条件；原始输出只保留在源结果中。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    result = summarize(ROOT / "runs/paired_r1")
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "RESULTS.md").write_text(markdown(result))
    print(json.dumps({"summary": str(ROOT / "summary.json"), "report": str(ROOT / "RESULTS.md"),
                      "pp_relative_to_ep": result["pp_relative_to_ep"]}, ensure_ascii=False))
