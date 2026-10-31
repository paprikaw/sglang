"""CPU-only archival analysis. Does not import torch or start GPU work."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
ARCHIVE = WORKSPACE.parents[2] / "experiment_manager" / "experiments"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, "/home/bxb1/.codex/skills/llm-experiment-guard/scripts")
from artifact_visibility import wait_for_stable_json

SOURCES = {}


def record(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    value, _ = wait_for_stable_json(path, timeout_s=3)
    record(path)
    return value


def read_csv(path):
    with record(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_group(ranks):
    values = [r["target"] for r in ranks]
    keys = ("graph_span_ms", "attention_ms", "expert_ms", "communication_ms",
            "other_graph_ms", "overlap_ms")
    result = {k: statistics.mean(v[k] for v in values) for k in keys}
    for key in ("expert_routed", "expert_shared", "deepep_dispatch",
                "deepep_combine", "nccl_tp"):
        result[key] = statistics.mean(v["exclusive_ms"].get(key, 0) for v in values)
    result["dispatch_combine"] = result["deepep_dispatch"] + result["deepep_combine"]
    partition = sum(result[k] for k in keys if k != "graph_span_ms")
    assert abs(partition - result["graph_span_ms"]) < 0.002
    return result


def compare_graphs(name, ep_path, pp_path):
    ep = read_json(ep_path)
    pp = read_json(pp_path)
    assert ep["passed"] and pp["passed"]
    assert sorted(r["rank"] for r in ep["ranks"]) == list(range(8))
    assert sorted(r["rank"] for r in pp["ranks"]) == list(range(8))
    e = graph_group(ep["ranks"])
    ranks = sorted(pp["ranks"], key=lambda r: r["rank"])
    p0, p1 = graph_group(ranks[:4]), graph_group(ranks[4:])
    rows = []
    for key in e:
        total = p0[key] + p1[key]
        rows.append(dict(cohort=name, component=key, ep8_ms=e[key],
                         pp_stage0_ms=p0[key], pp_stage1_ms=p1[key],
                         pp_path_accounting_ms=total, pp_minus_ep_ms=total-e[key]))
    budget = dict(
        cohort=name,
        dispatch_combine_saved_ms=e["dispatch_combine"]-p0["dispatch_combine"]-p1["dispatch_combine"],
        routed_expert_extra_ms=p0["expert_routed"]+p1["expert_routed"]-e["expert_routed"],
        all_internal_comm_saved_ms=e["communication_ms"]-p0["communication_ms"]-p1["communication_ms"],
        all_expert_ffn_extra_ms=p0["expert_ms"]+p1["expert_ms"]-e["expert_ms"],
        graph_path_saved_ms=e["graph_span_ms"]-p0["graph_span_ms"]-p1["graph_span_ms"],
        illustrative_two_slot_target_service_floor_ms=2*max(p0["graph_span_ms"],p1["graph_span_ms"]),
        ep_target_graph_ms=e["graph_span_ms"],
        note="Per-rank means, diagnostic target Graphs only; PP path sum is not TPOT. "
             "Two-slot floor is illustrative, excludes PP boundaries/feedback/draft, "
             "and is not an attainable wall-time claim. No addition across cohorts.")
    return rows, budget


def main():
    short = WORKSPACE / "experiments/glm47_semantic_bank_fourway_20261023_v8/runs/event_profile_followup_r1"
    ordinary, b0 = compare_graphs("decode_off_mtp_p4096_graph",
        short/"ep8_mtp_off/stage_activity_summary.json",
        short/"pp2ep4_mtp_off/stage_activity_summary.json")
    mtp, b1 = compare_graphs("decode_mtp_fixed_verify_p4096_graph_historical_reference",
        WORKSPACE/"experiments/glm47_mtp_verify_fixed_ep8_pp2ep4_20261023/runs/paired_r1/ep8_mtp_fixed/stage_activity_summary.json",
        WORKSPACE/"experiments/glm47_pp_stage_balance_20261023/runs/balance_r1/pp2ep4_balanced/stage_activity_summary.json")
    write_csv("graph_components.csv", ordinary + mtp)
    write_csv("graph_budgets.csv", [b0,b1])
    expert = read_csv(WORKSPACE/"experiments/glm47_ep8_pp2ep4_expert_compute_20261021/runs/r2/analysis/comparison.csv")
    for row in expert:
        row["pp_extra_ms"] = float(row["pp2ep4_mean_slot_median_ms"])-float(row["ep8_median_ms"])
    write_csv("expert_curve.csv", expert)
    long = read_csv(ARCHIVE/"glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/stage_metrics.csv")
    mapped = {r["configuration"]:r for r in long}
    long_rows=[]
    for key in ["mean_microbatch_cycle_ms","attention_ms","expert_ffn_ms","communication_ms","remaining_ms"]:
        e=float(mapped["EP8"][key]); p0=float(mapped["PP2xEP4 stage0"][key]); p1=float(mapped["PP2xEP4 stage1"][key])
        long_rows.append(dict(component=key,ep8_ms=e,pp_stage0_ms=p0,pp_stage1_ms=p1,
                              pp_path_accounting_ms=p0+p1,pp_minus_ep_ms=p0+p1-e))
    write_csv("long_context_eager_accounting.csv",long_rows)
    record(ARCHIVE/"glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/overhead_analysis.md")
    record(ARCHIVE/"glm47_a100_ep8_pp2ep4_cached_decode_components_20261022_r1/summary.md")
    record(WORKSPACE/"experiments/glm47_semantic_bank_fourway_20261023_v8/EVENT_PROTOCOL.md")
    lines=["# A100 GLM：EP8 与 PP2×EP4 的 Decode 取舍", "",
           "只复算保存的数据；本次没有启动 GPU。两种布局均使用 8 张 A100，Attention 在节点内 TP4。",
           "", "## 1. 普通 Decode 的计算与通信预算", "",
           "B8、P4096（Bank 实际保存4032）、Q1、64输出，两个B4 slot，CUDA Graph。",
           "以下只统计实际 Decode Graph，不包含开头的残余 Prefill；语义上下文相同，但不宣称跨布局输出逐token完全相同。",
           "PP 是两个半模型的分项工作量合计，不是 HTTP TPOT，也不是同一时刻的墙钟。Graph 外等待和 PP 边界尚未纳入。", "",
           "| 分项 | EP8 ms | PP 路径合计 ms | PP−EP ms |", "|---|---:|---:|---:|"]
    for row in ordinary:
        if row["component"] in ["attention_ms","expert_routed","expert_ms","dispatch_combine","communication_ms","graph_span_ms"]:
            lines.append(f'| {row["component"]} | {row["ep8_ms"]:.4f} | {row["pp_path_accounting_ms"]:.4f} | {row["pp_minus_ep_ms"]:+.4f} |')
    lines += ["",f'通信仅算 Dispatch+Combine，节省{b0["dispatch_combine_saved_ms"]:.4f}ms，已经足以覆盖 Routed Expert 多出的{b0["routed_expert_extra_ms"]:.4f}ms。',
              f'含TP的模型内部通信节省{b0["all_internal_comm_saved_ms"]:.4f}ms，全部Expert/FFN多出{b0["all_expert_ffn_extra_ms"]:.4f}ms，二者差额{b0["all_internal_comm_saved_ms"]-b0["all_expert_ffn_extra_ms"]:.4f}ms。',
              "这说明计算/通信侧存在余量，不证明最终TPOT改善。插桩改变执行节奏，均值掩盖最慢rank，不能直接把余量当作服务延迟。",
              "", "## 2. 稳态预测模型", "",
              "固定全局B与请求组成；EP8使用两个Attention TP4组，PP使用两个B/2 slot。设总专家数E、top-k为k。",
              "在生产D1 source scatter与均衡路由下，两者每卡每层实际Expert配对数均为S=B*k/8；PP每卡持有E/4个expert，EP8为E/8。",
              "因此有用FLOPs相同，Expert数量、每Expert行数、padding和权重复用不同。不要因为PP slot减半就把它每卡的Expert工作也减半。",
              "", "定义同一全局Decode轮次、同一计时协议：", "",
              "    预计PP净收益 = 暴露通信节省 − Expert额外计算 − Attention差额 − PP额外开销", "",
              "PP额外开销包含边界激活/反馈、CPU调度、阶段不均衡及无法隐藏的依赖；必须用时间线校准，不能将所有GPU空隙叫气泡。",
              "如果只看目标Graph，两个slot的理想吞吐服务下界约为2*max(G0,G1)，不是G0+G1；反馈/依赖可能使实际周期更大。",
              f'本次短context普通Decode的Graph均值是EP={b0["ep_target_graph_ms"]:.4f}ms，PP两stage={ordinary[0]["pp_stage0_ms"]:.4f}/{ordinary[0]["pp_stage1_ms"]:.4f}ms。',
              f'用这些诊断均值作说明，2*max={b0["illustrative_two_slot_target_service_floor_ms"]:.4f}ms，与EP还差约{b0["ep_target_graph_ms"]-b0["illustrative_two_slot_target_service_floor_ms"]:.4f}ms。该数不是实测PP overhead或TPOT门槛，只说明阶段不均衡还会减少可兑现余量。',
              "", "## 3. 长context普通Decode：收益去了哪里", "",
              "独立cohort：B8/P81920/Q1、32输出，065/167；Graph OFF、overlap OFF。先恢复并预热GPU前缀，采稳态；不是首次HiCache换入计时。", "",
              "| 分项 | EP8 ms | PP两stage核算 ms | PP−EP ms |", "|---|---:|---:|---:|"]
    for row in long_rows:
        lines.append(f'| {row["component"]} | {row["ep8_ms"]:.4f} | {row["pp_path_accounting_ms"]:.4f} | {row["pp_minus_ep_ms"]:+.4f} |')
    lines += ["", "原记录：GPU无活动间隙16.07→29.04ms，增加12.97ms，其中forward内+8.31、forward间+4.66ms。不能据此认定全部来自固有pipeline bubble。",
              "同一诊断实验的无profiler TPOT为EP201.162、PP205.379ms。CUDA-only profiler引入约6%扰动，以上预算只用于定位。",
              "通信省12.59ms而Expert/FFN仅多0.67ms，仍未转为TPOT优势；当前GLM更应检查额外提交/同步/反馈依赖。", "",
              "## 4. MTP Decode 单独记录", "",
              f'修复Verify Attention之后，目标Verify路径D+C节省{b1["dispatch_combine_saved_ms"]:.4f}ms，Routed Expert多{b1["routed_expert_extra_ms"]:.4f}ms。',
              "这是EP8已有同Bank参照与50/42层PP的历史对照，非新鲜严格配对实验。未计draft、接受率、PP边界与反馈，不能除以名义草稿数得到TPOT。",
              "正确比较应使用完整Verify+draft+反馈周期 / 实际接受输出token数，并验证语义/输出等价。较早错误Verify Attention后端的MTP数据没有纳入。", "",
              "## 5. Expert token扫描：保留原口径，不与上表相加", "",
              "单张A100真实GLM AWQ单层，Graph主表；S为每卡实际token–expert配对数。旧probe采用D4来源副本，原logical B=S/4，不是生产Decode请求B。",
              "生产D1下top8、全局B8 Decode对应S约8；低于这个匹配扫描最低S32，因此不能直接外推。", "",
              "| S | EP8 ms | PP slot平均 ms | PP−EP ms |", "|---:|---:|---:|---:|"]
    for row in expert:
        if row["mode"]=="cuda_graph":
            lines.append(f'| {row["local_assignments_each"]} | {float(row["ep8_median_ms"]):.5f} | {float(row["pp2ep4_mean_slot_median_ms"]):.5f} | {row["pp_extra_ms"]:+.5f} |')
    lines += ["", "大工作量下约1%的差异不足以认定胜者。小工作量的差异随活跃expert数、tile padding和调优改变，不保证随batch单调。只测单层热重放，不代表逐层冷权重流量。", "",
              "## 判断", "",
              "已有普通Decode实测支持：PP的通信节省能够超过其Expert计算代价。仍缺的是将剩余收益在稳态关键路径兑现。",
              "当前证据不足以给出通用的B阈值；需要同一Graph/传输协议、相同P与全局B的完整Decode周期来校准PP额外开销。",
              "长P增大Attention工作和KV需求，不直接增加每轮Decode的Expert/通信token数；不要把长P当成更大通信包。", "",
              "## 来源", ""]
    lines += [f'- [{Path(p).name}]({p}) SHA256 `{sha}`' for p,sha in SOURCES.items()]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    (OUT/"provenance.json").write_text(json.dumps(dict(cpu_only=True,new_gpu_measurements=0,sources=SOURCES),indent=2)+"\n")
    print(json.dumps(dict(report=str(OUT/"REPORT.md"),ordinary_decode=b0,mtp_target=b1),indent=2))


if __name__ == "__main__":
    main()
