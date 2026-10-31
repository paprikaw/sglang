"""Retrieve the saved four-H100 layout results; launch no experiments."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


SOURCE = Path(
    "/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/"
    "qwen35b_h100_five_layouts_hicache_20261028"
)
OUT = Path(__file__).resolve().parent
LAYOUTS = [
    ("ep4", "EP4", 1, "4", "64"),
    ("two_ep2", "2×EP2", 2, "2+2", "32 / 32"),
    ("four_ep1", "4×单卡（PP1/EP1）", 4, "1+1+1+1", "16 / 16 / 16 / 16"),
    ("two_pp2", "2×PP2", 2, "2+2", "32 / 32"),
    ("pp4", "PP4", 1, "4", "64"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_csv = SOURCE / "full_metrics.csv"
    with source_csv.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [r for r in source_rows if r["phase"] in ("prefill", "decode")]
    assert len(source_rows) == 130 and len(rows) == 120
    groups = defaultdict(list)
    raw_sources = {}
    for row in rows:
        assert row["B"] == "64"
        assert row["node"] == "spartan-gpgpu178"
        assert row["Q"] == ("512" if row["phase"] == "prefill" else "1")
        raw_path = Path(row["source"])
        if raw_path not in raw_sources:
            raw_sources[raw_path] = json.loads(raw_path.read_text())
        raw = raw_sources[raw_path]
        assert raw["valid"] is True
        assert len(raw["rounds"]) == 3
        trial = raw["rounds"][int(row["repeat"])]
        assert len(trial["requests"]) == 64
        assert abs(float(row["makespan_s"]) - trial["makespan_s"]) < 1e-10
        groups[(row["layout"], int(row["P"]), row["phase"], row["cache"])].append(row)
    assert len(groups) == 40 and len(raw_sources) == 40
    for group in groups.values():
        assert sorted(int(r["repeat"]) for r in group) == [0, 1, 2]

    def times(layout, prefix, phase, cache):
        group = sorted(groups[(layout, prefix, phase, cache)], key=lambda r: int(r["repeat"]))
        return [float(r["makespan_s"]) * 1000 for r in group]

    def average(layout, prefix, phase, cache):
        values = times(layout, prefix, phase, cache)
        return f"{fmean(values):,.2f}"

    lines = [
        "# 已有四卡五布局结果：三轮平均完成时间",
        "",
        "摘录日期：2026-10-29。全部来自2026-10-28已完成实验；没有新增GPU测量。",
        "",
        f"[原完整报告]({SOURCE / 'REPORT.md'}) · [原逐轮CSV]({source_csv}) · [三轮均值CSV](mean_metrics.csv) · [本次120条正式轮次](formal_rounds.csv) · [来源校验](provenance.json)",
        "",
        "## 固定实验条件",
        "",
        "Qwen3.6-35B-A3B BF16；同一178节点、4×H100 80GB、节点内NVLink；全局64请求。",
        "每请求缓存前缀P=8,256或130,304；Prefill新增Q=512、输出1；Decode新增Q=1、输出128。",
        "下表是整个四卡部署完成64请求的makespan，包含该批内排队和CPU→GPU缓存恢复，不包含批前准备。",
        "每格显示3轮正式批次完成时间的算术平均值，单位ms，越低越好；每轮等权，全部样本均计入，不归一化。",
        "平均的是每轮makespan，不是先平均tokens/s再反算时间，也不是对批内请求延迟求平均。",
        "CSV的repeat=0/1/2分别为第1/2/3正式轮；每项3轮都保留，另10条cold_prefill不并入这里。",
        "",
        "| 部署 | 模型实例数 | GPU分组 | 每实例请求数 |",
        "|---|---:|---|---|",
    ]
    lines += [f"| {label} | {replicas} | {gpus} | {batch} |" for _, label, replicas, gpus, batch in LAYOUTS]
    lines += [
        "",
        "EP的Attention TP=1，EP4/EP2的Attention DP分别为4/2；PP采用Attention TP=1、Expert EP=1。",
        "PP2为20+20层，PP4为10+10+10+10层。多副本表中的总工作量仍为64请求，不能把单个两卡实例与整套四卡部署混用。",
    ]
    for cache, title in [("natural_replay", "自然前缀重放"), ("cpu_replay", "受控CPU前缀重放")]:
        lines += ["", f"## {title}", ""]
        if cache == "natural_replay":
            lines += [
                "保留各布局原生缓存状态。短前缀全部命中GPU；长前缀下EP4与PP4仍命中GPU，多副本方案包含不同程度的CPU恢复。",
                "这组比较覆盖真实容量差异，不能单独归因为Expert计算或通信。",
                "",
            ]
        else:
            lines += [
                "所有布局的前缀token均从CPU命中。主动移出GPU及CPU写入fence在批前，计时包括传回GPU。",
                "与自然重放的缓存历史、生成分支及PP调优选择不同，跨表差额不能全部归因为CPU传输。",
                "",
            ]
        for phase, phase_title in [("prefill", "Prefill"), ("decode", "Decode")]:
            lines += [
                f"### {phase_title}：完整四卡批次平均ms",
                "",
                "| 部署 | P=8,256 | P=130,304 |",
                "|---|---:|---:|",
            ]
            for layout, label, *_ in LAYOUTS:
                lines.append(f"| {label} | {average(layout,8256,phase,cache)} | {average(layout,130304,phase,cache)} |")
            lines.append("")
    lines += [
        "## 长缓存CPU重放：保留三轮原值",
        "",
        "Decode，P=130,304，单位ms。EP4的第三轮长等待没有剔除，原因本摘录未定位。",
        "",
        "| 部署 | 第1轮 | 第2轮 | 第3轮 |",
        "|---|---:|---:|---:|",
    ]
    for layout, label, *_ in LAYOUTS:
        values = times(layout, 130304, "decode", "cpu_replay")
        lines.append(f"| {label} | " + " | ".join(f"{v:,.2f}" for v in values) + " |")
    lines += [
        "",
        "## 已有结果支持的判断",
        "",
        "- 四卡部署候选应包括EP4、2×EP2、4×单卡、2×PP2和PP4；只比较EP4与PP4不足以判断最佳部署。",
        "- 以下按三轮平均批次完成时间比较。自然重放短前缀：Prefill的4×单卡最快；Decode的EP4与4×单卡接近，均明显快于2×PP2和PP4。",
        "- 自然重放长前缀：Prefill的EP4与2×PP2接近；Decode的EP4明显领先，其余布局存在不同缓存容量与恢复成本。",
        "- 全CPU重放：两种前缀的Prefill均以2×PP2更快；长前缀Decode的PP4最快。因此不能预设多副本总是优于PP4。",
        "- 受控CPU重放内部确有随context改变的Decode均值排序反转：短前缀EP4最快，长前缀PP4最快。EP4长前缀的第三轮15,148.43 ms已计入8,519.33 ms均值；原值保留于上表。",
        "- 两个前缀点的全局Decode并发都为64，每步新token数量相同。上述反转同时涉及context、缓存工作集、恢复和条件对应的PP调优，不能视为仅由Decode token数量/活跃专家驱动。自然重放两个前缀点均没有PP方案反胜EP4的区域。",
        "",
        "## 实现边界",
        "",
        "全程开启原生HiCache，write_through/kernel/page_first；每GPU CPU池64 decimal GB Attention KV +8 decimal GB GDN。",
        "PP按阶段、前缀与缓存模式，从通过资格的native/early-send及microbatch16/8/4候选中选择。",
        "lookahead/tail-local/policy10未取得本次HiCache负载资格，不能把这批结果标成最新lookahead实现的排名。",
        "此数据与GPU-hot4096的新优化Decode实验属于不同协议；不拼接成同一性能曲线。",
        "",
        "## 离线校验",
        "",
        "从130条原CSV记录中提取120条Prefill/Decode正式记录；40个条件，每项3轮。",
        "逐条核对40份原始JSON的valid标志、64请求数、轮次索引及makespan与CSV相符。",
        "源CSV、原报告和原始JSON的SHA256保存于provenance.json。源文件未修改。",
        "",
    ]
    report = OUT / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "formal_rounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["makespan_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, makespan_ms=f"{float(row['makespan_s'])*1000:.9f}"))
    with (OUT / "mean_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layout", "P", "phase", "cache", "rounds", "mean_makespan_ms", "round1_ms", "round2_ms", "round3_ms"])
        writer.writeheader()
        for layout, prefix, phase, cache in sorted(groups):
            values = times(layout, prefix, phase, cache)
            writer.writerow({"layout": layout, "P": prefix, "phase": phase, "cache": cache, "rounds": len(values), "mean_makespan_ms": f"{fmean(values):.9f}", **{f"round{i+1}_ms": f"{v:.9f}" for i,v in enumerate(values)}})
    provenance = {
        "source_experiment": SOURCE.name,
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256(source_csv),
        "source_report_sha256": sha256(SOURCE / "REPORT.md"),
        "original_rows": len(source_rows),
        "formal_rows": len(rows),
        "conditions": len(groups),
        "checked_raw_sources": [{"path": str(path), "sha256": sha256(path)} for path in sorted(raw_sources)],
        "measurements_added": 0,
        "aggregation": "equal-weight arithmetic mean of 3 formal cohort makespans; all original rounds retained",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(report), "formal_rows": len(rows), "conditions": len(groups), "raw_sources_checked": len(raw_sources)}, ensure_ascii=False))
    print("Decode arithmetic mean, ms:")
    for layout, label, *_ in LAYOUTS:
        print(label, *(average(layout, prefix, "decode", cache) for cache in ("natural_replay", "cpu_replay") for prefix in (8256,130304)), sep=" | ")


if __name__ == "__main__":
    main()
