"""Derive equal two-GPU/B32 groups from the saved four-GPU/B64 runs."""
import ast
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
LABELS = {"two_ep1": "2×EP1", "two_ep2": "EP2", "two_pp2": "PP2"}
LAYOUTS = {"four_ep1": "two_ep1", "two_ep2": "two_ep2", "two_pp2": "two_pp2"}


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None


def main():
    parent = json.loads((ROOT / "summary.json").read_text())
    rows, metrics, proof = [], [], []
    for run in ["ep_178_r4", "ep_178_pp2_r1"]:
        path = ROOT / "runs" / run / "code/runner.py"
        raw = path.read_bytes()
        functions = {n.name: n for n in ast.parse(raw.decode()).body if isinstance(n, ast.FunctionDef)}
        proof.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                          endpoint_source=ast.get_source_segment(raw.decode(), functions["endpoint"]),
                          cohort_source=ast.get_source_segment(raw.decode(), functions["cohort"])))
    assert proof[0]["endpoint_source"] == proof[1]["endpoint_source"]
    assert proof[0]["cohort_source"] == proof[1]["cohort_source"]
    for m in parent["metrics"]:
        if m["layout"] not in LAYOUTS:
            continue
        layout = LAYOUTS[m["layout"]]
        path = Path(m["source"])
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data["workload"]["hicache"] is True
        assert len(data["rounds"]) == (1 if m["cache"] == "cold" else 3)
        current = []
        for repeat, cohort in enumerate(data["rounds"]):
            requests = cohort["requests"]
            assert len(requests) == 64
            span = max(q["finished"] for q in requests) - min(q["started"] for q in requests)
            assert math.isclose(span, cohort["makespan_s"], abs_tol=1e-7)
            for group in range(2):
                indices = [i for i in range(64) if (i % 2 if layout == "two_pp2" else (i % 4) // 2) == group]
                selected = [requests[i] for i in indices]
                assert len(selected) == 32
                expected = 128 if m["phase"] == "decode" else 1
                assert all(q["output_tokens"] == expected for q in selected)
                prompt = m["P"] if m["cache"] == "cold" else m["P"] + m["Q"]
                assert all(q["prompt_tokens"] == prompt for q in selected)
                details = [q["cached_tokens_details"] for q in selected]
                if m["cache"] != "cold":
                    assert all(isinstance(d, dict) for d in details)
                span = max(q["finished"] for q in selected) - min(q["started"] for q in selected)
                count = sum(q["output_tokens"] for q in selected) if m["phase"] == "decode" else 32 * m["Q"]
                row = dict(layout=layout, source_layout=m["layout"], P=m["P"], B=32, Q=m["Q"],
                           phase=m["phase"], cache=m["cache"], repeat=repeat, group=group,
                           makespan_s=span, throughput=count / span,
                           ttft_s=mean(selected, "ttft_s"), tpot_ms=mean(selected, "tpot_ms"),
                           cpu_prefix_fraction=sum((d or {}).get("host", 0) for d in details) / (32 * m["P"]),
                           gpu_prefix_fraction=sum((d or {}).get("device", 0) for d in details) / (32 * m["P"]),
                           source=str(path), source_sha256=hashlib.sha256(raw).hexdigest())
                rows.append(row)
                current.append(row)
        metrics.append(dict(layout=layout, label=LABELS[layout], P=m["P"], B=32, Q=m["Q"],
                            phase=m["phase"], cache=m["cache"], throughput=mean(current, "throughput"),
                            throughput_min=min(r["throughput"] for r in current),
                            throughput_max=max(r["throughput"] for r in current),
                            group_means=[mean([r for r in current if r["group"] == g], "throughput") for g in range(2)],
                            ttft_s=mean(current, "ttft_s"), tpot_ms=mean(current, "tpot_ms"),
                            cpu_prefix_fraction=mean(current, "cpu_prefix_fraction"),
                            gpu_prefix_fraction=mean(current, "gpu_prefix_fraction")))
    assert len(rows) == 156 and len(metrics) == 30
    previous = json.loads((ROOT / "two_gpu_analysis/summary.json").read_text())
    for old in previous["metrics"]:
        new = next(m for m in metrics if all(m[k] == old[k] for k in ["layout", "P", "cache", "phase"]))
        assert math.isclose(new["throughput"], old["throughput"], rel_tol=1e-12)
    capacities = {LAYOUTS[c["layout"]]: c["conservative_gpu_token_pool"] // 2
                  for c in parent["capacities"] if c["layout"] in LAYOUTS}
    result = dict(status="complete", created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  kind="offline extraction; no new GPU experiment", parent_experiment=str(ROOT),
                  scope="Each two-GPU group serves B32. Two groups concurrently share H100178 CPU/memory resources; not an isolated two-GPU B64 measurement.",
                  aggregation="Mean of independently timed two-GPU group throughputs across two groups and formal rounds; concurrent groups are not independent repeated trials.",
                  gpu_groups=[[0, 1], [2, 3]], replica_mapping_proof=proof,
                  group_mapping={"four_ep1": "(index % 4) // 2; two independent EP1 instances per GPU pair",
                                 "two_ep2": "(index % 4) // 2; one EP2 instance per GPU pair",
                                 "two_pp2": "index % 2; one PP2 instance per GPU pair"},
                  gpu_kv_tokens_per_two_gpu_group=capacities, metrics=metrics, rows=rows)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (OUT / "per_group_rounds.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    doc = ["# 35B：两卡预算下的2×EP1、EP2与PP2", "",
           "两卡候选应同时包括2×EP1（两个独立单卡实例）、EP2与PP2。所有数据来自本次Qwen3.6-35B-A3B BF16 / H10080GB实验，全部开启原生HiCache。", "",
           "这是既有四卡、64请求实验的离线拆分：固定GPU对[0,1]与[2,3]各承担32请求；2×EP1的每个EP1承担16请求，EP2/PP2各实例承担32请求。对每个GPU对使用自己的最早请求开始至最晚请求完成重算吞吐，再对两个GPU对、正式三轮取均值。未简单除二，未挑选最快GPU对。EP2与2×EP1的每GPU对请求组成相同；PP2采用原实验轮转分配，完整64请求组成相同。", "",
           "两个GPU对同时运行，共享节点CPU/内存；这是两卡组的观测值，不能代替独占两卡、承接全部64请求或固定到达率的实测。Prefill每请求新增Q512、输出1；Decode新增Q1、输出128。P为每请求缓存前缀。吞吐单位tokens/s，越高越好；Prefill只计新增输入，Decode计生成输出。CPU→GPU恢复与排队包含在请求时间中，批前准备不计入。", ""]
    def lookup(layout, p, cache, phase):
        return next(m for m in metrics if (m["layout"], m["P"], m["cache"], m["phase"]) == (layout, p, cache, phase))
    for cache, title in [("natural_replay", "自然缓存重放"), ("cpu_replay", "受控重放：前缀token全部从CPU命中")]:
        doc += ["## " + title, "", "| P | 阶段 | 2×EP1 | EP2 | PP2 | CPU前缀token比例：2×EP1 / EP2 / PP2 |", "|---|---|---:|---:|---:|---|"]
        for p in [8256, 130304]:
            for phase in ["prefill", "decode"]:
                group = [lookup(layout, p, cache, phase) for layout in LABELS]
                values = " | ".join(f"{m['throughput']:,.0f}" for m in group)
                hits = " / ".join(f"{m['cpu_prefix_fraction']:.1%}" for m in group)
                doc.append(f"| {p:,} | {phase} | {values} | {hits} |")
        doc.append("")
    doc += ["## GPU KV容量与策略含义", "", "| 两卡布局 | GPU KV token池 |", "|---|---:|"]
    for layout in LABELS:
        doc.append(f"| {LABELS[layout]} | {capacities[layout]:,} |")
    doc += ["", "每GPU对短前缀总计264,192 tokens，长前缀总计4,169,728 tokens。短前缀自然重放全部命中GPU，2×EP1的Prefill与Decode均高于另两者；该条件下复制单卡实例是应保留的扩容候选。长前缀下，2×EP1约99%的前缀token从CPU命中，Decode吞吐明显下降；Prefill优先PP2，Decode中EP2与PP2接近。受控全CPU重放下，短前缀Prefill的2×EP1和PP2均约5.2万tokens/s，差异很小；长前缀Prefill仍以PP2更高。因此不能只按模型能否单卡放下选择拓扑，还需看KV工作集、缓存层、并发和阶段。", "",
            "这补充并修正此前EP2/PP2二选一的适用范围：短前缀、GPU缓存充足时，不应默认一拿到第二张卡就重组为EP2或PP2。具体原地切换、KV/GDN迁移、在途请求以及缩容正确性和成本尚未实测；这些静态结果用于提出策略候选。", "",
            "PP2使用本轮通过资格的原生/early-send与microbatch调优，lookahead/tail-local未取得本负载资格。EP2为Attention TP1/DP2、Expert EP2；PP2为Attention TP1、Expert EP1、20+20层；2×EP1各自Attention TP1/DP1、Expert EP1。每GPU CPU池64 decimal GB KV +8 decimal GB GDN；EP/PP static memory fraction为0.94/0.92。合成重复前缀不能代替生产命中率，受控CPU重放与自然重放的历史和生成分支不同，表间差异不能单独归因H2D。", "",
            "## 核验与来源", "", "30项布局/上下文/缓存/阶段指标，156个GPU对-轮次记录。正式热重放每项3轮，另有冷Prefill每项1轮保存在JSON/CSV中。核对请求顺序与endpoint映射、数量、输入输出token、计时、HiCache字段和原始SHA256；EP2/PP2指标与上一份两卡拆分一致。两个并发GPU对不是独立重复实验，未建立统计置信区间。", "",
            f"[逐GPU对逐轮数据]({OUT}/per_group_rounds.csv) · [指标、容量与来源哈希]({OUT}/summary.json) · [可复现分析脚本]({OUT}/analyze.py) · [原四卡报告]({ROOT}/REPORT.md)", ""]
    (OUT / "REPORT.md").write_text("\n".join(doc))
    print("Saved", OUT, "metrics", len(metrics), "group-rounds", len(rows))
    for m in metrics:
        if m["phase"] != "cold_prefill":
            print(m["P"], m["cache"], m["phase"], m["label"], round(m["throughput"], 2), "CPU", round(m["cpu_prefix_fraction"], 4))


if __name__ == "__main__":
    main()
