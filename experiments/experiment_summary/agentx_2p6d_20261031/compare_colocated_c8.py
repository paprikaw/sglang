"""Recheck saved C8 episodes; no GPU work or changes to frozen runs."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent
RUNS = SUMMARY.parent / "agentx_pd_dynamic_20261030/runs"
AUDIT = SUMMARY / "agentx_data_audit_20261031"


def read_csv(path):
    return list(csv.DictReader(path.open()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels = {
        "col_ep4": "混部：2×EP4",
        "col_pp2": "混部：4×PP2",
        "col_single": "混部：8×单卡",
        "pd_ep4": "4P4D：EP4 P→EP4 D（历史完整 C8）",
        "pd_2p6d_pp2_ep4ep2": "2P6D：PP2 P→EP4＋EP2 D",
        "pd_2p6d_pp2_3ep2": "2P6D：PP2 P→3×EP2 D",
    }
    selected = [
        r for r in read_csv(AUDIT / "accepted_episodes.csv")
        if r["case"] == "agentx_c8" and r["arm"] in labels
    ]
    selected += [
        r for r in read_csv(AUDIT / "historical_episodes.csv")
        if r["run"] == "eight_gpu_pd_ep4_durable_slots8_r4"
        and r["case"] == "agentx_c8"
    ]
    primary = [
        r for r in read_csv(HERE / "full_metrics.csv")
        if r["case"] == "agentx_c8" and r["placement_cohort"] == "owner_balanced"
    ]
    assert len(selected) == 12 and len(primary) == 2
    for row in primary:
        run = RUNS / row["source_run"]
        cfg = json.loads((run / "config.json").read_text())
        commits = {n["engine_commit"] for n in cfg["node_configs"].values()}
        assert len(commits) == 1
        source = run / "results" / row["arm"] / "agentx_c8/sticky_hash/formal.r1.json"
        selected.append(dict(row, source=str(source), engine_commit=commits.pop()))

    groups = collections.defaultdict(list)
    evidence = []
    identities = []
    for seed in selected:
        source = Path(seed["source"])
        sha = digest(source)
        if seed.get("sha256"):
            assert sha == seed["sha256"], source
        marker = source.with_name("completed." + source.stem.split(".")[-1] + ".json")
        assert json.loads(marker.read_text())["passed"], marker
        raw = json.loads(source.read_text())
        rows = raw["requests"]
        assert len(rows) == raw["completed_requests"] == 171, source
        for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
            actual = statistics.fmean(r[metric] for r in rows if r[metric] is not None)
            assert abs(actual - float(seed[metric])) < 1e-6, (source, metric)
        identity = collections.Counter(
            json.dumps([r["conversation_id"].split(":copy0:", 1)[-1]] + [r.get(k) for k in
                        ("source_kind", "source_outer_idx", "source_inner_idx",
                         "turn", "input_tokens", "output_tokens")])
            for r in rows
        )
        identities.append(identity)
        groups[seed["arm"]].append((seed, raw))
        evidence.append(dict(
            source=str(source), sha256=sha, marker=str(marker),
            marker_sha256=digest(marker), engine_commit=seed["engine_commit"],
            arm=seed["arm"], requests=len(rows),
            provenance="historical_complete_c8_not_main_matrix"
            if seed["arm"] == "pd_ep4" else "accepted_complete_episode",
        ))
    assert all(x == identities[0] for x in identities), "Request identities/lengths differ"
    assert len({r["sha256"] for r in evidence}) == 14, "Duplicate formal artifacts"

    result = []
    for arm, label in labels.items():
        episodes = groups[arm]
        rows = [r for _, raw in episodes for r in raw["requests"]]
        item = dict(
            arm=arm, layout=label, cohort="E8/V4" if arm.startswith("pd_2p6d") else "E5/V3",
            engine_commit=episodes[0][0]["engine_commit"], repeats=len(episodes),
            requests_per_episode=171, gpu_budget=8,
        )
        for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
            item[metric] = statistics.fmean(r[metric] for r in rows if r[metric] is not None)
        intervals = [r for r in rows if r["output_tokens"] > 1]
        item["interval_weighted_tpot_ms"] = (
            sum(r["e2e_ms"] - r["ttft_ms"] for r in intervals)
            / sum(r["output_tokens"] - 1 for r in intervals)
        )
        item["wall_s"] = statistics.fmean(raw["wall_s"] for _, raw in episodes)
        item["mean_input_tokens"] = statistics.fmean(r["input_tokens"] for r in rows)
        item["mean_output_tokens"] = statistics.fmean(r["output_tokens"] for r in rows)
        cache = [r.get("prefill_meta") if arm.startswith("pd_") else r.get("meta_info")
                 for r in rows]
        available = all(m is not None and "cached_tokens" in m for m in cache)
        cached = sum(m["cached_tokens"] for m in cache) if available else None
        total_input = sum(r["input_tokens"] for r in rows)
        item["p_prefix_hit_pct"] = 100 * cached / total_input if available else None
        item["mean_new_prefill_tokens"] = (
            (total_input - cached) / len(rows) if available else None
        )
        result.append(item)
    with (HERE / "c8_comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    comparison = []
    for item in result:
        if not item["arm"].startswith("pd_2p6d"):
            continue
        for base in result[:4]:
            comparison.append(dict(
                candidate=item["arm"], baseline=base["arm"],
                **{key + "_change_pct": 100 * (item[key] / base[key] - 1)
                   for key in ("ttft_ms", "tpot_ms", "e2e_ms", "interval_weighted_tpot_ms")},
            ))
    payload = dict(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        complete_episodes=14, requests=2394, same_request_identities_and_lengths=True,
        independent_source_hashes=14, measurements=result, comparisons=comparison,
        sources=evidence, source_audit=str(AUDIT),
        limitations=[
            "E5/V3 versus E8/V4: engine, GDN/KV budget and branch release timing differ.",
            "Old nodes011/171 versus new011/013; all use eight H100 GPUs.",
            "C8 is eight root families, not eight continuously active Decode requests.",
            "4P4D E5 C8 completed three times; its whole run failed in C128.",
            "No causal PD benefit, P:D ratio gain, or dynamic policy gain established.",
            "Cross-cohort episode wall times are not used as a deployment speedup.",
        ],
    )
    (HERE / "c8_comparison_evidence.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    text = [
        "# AgentX C8：混部、4P4D 与 2P6D",
        "",
        "历史数值对照：2P6D 尚未在请求平均完成时间上超过最佳混部。"
        "相对旧 4P4D，2P6D 的 E2E 仅改善约 0.2%–1.5%，TTFT 增加约 3%–8%。"
        "这还不足以证明改变 P:D 配比有稳定收益。",
        "",
        "均为 Qwen3.6-35B-A3B BF16、8 张 H100、HiCache 开启；"
        "C8 是同一批 8 个根会话族，共 171 次请求。"
        "14 个完整 episode、2394 次请求已重新检查原始 SHA、独立完成 marker、"
        "请求身份／输入输出长度以及延迟均值。没有新增 GPU 实验。",
        "",
        "| 部署 | 引擎／回放 | 完整轮次 | 平均 TTFT ms | 请求平均 TPOT ms | 平均 E2E ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result:
        text.append(
            f'| {row["layout"]} | {row["cohort"]} | {row["repeats"]} | '
            f'{row["ttft_ms"]:.2f} | {row["tpot_ms"]:.3f} | {row["e2e_ms"]:.2f} |')
    text += [
        "",
        "延迟越低越好。TTFT／E2E 从 HTTP 发起计时；TPOT 先逐请求计算"
        "首 token 后平均间隔，再对请求等权取均值。短输出会影响这一均值。",
        "",
        "## Decode 与缓存补充",
        "",
        "| 部署 | 按输出 token 间隔加权的 TPOT ms | P 前缀命中 | 实际新增 P token／请求 |",
        "|---|---:|---:|---:|",
    ]
    for row in result:
        hit = f'{row["p_prefix_hit_pct"]:.4f}%' if row["p_prefix_hit_pct"] is not None else "P侧元数据缺失"
        new = f'{row["mean_new_prefill_tokens"]:,.2f}' if row["mean_new_prefill_tokens"] is not None else "缺失"
        text.append(f'| {row["layout"]} | {row["interval_weighted_tpot_ms"]:.3f} | {hit} | {new} |')
    text += [
        "",
        "PD 缓存只取 P 端 prefill_meta，不能用 D 端的命中数代替。"
        "未保存 P 端字段时标缺失。相近缓存命中仍不能消除引擎、预算和到达时序差异。",
        "",
        "## 如何解读",
        "",
        "- 相对混部单卡：2P6D 的请求 E2E 约慢 38%–40%，现有数据没有显示 PD 分离的整体优势。",
        "- 相对混部 PP2：2P6D 请求 E2E 约慢 8%–10%，改善部分 Decode 间隔没有抵消首 token 成本。",
        "- 相对混部 EP4：请求 E2E 接近；请求等权 TPOT 的较大变化不等于纯 Decode 算力等比例提升，应同时看间隔加权列。",
        "- 相对旧 4P4D：差别很小，且跨引擎／回放，不能归因为 2P6D 比例更优。"
        "旧同 E5/V3 的 EP4 混部与 4P4D 配对，也没有看到平均 E2E 改善。",
        "",
        "## 可比范围",
        "",
        "混部使用 E5/V3，旧 4P4D 同为 E5/V3；新 2P6D 使用 E8/V4。"
        "V4 修正分支触发／返回时序，GDN/KV 预算及第二节点也有变化。"
        "整轮 wall_s 留在 CSV 中供追溯，不用其跨版本差异声称吞吐提升。"
        "表中的旧 4P4D 是 E5 下三轮完整 C8 历史结果，其整组 C128 失败；"
        "并非后来 E7/V3、仅有 C128 的三轮压力点。旧 E7 C128 不混入本表。",
        "",
        "下一步若验证 PD 收益，应补同一 E8/V4 的最佳混部；"
        "若验证 P:D 比例，应补同一版本的 4P4D。保持同样工作负载和缓存协议，"
        "记录 P 计算等待、D 接收预分配及完整完成情况。",
        "",
        f'[绝对数 CSV]({HERE / "c8_comparison.csv"}) · '
        f'[逐轮来源、哈希与差值]({HERE / "c8_comparison_evidence.json"}) · '
        f'[复算脚本]({Path(__file__).resolve()}) · '
        f'[当前 2P6D 状态]({HERE / "STATUS.md"})',
        "",
        f'[单卡与PD的差距分解]({HERE / "SINGLE_VS_PD_DIAGNOSIS.md"})：'
        "进一步分开首token前后时间、P排队、低batch布局成本及同输出对子集。",
    ]
    (HERE / "C8_COLOCATED_COMPARISON.md").write_text("\n".join(text) + "\n")
    print(json.dumps(dict(episodes=14, requests=2394, metrics=result,
                          comparisons=comparison), ensure_ascii=False))


if __name__ == "__main__":
    main()
