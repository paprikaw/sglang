"""Offline AgentX corpus and accepted-cohort workload characterization.

No model server, GPU access, or experiment mutation. Quantiles interpolate
linearly. Source API intervals are not H100 batch sizes or service times.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
DATA = WORK / "datasets/agentx_20261030"
EARLY = WORK / "experiments/agentx_pd_feasibility_20261030"
DYNAMIC = WORK / "experiments/agentx_pd_dynamic_20261030"
PROFILE = HERE.parent / "agentx_workload_profile_20261031"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantile(values, fraction):
    a = sorted(values)
    if not a:
        return None
    position = (len(a) - 1) * fraction
    lo = int(position)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (position - lo)


def describe(values):
    a = list(values)
    if not a:
        return {"n": 0}
    return dict(n=len(a), mean=statistics.fmean(a), p50=quantile(a, .5),
                p90=quantile(a, .9), p99=quantile(a, .99), max=max(a))


def write_csv(name, rows):
    assert rows
    with (HERE / name).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with path.open() as src:
        return list(csv.DictReader(src))


def peak_intervals(intervals):
    events = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def lcp_tokens(a, b, block_size=64):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += block_size
    return count


def main():
    manifest = json.loads((PROFILE / "profile_manifest.json").read_text())
    raw_hashes = []
    for item in manifest["raw_sources"]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"], path
        raw_hashes.append(item)
    profile = csv_rows(PROFILE / "request_profiles.csv")
    assert len(profile) == 34371
    numeric = ["turn", "context_tokens", "cached_tokens", "device_cached_tokens",
               "host_cached_tokens", "new_prefill_tokens", "decode_tokens"]
    for row in profile:
        for key in numeric:
            row[key] = int(row[key])
        assert row["cached_tokens"] == row["device_cached_tokens"] + row["host_cached_tokens"]
        assert row["new_prefill_tokens"] == row["context_tokens"] - row["cached_tokens"]
    selected = [r for r in profile if r["arm"] == "col_ep4" and
                r["case"] == "agentx_c8" and r["repeat"] == "r1"]
    assert len(selected) == 171
    selected_ids = {r["source_family"] for r in selected}
    graph_file = DYNAMIC / "agentx_graph_candidates.json"
    graph_data = json.loads(graph_file.read_text())
    graphs = [g for g in graph_data["graphs"] if g["id"] in selected_ids]
    assert len(graphs) == 8
    conversations = {c["session_id"]: c for g in graphs for c in g["conversations"]}
    assert len(conversations) == 18
    source_turns = {}
    graph_families = []
    for graph in graphs:
        branches = [b for c in graph["conversations"] for b in c.get("branches", [])]
        for conv in graph["conversations"]:
            kinds = {t["source_kind"] for t in conv["turns"]}
            assert len(kinds) == 1
            kind = next(iter(kinds))
            for turn, data in enumerate(conv["turns"]):
                source_turns[conv["session_id"], turn] = dict(data, track_kind=kind)
        graph_families.append(dict(source_family=graph["id"],
            independent_tracks=len(graph["conversations"]),
            root_tracks=sum(c["is_root"] for c in graph["conversations"]),
            explicit_subagent_tracks=sum(c["turns"][0]["source_kind"] == "weka_subagent"
                                         for c in graph["conversations"]),
            reconstructed_flat_tracks=sum(c["turns"][0]["source_kind"] == "weka_flat"
                                         for c in graph["conversations"]),
            calls=sum(len(c["turns"]) for c in graph["conversations"]),
            branches=len(branches), background_branches=sum(b["is_background"] for b in branches),
            join_branches=sum(not b["is_background"] for b in branches)))
    for row in selected:
        turn = source_turns[row["source_conversation"], row["turn"]]
        assert row["context_tokens"] == turn["input_tokens"]
        assert row["decode_tokens"] == turn["output_tokens"]

    # Reuse the saved full-corpus accounting; recompute distributions of session sizes.
    session_sizes, corpora = [], {}
    for name in ("cc-traces-weka-062126", "cc-traces-weka-062126-256k"):
        stats = json.loads((EARLY / (name + "-statistics.json")).read_text())
        sessions = csv_rows(EARLY / (name + "-sessions.csv"))
        assert len(sessions) == stats["sessions"] == 393
        assert sum(int(r["requests"]) for r in sessions) == stats["requests"]
        assert sum(int(r["input_tokens"]) for r in sessions) == stats["total_input_tokens"]
        corpora[name] = stats
        fields = ["requests", "top_level", "nested", "input_tokens", "recorded_api_peak_overlap"]
        for field in fields:
            session_sizes.append(dict(dataset=name, metric=field,
                                      **describe(int(r[field]) for r in sessions)))

    # Stream the 256k corpus once, with a digest computed from the exact parsed bytes.
    types = collections.Counter()
    models = collections.Counter()
    group_types = collections.Counter()
    statuses = collections.Counter()
    request_fields = collections.Counter()
    group_durations, group_calls, original_think = [], [], []
    input_values, output_values, original_ttft = [], [], []
    selected_raw = []
    selected_group_durations = []
    raw_root_rows = []
    counters = collections.Counter()
    digest = hashlib.sha256()
    source = DATA / "cc-traces-weka-062126-256k/traces.jsonl"
    with source.open("rb") as src:
        for line in src:
            digest.update(line)
            trace = json.loads(line)
            assert trace["block_size"] == 64 and trace["hash_id_scope"] == "local"
            counters["roots"] += 1
            selected_trace = trace["id"] in selected_ids
            flat = []
            groups = []
            for outer in trace["requests"]:
                if outer["type"] == "subagent":
                    groups.append(outer)
                    group_types[outer.get("subagent_type") or "unlabeled"] += 1
                    statuses[outer.get("status") or "unlabeled"] += 1
                    group_calls.append(len(outer["requests"]))
                    if outer.get("duration_ms") is not None:
                        group_durations.append(outer["duration_ms"] / 1000)
                        if selected_trace:
                            selected_group_durations.append(outer["duration_ms"] / 1000)
                    entries = [(r, "explicit_subagent") for r in outer["requests"]]
                else:
                    entries = [(outer, "source_main_stream")]
                for row, role in entries:
                    assert row["in"] == 64 * len(row["hash_ids"])
                    counters["calls"] += 1
                    counters[role] += 1
                    types[role, row.get("type", "missing")] += 1
                    models[role, row.get("model", "missing")] += 1
                    request_fields.update(row.keys())
                    input_values.append(row["in"])
                    output_values.append(row["out"])
                    if row.get("think_time") is not None:
                        original_think.append(row["think_time"])
                    if row.get("ttft") is not None:
                        original_ttft.append(row["ttft"])
                    flat.append(row)
                    if selected_trace:
                        selected_raw.append(dict(source_family=trace["id"], role=role,
                            model=row.get("model"), type=row.get("type"),
                            input_tokens=row["in"], output_tokens=row["out"]))
            counters["groups"] += len(groups)
            counters["roots_with_explicit_subagents"] += bool(groups)
            intervals = [(r["t"], r["t"] + max(r.get("api_time") or 0, 0)) for r in flat]
            root_duration = max(end for _, end in intervals) - min(start for start, _ in intervals)
            raw_root_rows.append(dict(dataset="selected" if selected_trace else "256k_only",
                source_family=trace["id"], calls=len(flat), explicit_subagent_groups=len(groups),
                explicit_subagent_calls=sum(len(g["requests"]) for g in groups),
                recorded_span_s=root_duration, recorded_api_peak_overlap=peak_intervals(intervals)))
    assert digest.hexdigest() == "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"
    assert counters["roots"] == 393 and counters["calls"] == 68266
    assert counters["groups"] == 1697 and counters["explicit_subagent"] == 39822
    assert len(selected_raw) == 171
    corpus_extra = dict(counters=counters, input=describe(input_values), output=describe(output_values),
        subagent_duration_s=describe(group_durations), subagent_calls=describe(group_calls),
        recorded_root_span_s=describe(r["recorded_span_s"] for r in raw_root_rows),
        recorded_think_time_s=describe(original_think), original_provider_ttft_s=describe(original_ttft),
        request_field_coverage=dict(request_fields), group_types=dict(group_types), group_statuses=dict(statuses))
    labels = []
    for (role, label), n in sorted(models.items()):
        labels.append(dict(scope="256k", label_family="model", role=role, label=label, calls=n))
    for (role, label), n in sorted(types.items()):
        labels.append(dict(scope="256k", label_family="request_type", role=role, label=label, calls=n))
    for (role, label), n in sorted(collections.Counter((r["role"], r["model"]) for r in selected_raw).items()):
        labels.append(dict(scope="selected", label_family="model", role=role, label=label, calls=n))

    # Request count probability and token coverage are distinct cache measurements.
    by_case = collections.defaultdict(list)
    for row in profile:
        by_case[row["cohort"], row["arm"], row["case"]].append(row)
    cache_rows, tail_rows, partitions = [], [], []
    for (cohort, arm, case), rows in sorted(by_case.items()):
        count = len(rows)
        ctx = sum(r["context_tokens"] for r in rows)
        cached = sum(r["cached_tokens"] for r in rows)
        new = sum(r["new_prefill_tokens"] for r in rows)
        out = sum(r["decode_tokens"] for r in rows)
        cache_rows.append(dict(cohort=cohort, arm=arm, case=case, requests=count,
            any_prefix_hit_requests_pct=100 * sum(r["cached_tokens"] > 0 for r in rows) / count,
            any_host_hit_requests_pct=100 * sum(r["host_cached_tokens"] > 0 for r in rows) / count,
            zero_hit_requests_pct=100 * sum(r["cached_tokens"] == 0 for r in rows) / count,
            token_weighted_prefix_hit_pct=100 * cached / ctx,
            request_mean_prefix_hit_pct=statistics.fmean(100*r["cached_tokens"]/r["context_tokens"] for r in rows),
            token_weighted_device_hit_pct=100 * sum(r["device_cached_tokens"] for r in rows) / ctx,
            token_weighted_host_hit_pct=100 * sum(r["host_cached_tokens"] for r in rows) / ctx,
            mean_new_prefill_tokens=new / count,
            new_prefill_to_output_ratio=new / out,
            zero_hit_share_of_new_prefill_pct=100 * sum(r["new_prefill_tokens"] for r in rows if r["cached_tokens"] == 0) / new))
        ranked = sorted(rows, key=lambda r: r["new_prefill_tokens"], reverse=True)
        for fraction in (.01, .1):
            n = math.ceil(count * fraction)
            tail_rows.append(dict(cohort=cohort, arm=arm, case=case, top_request_fraction=fraction,
                requests_selected=n, selected_share_of_new_prefill_pct=100 * sum(r["new_prefill_tokens"] for r in ranked[:n]) / new))
        for label, keep in [("first_turn", lambda r: r["turn"] == 0),
                            ("later_turn", lambda r: r["turn"] > 0),
                            ("weka_main", lambda r: source_turns[r["source_conversation"],r["turn"]]["track_kind"] == "weka_main"),
                            ("weka_flat", lambda r: source_turns[r["source_conversation"],r["turn"]]["track_kind"] == "weka_flat"),
                            ("weka_subagent", lambda r: source_turns[r["source_conversation"],r["turn"]]["track_kind"] == "weka_subagent")]:
            part = [r for r in rows if keep(r)]
            total = sum(r["context_tokens"] for r in part)
            partitions.append(dict(cohort=cohort, arm=arm, case=case, partition=label, requests=len(part),
                mean_context_tokens=statistics.fmean(r["context_tokens"] for r in part),
                mean_output_tokens=statistics.fmean(r["decode_tokens"] for r in part),
                mean_new_prefill_tokens=statistics.fmean(r["new_prefill_tokens"] for r in part),
                token_weighted_prefix_hit_pct=100*sum(r["cached_tokens"] for r in part)/total,
                share_of_new_prefill_pct=100*sum(r["new_prefill_tokens"] for r in part)/new))

    # Intrinsic selected-input shapes, without counting clones/repeats as new traces.
    bins = []
    for metric, thresholds in [("context_tokens", [32768, 65536, 131072]),
                               ("decode_tokens", [128, 512, 2048])]:
        lower = 0
        for upper in thresholds + [None]:
            rows = [r for r in selected if r[metric] >= lower and (upper is None or r[metric] < upper)]
            bins.append(dict(metric=metric, lower_inclusive=lower, upper_exclusive=upper,
                requests=len(rows), request_fraction_pct=100*len(rows)/len(selected),
                input_token_fraction_pct=100*sum(r["context_tokens"] for r in rows)/sum(r["context_tokens"] for r in selected),
                output_token_fraction_pct=100*sum(r["decode_tokens"] for r in rows)/sum(r["decode_tokens"] for r in selected)))
            lower = upper
    track_sizes = [len(c["turns"]) for c in conversations.values()]
    first_inputs = [c["turns"][0]["input_tokens"] for c in conversations.values()]
    adjacent = []
    for conv in conversations.values():
        for prev, turn in zip(conv["turns"], conv["turns"][1:]):
            shared = lcp_tokens(prev["hash_ids"], turn["hash_ids"])
            adjacent.append(dict(shared_tokens=shared, input_tokens=turn["input_tokens"],
                new_vs_previous_input=turn["input_tokens"]-shared,
                context_shrunk=turn["input_tokens"] < prev["input_tokens"]))
    selected_description = dict(
        families=describe(g["calls"] for g in graph_families),
        independent_tracks=describe(track_sizes), one_call_tracks=sum(n == 1 for n in track_sizes),
        input=describe(r["context_tokens"] for r in selected), output=describe(r["decode_tokens"] for r in selected),
        total_input_tokens=sum(r["context_tokens"] for r in selected),
        total_output_tokens=sum(r["decode_tokens"] for r in selected),
        mean_family_cumulative_input_tokens=sum(r["context_tokens"] for r in selected)/8,
        mean_family_cumulative_output_tokens=sum(r["decode_tokens"] for r in selected)/8,
        mean_track_cumulative_input_tokens=sum(r["context_tokens"] for r in selected)/18,
        mean_track_cumulative_output_tokens=sum(r["decode_tokens"] for r in selected)/18,
        explicit_subagent_calls=sum(r["role"] == "explicit_subagent" for r in selected_raw),
        explicit_subagent_families=sum(r["explicit_subagent_groups"] > 0 for r in raw_root_rows if r["dataset"] == "selected"),
        explicit_subagent_groups=sum(r["explicit_subagent_tracks"] for r in graph_families),
        reconstructed_flat_tracks=sum(r["reconstructed_flat_tracks"] for r in graph_families),
        first_input=describe(first_inputs), adjacent_pairs=len(adjacent),
        adjacent_shared_input_token_pct=100*sum(r["shared_tokens"] for r in adjacent)/sum(r["input_tokens"] for r in adjacent),
        context_decreases=sum(r["context_shrunk"] for r in adjacent),
        adjacent_new_vs_previous_input=describe(r["new_vs_previous_input"] for r in adjacent),
        source_subagent_duration_s=describe(selected_group_durations),
        source_recorded_span_s=describe(r["recorded_span_s"] for r in raw_root_rows if r["dataset"] == "selected"))
    for row in cache_rows:
        prior = next(r for r in csv_rows(PROFILE / "summary.csv")
                     if (r["cohort"],r["arm"],r["case"]) == (row["cohort"],row["arm"],row["case"]))
        for key in ("token_weighted_prefix_hit_pct", "token_weighted_device_hit_pct",
                    "token_weighted_host_hit_pct", "mean_new_prefill_tokens"):
            assert math.isclose(row[key],float(prior[key]),rel_tol=1e-12), (key,row,prior)
    for name, rows in [("session_size_distributions.csv",session_sizes),
                       ("source_models_and_types.csv",labels), ("source_roots.csv",raw_root_rows),
                       ("selected_graph_families.csv",graph_families),
                       ("cache_hit_definitions.csv",cache_rows), ("new_prefill_tails.csv",tail_rows),
                       ("request_partitions.csv",partitions), ("selected_length_bins.csv",bins)]:
        write_csv(name, rows)
    sources = [PROFILE / "request_profiles.csv", PROFILE / "summary.csv", PROFILE / "profile_manifest.json", graph_file]
    sources += [EARLY / (name+suffix) for name in corpora for suffix in ("-statistics.json","-sessions.csv")]
    result = dict(id="qwen35b_h100_agentx_workload_exploration_20261031", cpu_only=True,
        scope="Official corpus characterization plus exactly the prior 21 accepted E5/V3 and E7/V3 episodes; newer runs excluded",
        quantiles="Linear interpolation; legacy saved corpus percentile fields retain their original nearest-rank convention",
        corpora=corpora, corpus_256k_extra=corpus_extra, selected=selected_description,
        source_256k=dict(path=str(source), sha256=digest.hexdigest()),
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        verified_raw_episodes=raw_hashes,
        caveats=["Cumulative input counts repeat prefixes and are not unique KV memory",
                 "Original provider intervals, think times and TTFT are source-trace metadata, not H100 measurements",
                 "Active prefetch accuracy, timeliness, transfer bytes and hidden latency are not recorded",
                 "V3 branch timings and engine/capacity differences remain; these cache rates are observations, not causal layout effects"])
    (HERE / "exploration.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(dict(corpus_256k_extra=corpus_extra,selected=selected_description),indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
