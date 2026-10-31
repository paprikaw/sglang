"""Recompute formal metrics and independently profiled PP token cycles."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def read_trace(path, digest, local):
    assert sha(path) == digest, path
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and f"step[DECODE bs={local} " in e["name"]), key=lambda e: e["ts"])
    launches = [e for e in events if e["name"] == "cudaGraphLaunch"]
    by_correlation = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel":
            by_correlation[event.get("args", {}).get("correlation")].append(event)
    occurrences, graphs = Counter(), []
    for step in steps:
        seq = int(re.search(r"g_sk=(\d+)", step["name"])[1])
        key = (seq, occurrences[seq])
        occurrences[seq] += 1
        found = [e for e in launches if e.get("tid") == step.get("tid")
                 and step["ts"] <= e["ts"] < step["ts"] + step["dur"]]
        if not found:
            continue
        assert len(found) == 1, (path, key)
        launch = found[0]
        kernels = by_correlation[launch["args"]["correlation"]]
        assert kernels and all(e["args"].get("graph id") for e in kernels)
        graphs.append(dict(key=key, launch=launch["ts"], start=min(e["ts"] for e in kernels),
                           end=max(e["ts"] + e["dur"] for e in kernels)))
    return dict(graphs=graphs, events=events, host=data["host_name"], base_time=data["baseTimeNanoseconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--skip-profiles", action="store_true")
    args = ap.parse_args()
    run, out = args.run.absolute(), ROOT / "analysis" / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = read(run / "config.json")
    progress = read(run / "task/progress.json")
    audit = dict(stage=progress["stage"], partial=args.partial)
    if not args.partial:
        assert progress["stage"] == "complete", progress
        assert read(run / "task/cleanup.json")["passed"]
        checks = [read(p) for p in (run / "dashboard_guard/checks").glob("*-post.json")]
        assert len(checks) == 1 and checks[0]["passed"] and checks[0]["cleanup_verified"]
        records = []
        for line in (run / "supervisor.log").read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and "gpu_registry_state" in record:
                records.append(record)
        assert records and records[-1]["gpu_registry_state"] == "released" and records[-1]["postcheck"] == "passed"
        for entry in read(run / "dashboard_spec.json")["frozen_inputs"]:
            assert sha(entry["path"]) == entry["sha256"], entry
        audit.update(guard_postcheck="passed", cleanup_verified=True, gpu_registry_state="released",
                     all_frozen_hashes_verified=True)
    timing = load("lookahead_timing", ROOT.parent / "h100_ep_pp_fair_expert_20261026/tpot_analysis_20261027/analyze.py")
    profile_loader = load("lookahead_components", ROOT.parent / "h100_pp_expert_fragmentation_20261026/analyze.py")
    tools = profile_loader.load_analysis()
    old_bucket = tools.graph_bucket
    tools.graph_bucket = lambda name: "expert.layout_adapter" if name == "_metadata" or name.startswith("_metadata<") else old_bucket(name)
    visibility = load("lookahead_visibility", Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py"))
    metrics, cohorts, requests, components, profiles, cycles, sources, receipts, stream_audits, qualifications = [], [], [], [], [], [], [], [], [], []
    for arm in cfg["arm_order"]:
        if not args.partial:
            assert read(run / arm / "cleanup.json")["cleanup_verified"]
            assert read(run / arm / "completed.json")["passed"]
        for case in cfg["workloads"]:
            root = run / arm / "workloads" / case["id"]
            common = dict(workload=case["id"], family=case["family"], arm=arm,
                          global_requests=case["requests"], local_tokens=case["requests"] // 2)
            if not (root / "protocol.json").exists():
                continue
            protocol = read(root / "protocol.json")
            previous = ROOT.parent / "h100_expert_workload_validation_20261027/runs/triton_r1" / arm / "workloads" / case["id"] / "protocol.json"
            assert protocol["prompt_sha256"] == read(previous)["prompt_sha256"], root
            if arm == "pp2":
                info = read(root / "server_info.json")
                values = [s["pp_max_micro_batch_size"] for s in info["internal_states"] if "pp_max_micro_batch_size" in s]
                assert values and set(values) == {common["local_tokens"]}, values
            if not args.partial:
                assert read(root / "completed.json")["passed"]
                if arm == "pp2":
                    assert read(root / "output_agreement.json")["all_modes_equal"]
            variants = [("prefill", 0, root / "metrics.prefill.json")]
            policies = cfg["gap_policies"] if arm == "pp2" else [0]
            variants += [("decode", p, root / f"metrics.decode.p{p}.json") for p in policies]
            for phase, p, path in variants:
                if not path.exists():
                    continue
                data, receipt = visibility.wait_for_stable_json(path, timeout_s=20)
                receipts.append(receipt)
                assert data["valid"] and not data["profiler_enabled"]
                if args.partial and len(data["rounds"]) < cfg["repeats"]:
                    continue
                assert len(data["rounds"]) == cfg["repeats"]
                rows, rates = [], []
                for index, group in enumerate(data["rounds"]):
                    assert len(group["requests"]) == case["requests"]
                    for j, row in enumerate(group["requests"]):
                        assert (row.get("cached_tokens") or 0) == (4096 if phase == "decode" else 0)
                        assert row["output_tokens"] == (128 if phase == "decode" else 1)
                        assert row["prompt_tokens"] == (4097 if phase == "decode" else 4096)
                        if phase == "decode":
                            assert len(row["output_ids"]) == 128
                            counts = [e["completion_tokens"] for e in row["events"]]
                            assert counts == list(range(1, 129)), (path, index, j, counts)
                            assert abs(row["tpot_ms"] - (row["e2e_s"] - row["ttft_s"]) * 1000 / 127) < 1e-6
                        rows.append(row)
                        requests.append(dict(**common, phase=phase, policy=p, cohort=index, request=j,
                            e2e_ms=row["e2e_s"] * 1000, ttft_ms=row["ttft_s"] * 1000, tpot_ms=row.get("tpot_ms")))
                    if arm == "pp2" and phase == "decode":
                        observed = group["scheduling_receipt"]
                        assert observed["policy"] == p
                        if p == 9:
                            assert all(d["ahead_produced"] > 0 and d["ahead_produced"] == d["ahead_consumed"] for d in observed["deltas"])
                    rate = case["requests"] * (128 if phase == "decode" else 4096) / group["makespan_s"]
                    rates.append(rate)
                    cohorts.append(dict(**common, phase=phase, policy=p, cohort=index,
                        throughput_tok_s=rate, tpot_ms=st.mean(r["tpot_ms"] for r in group["requests"]) if phase == "decode" else None,
                        makespan_ms=group["makespan_s"] * 1000))
                metrics.append(dict(**common, phase=phase, policy=p, repeats=len(rates),
                    throughput_tok_s=st.mean(rates), throughput_stdev=st.stdev(rates),
                    throughput_min=min(rates), throughput_max=max(rates),
                    tpot_ms=st.mean(r["tpot_ms"] for r in rows) if phase == "decode" else None,
                    ttft_ms=st.mean(r["ttft_s"] for r in rows) * 1000,
                    e2e_ms=st.mean(r["e2e_s"] for r in rows) * 1000, source=str(path), sha256=sha(path)))
                sources.append(dict(path=str(path), sha256=sha(path), kind="formal_metrics"))
                if phase == "decode":
                    check = timing.formal_stream_audit(path)
                    assert check["repeated_completion_count_events"] == 0
                    assert abs(check["reported_tpot_ms"] - check["unique_token_tpot_ms"]) < 1e-8
                    stream_audits.append(dict(**common, policy=p, **check))
            for p in policies:
                if arm == "pp2" and (root / f"qualification.p{p}.json").exists():
                    qualification = read(root / f"qualification.p{p}.json")
                    assert qualification["passed"] and qualification["wire"]["passed"]
                    qualifications.append(dict(**common, **qualification))
                path = root / "traces" / f"p{p}" / "manifest.json"
                if args.skip_profiles or not path.exists():
                    continue
                manifest, receipt = visibility.wait_for_stable_json(path, timeout_s=20)
                receipts.append(receipt)
                entries = sorted(manifest["files"], key=lambda x: x["path"])
                assert len(entries) == 2
                local = common["local_tokens"]
                traces = [read_trace(Path(e["path"]), e["sha256"], local) for e in entries]
                sources += [dict(**e, kind="trace") for e in entries]
                assert traces[0]["host"] == traces[1]["host"] and traces[0]["base_time"] == traces[1]["base_time"]
                drop, selected_count = cfg["drop_initial_graphs"], cfg["selected_forwards"]
                stop = drop + selected_count
                for rank, (entry, trace) in enumerate(zip(entries, traces)):
                    assert len(trace["graphs"]) > stop + 2
                    tools.steady_step = lambda event, n=local: event is not None and f"step[DECODE bs={n} " in event["name"]
                    value, detail = tools.trace_components(Path(entry["path"]), arm, "native", "H100", stop - 4)
                    selected = detail["samples"][drop - 4:]
                    assert len(selected) == selected_count
                    value.update({k: st.mean(row.get(k, 0) for row in selected) for k in {k for row in selected for k in row}})
                    value.update(unit=f"ms/rank/actual-layer-call; local Decode batch {local}",
                                 actual_layer_invocations=selected_count * value["layers_per_rank"])
                    for k in list(value):
                        if k.startswith("pp_boundary_"):
                            value.pop(k)
                    assert [g["launch"] for g in trace["graphs"][drop:stop]] == [g["cpu_start_us"] for g in detail["actual_graphs"][drop - 4:]]
                    components.append({**value, **common, "policy": p, "rank": rank,
                        "selected_forwards": selected_count, "dropped_initial_decode_forwards": drop})
                samples = []
                if arm == "pp2":
                    a, b = [t["graphs"] for t in traces]
                    remote = {g["key"]: g for g in b}
                    for i in range(drop, stop):
                        g, n = a[i], a[i + 2]
                        r = remote[g["key"]]
                        assert n["key"] == (g["key"][0] + local, g["key"][1]), (g, n)
                        assert g["end"] <= r["start"] < r["end"] <= n["start"]
                        row = dict(stage0_graph_ms=(g["end"] - g["start"]) / 1000,
                            forward_handoff_ms=(r["start"] - g["end"]) / 1000,
                            stage1_graph_ms=(r["end"] - r["start"]) / 1000,
                            feedback_ms=(n["start"] - r["end"]) / 1000,
                            cycle_ms=(n["start"] - g["start"]) / 1000)
                        assert abs(row["cycle_ms"] - sum(row[k] for k in ("stage0_graph_ms", "stage1_graph_ms", "forward_handoff_ms", "feedback_ms"))) < 1e-8
                        samples.append(dict(sequence_sum=g["key"][0], occurrence=g["key"][1], **row))
                else:
                    for rank, trace in enumerate(traces):
                        a = trace["graphs"]
                        for i in range(drop, stop):
                            g, n = a[i], a[i + 1]
                            assert n["key"][0] == g["key"][0] + local
                            samples.append(dict(rank=rank, sequence_sum=g["key"][0],
                                model_graphs_ms=(g["end"] - g["start"]) / 1000,
                                between_graphs_ms=(n["start"] - g["end"]) / 1000,
                                cycle_ms=(n["start"] - g["start"]) / 1000))
                record = dict(**common, policy=p, samples=len(samples))
                record.update({k: st.mean(row[k] for row in samples) for k in samples[0] if k.endswith("_ms")})
                if arm == "pp2":
                    record["model_graphs_ms"] = record["stage0_graph_ms"] + record["stage1_graph_ms"]
                    record["between_graphs_ms"] = record["forward_handoff_ms"] + record["feedback_ms"]
                for rank, trace in enumerate(traces):
                    a = trace["graphs"]
                    lo, hi = a[drop]["start"], a[stop]["start"]
                    record[f"rank{rank}_graph_fraction"] = timing.union(timing.clipped([(g["start"], g["end"]) for g in a], lo, hi)) / (hi - lo)
                profiles.append(record)
                cycles += [dict(**common, policy=p, **row) for row in samples]
                # Large raw traces can be released from this analyzer's memory.
                del traces
    comparisons = []
    for case in cfg["workloads"]:
        data = [m for m in metrics if m["workload"] == case["id"] and m["phase"] == "decode"]
        if len(data) != 3:
            continue
        ep = next(m for m in data if m["arm"] == "ep2")
        baseline = next(m for m in data if m["arm"] == "pp2" and m["policy"] == 0)
        lookahead = next(m for m in data if m["arm"] == "pp2" and m["policy"] == 9)
        comparisons.append(dict(workload=case["id"], requests=case["requests"], local_tokens=case["requests"] // 2,
            ep_tok_s=ep["throughput_tok_s"], pp_baseline_tok_s=baseline["throughput_tok_s"],
            pp_lookahead_tok_s=lookahead["throughput_tok_s"],
            pp_lookahead_gain_pct=(lookahead["throughput_tok_s"] / baseline["throughput_tok_s"] - 1) * 100,
            pp_vs_ep_pct=(lookahead["throughput_tok_s"] / ep["throughput_tok_s"] - 1) * 100,
            ep_tpot_ms=ep["tpot_ms"], pp_baseline_tpot_ms=baseline["tpot_ms"], pp_lookahead_tpot_ms=lookahead["tpot_ms"]))
    if not args.partial:
        assert len(metrics) == 20 and len(comparisons) == 4 and len(qualifications) == 8
        if not args.skip_profiles:
            assert len(components) == 24 and len(profiles) == 12
    result = dict(run=str(run), partial=args.partial, profiles_skipped=args.skip_profiles,
        config=cfg, run_audit=audit, metrics=metrics, comparisons=comparisons,
        components=components, profiles=profiles, qualifications=qualifications,
        stream_audits=stream_audits, visibility=receipts, source_descriptors=sources,
        analyzer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    save(out / "results.json", result)
    for name, data in (("metrics", metrics), ("cohorts", cohorts), ("requests", requests),
                       ("components", components), ("profile_summary", profiles),
                       ("profile_cycles", cycles), ("comparisons", comparisons)):
        table(out / f"{name}.csv", data)
    print(json.dumps(dict(audit=audit, comparisons=comparisons, metrics=len(metrics), components=len(components)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
