"""Offline token-cycle reconstruction from the immutable fair-comparison traces.

No GPU work. Formal streaming measurements and profiled GPU timelines remain
separate. PP graphs are matched by sequence-length sum and occurrence order,
then checked for causal order and a complete return to the same ring slot.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics as st

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RUNS = ("fair_native_20261026_v3", "fair_triton_20261026_v1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def union(spans):
    answer, end = 0., float("-inf")
    for a, b in sorted(spans):
        answer += max(0., b - max(a, end))
        end = max(end, b)
    return answer


def clipped(spans, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in spans if a < hi and b > lo]


def read_trace(path, expected_hash):
    assert digest(path) == expected_hash, path
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    events = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and "step[DECODE bs=4 " in e["name"]), key=lambda e: e["ts"])
    launches = [e for e in events if e["name"] == "cudaGraphLaunch"]
    by_correlation = defaultdict(list)
    for event in events:
        if event.get("cat") == "kernel":
            by_correlation[event.get("args", {}).get("correlation")].append(event)
    occurrences = Counter()
    graphs = []
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
        graphs.append(dict(key=key, cpu=step["ts"], cpu_end=step["ts"] + step["dur"],
                           launch=launch["ts"], start=min(e["ts"] for e in kernels),
                           end=max(e["ts"] + e["dur"] for e in kernels)))
    assert len(graphs) >= 22, path
    return dict(path=str(path), sha256=expected_hash, graphs=graphs, events=events,
                host=data["host_name"], base_time=data["baseTimeNanoseconds"])


def formal_stream_audit(path):
    data = json.loads(path.read_text())
    unique_tp, duplicate_counts, deltas = [], [], Counter()
    rows = [r for group in data["rounds"] for r in group["requests"]]
    for row in rows:
        events = row["events"]
        count, first, last = 0, None, None
        duplicate = 0
        for event in events:
            new = event["completion_tokens"]
            assert new >= count
            if new > count:
                deltas[new - count] += 1
                first = event["time_s"] if first is None else first
                last = event["time_s"]
            else:
                duplicate += 1
            count = new
        assert count == 128
        unique_tp.append((last - first) * 1000 / 127)
        duplicate_counts.append(duplicate)
    return dict(path=str(path), sha256=digest(path), requests=len(rows),
                reported_tpot_ms=st.mean(r["tpot_ms"] for r in rows),
                unique_token_tpot_ms=st.mean(unique_tp),
                repeated_completion_count_events=sum(duplicate_counts),
                count_increment_distribution=dict(deltas))


def main():
    detail = json.loads((ROOT / "analysis/profile_details.json").read_text())
    parsed, summaries, cycles, stream, communication, switches = {}, [], [], [], [], []
    for run in RUNS:
        for arm in ("ep2", "pp2"):
            selected = [d for d in detail if d["run"] == run and d["arm"] == arm]
            traces = [read_trace(Path(d["trace"]), d["trace_sha256"])
                      for d in sorted(selected, key=lambda d: d["rank_trace"])]
            assert len(traces) == 2
            assert traces[0]["host"] == traces[1]["host"]
            assert traces[0]["base_time"] == traces[1]["base_time"]
            for trace, old in zip(traces, sorted(selected, key=lambda d: d["rank_trace"])):
                assert [g["launch"] for g in trace["graphs"][4:20]] == [g["cpu_start_us"] for g in old["actual_graphs"]]
            parsed[run, arm] = traces
            record = dict(run=run, arm=arm, samples=16, timing_scope="separate profiled token cycle; not formal TPOT")
            local = []
            if arm == "pp2":
                a, b = (t["graphs"] for t in traces)
                b_by_key = {g["key"]: g for g in b}
                for i in range(4, 20):
                    current, nxt = a[i], a[i + 2]
                    remote = b_by_key[current["key"]]
                    assert nxt["key"] == (current["key"][0] + 4, current["key"][1]), (current, nxt)
                    assert current["end"] <= remote["start"] < remote["end"] <= nxt["start"]
                    segments = dict(stage0_graph_ms=(current["end"] - current["start"]) / 1000,
                                    forward_handoff_ms=(remote["start"] - current["end"]) / 1000,
                                    stage1_graph_ms=(remote["end"] - remote["start"]) / 1000,
                                    return_and_next_input_ms=(nxt["start"] - remote["end"]) / 1000)
                    total = (nxt["start"] - current["start"]) / 1000
                    assert abs(sum(segments.values()) - total) < 1e-8
                    local.append(dict(run=run, arm=arm, sequence_sum=current["key"][0],
                                      occurrence=current["key"][1], token_cycle_ms=total, **segments))
                record.update({k: st.mean(r[k] for r in local) for k in local[0] if k.endswith("_ms")})
                record["model_graphs_ms"] = record["stage0_graph_ms"] + record["stage1_graph_ms"]
                record["between_graphs_ms"] = record["forward_handoff_ms"] + record["return_and_next_input_ms"]
                record["compute_only_pipeline_bound_ms"] = 2 * max(record["stage0_graph_ms"], record["stage1_graph_ms"])
                record["bound_scope"] = "conditional model: unchanged stage times, perfect scheduling; not a measured optimized TPOT"
            else:
                for rank, trace in enumerate(traces):
                    a = trace["graphs"]
                    for i in range(4, 20):
                        current, nxt = a[i], a[i + 1]
                        assert nxt["key"][0] == current["key"][0] + 4
                        model = (current["end"] - current["start"]) / 1000
                        gap = (nxt["start"] - current["end"]) / 1000
                        assert gap >= 0
                        total = (nxt["start"] - current["start"]) / 1000
                        assert abs(model + gap - total) < 1e-8
                        local.append(dict(run=run, arm=arm, rank=rank,
                                          sequence_sum=current["key"][0], occurrence=current["key"][1],
                                          model_graphs_ms=model, between_graphs_ms=gap, token_cycle_ms=total))
                record.update({k: st.mean(r[k] for r in local) for k in local[0] if k.endswith("_ms")})
            cycles += local
            for rank, trace in enumerate(traces):
                a = trace["graphs"]
                lo, hi = a[4]["start"], a[20]["start"]
                graph_us = union(clipped([(g["start"], g["end"]) for g in a], lo, hi))
                record[f"rank{rank}_model_graph_fraction"] = graph_us / (hi - lo)
                nccl = [(e["ts"], e["ts"] + e["dur"]) for e in trace["events"]
                        if e.get("cat") == "kernel" and "nccl" in e["name"].lower()]
                record[f"rank{rank}_nccl_residency_per_forward_ms"] = union(clipped(nccl, lo, hi)) / 1000 / 16
                if arm == "pp2":
                    inventory = Counter()
                    for event in trace["events"]:
                        if event.get("cat") == "user_annotation" and event["name"].startswith("nccl:") and lo <= event["ts"] < hi:
                            args = event.get("args", {})
                            inventory[json.dumps(dict(operation=event["name"], dtype=args.get("Input type"),
                                                      dimensions=args.get("Input Dims")), sort_keys=True)] += 1
                    durations = sorted(e["dur"] / 1000 for e in trace["events"]
                                       if e.get("cat") == "kernel" and "nccl" in e["name"].lower()
                                       and lo <= e["ts"] < hi)
                    communication.append(dict(run=run, rank=rank,
                                              message_inventory=[dict(**json.loads(k), count=v) for k, v in inventory.items()],
                                              kernel_residency_ms=dict(min=min(durations), median=st.median(durations),
                                                                       max=max(durations)),
                                              interpretation="NCCL residency includes peer waits and overlaps model graphs; not link transfer time"))
            summaries.append(record)
            stream.append(dict(run=run, arm=arm, **formal_stream_audit(ROOT / "runs" / run / arm / "metrics.decode.json")))
            log = ROOT / "runs" / run / "cache" / f"{arm}.log"
            arg_line = next(line for line in log.read_text().splitlines() if "server_args=ServerArgs" in line)
            fields = ("disable_overlap_schedule", "pp_size", "pp_max_micro_batch_size", "pp_async_batch_depth", "stream_interval", "enable_dp_attention")
            switches.append(dict(run=run, arm=arm, source=str(log), sha256=digest(log),
                                 **{k: re.search(r"\b" + k + r"=([^,\)]*)", arg_line)[1] for k in fields}))
    metrics = list(csv.DictReader((ROOT / "analysis/metrics.csv").open()))
    formal = [dict(run=r["run"], arm=r["arm"], tpot_ms=float(r["tpot_ms"]), throughput_tok_s=float(r["throughput_tok_s"]))
              for r in metrics if r["phase"] == "decode"]
    sources = [dict(path=t["path"], sha256=t["sha256"]) for ts in parsed.values() for t in ts]
    scheduler = ROOT.parents[1] / ".engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/managers/scheduler_pp_mixin.py"
    assert scheduler.exists(), scheduler
    result = dict(protocol="Offline reuse of same H100178 NVLink GPU0/1, B8, cached4096+new1->128, BF16; no new experiment launched",
                  selection="Same first 16 B4 Graphs after discarding four as original component analysis",
                  matching="PP matches sequence-length sum and occurrence across ranks; next same ring slot is two forwards and four sequence-sum units later",
                  clock_validation="same host, same Kineto baseTimeNanoseconds; matching requests satisfy ordered stage0->stage1->same-slot-next-token",
                  caveat="All temporal decomposition numbers are from profiler runs. Do not subtract these from formal TPOT as an exact causal decomposition.",
                  formal=formal, profile=summaries, stream_audit=stream, trace_sources=sources,
                  communication=communication, runtime_switches=switches,
                  scheduler_source=dict(path=str(scheduler), sha256=digest(scheduler),
                                        launch_lines="631-641", retirement_lines="648-668", proxy_send_lines="670-685"))
    save(OUT / "timing.json", result)
    table(OUT / "profile_summary.csv", summaries)
    table(OUT / "profile_cycles.csv", cycles)
    optimized = RUNS[1]
    timeline = {}
    for arm in ("ep2", "pp2"):
        traces = parsed[optimized, arm]
        anchor = traces[0]["graphs"][8]
        origin = anchor["start"]
        upper = origin + 24000
        timeline[arm] = dict(anchor_key=anchor["key"], rows=[])
        for rank, trace in enumerate(traces):
            bars = [dict(key=g["key"], start_ms=(g["start"] - origin) / 1000,
                         end_ms=(g["end"] - origin) / 1000)
                    for g in trace["graphs"] if g["start"] < upper and g["end"] > origin]
            timeline[arm]["rows"].append(dict(rank=rank, graphs=bars))
    save(OUT / "timeline.json", timeline)
    print(json.dumps(dict(profile=summaries, formal=formal, stream_audit=stream), indent=2))


if __name__ == "__main__":
    main()
