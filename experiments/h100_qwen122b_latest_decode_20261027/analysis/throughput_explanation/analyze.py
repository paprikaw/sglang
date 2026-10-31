"""Read-only trace analysis; keep profiler samples separate from formal timing."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics as st

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = load("explanation_reader", ROOT.parent / "h100_pp_lookahead_workloads_20261027/analyze.py")
intervals = load("explanation_intervals", ROOT / "analyze.py")
buckets = load("explanation_buckets", ROOT.parent / "h100_pp_expert_fragmentation_20261026/analyze.py").load_analysis()
result = {"workload": "b32_repeated", "new_gpu_experiment": False, "arms": []}

for run, layout in (("ep2_native_r5", "ep2"), ("pp2_lookahead_r6", "pp2")):
    data = json.loads((ROOT / "analysis" / run / "results.json").read_text())
    ranks = [c for c in data["components"] if c["workload"] == result["workload"]]
    traces = [reader.read_trace(Path(c["trace"]["path"]), c["trace"]["sha256"], 16) for c in ranks]
    arm = {"run": run, "sources": [c["trace"] for c in ranks], "rank_samples": []}
    for component, trace in zip(ranks, traces):
        graphs = trace["graphs"][20:36]
        launches = {e["ts"]: e for e in trace["events"] if e["name"] == "cudaGraphLaunch"}
        correlations = {launches[g["launch"]]["args"]["correlation"] for g in graphs}
        spans = defaultdict(list)
        for event in trace["events"]:
            if event.get("cat") == "kernel" and event["args"].get("correlation") in correlations:
                spans[buckets.graph_bucket(event["name"])].append((event["ts"], event["ts"] + event["dur"]))
        comm = intervals.union(spans["communication.dispatch"] + spans["communication.combine"])
        compute = intervals.union([s for k, ss in spans.items() if not k.startswith("communication.") for s in ss])
        divisor = 16 * component["layers_per_rank"]
        arm["rank_samples"].append({
            "rank": component["rank"], "selected_graphs": 16,
            "layers_per_graph": component["layers_per_rank"],
            "captured_target_graphs": len(trace["graphs"]),
            "selected_sequence_sum_keys": [g["key"] for g in graphs],
            "comm_union_us_per_layer": sum(b - a for a, b in comm) / divisor,
            "comm_intersection_all_compute_us_per_layer": intervals.intersect(comm, compute) / divisor,
            "comm_intersection_routed_gemm_us_per_layer": component["comm_gemm_overlap_us_per_layer"],
        })
    cycles = []
    if layout == "pp2":
        assert traces[0]["host"] == traces[1]["host"]
        assert traces[0]["base_time"] == traces[1]["base_time"]
        # Use local GPU clocks and both alternating slots for the budget.
        # Sequence-sum/occurrence matching alone did not establish a causal
        # cross-rank stage boundary on these policy9 traces.
        for trace in traces:
            a = trace["graphs"]
            for i in range(20, 36):
                g, middle, n = a[i:i + 3]
                assert n["key"] == (g["key"][0] + 16, g["key"][1])
                assert g["end"] <= middle["start"] and middle["end"] <= n["start"]
                model = (g["end"] - g["start"] + middle["end"] - middle["start"]) / 1000
                gap = (middle["start"] - g["end"] + n["start"] - middle["end"]) / 1000
                cycle = (n["start"] - g["start"]) / 1000
                assert abs(model + gap - cycle) < 1e-8
                cycles.append({"two_local_graphs_ms": model,
                               "between_graphs_ms": gap,
                               "cycle_ms": cycle})
    else:
        for trace in traces:
            for i in range(20, 36):
                g, n = trace["graphs"][i:i + 2]
                assert n["key"][0] == g["key"][0] + 16
                cycles.append({"model_graph_ms": (g["end"] - g["start"]) / 1000,
                               "between_graphs_ms": (n["start"] - g["end"]) / 1000,
                               "cycle_ms": (n["start"] - g["start"]) / 1000})
    arm["profile_cycle_mean"] = {k: st.mean(c[k] for c in cycles) for k in cycles[0]}
    formal = ROOT / "runs" / run / layout / "workloads" / result["workload"] / "metrics.decode.json"
    data = json.loads(formal.read_text())
    bands = defaultdict(list)
    for row in (r for group in data["rounds"] for r in group["requests"]):
        events = sorted(row["events"], key=lambda e: e["completion_tokens"])
        assert [e["completion_tokens"] for e in events] == list(range(1, 129))
        for a, b in zip(events, events[1:]):
            label = "2_to_8" if b["completion_tokens"] <= 8 else "9_to_120" if b["completion_tokens"] <= 120 else "121_to_128"
            bands[label].append((b["time_s"] - a["time_s"]) * 1000)
    arm["formal_source"] = {"path": str(formal), "sha256": reader.sha(formal)}
    arm["formal_inter_token_mean_ms"] = {k: st.mean(v) for k, v in bands.items()}
    arm["formal_profile_off_output_tok_s"] = st.mean(g["output_tokens_per_s"] for g in data["rounds"])
    result["arms"].append(arm)
    print(json.dumps({k: v for k, v in arm.items() if k not in ("sources", "rank_samples")}, indent=2))

result["limitations"] = [
    "Kernel duration sums and interval intersections do not measure counterfactual communication cost.",
    "Identical 20-graph drop and 16-graph selection cover different token ranges in EP and two-slot PP.",
    "Independent profiler-on cycle samples cannot be subtracted from full profiler-off TPOT.",
    "Between-graph intervals include dependencies, scheduling and boundary work; they are not pure wire times.",
    "No causal cross-rank handoff/feedback decomposition is claimed from sequence-sum matching alone.",
]
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
