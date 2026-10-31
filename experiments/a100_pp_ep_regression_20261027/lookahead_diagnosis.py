"""Offline scheduling diagnosis from accepted A100 PP traces; launches no GPU work."""
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/a10035_nvlink_r4"
OUT = ROOT / "analysis/lookahead_diagnosis"


def read(path):
    return json.loads(path.read_text())


spec = importlib.util.spec_from_file_location(
    "trace_tools", ROOT.parent / "pp_ep_steady_decode_20261026/analyze.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def trace(path, expected_sha, local):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected_sha
    cache = OUT / "trace_summaries" / f"{path.name}.json"
    if cache.exists():
        value = read(cache)
        assert value["sha256"] == digest
        for forward in value["forwards"]:
            forward["key"] = tuple(forward["key"])
        return value
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    events = [e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    assert not any(e["name"] == "cudaGraphLaunch" for e in events)
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and e["name"].startswith(f"step[DECODE bs={local} ")), key=lambda e: e["ts"])
    assert len(steps) >= 38
    index = tools.IntervalIndex(steps)
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and e.get("args", {}).get("correlation") is not None}
    gpu = defaultdict(list)
    kernel_count = Counter()
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            continue
        step = index.find(launch["ts"], launch.get("tid"))
        if step is not None:
            gpu[id(step)].append((event["ts"], event["ts"] + event["dur"]))
            kernel_count[id(step)] += event.get("cat") == "kernel"
    occurrences = Counter()
    forwards = []
    for step in steps:
        seq = int(re.search(r"g_sk=(\d+)", step["name"])[1])
        key = (seq, occurrences[seq])
        occurrences[seq] += 1
        spans = gpu[id(step)]
        assert spans, (path, key)
        start = min(a for a, _ in spans)
        end = max(b for _, b in spans)
        active = tools.union_us(spans)
        forwards.append(dict(key=key, start=start, end=end,
                             cpu_ms=step["dur"] / 1000,
                             gpu_span_ms=(end - start) / 1000,
                             gpu_kernel_union_ms=active / 1000,
                             within_forward_no_kernel_ms=(end - start - active) / 1000,
                             kernel_count=kernel_count[id(step)]))
    value = dict(forwards=forwards, host=data["host_name"],
                 base_time=data["baseTimeNanoseconds"], source=str(path), sha256=digest)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(value, indent=2) + "\n")
    return value


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, cycles, sources = [], [], []
    for workload, local in [("decode_b8_repeated", 4), ("decode_b32_repeated", 16)]:
        for policy in (0, 9):
            folder = RUN / "pp2/workloads" / workload / "traces" / f"p{policy}"
            entries = sorted(read(folder / "manifest.json")["files"], key=lambda e: e["path"])
            assert [int(re.search(r"PP-(\d+)", e["path"])[1]) for e in entries] == [0, 1]
            traces = [trace(Path(e["path"]), e["sha256"], local) for e in entries]
            assert traces[0]["host"] == traces[1]["host"]
            assert traces[0]["base_time"] == traces[1]["base_time"]
            sources.extend({k: t[k] for k in ("source", "sha256")} for t in traces)
            head, tail = [t["forwards"] for t in traces]
            remote = {f["key"]: f for f in tail}
            selected = []
            for i in range(20, 36):
                first, following = head[i], head[i + 2]
                last = remote[first["key"]]
                assert following["key"] == (first["key"][0] + local, first["key"][1])
                # An eager step also includes independent preparation kernels;
                # next-step preparation can precede the tail's last GPU op.
                # Preserve this small overlap as a signed interval, not idle time.
                assert first["start"] < last["start"] < following["start"], (
                    workload, policy, i, first, last, following)
                row = dict(workload=workload, policy=policy, sequence_sum=first["key"][0],
                           occurrence=first["key"][1],
                           stage0_span_ms=first["gpu_span_ms"],
                           signed_forward_handoff_ms=(last["start"] - first["end"]) / 1000,
                           stage1_span_ms=last["gpu_span_ms"],
                           signed_feedback_ms=(following["start"] - last["end"]) / 1000,
                           cycle_ms=(following["start"] - first["start"]) / 1000,
                           stage0_cpu_ms=first["cpu_ms"], stage1_cpu_ms=last["cpu_ms"],
                           stage0_kernel_union_ms=first["gpu_kernel_union_ms"],
                           stage1_kernel_union_ms=last["gpu_kernel_union_ms"],
                           stage0_no_kernel_inside_span_ms=first["within_forward_no_kernel_ms"],
                           stage1_no_kernel_inside_span_ms=last["within_forward_no_kernel_ms"],
                           stage0_kernel_count=first["kernel_count"], stage1_kernel_count=last["kernel_count"])
                assert abs(row["cycle_ms"] - sum(row[k] for k in (
                    "stage0_span_ms", "signed_forward_handoff_ms", "stage1_span_ms", "signed_feedback_ms"))) < 1e-7
                selected.append(row)
            summary = dict(workload=workload, policy=policy, samples=len(selected))
            summary.update({k: st.mean(r[k] for r in selected) for k in selected[0]
                            if k.endswith("_ms") or k.endswith("_count")})
            summary["signed_between_forward_ms"] = summary["signed_forward_handoff_ms"] + summary["signed_feedback_ms"]
            summaries.append(summary)
            cycles.extend(selected)
            print(json.dumps(summary), flush=True)
    save_csv(OUT / "profile_summary.csv", summaries)
    save_csv(OUT / "profile_cycles.csv", cycles)
    result = dict(summaries=summaries, sources=sources, drop_initial_forwards=20,
                  selected_forwards=16, description=(
                      "Separate profiler-on cohort. Eager forward span covers GPU kernels/memcpy/memset "
                      "correlated to host CUDA calls inside each Decode step; includes launch gaps. "
                      "Kernel union excludes gaps and is not a compute utilization metric. "
                      "Eager preparation can overlap a prior stage's final GPU operations, so "
                      "inter-forward intervals are signed and are not pure idle or wire times. "
                      "Within-forward gaps are not attributable to lookahead without a causal experiment. "
                      "These profiled cycles are not profiler-off request TPOT."))
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
