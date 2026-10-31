"""CPU-only diagnosis of saved Prefill traces; no model/GPU imports or launches."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "ep2_pp2_qwen_a100_20261024/runs"
OUT = HERE / "qwen_prefill_diagnosis"
ARMS = [("EP2", ROOT / "qwen122_ep2_pp2_r7", "ep2"),
        ("PP2_4slots", ROOT / "qwen122_ep2_pp2_r7", "pp2"),
        ("PP2_2slots", ROOT / "qwen122_pp2_two_slots_r8", "pp2")]


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def duration(intervals):
    return sum(hi - lo for lo, hi in merge(intervals))


def overlap(first, second):
    a, b = merge(first), merge(second)
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def enclosing(events, timestamp, tid):
    candidates = [e for e in events if e.get("tid") == tid and
                  e["ts"] <= timestamp < e["ts"] + e["dur"]]
    return min(candidates, key=lambda e: e["dur"]) if candidates else None


def csv_file(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    scope_rows, matched, summary, sources = [], [], [], []
    for label, run, arm in ARMS:
        for path in sorted((run / arm / "traces/prefill").glob("*.trace.json.gz")):
            with gzip.open(path, "rt") as handle:
                raw = json.load(handle)
                device_properties = raw.get("deviceProperties", [])
                events = [e for e in raw["traceEvents"]
                          if e.get("ph") == "X" and e.get("dur", 0) > 0]
            scopes = [e for e in events if e.get("cat") == "user_annotation"
                      and e.get("name", "").startswith("component.attention.")]
            runtime = {e["args"]["correlation"]: e for e in events
                       if e.get("cat") in ("cuda_runtime", "cuda_driver")
                       and "correlation" in e.get("args", {})}
            gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
            origin = min(e["ts"] for e in gpu)
            nccl = merge([(e["ts"], e["ts"] + e["dur"]) for e in gpu
                          if "nccl" in e.get("name", "").lower()])
            intervals, signatures, name_totals = defaultdict(list), defaultdict(list), defaultdict(float)
            ordinals = defaultdict(int)
            parents = {}
            projection_example = None
            for event in sorted(gpu, key=lambda e: e["ts"]):
                launch = runtime.get(event.get("args", {}).get("correlation"))
                if not launch:
                    continue
                scope = enclosing(scopes, launch["ts"], launch.get("tid"))
                if not scope:
                    continue
                key = id(scope)
                parents[key] = scope
                interval = (event["ts"], event["ts"] + event["dur"])
                intervals[key].append(interval)
                category = ".".join(scope["name"].split(".")[1:-1])
                args = event.get("args", {})
                if event.get("cat") == "kernel":
                    if (projection_example is None and "::marlin::Marlin" in event["name"]
                            and args.get("grid") == [108, 1, 1]):
                        projection_example = {k: args.get(k) for k in
                            ["grid", "block", "registers per thread", "shared memory"]}
                    name_totals[(category, event["name"])] += event["dur"]
                    fraction = overlap([interval], nccl) / event["dur"]
                    state = "overlap" if fraction >= .9 else "no_overlap" if fraction <= .01 else "partial"
                    geometry = (event["name"], tuple(args.get("grid", [])), tuple(args.get("block", [])))
                    ordinal = ordinals[(key, geometry)]
                    ordinals[(key, geometry)] += 1
                    signature = (category, int(scope["name"].split(".")[-1]), *geometry, ordinal)
                    signatures[signature].append((state, event["dur"]))
            for key, spans in intervals.items():
                scope = parents[key]
                active_us = duration(spans)
                overlap_us = overlap(spans, nccl)
                scope_rows.append(dict(configuration=label, trace=path.name,
                    component=".".join(scope["name"].split(".")[1:-1]),
                    layer=int(scope["name"].split(".")[-1]),
                    gpu_start_ms=(min(x[0] for x in spans) - origin) / 1000,
                    gpu_end_ms=(max(x[1] for x in spans) - origin) / 1000,
                    gpu_active_ms=active_us / 1000, nccl_overlap_ms=overlap_us / 1000,
                    nccl_overlap_fraction=overlap_us / active_us))
            for (category, layer, name, grid, block, ordinal), timings in signatures.items():
                high = [value for state, value in timings if state == "overlap"]
                low = [value for state, value in timings if state == "no_overlap"]
                if high and low:
                    matched.append(dict(configuration=label, trace=path.name,
                        component=category, layer=layer, kernel=name,
                        grid=str(grid), block=str(block), same_name_shape_ordinal=ordinal,
                        overlap_samples=len(high),
                        no_overlap_samples=len(low), overlap_median_us=statistics.median(high),
                        no_overlap_median_us=statistics.median(low),
                        median_ratio=statistics.median(high) / statistics.median(low)))
            selected = [r for r in scope_rows if r["trace"] == path.name and r["configuration"] == label]
            assert len(selected) == 192, (path, len(selected))
            existing_path = run / "analysis" / path.name.replace(".trace.json.gz", ".json")
            existing = json.loads(existing_path.read_text())
            expected = sum(sum(v for k, v in r["gpu_component_union_ms"].items()
                               if k.startswith("attention.")) for r in existing["invocations"]
                           if "EXTEND bs=1 toks=4096 " in r["step"])
            assert math.isclose(sum(r["gpu_active_ms"] for r in selected), expected,
                                abs_tol=1e-5), (path, expected)
            long_nccl = sorted([e for e in gpu if "nccl" in e.get("name", "").lower()
                                and e["dur"] >= 50000], key=lambda e: e["dur"], reverse=True)
            top = sorted(name_totals.items(), key=lambda item: item[1], reverse=True)[:5]
            item = dict(configuration=label, trace=path.name,
                attention_union_ms=sum(r["gpu_active_ms"] for r in selected),
                attention_nccl_overlap_ms=sum(r["nccl_overlap_ms"] for r in selected),
                nccl_kernel_union_ms=duration(nccl) / 1000,
                nccl_kernels_top_ms=sorted([e["dur"] / 1000 for e in gpu
                    if "nccl" in e.get("name", "").lower()], reverse=True)[:3],
                device_properties=device_properties,
                marlin_projection_resource_example=projection_example,
                long_nccl_resource_examples=[dict(duration_ms=e["dur"] / 1000,
                    **{k: e.get("args", {}).get(k) for k in
                       ["grid", "block", "registers per thread", "shared memory"]})
                    for e in long_nccl[:2]],
                top_attention_kernels=[dict(component=k[0], kernel=k[1], total_ms=v / 1000)
                                       for k, v in top])
            summary.append(item)
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    csv_file("attention_scopes.csv", scope_rows)
    csv_file("matched_kernel_overlap.csv", matched)
    result = dict(no_gpu_launch=True, method="GPU launch correlation to CPU Attention annotation; interval intersection with NCCL kernels",
        scope_count=len(scope_rows), matched_kernel_groups=len(matched), sources=sources,
        caveat="Overlap correlation is not a causal SM, clock or bandwidth isolation experiment; kernel grid/block equality does not attest every tensor stride.",
        traces=summary)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for item in summary:
        print(json.dumps({k: v for k, v in item.items() if k not in
                         ("top_attention_kernels", "device_properties") }))
        for k in item["top_attention_kernels"][:2]:
            print("TOP_ATTENTION_KERNEL", k["component"], k["total_ms"], k["kernel"][:140])
    for row in sorted(matched, key=lambda r: r["no_overlap_median_us"], reverse=True)[:12]:
        print("MATCHED", json.dumps({k: (v[:120] if k == "kernel" else v) for k, v in row.items()}))


if __name__ == "__main__":
    main()
