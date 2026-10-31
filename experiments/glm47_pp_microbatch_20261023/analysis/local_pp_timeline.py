"""Offline local PP timeline analysis; never launches or modifies a run.

Uses the existing calibrated exact-name map.  Each rank has its own CUDA clock:
cross-host overlap is deliberately not estimated from timestamp subtraction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("microbatch_attribution", ROOT / "analyze.py")
ATTR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ATTR)


def mean(values):
    return statistics.mean(values) if values else None


def overlap_partition(events, low, high, known, marker):
    """Non-overlapping wall timeline, preserving all active category sets."""
    changes = defaultdict(Counter)
    pp_kernels = []
    for event in events:
        category = event.get("cat")
        if category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        start, end = ATTR.interval(event)
        start, end = max(start, low), min(end, high)
        if start >= end:
            continue
        name = event.get("name", "")
        label = ("marker" if name == marker else known.get(name, "unknown_gpu")) if category == "kernel" else category
        if label == "pp_communication":
            pp_kernels.append({"name": name, "window_relative_start_ms": (start-low)/1000,
                               "clipped_duration_ms": (end-start)/1000})
        changes[start][label] += 1
        changes[end][label] -= 1
    active, totals, previous = Counter(), Counter(), low
    for timestamp, delta in sorted(changes.items()):
        labels = tuple(sorted(key for key, count in active.items() if count > 0))
        totals[labels] += timestamp - previous
        active.update(delta)
        previous = timestamp
    totals[()] += high - previous
    if abs(sum(totals.values()) - (high - low)) > 1e-3:
        raise ValueError("Local partition does not cover the complete window")
    pp_any = sum(value for labels, value in totals.items() if "pp_communication" in labels)
    pp_only = totals.get(("pp_communication",), 0)
    compute = {"attention", "expert_routed", "expert_shared_or_dense"}
    compute_any = sum(value for labels, value in totals.items() if set(labels) & compute)
    pp_compute = sum(value for labels, value in totals.items() if "pp_communication" in labels and set(labels) & compute)
    pp_durations = sorted(row["clipped_duration_ms"] for row in pp_kernels)
    return {
        "window_ms": (high-low)/1000,
        "category_sets_ms": {" + ".join(labels) if labels else "no_traced_gpu_activity": value/1000 for labels, value in sorted(totals.items())},
        "pp_kernel_union_ms": pp_any/1000,
        "pp_kernel_exclusive_ms": pp_only/1000,
        "pp_kernel_overlapping_compute_ms": pp_compute/1000,
        "compute_kernel_union_ms": compute_any/1000,
        "no_traced_gpu_activity_ms": totals.get((), 0)/1000,
        "pp_kernel_union_pct": pp_any/(high-low)*100,
        "pp_kernel_exclusive_pct": pp_only/(high-low)*100,
        "compute_kernel_union_pct": compute_any/(high-low)*100,
        "no_traced_gpu_activity_pct": totals.get((), 0)/(high-low)*100,
        "pp_kernel_count_intersecting_window": len(pp_kernels),
        "pp_kernel_clipped_duration_median_ms": statistics.median(pp_durations) if pp_durations else None,
        "pp_kernel_clipped_duration_p99_ms": pp_durations[math.ceil(.99*len(pp_durations))-1] if pp_durations else None,
        "pp_kernel_clipped_duration_max_ms": max(pp_durations) if pp_durations else None,
        "pp_kernels_longer_than_1_ms": [row for row in pp_kernels if row["clipped_duration_ms"] > 1],
    }


def rank_report(folder, rank, known):
    receipt = ATTR.read(folder/f"mtp_activity_receipt_rank{rank}.json")
    signature = ATTR.read(folder/f"mtp_marker_signature_rank{rank}.json")["name"]
    trace = ATTR.read(Path(receipt["trace"]))
    events = trace["traceEvents"]
    markers = sorted((event for event in events if event.get("cat") == "kernel" and event.get("name") == signature), key=lambda row: row["ts"])
    if not receipt["passed"] or len(markers) != 2*len(receipt["segments"]):
        return {"rank": rank, "stage": rank//4, "valid": False,
                "error": "Marker/receipt mismatch; no attempt to infer or repair missing marker",
                "receipt_claims_passed": receipt["passed"],
                "actual_markers": len(markers), "expected_markers": 2*len(receipt["segments"])}
    segments = []
    for index, metadata in enumerate(receipt["segments"]):
        start, end = markers[2*index]["ts"]+markers[2*index]["dur"], markers[2*index+1]["ts"]
        if start >= end:
            raise ValueError("Reversed marker order")
        segments.append(dict(metadata, local_start_us=start, local_end_us=end, envelope_ms=(end-start)/1000))
    verify = [row for row in segments if row["phase"] == "target_verify"]
    interior = verify[1:-1]
    if not interior:
        raise ValueError("No interior target-verify sample")
    low, high = interior[0]["local_start_us"], interior[-1]["local_end_us"]
    cadence = []
    by_cohort = defaultdict(list)
    for row in verify:
        by_cohort[tuple(row.get("request_ids", []))].append(row)
    for ids, rows in by_cohort.items():
        for first, second in zip(rows, rows[1:]):
            cadence.append({
                "request_ids": list(ids),
                "from_iteration": first["cohort_iteration"],
                "to_iteration": second["cohort_iteration"],
                "local_verify_start_interval_ms": (second["local_start_us"]-first["local_start_us"])/1000,
                "local_between_verify_envelopes_ms": (second["local_start_us"]-first["local_end_us"])/1000,
            })
    attributed = ATTR.rank_result(folder, rank, known)
    phase_partition = attributed["phases"]
    return {
        "rank": rank, "stage": rank//4, "valid": True,
        "host": trace.get("host_name"),
        "window_boundary": "Start of second local target-verify envelope through end of penultimate local target-verify envelope",
        "interior_verify_count": len(interior),
        "target_verify_phases": phase_partition,
        "interior_target_verify_segments": [row for row in attributed["segments"] if row["phase"] == "target_verify"][1:-1],
        "local_window": overlap_partition(events, low, high, known, signature),
        "local_verify_cadence": cadence,
        "mean_local_same_cohort_verify_start_interval_ms": mean([row["local_verify_start_interval_ms"] for row in cadence]),
        "mean_local_same_cohort_between_verify_envelopes_ms": mean([row["local_between_verify_envelopes_ms"] for row in cadence]),
    }


def analyze(run):
    folder = run/"pp2ep4"
    known, audit = ATTR.calibrate(folder)
    if not any(label == "pp_communication" for label in known.values()):
        raise ValueError("No exact calibrated PP transport name; cannot attribute PP kernel time")
    result = {
        "run": str(run.resolve()),
        "calibration": audit,
        "ranks": [rank_report(folder, rank, known) for rank in range(8)],
        "caveats": [
            "Every rank is a local clock domain. Cross-host simultaneous execution and pipeline overlap are not calculated.",
            "PP kernel duration includes transfer and device polling/waiting; it is not pure wire latency or an exact pipeline bubble duration.",
            "No traced GPU activity means no captured kernel/copy/set activity; it is not a measured SM occupancy or proof that the GPU is globally idle.",
            "Category-set partitions are exclusive wall-time bins. Union metrics intentionally overlap and must not be summed.",
            "Local same-cohort verify cadence includes the other microbatch, draft work, communication and host scheduling. It is not per output token TPOT.",
            "Representative rank 0 and rank 4 measure different stages. TP peer ranks are not independent samples and must not be summed as serial model execution.",
            "Production latency is unprofiled and separate from these bounded profiled intervals.",
        ],
    }
    by_stage = {stage: [row for row in result["ranks"] if row.get("valid") and row["stage"] == stage] for stage in (0, 1)}
    if all(by_stage.values()):
        representatives = [by_stage[stage][0] for stage in (0, 1)]
        keyed = [
            {(tuple(row.get("request_ids", [])), row["cohort_iteration"]): row for row in rank["interior_target_verify_segments"] if row.get("request_ids")}
            for rank in representatives
        ]
        pairs = []
        for key in sorted(keyed[0].keys() & keyed[1].keys()):
            first, second = keyed[0][key], keyed[1][key]
            categories = first["partition_ms"].keys() | second["partition_ms"].keys()
            pairs.append({"request_ids": list(key[0]), "cohort_iteration": key[1],
                          "serial_stage_envelope_sum_ms": first["envelope_ms"]+second["envelope_ms"],
                          "serial_stage_partition_sum_ms": {category: first["partition_ms"].get(category, 0)+second["partition_ms"].get(category, 0) for category in categories}})
        categories = {category for row in pairs for category in row["serial_stage_partition_sum_ms"]}
        result["matched_serial_target_verify"] = {
            "representative_ranks": [row["rank"] for row in representatives],
            "meaning": "Same RID/cohort ordinal, one valid TP peer per stage. Serial local API sums exclude stage boundary and draft; not elapsed wall time or TPOT.",
            "pair_count": len(pairs),
            "mean_serial_stage_envelope_sum_ms": mean([row["serial_stage_envelope_sum_ms"] for row in pairs]),
            "mean_serial_stage_partition_sum_ms": {category: mean([row["serial_stage_partition_sum_ms"].get(category, 0) for row in pairs]) for category in categories},
            "pairs": pairs,
        }
    task = run/"task/results.json"
    if task.exists():
        rows = ATTR.read(task).get("arms", [])
        result["production_rows"] = [
            {key: row[key] for key in ("microbatch_mode", "prefix_tokens_per_request", "actual_cached_tokens_per_request", "first_forward_new_tokens_per_request", "request_count", "output_tokens_per_request", "throughput_tok_s", "ttft_s", "tpot_s", "end_to_end_s") if key in row}
            for arm in rows for row in arm.get("results", [])
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output outside immutable run/runtime directories")
    args = parser.parse_args()
    output = args.out.resolve()
    for run in args.run:
        if output.is_relative_to(run.resolve()) or output.is_relative_to((ROOT/"runtime").resolve()):
            parser.error("Output must not modify a frozen run or runtime directory")
    result = {"runs": [analyze(run) for run in args.run]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    for run in result["runs"]:
        print(run["run"])
        for rank in run["ranks"]:
            if not rank["valid"]:
                print(f"  rank {rank['rank']}: INVALID profile markers {rank['actual_markers']}/{rank['expected_markers']}")
                continue
            if rank["rank"] not in (0, 4, 5):
                continue
            window = rank["local_window"]
            print(f"  rank {rank['rank']}: window={window['window_ms']:.3f} ms; PP union={window['pp_kernel_union_pct']:.2f}%, exclusive={window['pp_kernel_exclusive_pct']:.2f}%; A/E union={window['compute_kernel_union_pct']:.2f}%; no-traced={window['no_traced_gpu_activity_pct']:.2f}%")
    print(output)


if __name__ == "__main__":
    main()
