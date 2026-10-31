"""CPU-only Kineto and deferred CUDA-event attribution; imports no GPU modules.

Classification requires a trace correlation/external-id link to an actual CPU
attr scope. Kernel-name guesses are deliberately forbidden. GPU activity and
CUDA-event API intervals are separate clocks/measurements. No cross-rank or
cross-stream sum is reported as an end-to-end fraction.
"""
from __future__ import annotations

import argparse
from decode_components import summarize as summarize_decode_components
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


GPU_CATEGORIES = {
    "kernel": "kernel", "cuda_kernel": "kernel",
    "gpu_memcpy": "memcpy", "gpu_memset": "memset",
}
RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver", "hip_runtime"}


def arg(row, *names):
    values = {str(k).lower().replace("_", " "): v
              for k, v in row.get("args", {}).items()}
    for name in names:
        key = name.lower().replace("_", " ")
        if key in values:
            return str(values[key])
    return None


def interval(row):
    start, duration = float(row["ts"]), float(row.get("dur", 0))
    if duration < 0:
        raise ValueError("Negative trace duration")
    return start, start + duration


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Backwards interval")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def union_length(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_length(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    value = 0.0
    while i < len(left) and j < len(right):
        value += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return value


def activity_metrics(intervals, window=None):
    intervals = list(intervals)
    merged = merge_intervals(intervals)
    if not merged:
        return dict(event_count=0, union_us=0.0, duration_sum_us=0.0,
                    envelope_us=0.0, no_traced_activity_us=0.0, gaps=[],
                    overlap_at_least_two_us=0.0, concurrency_histogram_us={})
    start, end = window or (merged[0][0], merged[-1][1])
    if start > merged[0][0] or end < merged[-1][1]:
        raise ValueError("Activity window clips a device event")
    gaps, cursor = [], start
    for low, high in merged:
        if low > cursor:
            gaps.append(dict(start_us=cursor, end_us=low, duration_us=low - cursor))
        cursor = max(cursor, high)
    if end > cursor:
        gaps.append(dict(start_us=cursor, end_us=end, duration_us=end - cursor))
    changes = Counter()
    for low, high in intervals:
        changes[low] += 1
        changes[high] -= 1
    concurrency, last = 0, None
    histogram = defaultdict(float)
    for point, change in sorted(changes.items()):
        if last is not None and concurrency > 0:
            histogram[concurrency] += point - last
        concurrency += change
        last = point
    busy = union_length(intervals)
    return dict(event_count=len(intervals), start_us=start, end_us=end,
                union_us=busy, duration_sum_us=sum(high - low for low, high in intervals),
                envelope_us=end - start, no_traced_activity_us=max(0.0, end - start - busy),
                gaps=gaps, overlap_at_least_two_us=sum(v for k, v in histogram.items() if k >= 2),
                concurrency_histogram_us={str(k): v for k, v in sorted(histogram.items())})


def parse_scope(row):
    name = str(row.get("name", ""))
    if not name.startswith("attr::"):
        return None
    fields = name.split("::", 2)
    return (fields[1], fields[2] if len(fields) > 2 else "")


def scope_category(scopes):
    """Scopes are ordered innermost first; return the first explicit phase."""
    for category, label in scopes:
        if category.startswith("deep_ep.dispatch"):
            return "deep_ep.dispatch"
        if category.startswith("deep_ep.combine"):
            return "deep_ep.combine"
        if category == "tp_collective":
            return "tp_collective"
        if category.startswith("pp_"):
            return "pp_transport_api"
        if category == "routed_expert_compute" or category == "routed_activation":
            return "expert.routed"
        if category == "dense_or_shared_mlp":
            return "expert.shared" if ".shared_experts" in label else "dense_mlp"
        if category == "shared_expert":
            return "expert.shared"
        if category in ("router_gate", "router_topk"):
            return "router"
        if category in ("attention_full", "attention_prepare", "attention_core"):
            return "attention"
        # A prefix is an explicit CPU scope label, never a GPU-name heuristic.
        if category in ("linear_projection", "humming_gemm"):
            if ".self_attn." in label:
                return "attention"
            if ".shared_experts" in label:
                return "expert.shared"
            if ".mlp.experts" in label:
                return "expert.routed"
            if ".mlp." in label:
                return "dense_mlp"
    return "unknown"


class ScopeIndex:
    def __init__(self, rows):
        self.threads = {}
        groups = defaultdict(list)
        for i, row in rows:
            if parse_scope(row) is not None and row.get("ph") == "X":
                groups[(str(row.get("pid")), str(row.get("tid")))].append((i, row))
        for thread, items in groups.items():
            items.sort(key=lambda value: (interval(value[1])[0], -interval(value[1])[1]))
            stack, nodes = [], []
            for i, row in items:
                low, high = interval(row)
                while stack and not (nodes[stack[-1]]["low"] <= low
                                     and high <= nodes[stack[-1]]["high"]):
                    stack.pop()
                nodes.append(dict(index=i, row=row, low=low, high=high,
                                  parent=stack[-1] if stack else None))
                stack.append(len(nodes) - 1)
            self.threads[thread] = ([node["low"] for node in nodes], nodes)

    def containing(self, row):
        item = self.threads.get((str(row.get("pid")), str(row.get("tid"))))
        if item is None:
            return []
        starts, nodes = item
        low, high = interval(row)
        position = bisect_right(starts, low) - 1
        result = []
        while position >= 0:
            node = nodes[position]
            if node["low"] <= low and high <= node["high"] + 0.001:
                result.append(node)
            position = node["parent"] if node["parent"] is not None else -1
        return result


def real_local_tokens(meta):
    if meta.get("forward_mode") == "2":
        return int(meta["batch_size"])
    values = meta.get("extend_seq_lens_cpu")
    if isinstance(values, list) and all(isinstance(value, (int, float)) for value in values):
        return int(sum(values))
    # Extend metadata is authoritative; input_ids_shape can contain padding.
    return None


def _event_ancestors(row, rows):
    chain, seen = [], set()
    current = row
    while current is not None:
        identifier = current.get("index")
        if identifier in seen:
            raise ValueError("CUDA-event parent cycle")
        seen.add(identifier)
        chain.append((current.get("category", ""), str(current.get("label", ""))))
        current = rows.get(current.get("parent"))
    return chain


def _exclusive_partition(spans, window):
    """Partition nested same-stream API spans; deepest explicit phase wins."""
    points = sorted({window[0], window[1]} | {value for low, high, _, _ in spans
                                            for value in (max(low, window[0]), min(high, window[1]))
                                            if window[0] <= value <= window[1]})
    totals = defaultdict(float)
    for low, high in zip(points, points[1:]):
        candidates = [(depth, category) for start, end, category, depth in spans
                      if start <= low and high <= end]
        selected = max(candidates)[1] if candidates else "unassigned"
        totals[selected] += high - low
    return dict(totals)


def summarize_cuda_events(sample):
    lookup = {row["index"]: row for row in sample.get("events", []) + sample.get("cpu", [])
              if "index" in row}
    forward_meta = {row["forward_index"]: row.get("metadata", {})
                    for row in sample.get("forwards", [])}
    grouped = defaultdict(list)
    excluded = []
    for row in sample.get("events", []):
        if row.get("start_stream") != row.get("end_stream") or row.get("duration_ms") is None:
            excluded.append(row.get("index"))
            continue
        layer = str(row["label"]) if row.get("category") == "layer" else row.get("layer")
        # Component detail has a layer; global/PP envelopes are reported separately.
        grouped[(row.get("forward_index"), layer, str(row.get("start_stream")))].append(row)
    result = []
    for (forward, layer, stream), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        if layer is None:
            continue
        spans, phases, envelope = [], defaultdict(list), []
        for row in rows:
            start, end = float(row["start_ms"]), float(row["end_ms"])
            if end < start:
                raise ValueError("Backwards CUDA-event API span")
            ancestors = _event_ancestors(row, lookup)
            category = scope_category(ancestors)
            if category != "unknown":
                phases[category].append((start, end))
                spans.append((start, end, category, len(ancestors)))
            if row.get("category") == "layer":
                envelope.append((start, end))
        if not envelope:
            continue
        low, high = min(a for a, _ in envelope), max(b for _, b in envelope)
        exclusive = _exclusive_partition(spans, (low, high))
        tokens = real_local_tokens(forward_meta.get(forward, {}))
        result.append(dict(forward_index=forward, layer=layer, stream=stream,
                           local_real_new_tokens=tokens, layer_api_span_ms=high - low,
                           inclusive_phase_union_ms={key: union_length(value) for key, value in phases.items()},
                           exclusive_api_partition_ms=exclusive,
                           per_local_real_token_ms={key: value / tokens if tokens else None
                                                    for key, value in exclusive.items()},
                           metadata=forward_meta.get(forward, {})))
    return dict(per_forward_layer_stream=result, excluded_cross_stream_event_indices=excluded,
                semantics="CUDA-event API intervals contain waits and CPU feed gaps. Nested scopes are unioned/partitioned within one stream only. These are not GPU kernel execution or wire times; different stream origins cannot be aligned. Per-token denominators are local real new Attention tokens, not aggregate cluster tokens or received expert assignments.")


def calibrated_name_view(gpu, sample):
    """Strict, separately reported exact-name extrapolation with held-out scopes.

    CPU correlations remain the primary evidence. No substring classification,
    cross-device learning, or extrapolation to activities outside model_forward.
    """
    detailed = sorted([row for row in sample.get("forwards", []) if row.get("detailed")],
                      key=lambda row: row.get("extend_index", -1))
    if len(detailed) != 3:
        return dict(available=False, reason="Exactly three detailed forward receipts required",
                    training_forward_indices=[], holdout_forward_index=None, names=[], devices={})
    training = {row["forward_index"] for row in detailed[:2]}
    holdout = detailed[2]["forward_index"]
    grouped = defaultdict(list)
    for row in gpu:
        if row["kind"] == "kernel" and row["category"] != "unknown":
            grouped[(row["device"], row["name"])].append(row)
    learned, audit = {}, []
    for (device, name), rows in sorted(grouped.items()):
        categories = {row["category"] for row in rows}
        train = [row for row in rows if row["forward_index"] in training]
        held = [row for row in rows if row["forward_index"] == holdout]
        train_phases = {row["category"] for row in train}
        held_phases = {row["category"] for row in held}
        witnessed = {row["forward_index"] for row in train}
        passed = (len(categories) == 1 and witnessed == training
                  and len(train_phases) == 1 and held_phases == train_phases)
        reason = ("phase_conflict" if len(categories) > 1 else
                  "missing_both_training_forwards" if witnessed != training else
                  "holdout_missing_or_disagreement" if held_phases != train_phases else
                  "holdout_passed")
        item = dict(device=device, exact_kernel_name=name, directly_observed_phases=sorted(categories),
                    training_phases=sorted(train_phases), training_event_count=len(train),
                    holdout_phases=sorted(held_phases), holdout_event_count=len(held),
                    eligible=passed, reason=reason)
        audit.append(item)
        if passed:
            learned[(device, name)] = next(iter(train_phases))
    expanded = []
    counts = Counter()
    for row in gpu:
        category = row["category"]
        method = "direct_cpu_scope" if category != "unknown" else "unknown"
        if (category == "unknown" and row["kind"] == "kernel"
                and row["forward_index"] is not None and (row["device"], row["name"]) in learned):
            category = learned[(row["device"], row["name"])]
            method = "calibrated_exact_name"
        expanded.append(dict(row, category=category, classification_method=method))
        counts[method] += 1
    devices, per_forward = {}, defaultdict(list)
    for device in sorted({row["device"] for row in expanded}):
        selected = [row for row in expanded if row["device"] == device]
        by_category, by_method = defaultdict(list), defaultdict(list)
        model_categories=defaultdict(list)
        for row in selected:
            by_category[row["category"]].append(row["interval"])
            by_method[row["classification_method"]].append(row["interval"])
            if row["forward_index"] is not None:
                per_forward[(device, row["forward_index"], row["category"])].append(row["interval"])
                model_categories[row["category"]].append(row["interval"])
        devices[device] = dict(
            categories={key: dict(event_count=len(value), union_us=union_length(value),
                                  duration_sum_us=sum(high - low for low, high in value))
                        for key, value in sorted(by_category.items())},
            evidence={key: dict(event_count=len(value), union_us=union_length(value),
                                duration_sum_us=sum(high - low for low, high in value))
                      for key, value in sorted(by_method.items())},
            model_forward_categories={key:dict(event_count=len(value),union_us=union_length(value),
                duration_sum_us=sum(high-low for low,high in value)) for key,value in sorted(model_categories.items())},
            category_pair_overlap_us={f"{left}|{right}": intersection_length(by_category[left], by_category[right])
                for pos, left in enumerate(sorted(by_category))
                for right in sorted(by_category)[pos + 1:]
                if intersection_length(by_category[left], by_category[right]) > 0})
    return dict(available=True, training_forward_indices=sorted(training), holdout_forward_index=holdout,
                eligible_name_count=len(learned), classification_counts=dict(counts), names=audit,
                devices=devices,
                per_forward_category=[dict(device=key[0], forward_index=key[1], category=key[2],
                    event_count=len(value), union_us=union_length(value),
                    duration_sum_us=sum(high - low for low, high in value))
                    for key, value in sorted(per_forward.items(), key=lambda item: str(item[0]))],
                semantics="Separate exact-name extrapolation, not direct CPU-scope evidence: one phase in all directly annotated occurrences, present in both training detailed forwards, validated against the third held-out forward, same rank/device. Name collisions remain unknown; memcpy/memset and model-external events are never extrapolated. Still not wire, occupancy, or additive E2E fractions.")


def summarize_trace(trace, sample=None):
    sample = sample or {}
    events = [(i, row) for i, row in enumerate(trace.get("traceEvents", []))
              if row.get("ph") == "X" and "ts" in row and "dur" in row]
    scopes = ScopeIndex(events)
    runtime_by_correlation, external_events = defaultdict(list), defaultdict(list)
    for i, row in events:
        if row.get("cat") in RUNTIME_CATEGORIES:
            key = arg(row, "correlation", "correlation id")
            if key is not None:
                runtime_by_correlation[key].append((i, row))
        if row.get("cat") not in GPU_CATEGORIES:
            key = arg(row, "External id", "External ID")
            if key is not None:
                external_events[key].append((i, row))
    model_scopes = sorted([(i, row) for i, row in events
                           if (parse_scope(row) or (None,))[0] == "model_forward"],
                          key=lambda item: interval(item[1])[0])
    model_receipts = sorted([row for row in sample.get("events", [])
                             if row.get("category") == "model_forward"],
                            key=lambda row: row.get("cpu_start_ns", 0))
    forward_mapping_valid = bool(model_scopes) and len(model_scopes) == len(model_receipts)
    forward_map = {scope[0]: receipt.get("forward_index") for scope, receipt in zip(model_scopes, model_receipts)} if forward_mapping_valid else {}
    gpu = []
    linkage = Counter()
    unknown = Counter()
    example_unknown = []
    known_device_by_pid = {}
    for _, row in events:
        if row.get("cat") in GPU_CATEGORIES:
            device = arg(row, "device", "device id")
            if device is not None:
                known_device_by_pid[str(row.get("pid"))] = device
    for i, row in events:
        kind = GPU_CATEGORIES.get(row.get("cat"))
        if kind is None:
            continue
        device = arg(row, "device", "device id") or known_device_by_pid.get(str(row.get("pid")), "unknown")
        correlation = arg(row, "correlation", "correlation id")
        runtime = runtime_by_correlation.get(correlation, []) if correlation is not None else []
        containers, method = [], "unlinked"
        if len(runtime) == 1:
            containers = scopes.containing(runtime[0][1])
            method = "correlation_runtime_scope" if containers else "correlation_runtime_without_attr"
        elif len(runtime) > 1:
            method = "ambiguous_runtime_correlation"
        else:
            external = arg(row, "External id", "External ID")
            candidates = external_events.get(external, []) if external is not None else []
            if len(candidates) == 1:
                containers = scopes.containing(candidates[0][1])
                method = "external_id_cpu_scope" if containers else "external_id_without_attr"
        category = scope_category([parse_scope(node["row"]) for node in containers])
        lineage = [parse_scope(node["row"]) for node in containers]
        layer = next((label for cat, label in lineage if cat == "layer"), None)
        model_node = next((node["index"] for node in containers
                           if parse_scope(node["row"])[0] == "model_forward"), None)
        gpu.append(dict(interval=interval(row), category=category, kind=kind, device=device,
                        forward_index=forward_map.get(model_node), layer=layer,
                        name=str(row.get("name", "")), trace_index=i, linkage=method,
                        runtime_trace_index=runtime[0][0] if len(runtime)==1 else None,
                        runtime_start_us=interval(runtime[0][1])[0] if len(runtime)==1 else None,
                        cpu_scope_lineage=lineage))
        linkage[method] += 1
        if category == "unknown":
            unknown[str(row.get("name", ""))] += 1
            if len(example_unknown) < 12:
                example_unknown.append(dict(trace_index=i, name=row.get("name"), kind=kind,
                                            linkage=method, correlation=correlation, scopes=lineage))
    devices = {}
    expected = str(sample["cuda_device"]) if "cuda_device" in sample else None
    for device in sorted({item["device"] for item in gpu}):
        selected = [row for row in gpu if row["device"] == device]
        categories, kinds = defaultdict(list), defaultdict(list)
        for row in selected:
            categories[row["category"]].append(row["interval"])
            kinds[row["kind"]].append(row["interval"])
        intervals = [row["interval"] for row in selected]
        model_window = None
        if model_scopes and intervals:
            model_window = (min(interval(model_scopes[0][1])[0], min(low for low, _ in intervals)),
                            max(max(interval(row)[1] for _, row in model_scopes), max(high for _, high in intervals)))
        model_selected=[row for row in selected if row["forward_index"] is not None]
        model_categories=defaultdict(list)
        for row in model_selected:model_categories[row["category"]].append(row["interval"])
        excluded=[row for row in selected if row["forward_index"] is None]
        excluded_top=sorted(excluded,key=lambda row:row["interval"][1]-row["interval"][0],reverse=True)[:12]
        devices[device] = dict(matches_expected_device=expected is None or device == expected,
            activity_envelope=activity_metrics(intervals),
            observed_model_envelope=activity_metrics(intervals, model_window) if model_window else None,
            activity_kinds={key: activity_metrics(value) for key, value in sorted(kinds.items())},
            categories={key: dict(event_count=len(value), union_us=union_length(value),
                                  duration_sum_us=sum(high - low for low, high in value))
                        for key, value in sorted(categories.items())},
            model_forward_gpu_work=dict(
                activity=activity_metrics([row["interval"] for row in model_selected]),
                categories={key:dict(event_count=len(value),union_us=union_length(value),
                    duration_sum_us=sum(high-low for low,high in value)) for key,value in sorted(model_categories.items())},
                included_event_count=len(model_selected),excluded_event_count=len(excluded),
                excluded_top_duration_events=excluded_top,
                semantics="Only GPU activities whose actual CPU launch falls inside a model_forward attr scope matched to a formal receipt ordinal. This excludes PP boundary work and model-external profiler-finish/wakeup activity; PP boundary is retained separately in the raw view. Window is matched model GPU launch completion envelope, not HTTP E2E."),
            category_pair_overlap_us={f"{left}|{right}": intersection_length(categories[left], categories[right])
                                      for pos, left in enumerate(sorted(categories))
                                      for right in sorted(categories)[pos + 1:]
                                      if intersection_length(categories[left], categories[right]) > 0})
    per_layer = defaultdict(list)
    for row in gpu:
        if row["forward_index"] is not None and row["layer"] is not None:
            per_layer[(row["device"], row["forward_index"], row["layer"], row["category"], row["kind"])].append(row["interval"])
    return dict(schema_version=1, arm=sample.get("arm"), rank=sample.get("rank"),
                sample_passed=sample.get("passed"), hostname=sample.get("hostname"),
                gpu_devices=devices, linkage=dict(linkage),
                unknown_event_count=sum(unknown.values()), unknown_name_counts=dict(unknown),
                unknown_examples=example_unknown,
                forward_mapping=dict(valid=forward_mapping_valid, trace_model_scope_count=len(model_scopes),
                                     receipt_model_scope_count=len(model_receipts),
                                     method="Same-rank ordered model_forward scopes matched to receipt ordinals; no host clock offset conversion."),
                gpu_per_forward_layer_category=[dict(device=key[0], forward_index=key[1], layer=key[2],
                    category=key[3], activity_kind=key[4], event_count=len(value),
                    union_us=union_length(value), duration_sum_us=sum(high - low for low, high in value))
                    for key, value in sorted(per_layer.items(), key=lambda item: str(item[0]))],
                decode_components=summarize_decode_components(gpu, sample),
                calibrated_exact_name=calibrated_name_view(gpu, sample),
                cuda_event_api_detail=summarize_cuda_events(sample),
                semantics="Kineto activity includes communication kernels that may spin/wait: union is traced device activity, not useful compute, SM occupancy, or wire time. Idle gaps mean no traced kernel/memcpy/memset in this rank's window. Per-category unions can overlap across streams and are not additive end-to-end fractions. Unknowns remain unknown. Independent node clocks are never aligned or rank durations summed.")


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def file_receipt(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed while hashing: {path}")
    return dict(path=str(path.resolve()), bytes=after.st_size,
                mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def summarize_file(trace_path, sample_path):
    trace_before, sample_before = Path(trace_path).stat(), Path(sample_path).stat()
    sample = read_json(sample_path)
    if sample.get("passed") is not True:
        raise ValueError(f"Attribution receipt did not pass: {sample_path}")
    result = summarize_trace(read_json(trace_path), sample)
    result["sources"] = dict(trace=file_receipt(trace_path), attribution=file_receipt(sample_path))
    for key, before in (("trace", trace_before), ("attribution", sample_before)):
        receipt = result["sources"][key]
        if (before.st_size, before.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
            raise ValueError(f"Artifact changed during analysis: {receipt['path']}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--arm", choices=("pp2ep4", "ep8"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.run:
        if args.trace or args.attribution:
            parser.error("Use --run or --trace/--attribution, not both")
        samples = sorted(path for arm in ((args.arm,) if args.arm else ("pp2ep4", "ep8"))
                         for path in (args.run / arm).glob("attribution_rank*.json"))
        if not samples:
            raise ValueError("No completed attribution receipts found")
        results = []
        for sample_path in samples:
            sample = read_json(sample_path)
            trace = sample_path.parent / sample["trace_filename"]
            results.append(summarize_file(trace, sample_path))
    else:
        if args.arm:
            parser.error("--arm requires --run")
        if not args.trace or not args.attribution:
            parser.error("--trace and --attribution are required without --run")
        results = [summarize_file(args.trace, args.attribution)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(schema_version=1, ranks=results,
        aggregate_end_to_end_fraction=None,
        aggregation_warning="Per-rank/per-stream observations only; do not add durations across ranks, streams, or independent host clocks."),
        ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Saved {len(results)} rank summaries: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
