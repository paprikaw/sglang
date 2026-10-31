"""CPU-only partitions of directly correlated, detailed Decode GPU activity.

No kernel-name classification, extrapolation or cross-rank clock alignment.
This is an execution-window composition, not an HTTP TPOT decomposition.
"""
from collections import Counter, defaultdict


def component(category):
    if category == "attention":
        return "attention"
    if category in ("expert.routed", "expert.shared", "dense_mlp"):
        return "expert_and_dense_mlp"
    if category.startswith("deep_ep.") or category in ("tp_collective", "pp_transport_api"):
        return "communication"
    return "other_gpu"


def partition(rows, window):
    """Disjoint timeline buckets, preserving cross-component overlap and gaps."""
    changes = defaultdict(Counter)
    for row in rows:
        low, high = row["interval"]
        low, high = max(low, window[0]), min(high, window[1])
        if high <= low:
            continue
        label = component(row["category"])
        changes[low][label] += 1
        changes[high][label] -= 1
    changes[window[0]]
    changes[window[1]]
    active, totals, previous = Counter(), defaultdict(float), window[0]
    for point in sorted(changes):
        labels = [label for label, count in active.items() if count > 0]
        bucket = labels[0] if len(labels) == 1 else "component_overlap" if labels else "no_traced_gpu_activity"
        totals[bucket] += point - previous
        active.update(changes[point])
        previous = point
    if abs(sum(totals.values()) - (window[1] - window[0])) > 1e-4:
        raise ValueError("GPU execution window partition does not close")
    return {key: value / 1000 for key, value in sorted(totals.items())}


def summarize(gpu, sample):
    detailed = {row["forward_index"]: row for row in sample.get("forwards", []) if row.get("detailed")}
    groups = defaultdict(list)
    for row in gpu:
        if row["forward_index"] in detailed:
            groups[(row["device"], row["forward_index"])].append(row)
    forwards = []
    for (device, index), rows in sorted(groups.items()):
        window = min(row["interval"][0] for row in rows), max(row["interval"][1] for row in rows)
        by_category = defaultdict(float)
        for row in rows:
            by_category[row["category"]] += (row["interval"][1] - row["interval"][0]) / 1000
        foreign = [row for row in gpu if row["device"] == device and row["forward_index"] != index
                   and row["interval"][0] < window[1] and window[0] < row["interval"][1]]
        forwards.append(dict(device=device, forward_index=index,
            decode_ordinal=detailed[index]["extend_index"], metadata=detailed[index].get("metadata"),
            window_ms=(window[1] - window[0]) / 1000, gpu_activity_count=len(rows),
            partition_ms=partition(rows, window),
            category_duration_sum_ms=dict(sorted(by_category.items())),
            foreign_overlapping_gpu_events=len(foreign),
            foreign_overlapping_categories=dict(Counter(row["category"] for row in foreign))))
    boundary = [row for row in gpu if row["category"] == "pp_transport_api"]
    return dict(forwards=forwards, detailed_forward_count=len(detailed),
        directly_mapped_forward_count=len(forwards),
        pp_boundary_kernel_event_count=len(boundary),
        pp_boundary_kernel_duration_sum_ms=sum(row["interval"][1] - row["interval"][0] for row in boundary) / 1000,
        semantics="Detailed model-forward GPU windows only, direct CPU launch-scope correlation. Per-window timeline is disjoint including overlap and no-traced-activity gaps. Dense/shared/routed MLP are grouped as Expert/FFN. Communication includes DeepEP and TP collectives; spinning/waiting kernels are not pure wire time. PP boundary kernels are additionally retained separately for the entire profile, not normalized by or added to HTTP TPOT. Foreign overlapping activities must be inspected; no cross-rank durations are added.")
