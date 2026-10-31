"""Matched local-work geometry; no GPU or model imports."""

from collections import Counter
import math
import statistics

EXPERTS = 160
TOP_K = 8
HIDDEN = 5120
INTERMEDIATE = 1536
GLOBAL_LOGICAL_TOKENS = (8, 32, 128, 512, 2048, 4096)


def routes(token):
    # One expert from each EP8 owner; every top-k entry is unique.
    return tuple(owner * 20 + token % 20 for owner in range(8))


def layout(global_tokens, name, slot=0, duplication=1):
    """EP8 full batch vs one of two PP microbatches, not two replicas.

    D=4 reproduces the TP4 physical source-copy convention of the saved
    GLM expert benchmark. Unique user-token counts remain explicit.
    """
    if global_tokens <= 0 or global_tokens % 2:
        raise ValueError('global_tokens must be positive and even')
    if duplication not in (1, 4):
        raise ValueError('duplication must be 1 or 4')
    if name == 'ep8':
        if slot != 0:
            raise ValueError('EP8 has no PP slot index')
        ep, local_experts, start, stop = 8, 20, 0, global_tokens
    elif name == 'pp2_ep4':
        if slot not in (0, 1):
            raise ValueError('PP2 slot must be 0 or 1')
        ep, local_experts = 4, 40
        start, stop = slot * (global_tokens // 2), (slot + 1) * (global_tokens // 2)
    else:
        raise ValueError('unknown layout')
    counts = Counter()
    for token in range(start, stop):
        for expert in routes(token):
            if expert < local_experts:
                counts[expert] += duplication
    count_list = [counts[e] for e in range(local_experts)]
    physical_source_rows = (stop - start) * duplication
    assignments = sum(count_list)
    assert assignments == global_tokens * duplication
    assert max(count_list) - min(count_list) <= duplication
    return {
        'layout': name, 'pp_slot': slot, 'global_logical_tokens': global_tokens,
        'microbatch_logical_tokens': stop - start,
        'attention_tp_physical_copies': duplication,
        'physical_source_rows_in_ep_group': physical_source_rows,
        'source_rows_per_ep_rank': physical_source_rows / ep,
        'ep_size': ep, 'local_experts': local_experts,
        'active_experts': sum(c > 0 for c in count_list),
        'counts': count_list, 'local_assignments': assignments,
        'min_rows_per_resident_expert': min(count_list),
        'max_rows_per_expert': max(count_list),
        'mean_rows_per_active_expert': assignments / sum(c > 0 for c in count_list),
        'expected_m': (physical_source_rows * TOP_K + EXPERTS) // EXPERTS,
        'gemm_flops': 6 * HIDDEN * INTERMEDIATE * assignments,
    }


def grid():
    return [layout(n, name, slot) for n in GLOBAL_LOGICAL_TOKENS
            for name, slots in [('ep8', (0,)), ('pp2_ep4', (0, 1))]
            for slot in slots]


def summarize(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('timings must be finite and positive')
    x = sorted(values)
    def q(p):
        i = (len(x) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return x[lo] + (x[hi] - x[lo]) * (i - lo)
    mean = statistics.fmean(x)
    return {'samples': len(x), 'mean_ms': mean, 'median_ms': q(.5),
            'p95_ms': q(.95), 'min_ms': x[0], 'max_ms': x[-1],
            'cv': statistics.stdev(x) / mean if len(x) > 1 else 0.0}
