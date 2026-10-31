"""Actual local token-expert rows, without logical-source duplication."""
import math
import statistics

HIDDEN = 5120
INTERMEDIATE = 1536
RESIDENT_EXPERTS = 40
CAPACITY = 4096
TOTAL_ROWS = (128, 512, 8192)
ACTIVE_EXPERTS = (2, 4, 8, 10, 20, 40)
BYTES_PER_EXPERT = 12_257_280


def point(total, active):
    assert total in TOTAL_ROWS and active in ACTIVE_EXPERTS
    ids = [i * RESIDENT_EXPERTS // active for i in range(active)]
    counts = [0] * RESIDENT_EXPERTS
    for i, expert in enumerate(ids):
        counts[expert] = total // active + (i < total % active)
    assert sum(counts) == total and max(counts) <= CAPACITY
    assert max(counts[i] for i in ids) - min(counts[i] for i in ids) <= 1
    return dict(total_assignments=total, active_experts=active,
                resident_experts=RESIDENT_EXPERTS, active_ids=ids, counts=counts,
                mean_rows_per_active_expert=total / active,
                active_checkpoint_bytes=BYTES_PER_EXPERT * active,
                expected_m=math.ceil(total / RESIDENT_EXPERTS),
                gemm_flops=6 * HIDDEN * INTERMEDIATE * total)


def summarize(values):
    values = sorted(values)
    def percentile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(n=len(values), mean_ms=statistics.mean(values),
                median_ms=statistics.median(values), p95_ms=percentile(.95),
                p99_ms=percentile(.99), min_ms=values[0], max_ms=values[-1])
