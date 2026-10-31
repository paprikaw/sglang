"""CPU-only protocol for equal-resource Hopper EP2 versus PP2."""
from __future__ import annotations


def cases(capacities):
    """Capacity means logical tokens across the whole two-GPU configuration.

    EP2 has two independent Attention/KV owners. PP2's two partial-layer
    pools describe ONE set of requests and must never be added together.
    Leave space for two fresh Prefill branches and the fixed Decode output.
    """
    if set(capacities) != {'ep2', 'pp2'}:
        raise ValueError('Both actual allocator observations are mandatory')
    total = min(capacities.values())
    if total < 32768:
        raise ValueError('Unexpectedly small measured KV capacity')
    rows = [dict(id='moderate', requests=8, cached_context_tokens=4096,
                 prefill_cached_context_tokens=0, new_prefill_tokens=4096)]
    for batch in (16, 32):
        prefix = min(240000, int(min(.90 * total, total - batch * (2 * 512 + 128)) / batch))
        prefix = prefix // 64 * 64
        if prefix < 1024:
            raise ValueError('Insufficient capacity for a meaningful long-context cell')
        rows.append(dict(id=f'near_capacity_b{batch}', requests=batch,
                         cached_context_tokens=prefix,
                         prefill_cached_context_tokens=prefix, new_prefill_tokens=512))
    for row in rows:
        row.update(new_decode_tokens=1, output_tokens=128,
                   global_cached_tokens=row['requests'] * row['cached_context_tokens'],
                   common_capacity_tokens=total)
    return rows


def own_capacity_case(capacity, batch=32):
    prefix = int(min(.90 * capacity, capacity - batch * (2 * 512 + 128)) / batch)
    prefix = min(240000, prefix) // 64 * 64
    return dict(id='pp_own_capacity_b32', requests=batch, cached_context_tokens=prefix,
                prefill_cached_context_tokens=prefix, new_prefill_tokens=512,
                new_decode_tokens=1, output_tokens=128,
                global_cached_tokens=batch * prefix, own_capacity_tokens=capacity,
                matched_workload=False)


def logical_capacity(arm, allocator_tokens):
    if type(allocator_tokens) is not int or allocator_tokens <= 0:
        raise ValueError('Positive actual allocator capacity required')
    if arm not in ('ep2', 'pp2'):
        raise ValueError('Unknown layout')
    return allocator_tokens * (2 if arm == 'ep2' else 1)


def microbatch_size(requests):
    if requests < 2 or requests % 2:
        raise ValueError('This protocol uses two equal nonempty pipeline slots')
    return requests // 2
