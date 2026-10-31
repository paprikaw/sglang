"""Startup-validated layer identity shared with CPU HiCache construction.

State is process-local, immutable after registration and read without artifact
polling. No GPU package is imported. This candidate is restricted to GLM's
92 layers, unchanged EP8, or static PP2 partition 47/45.
"""
from __future__ import annotations

_VALIDATED_INTERVAL = None


def validated_model_interval(model, layout, pp_rank, moe_layer_count):
    if layout == 'ep8':
        if pp_rank != 0:
            raise RuntimeError('EP8 must retain PP1')
        expected = (0, 92)
    elif layout == 'pp2ep4' and pp_rank in (0, 1):
        expected = ((0, 47), (47, 92))[pp_rank]
    else:
        raise RuntimeError('Unsupported layer-balance topology')
    start = getattr(model, 'start_layer', None)
    end = getattr(model, 'end_layer', None)
    if type(start) is not int or type(end) is not int or (start, end) != expected:
        raise RuntimeError(f'Actual model layer interval {(start, end)} differs from {expected}')
    dense_layers = max(0, min(end, 3) - start)
    expected_moe = end - start - dense_layers
    if type(moe_layer_count) is not int or moe_layer_count != expected_moe:
        raise RuntimeError(f'Expected {expected_moe} real MoE layers in interval {expected}, got {moe_layer_count}')
    return dict(passed=True, layout=layout, pp_rank=pp_rank, start_layer=start,
                end_layer=end, layer_count=end-start, dense_layer_count=dense_layers,
                moe_layer_count=moe_layer_count)


def register_model_interval(model, layout, pp_rank, moe_layer_count):
    global _VALIDATED_INTERVAL
    record = validated_model_interval(model, layout, pp_rank, moe_layer_count)
    identity = (layout, pp_rank, record['start_layer'], record['end_layer'])
    if _VALIDATED_INTERVAL is not None and _VALIDATED_INTERVAL != identity:
        raise RuntimeError('Cannot replace a validated layer interval in a live worker')
    _VALIDATED_INTERVAL = identity
    return record


def host_layer_interval(host, layout, pp_rank):
    if _VALIDATED_INTERVAL is None:
        raise RuntimeError('Model layer interval was not validated before HiCache use')
    actual_layout, actual_pp_rank, start, end = _VALIDATED_INTERVAL
    if (layout, pp_rank) != (actual_layout, actual_pp_rank):
        raise RuntimeError('HiCache topology differs from startup-validated model')
    count = getattr(host, 'layer_num', None)
    if type(count) is not int or count != end-start:
        raise RuntimeError(f'Native host KV layer count {count} differs from model interval {(start, end)}')
    return start, end
