from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

path = Path(__file__).parents[4] / "python/sglang/srt/elastic_pp/pd_reactive.py"
spec = importlib.util.spec_from_file_location("pd_reactive_unit", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def tick(policy, sample, now=None, occupied=4, free=2):
    return policy.observe([sample], now=sample.observed_at if now is None else now,
                          occupied_gpus=occupied, free_by_role={sample.role: free})


def high(t, **kw):
    return m.Observation("decode", t, capacity_blocked=True, capacity_wait_s=3, **kw)


@pytest.mark.parametrize("mechanism,delta", [("inplace", 1), ("replica", 2)])
def test_common_trigger_different_granularity(mechanism, delta):
    p = m.ReactivePDPolicy(mechanism)
    assert tick(p, high(0)) is None
    assert tick(p, high(1)) is None
    d = tick(p, high(2))
    assert (d.role, d.direction, d.gpu_delta) == ("decode", "up", delta)


def test_cache_occupancy_and_transfer_wait_are_not_capacity_failure():
    p = m.ReactivePDPolicy("inplace")
    for t in range(10):
        assert tick(p, m.Observation("decode", t, capacity_wait_s=20)) is None


def test_d_backpressure_does_not_expand_p():
    p = m.ReactivePDPolicy("inplace")
    for t in range(4):
        assert tick(p, m.Observation("prefill", t, compute_wait_s=20,
                                    compute_queue=3, blocked_by_decode=True)) is None


def test_decode_blocking_prefill_has_priority():
    p = m.ReactivePDPolicy("inplace")
    for t in range(3):
        d = p.observe([m.Observation("prefill", t, compute_queue=2,
                                    compute_wait_s=10, blocked_by_decode=True), high(t)],
                      now=t, occupied_gpus=4, free_by_role={"prefill": 2, "decode": 2})
    assert d.role == "decode"


@pytest.mark.parametrize("missing", ["target_fits_state", "target_fits_load"])
def test_shrink_requires_positive_target_feasibility(missing):
    p = m.ReactivePDPolicy("inplace")
    s = m.Observation("decode", 0, expanded=True,
                      target_fits_state=True, target_fits_load=True)
    for t in range(40):
        assert tick(p, replace(s, observed_at=t, **{missing: None})) is None


def test_handoff_blocks_shrink():
    p = m.ReactivePDPolicy("inplace")
    for t in range(40):
        s = m.Observation("prefill", t, expanded=True, target_fits_state=True,
                          target_fits_load=True, in_flight_handoff=True)
        assert tick(p, s) is None


def test_long_low_period_required_and_no_parallel_operation():
    p = m.ReactivePDPolicy("inplace")
    for t in range(30):
        s = m.Observation("decode", t, expanded=True,
                          target_fits_state=True, target_fits_load=True)
        assert tick(p, s) is None
    d = tick(p, replace(s, observed_at=30))
    assert d.direction == "down"
    assert tick(p, high(31)) is None


def test_cooldown_starts_after_completion():
    p = m.ReactivePDPolicy("inplace")
    tick(p, high(0)); assert tick(p, high(2))
    p.complete(now=100, success=True)
    for t in range(101, 115):
        assert tick(p, high(t)) is None
    assert tick(p, high(115))


def test_real_budget_and_physical_placement_are_both_enforced():
    for occupied, free in [(8, 2), (4, 0)]:
        p = m.ReactivePDPolicy("inplace")
        for t in range(5):
            assert tick(p, high(t), occupied=occupied, free=free) is None


def test_gap_duplicate_or_stale_data_do_not_establish_persistence():
    p = m.ReactivePDPolicy("inplace")
    assert tick(p, high(0)) is None
    assert tick(p, high(0), now=2) is None
    assert tick(p, high(0), now=100) is None
    assert tick(p, high(101)) is None
    assert tick(p, high(102)) is None
    assert tick(p, high(103))


def test_failure_retries_after_backoff_with_new_observations():
    p = m.ReactivePDPolicy("inplace")
    tick(p, high(0)); assert tick(p, high(2))
    p.complete(now=2, success=False)
    for t in range(3, 12):
        assert tick(p, high(t)) is None
    assert tick(p, high(12))


def test_slower_deeper_pp_is_not_assumed_to_cure_decode_load():
    p = m.ReactivePDPolicy("inplace")
    for t in range(5):
        assert tick(p, m.Observation("decode", t, overloaded=True)) is None
