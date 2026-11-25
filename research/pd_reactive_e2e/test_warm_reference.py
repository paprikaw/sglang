import pytest

from concurrent_smoke import (
    matching_prefill_caches,
    prefill_cache_signature,
    warm_reference_ready,
)


def response(cached=1984, device=1984, host=0, output=(11, 12)):
    return {
        "prefill_meta": {
            "prompt_tokens": 2048,
            "cached_tokens": cached,
            "cached_tokens_details": {"device": device, "host": host},
        },
        "output_ids": list(output),
    }


def test_repeated_cold_output_does_not_establish_a_warm_reference():
    cold = response(0, 0)
    cold["prefill_meta"]["cached_tokens_details"] = None
    assert not warm_reference_ready(cold, cold)
    assert not warm_reference_ready(cold, response())
    assert warm_reference_ready(response(), response())


def test_warm_reference_requires_matching_cache_source_and_generation():
    assert not warm_reference_ready(response(), response(1984, 0, 1984))
    assert not warm_reference_ready(response(), response(output=(11, 13)))
    assert not warm_reference_ready(response(1024, 1024), response())


def test_static_control_detects_cold_warm_and_device_host_confounders():
    assert matching_prefill_caches(
        [response(0, 0), response()], [response(), response(1984, 0, 1984)]
    ) == [False, False]
    assert matching_prefill_caches(
        [response(), response()], [response(), response()]
    ) == [True, True]


@pytest.mark.parametrize(
    "bad", [response(-1, -1), response(1984, 0), response(2049, 2049)]
)
def test_invalid_or_inconsistent_counters_fail_closed(bad):
    with pytest.raises(ValueError, match="Prefill cache counters"):
        prefill_cache_signature(bad)
