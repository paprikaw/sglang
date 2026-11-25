from stream_timing import StreamGaps


def content(field="content"):
    return {"choices": [{"delta": {field: "payload"}}]}


def test_usage_and_empty_deltas_do_not_create_content_gaps_or_include_ttft():
    tracker = StreamGaps(10, 100_000_000_000)
    tracker.observe({"usage": {"completion_tokens": 1}}, 11)
    tracker.observe(content(), 12)
    tracker.observe({"choices": [{"delta": {"content": ""}}]}, 12.5)
    tracker.observe(content("reasoning_content"), 13)
    tracker.observe(content("tool_calls"), 13.25)
    summary = tracker.summary()
    assert summary["content_events"] == 3
    assert summary["largest_content_gaps"] == [
        dict(gap_ms=1000, started_unix_ns=102_000_000_000,
             finished_unix_ns=103_000_000_000),
        dict(gap_ms=250, started_unix_ns=103_000_000_000,
             finished_unix_ns=103_250_000_000),
    ]


def test_timing_storage_is_bounded_and_preserves_largest_observed_interval():
    tracker = StreamGaps(0, 0)
    for when in range(100):
        tracker.observe(content(), when)
    tracker.observe(content(), 199)
    summary = tracker.summary()
    assert summary["content_events"] == 101
    assert len(summary["largest_content_gaps"]) == 8
    assert summary["largest_content_gaps"][0] == dict(
        gap_ms=100_000, started_unix_ns=99_000_000_000,
        finished_unix_ns=199_000_000_000)
