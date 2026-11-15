"""Bounded upstream content-event timing; chunks are not necessarily tokens."""

import heapq


class StreamGaps:
    def __init__(self, started, started_unix_ns, limit=8):
        self.started = started
        self.started_unix_ns = started_unix_ns
        self.limit = limit
        self.last = None
        self.events = 0
        self.largest = []

    def observe(self, event, now):
        if not any(
            choice.get("delta", {}).get(field)
            for choice in event.get("choices", [])
            for field in ("content", "reasoning_content", "tool_calls")
        ):
            return
        if self.last is not None:
            span = (1000 * (now - self.last), self.last, now)
            if len(self.largest) < self.limit:
                heapq.heappush(self.largest, span)
            else:
                heapq.heappushpop(self.largest, span)
        self.last = now
        self.events += 1

    def summary(self):
        def unix_ns(when):
            return self.started_unix_ns + round((when - self.started) * 1e9)

        return dict(
            content_events=self.events,
            largest_content_gaps=[
                dict(gap_ms=gap, started_unix_ns=unix_ns(start),
                     finished_unix_ns=unix_ns(end))
                for gap, start, end in sorted(self.largest, reverse=True)
            ],
            boundary="upstream SSE content events at the router; not individual token timestamps",
        )


def instrument_adapter(source):
    replacements = [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\nfrom stream_timing import StreamGaps\n"),
        ("        cache_details: dict[str, Any] | None = None\n",
         "        cache_details: dict[str, Any] | None = None\n"
         "        stream_gaps = StreamGaps(started, started_unix_ns)\n"),
        ("                        event = json.loads(line[5:].strip())\n",
         "                        event = json.loads(line[5:].strip())\n"
         "                        stream_gaps.observe(event, time.monotonic())\n"),
        ('                    "transport_failure": transport_failure,\n',
         '                    "transport_failure": transport_failure,\n'
         '                    "stream_timing": stream_gaps.summary(),\n'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("unknown AgentX adapter content timing boundary")
        source = source.replace(before, after)
    compile(source, "sglang_adapter.py", "exec")
    return source
