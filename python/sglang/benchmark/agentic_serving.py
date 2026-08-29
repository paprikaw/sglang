# SPDX-License-Identifier: Apache-2.0
"""Closed-loop TraceLab replay against SGLang's native streaming API.

Example: python -m sglang.benchmark.agentic_serving --workload selected.json
  --worker-url http://node-a:30000 --worker-url http://node-b:30000
  --output-dir results --flush-cache --prewarm-initial-prefix

Worker assignments belong to sessions, including their initial-prefix prewarm.
The single-worker case maps the same frozen workload to one PP8 server.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from sglang.benchmark.datasets.tracelab import (
    SyntheticPromptBuilder,
    load_tracelab_sessions,
    sha256_file,
)


def token_hash(ids):
    values = array("I", ids)
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def stats(values):
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def parse_prometheus(raw):
    pattern = re.compile(r"^(sglang:[\w:]+)(?:\{(.*)\})?\s+(\S+)")
    label_pattern = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
    records = []
    for line in raw.splitlines():
        match = pattern.match(line)
        if match:
            name, labels, value = match.groups()
            records.append(
                (name, dict(label_pattern.findall(labels or "")), float(value))
            )
    return records


def counter_totals(raw):
    result = {
        "requests": 0.0,
        "prompt_tokens": 0.0,
        "output_tokens": 0.0,
        "cached_tokens": 0.0,
    }
    names = {
        "sglang:num_requests_total": "requests",
        "sglang:prompt_tokens_total": "prompt_tokens",
        "sglang:generation_tokens_total": "output_tokens",
        "sglang:cached_tokens_total": "cached_tokens",
    }
    for name, labels, value in parse_prometheus(raw):
        if name not in names:
            continue
        # Cached tokens have no is_streaming label. SGLang emits either source
        # breakdowns OR a total fallback for each request, not both.
        if (
            name != "sglang:cached_tokens_total"
            and labels.get("is_streaming") != "true"
        ):
            continue
        result[names[name]] += value
    return result


@dataclass
class CompletionResult:
    success: bool
    started_at: float
    ended_at: float
    latency_s: float
    ttft_s: float | None
    output_ids: list[int]
    meta_info: dict[str, Any]
    error: str = ""


async def request_completion(
    client,
    *,
    url,
    session_id,
    rid,
    prompt_ids,
    output_tokens,
    first_token_callback=None,
):
    payload = {
        "input_ids": list(prompt_ids),
        "session_id": session_id,
        "rid": rid,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "min_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }
    started_at, started = time.time(), time.perf_counter()
    first = None
    ids, meta, error = [], {}, ""
    try:
        async with client.post(url.rstrip("/") + "/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:1000]}"
                )
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                text = line.removeprefix("data:").strip()
                if text == "[DONE]":
                    continue
                chunk = json.loads(text)
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"]))
                current = chunk.get("output_ids", [])
                if len(current) < len(ids) or current[: len(ids)] != ids:
                    raise ValueError(
                        "Expected cumulative output_ids; incremental streaming must be disabled"
                    )
                is_first_token = len(current) > len(ids) and first is None
                if is_first_token:
                    first = time.perf_counter()
                ids = current
                meta.update(chunk.get("meta_info") or {})
                if is_first_token and first_token_callback is not None:
                    first_token_callback(
                        {
                            "timestamp": time.time(),
                            "session_id": session_id,
                            "rid": rid,
                            "worker_url": url.rstrip("/"),
                            "prompt_tokens": len(prompt_ids),
                            "output_tokens_seen": len(ids),
                            "ttft_s": first - started,
                            "cached_tokens": meta.get("cached_tokens"),
                            "meta_info": dict(meta),
                        }
                    )
        if len(ids) != output_tokens:
            raise ValueError(f"Output length {len(ids)} != {output_tokens}")
        if meta.get("prompt_tokens") != len(prompt_ids):
            raise ValueError(
                f"Prompt length {meta.get('prompt_tokens')} != {len(prompt_ids)}"
            )
        if meta.get("completion_tokens") != output_tokens:
            raise ValueError(
                f"Server output length {meta.get('completion_tokens')} != {output_tokens}"
            )
        if first is None:
            raise ValueError("No token-bearing streaming response")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return CompletionResult(
        success=not error,
        started_at=started_at,
        ended_at=time.time(),
        latency_s=time.perf_counter() - started,
        ttft_s=first - started if first is not None else None,
        output_ids=ids,
        meta_info=meta,
        error=error,
    )


class DemandTracker:
    """Reusable frontier of unfinished sessions, not cumulative input tokens."""

    def __init__(self, handle, origin, workers):
        self.handle, self.origin, self.workers = handle, origin, workers
        self.sessions = {}

    def update(self, sid, worker, prefix, remaining, phase, round_index):
        if remaining == 0:
            self.sessions.pop(sid, None)
        else:
            self.sessions[sid] = {
                "worker": worker,
                "prefix": prefix,
                "remaining": remaining,
            }
        totals = []
        for index in range(self.workers):
            active = [x for x in self.sessions.values() if x["worker"] == index]
            totals.append(
                {
                    "worker": index,
                    "sessions": len(active),
                    "prefix_tokens": sum(x["prefix"] for x in active),
                    "prefix_tokens_with_two_rounds_remaining": sum(
                        x["prefix"] for x in active if x["remaining"] >= 2
                    ),
                }
            )
        self.handle.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "time_s": time.perf_counter() - self.origin,
                    "session_id": sid,
                    "worker": worker,
                    "phase": phase,
                    "round_index": round_index,
                    "session_prefix_tokens": prefix,
                    "remaining_rounds": remaining,
                    "workers": totals,
                }
            )
            + "\n"
        )


async def replay_session(
    client,
    *,
    session,
    worker,
    url,
    seed,
    prewarmed,
    start_delay,
    tracker,
    record,
    request_fn=request_completion,
):
    if start_delay:
        await asyncio.sleep(start_delay)
    sid = session.session_id
    builder = SyntheticPromptBuilder(f"seed-{seed}:{sid}")
    session_started = time.time()
    previous_end, previous_delay = None, 0.0
    for index, turn in enumerate(session.turns):
        prompt = builder.build(turn)
        replayable = (
            turn.prefix_tokens
            if index == 0 and prewarmed
            else prompt.replayable_prefix_tokens
        )
        tracker.update(
            sid,
            worker,
            replayable,
            len(session.turns) - index,
            "request",
            turn.round_index,
        )
        result = await request_fn(
            client,
            url=url,
            session_id=sid,
            rid=f"{sid}:r{turn.round_index}",
            prompt_ids=prompt.prompt_token_ids,
            output_tokens=turn.output_tokens,
        )
        gap = result.started_at - previous_end if previous_end is not None else None
        row = {
            **asdict(turn),
            "worker": worker,
            "worker_url": url,
            "session_started_at": session_started,
            "request_started_at": result.started_at,
            "request_ended_at": result.ended_at,
            "latency_s": result.latency_s,
            "ttft_s": result.ttft_s,
            "tpot_s": (
                (result.latency_s - result.ttft_s) / (turn.output_tokens - 1)
                if result.ttft_s is not None and turn.output_tokens > 1
                else None
            ),
            "actual_input_tokens": result.meta_info.get("prompt_tokens"),
            "actual_output_tokens": len(result.output_ids),
            "cached_tokens": result.meta_info.get("cached_tokens", 0),
            "num_retractions": result.meta_info.get("num_retractions", 0),
            "expected_replayable_prefix_tokens": replayable,
            "history_tokens_available": prompt.history_tokens_available,
            "prompt_sha256": token_hash(prompt.prompt_token_ids),
            "output_sha256": token_hash(result.output_ids),
            "output_ids": result.output_ids,
            "observed_gap_before_s": gap,
            "required_gap_before_s": previous_delay,
            "applied_tool_delay_s": (
                turn.tool_delay_s if index + 1 < len(session.turns) else 0.0
            ),
            "success": result.success,
            "error": result.error,
            "meta_info": result.meta_info,
        }
        record(row)
        if not result.success:
            raise RuntimeError(f"{sid} round {turn.round_index}: {result.error}")
        if gap is not None and gap + 0.005 < previous_delay:
            raise AssertionError("Tool-delay dependency violated")
        builder.observe_completion(prompt.prompt_token_ids, result.output_ids)
        previous_end, previous_delay = result.ended_at, row["applied_tool_delay_s"]
        remaining = len(session.turns) - index - 1
        prefix = (
            min(session.turns[index + 1].prefix_tokens, len(builder.history))
            if remaining
            else 0
        )
        tracker.update(
            sid,
            worker,
            prefix,
            remaining,
            "tool_wait" if remaining else "finished",
            turn.round_index,
        )
        if remaining:
            await asyncio.sleep(previous_delay)
    return {
        "session_id": sid,
        "worker": worker,
        "latency_s": time.time() - session_started,
        "completed_turns": len(session.turns),
    }


async def get_text(client, url, method="GET"):
    async with client.request(method, url) as response:
        response.raise_for_status()
        return await response.text()


def load_workload(path, worker_count):
    workload = json.loads(Path(path).read_text())
    if sha256_file(workload["dataset_path"]) != workload["dataset_sha256"]:
        raise ValueError("Trace hash differs from frozen workload")
    rows = workload["sessions"]
    ids = [row["session_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source sessions are not allowed")
    warm_first = workload.get("prewarm_first_session_ids", [])
    if len(warm_first) != len(set(warm_first)) or not set(warm_first).issubset(ids):
        raise ValueError("Prewarm-first sessions must be distinct selected session IDs")
    assignments = {
        row["session_id"]: (0 if worker_count == 1 else int(row["worker"]))
        for row in rows
    }
    if set(assignments.values()) != set(range(worker_count)):
        raise ValueError("Every worker must receive a valid session assignment")
    sessions = load_tracelab_sessions(
        workload["dataset_path"],
        session_ids=ids,
        provider="codex",
        min_rounds=workload["rounds"],
        max_rounds_per_session=workload["rounds"],
        max_input_tokens=workload["max_input_plus_output_tokens"],
        max_output_tokens=workload["max_output_tokens"],
        tool_delay_mode="wall-span",
    )
    for session in sessions:
        if (
            sum(t.tool_delay_s for t in session.turns[:-1])
            > workload["max_tool_delay_total_s"]
        ):
            raise ValueError(
                f"Tool delay exceeds frozen selection bound: {session.session_id}"
            )
    return workload, sessions, assignments


async def run(args):
    urls = [url.rstrip("/") for url in args.worker_url]
    workload, sessions, assignments = load_workload(args.workload, len(urls))
    selected = [asdict(t) for session in sessions for t in session.turns]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sessions": len(sessions),
                    "requests": len(selected),
                    "selected_trace": selected,
                },
                default=str,
            )
        )
        return 0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    (out / "selected_trace.json").write_text(
        json.dumps(selected, indent=2, default=str) + "\n"
    )
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    records, prewarm, completed_sessions, before, after = [], [], [], [], []
    error = ""
    origin_wall = origin = end_wall = None
    measured_s = 0.0
    whole_started = time.perf_counter()
    with (out / "requests.jsonl").open("w", buffering=1) as requests_file, (
        out / "demand_events.jsonl"
    ).open("w", buffering=1) as events, (out / "first_token_events.jsonl").open(
        "w", buffering=1
    ) as first_tokens:

        def record(row):
            records.append(row)
            requests_file.write(json.dumps(row, default=str) + "\n")
            if len(records) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "completed_turns": len(records),
                            "expected": len(selected),
                            "timestamp": time.time(),
                        }
                    ),
                    flush=True,
                )

        def record_first_token(row):
            first_tokens.write(json.dumps(row, default=str) + "\n")

        async def measured_request(client, **kwargs):
            return await request_completion(
                client, **kwargs, first_token_callback=record_first_token
            )

        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            try:
                for worker, url in enumerate(urls):
                    info = json.loads(await get_text(client, url + "/get_server_info"))
                    (out / f"worker{worker}_info.json").write_text(
                        json.dumps(info, indent=2) + "\n"
                    )
                    if info.get("disable_radix_cache") is not False:
                        raise ValueError(
                            "Server did not confirm prefix caching enabled"
                        )
                    if info.get("incremental_streaming_output", False):
                        raise ValueError(
                            "Cumulative native streaming output is required"
                        )
                    if args.flush_cache:
                        await get_text(client, url + "/flush_cache?timeout=30", "POST")
                prewarm_started = time.perf_counter()
                if args.prewarm_initial_prefix:

                    async def prewarm_session(session):
                        turn = session.turns[0]
                        if not turn.prefix_tokens:
                            return
                        builder = SyntheticPromptBuilder(
                            f"seed-{workload['seed']}:{session.session_id}"
                        )
                        prefix = builder.build(turn).prompt_token_ids[
                            : turn.prefix_tokens
                        ]
                        worker = assignments[session.session_id]
                        result = await request_completion(
                            client,
                            url=urls[worker],
                            session_id=session.session_id,
                            rid="prewarm:" + session.session_id,
                            prompt_ids=prefix,
                            output_tokens=1,
                        )
                        prewarm.append(
                            {
                                "session_id": session.session_id,
                                "worker": worker,
                                "prefix_tokens": len(prefix),
                                **asdict(result),
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "prewarmed_sessions": len(prewarm),
                                    "expected_sessions": len(sessions),
                                }
                            ),
                            flush=True,
                        )
                        if not result.success:
                            raise RuntimeError(
                                f"Initial-prefix prewarm failed: {result.error}"
                            )

                    # One initial-prefix request per session, with the same
                    # sticky assignment as the measured replay. Complete the
                    # entire prewarm wave before starting any measured turn.
                    warm_first = workload.get("prewarm_first_session_ids", [])
                    by_id = {s.session_id: s for s in sessions}
                    warm_tasks = []
                    try:
                        # Complete the frozen anchor order before admitting
                        # other prefixes. Each session is still warmed once.
                        for sid in warm_first:
                            await prewarm_session(by_id[sid])
                        warm_tasks = [
                            asyncio.create_task(prewarm_session(s))
                            for s in sessions
                            if s.session_id not in warm_first
                        ]
                        await asyncio.gather(*warm_tasks)
                    finally:
                        for task in warm_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*warm_tasks, return_exceptions=True)
                        (out / "prewarm.json").write_text(
                            json.dumps(
                                {
                                    "duration_s": time.perf_counter() - prewarm_started,
                                    "requests": prewarm,
                                    "concurrency": "one_per_session",
                                    "prewarm_first_session_ids": warm_first,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                prewarm_s = time.perf_counter() - prewarm_started
                (out / "prewarm.json").write_text(
                    json.dumps(
                        {
                            "duration_s": prewarm_s,
                            "requests": prewarm,
                            "concurrency": "one_per_session",
                            "prewarm_first_session_ids": workload.get(
                                "prewarm_first_session_ids", []
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                for worker, url in enumerate(urls):
                    raw = await get_text(client, url + "/metrics")
                    (out / f"worker{worker}_before.prom").write_text(raw)
                    before.append(counter_totals(raw))
                origin_wall, origin = time.time(), time.perf_counter()
                tracker = DemandTracker(events, origin, len(urls))
                delays = {
                    row["session_id"]: row.get("start_delay_s", 0.0)
                    for row in workload["sessions"]
                }
                tasks = [
                    asyncio.create_task(
                        replay_session(
                            client,
                            session=s,
                            worker=assignments[s.session_id],
                            url=urls[assignments[s.session_id]],
                            seed=workload["seed"],
                            prewarmed=args.prewarm_initial_prefix,
                            start_delay=delays[s.session_id],
                            tracker=tracker,
                            record=record,
                            request_fn=measured_request,
                        )
                    )
                    for s in sessions
                ]
                try:
                    completed_sessions = await asyncio.gather(*tasks)
                finally:
                    end_wall = time.time()
                    measured_s = time.perf_counter() - origin
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                for worker, url in enumerate(urls):
                    raw = await get_text(client, url + "/metrics")
                    (out / f"worker{worker}_after.prom").write_text(raw)
                    after.append(counter_totals(raw))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    deltas = [{k: a[k] - b[k] for k in b} for b, a in zip(before, after)]
    totals = {
        k: sum(row[k] for row in deltas)
        for k in ["requests", "prompt_tokens", "output_tokens", "cached_tokens"]
    }
    expected = {
        "requests": len(selected),
        "prompt_tokens": sum(t["input_tokens"] for t in selected),
        "output_tokens": sum(t["output_tokens"] for t in selected),
        "cached_tokens": sum(r["cached_tokens"] for r in records),
    }
    if not error and totals != expected:
        error = f"Server counter accounting mismatch: {totals} != {expected}"
    if not error and expected["cached_tokens"] <= 0:
        error = "No measured prefix cache hits"
    duration = measured_s if origin is not None else 0.0
    result = {
        "status": (
            "completed" if not error and len(records) == len(selected) else "failed"
        ),
        "error": error,
        "started_at": origin_wall,
        "ended_at": end_wall,
        "duration_s": duration,
        "including_setup_prewarm_s": time.perf_counter() - whole_started,
        "sessions": len(sessions),
        "completed_sessions": len(completed_sessions),
        "expected_turns": len(selected),
        "completed_turns": sum(r["success"] for r in records),
        "output_tps": (
            expected["output_tokens"] / duration if duration and not error else None
        ),
        "prefix_hit_rate": expected["cached_tokens"] / expected["prompt_tokens"],
        "counter_totals": totals,
        "expected_totals": expected,
        "counter_deltas_per_worker": deltas,
        "ttft_s": stats([r["ttft_s"] for r in records if r["success"]]),
        "tpot_s": stats(
            [r["tpot_s"] for r in records if r["success"] and r["tpot_s"] is not None]
        ),
        "request_latency_s": stats([r["latency_s"] for r in records if r["success"]]),
        "session_latency_s": stats([r["latency_s"] for r in completed_sessions]),
        "session_results": completed_sessions,
        "tool_delay_violations": sum(
            r["observed_gap_before_s"] is not None
            and r["observed_gap_before_s"] + 0.005 < r["required_gap_before_s"]
            for r in records
        ),
        "unreplayable_prefix_tokens": sum(
            r["prefix_tokens"] - r["expected_replayable_prefix_tokens"] for r in records
        ),
        "num_retractions": sum(r["num_retractions"] for r in records),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if records:
        columns = [
            k for k in records[0] if k not in {"tools", "meta_info", "output_ids"}
        ]
        with (out / "request_metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--worker-url", action="append", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--prewarm-initial-prefix", action="store_true")
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.output_dir:
        parser.error("--output-dir is required for a live replay")
    if len(args.worker_url) not in (1, 2):
        parser.error("Supply one PP8 URL or two independent PP4 URLs")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
