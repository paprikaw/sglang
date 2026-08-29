"""Trace fidelity and native HTTP replay tests; no model or GPU required."""

import argparse
import asyncio
import gzip
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web

from sglang.benchmark.agentic_serving import (
    DemandTracker,
    counter_totals,
    load_workload,
    request_completion,
    run,
)
from sglang.benchmark.datasets.tracelab import (
    AgenticTraceTurn,
    SyntheticPromptBuilder,
    compute_tool_delay_s,
    load_tracelab_sessions,
    sha256_file,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-c-test-cpu")


def row(sid, round_index, length, prefix, output=2):
    return {
        "provider": "codex",
        "session_id": sid,
        "round_index": round_index,
        "round_id": f"{sid}:{round_index}",
        "input_tokens_total": length,
        "prefix_tokens": prefix,
        "newly_append_tokens": length - prefix,
        "output_tokens": output,
        "tools": [{"tool_wall_latency_ms": 10}],
    }


def write_trace(path, rows):
    with gzip.open(path, "wt") as handle:
        for item in rows:
            handle.write(json.dumps(item) + "\n")


class TestTraceLab(unittest.TestCase):
    def test_parallel_tools_and_nonfinite_fallback(self):
        tools = [
            {
                "emitted_at": "2026-01-01T00:00:00Z",
                "result_at": "2026-01-01T00:00:00.100Z",
                "tool_wall_latency_ms": 100,
            },
            {
                "emitted_at": "2026-01-01T00:00:00.020Z",
                "result_at": "2026-01-01T00:00:00.250Z",
                "tool_wall_latency_ms": 230,
            },
        ]
        self.assertAlmostEqual(compute_tool_delay_s(tools), 0.25)
        self.assertAlmostEqual(compute_tool_delay_s(tools, "max-wall"), 0.23)
        self.assertEqual(
            compute_tool_delay_s([{"tool_wall_latency_ms": float("nan")}]), 0
        )

    def test_loader_preserves_tools_and_rejects_broken_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.gz"
            write_trace(
                path, [row("a", 1, 12, 10), row("a", 0, 8, 4), row("bad", 0, 0, 0)]
            )
            sessions = load_tracelab_sessions(path, session_ids=["a"], min_rounds=2)
            self.assertEqual([t.round_index for t in sessions[0].turns], [0, 1])
            self.assertEqual(sessions[0].turns[0].tools[0]["tool_wall_latency_ms"], 10)
            with self.assertRaisesRegex(ValueError, "zero-length"):
                load_tracelab_sessions(path, session_ids=["bad"], min_rounds=1)
            write_trace(path, [row("a", 0, 8, 4), row("a", 2, 12, 10)])
            with self.assertRaisesRegex(ValueError, "consecutive"):
                load_tracelab_sessions(path, session_ids=["a"], min_rounds=2)
            broken = row("a", 0, 8, 4)
            broken["newly_append_tokens"] = 3
            write_trace(path, [broken])
            with self.assertRaisesRegex(ValueError, "input=prefix"):
                load_tracelab_sessions(path, min_rounds=1)

    def test_prompt_preserves_actual_output_and_fork(self):
        builder = SyntheticPromptBuilder("a")
        first = AgenticTraceTurn("a", "codex", 0, "0", 8, 4, 4, 2, 0)
        prompt = builder.build(first)
        builder.observe_completion(prompt.prompt_token_ids, [99, 98])
        turn = AgenticTraceTurn("a", "codex", 1, "1", 12, 10, 2, 2, 0)
        second = builder.build(turn)
        self.assertEqual(
            second.prompt_token_ids[:10], prompt.prompt_token_ids + (99, 98)
        )
        self.assertEqual(second.replayable_prefix_tokens, 10)
        fork = AgenticTraceTurn("a", "codex", 2, "2", 9, 4, 5, 2, 0)
        self.assertNotEqual(
            builder.build(fork).prompt_token_ids[4], prompt.prompt_token_ids[4]
        )

    def test_independent_sessions_do_not_collapse_to_1000_sequences(self):
        sequences = {
            tuple(SyntheticPromptBuilder(str(i))._tokens("prefix", 16))
            for i in range(1100)
        }
        self.assertEqual(len(sequences), 1100)
        self.assertEqual(
            SyntheticPromptBuilder("same")._tokens("p", 16),
            SyntheticPromptBuilder("same")._tokens("p", 16),
        )

    def test_resident_frontier_does_not_sum_history_or_finished_sessions(self):
        stream = io.StringIO()
        tracker = DemandTracker(stream, time.perf_counter(), 2)
        tracker.update("a", 0, 100, 3, "request", 0)
        tracker.update("a", 0, 150, 2, "tool_wait", 0)
        tracker.update("b", 1, 200, 1, "request", 2)
        tracker.update("a", 0, 0, 0, "finished", 2)
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(events[1]["workers"][0]["prefix_tokens"], 150)
        self.assertEqual(
            events[2]["workers"][1]["prefix_tokens_with_two_rounds_remaining"], 0
        )
        self.assertEqual(events[-1]["workers"][0]["prefix_tokens"], 0)

    def test_cached_counter_has_no_streaming_label(self):
        raw = "\n".join(
            [
                'sglang:num_requests_total{is_streaming="true"} 3',
                'sglang:num_requests_total{is_streaming="false"} 1',
                'sglang:prompt_tokens_total{is_streaming="true"} 30',
                'sglang:generation_tokens_total{is_streaming="true"} 6',
                'sglang:cached_tokens_total{cache_source="device"} 10',
                'sglang:cached_tokens_total{cache_source="host"} 2',
                'sglang:cached_tokens_total{cache_source="total"} 4',
            ]
        )
        self.assertEqual(
            counter_totals(raw),
            {
                "requests": 3,
                "prompt_tokens": 30,
                "output_tokens": 6,
                "cached_tokens": 16,
            },
        )


class FakeWorker:
    def __init__(self):
        self.cache, self.calls = [], []
        self.counts = {
            "requests": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

    async def info(self, request):
        return web.json_response(
            {"disable_radix_cache": False, "incremental_streaming_output": False}
        )

    async def flush(self, request):
        self.cache.clear()
        return web.Response(text="Cache flushed.")

    async def metrics(self, request):
        names = {
            "requests": "num_requests_total",
            "prompt_tokens": "prompt_tokens_total",
            "output_tokens": "generation_tokens_total",
            "cached_tokens": "cached_tokens_total",
        }
        return web.Response(
            text="\n".join(
                f"sglang:{names[k]}{{"
                + (
                    'cache_source="device"'
                    if k == "cached_tokens"
                    else 'is_streaming="true"'
                )
                + f"}} {v}"
                for k, v in self.counts.items()
            )
        )

    async def generate(self, request):
        payload = await request.json()
        self.calls.append(payload)
        prompt = payload["input_ids"]
        count = payload["sampling_params"]["max_new_tokens"]
        cached = 0
        for previous in self.cache:
            common = 0
            for a, b in zip(prompt, previous):
                if a != b:
                    break
                common += 1
            cached = max(cached, common)
        output = [99] * count
        self.cache.append(prompt + output)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for length in range(1, count + 1):
            chunk = {
                "output_ids": output[:length],
                "meta_info": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": length,
                    "cached_tokens": cached,
                    "num_retractions": 0,
                },
            }
            await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            await asyncio.sleep(0.001)
        for key, value in {
            "requests": 1,
            "prompt_tokens": len(prompt),
            "output_tokens": count,
            "cached_tokens": cached,
        }.items():
            self.counts[key] += value
        await response.write(b"data: [DONE]\n\n")
        return response


class TestNativeReplay(unittest.IsolatedAsyncioTestCase):
    async def test_native_stream_rejects_bad_lengths_and_non_cumulative_ids(self):
        cases = [
            (
                [
                    {
                        "output_ids": [7],
                        "meta_info": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                ],
                "Output length",
            ),
            ([{"output_ids": [7]}, {"output_ids": [8, 9]}], "cumulative"),
            (
                [
                    {
                        "output_ids": [7, 8],
                        "meta_info": {"prompt_tokens": 3, "completion_tokens": 2},
                    }
                ],
                "Prompt length",
            ),
            ([{"error": "request aborted"}], "request aborted"),
        ]
        for chunks, message in cases:
            with self.subTest(message=message):

                async def generate(request):
                    return web.Response(
                        text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                    )

                app = web.Application()
                app.router.add_post("/generate", generate)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                try:
                    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                    async with aiohttp.ClientSession() as client:
                        result = await request_completion(
                            client,
                            url=url,
                            session_id="s",
                            rid="r",
                            prompt_ids=[1, 2],
                            output_tokens=2,
                        )
                    self.assertFalse(result.success)
                    self.assertIn(message, result.error)
                finally:
                    await runner.cleanup()

    async def test_two_workers_sticky_sessions_prewarm_and_closed_loop(self):
        workers, runners, urls = [FakeWorker(), FakeWorker()], [], []
        try:
            for worker in workers:
                app = web.Application()
                app.router.add_get("/get_server_info", worker.info)
                app.router.add_post("/flush_cache", worker.flush)
                app.router.add_get("/metrics", worker.metrics)
                app.router.add_post("/generate", worker.generate)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                runners.append(runner)
                urls.append(
                    f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
                )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "trace.gz"
                write_trace(
                    trace,
                    [
                        item
                        for sid in ["a", "b"]
                        for item in [
                            row(sid, 0, 8, 4),
                            row(sid, 1, 12, 10),
                            row(sid, 2, 15, 14),
                        ]
                    ],
                )
                workload = {
                    "dataset_path": str(trace),
                    "dataset_sha256": sha256_file(trace),
                    "sessions": [
                        {"session_id": "a", "worker": 0},
                        {"session_id": "b", "worker": 1},
                    ],
                    "rounds": 3,
                    "max_input_plus_output_tokens": 32,
                    "max_output_tokens": 4,
                    "max_tool_delay_total_s": 1,
                    "seed": 42,
                }
                path = root / "workload.json"
                path.write_text(json.dumps(workload))
                args = argparse.Namespace(
                    worker_url=urls,
                    workload=str(path),
                    output_dir=str(root / "out"),
                    dry_run=False,
                    request_timeout_s=10,
                    flush_cache=True,
                    prewarm_initial_prefix=True,
                )
                self.assertEqual(await run(args), 0)
                result = json.loads((root / "out/result.json").read_text())
                self.assertEqual(result["completed_turns"], 6)
                self.assertEqual(result["counter_totals"]["output_tokens"], 12)
                self.assertEqual(result["tool_delay_violations"], 0)
                self.assertEqual(result["unreplayable_prefix_tokens"], 0)
                self.assertGreater(result["prefix_hit_rate"], 0)
                first_tokens = [
                    json.loads(line)
                    for line in (root / "out/first_token_events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(first_tokens), 6)
                self.assertTrue(
                    all(not row["rid"].startswith("prewarm:") for row in first_tokens)
                )
                self.assertTrue(
                    all(row["cached_tokens"] is not None for row in first_tokens)
                )
                for worker, sid in zip(workers, ["a", "b"]):
                    self.assertEqual(len(worker.calls), 4)
                    self.assertEqual({c["session_id"] for c in worker.calls}, {sid})
                    self.assertTrue(worker.calls[0]["rid"].startswith("prewarm:"))
                _, _, mapped = load_workload(path, 1)
                self.assertEqual(mapped, {"a": 0, "b": 0})
                with self.assertRaisesRegex(FileExistsError, "exists"):
                    await run(args)
                workers[0].calls.clear()
                single = argparse.Namespace(**vars(args))
                single.worker_url = urls[:1]
                single.output_dir = str(root / "single")
                workload["prewarm_first_session_ids"] = ["b"]
                path.write_text(json.dumps(workload))
                self.assertEqual(await run(single), 0)
                self.assertTrue(
                    all(c["rid"].startswith("prewarm:") for c in workers[0].calls[:2])
                )
                self.assertEqual(
                    {c["session_id"] for c in workers[0].calls}, {"a", "b"}
                )
                result = json.loads((root / "single/result.json").read_text())
                self.assertEqual(result["counter_totals"]["requests"], 6)
                self.assertEqual(result["counter_totals"]["output_tokens"], 12)
                warm = json.loads((root / "single/prewarm.json").read_text())
                self.assertEqual(len(warm["requests"]), 2)
                self.assertEqual(warm["requests"][0]["session_id"], "b")
                self.assertLessEqual(
                    warm["requests"][0]["ended_at"],
                    warm["requests"][1]["started_at"],
                )
                self.assertEqual(workers[0].calls[0]["rid"], "prewarm:b")
                for invalid in [["a", "a"], ["missing"]]:
                    workload["prewarm_first_session_ids"] = invalid
                    path.write_text(json.dumps(workload))
                    with self.assertRaisesRegex(ValueError, "Prewarm-first"):
                        load_workload(path, 1)
        finally:
            for runner in runners:
                await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
