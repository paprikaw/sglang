"""CPU-only deterministic tests; local fake HTTP servers never touch a GPU."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

SPEC = importlib.util.spec_from_file_location("idle_benchmark_loadgen", Path(__file__).with_name("loadgen.py"))
loadgen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loadgen)
try:
    from aiohttp import web
except ImportError:
    web = None


def request(index=0, offset=0.0, output=3):
    return {"sample_id": str(index), "input_ids": [100, 200], "output_tokens": output, "offset_s": offset}


def endpoint(node="a"):
    return {"node": node, "url": "http://unused", "model_id": "model-" + node}


class Helpers(unittest.TestCase):
    def test_round_robin_and_preserve_bursts(self):
        schedule, window = loadgen.build_schedule([endpoint("a"), endpoint("b"), endpoint("c")],
            [request(i, offset) for i, offset in enumerate([4, 4, 5, 8, 8, 8, 9])], 2)
        self.assertEqual([entry["endpoint"]["node"] for entry in schedule], list("abcabca"))
        self.assertEqual([entry["offset_s"] for entry in schedule], [0, 0, .6, 2.4, 2.4, 2.4, 3])
        self.assertEqual(window, 3.5)

    def test_partition_preserves_common_global_offsets(self):
        schedule, window = loadgen.build_schedule([endpoint()], [request(1, 1.25), request(3, 3.25)],
                                                 1, preserve_offsets=True, duration_s=4)
        self.assertEqual([entry["offset_s"] for entry in schedule], [1.25, 3.25])
        self.assertEqual(window, 4)

    def test_zero_span_keeps_burst(self):
        schedule, _ = loadgen.build_schedule([endpoint()], [request(0), request(1)], 3)
        self.assertEqual([entry["offset_s"] for entry in schedule], [0, 0])

    def test_percentiles_interpolate(self):
        result = loadgen.percentiles([4, 1, 2, 3])
        self.assertEqual(result["p50"], 2.5)
        self.assertAlmostEqual(result["p90"], 3.7)
        self.assertAlmostEqual(result["p99"], 3.97)
        self.assertEqual(loadgen.percentiles([])["count"], 0)
        self.assertIsNone(loadgen.percentiles([])["p50"])

    def test_metrics_tp_rank_and_model_filter(self):
        text = '\n'.join([
            'sglang:queue_time_seconds_count{tp_rank="0",pp_rank="0",model_name="model-a"} 8',
            'sglang:queue_time_seconds_bucket{tp_rank="0",pp_rank="0",model_name="model-a",le="0.005"} 7',
            'sglang:queue_time_seconds_count{tp_rank="1",model_name="model-a"} 800',
            'sglang:queue_time_seconds_bucket{tp_rank="1",model_name="model-a",le="0.005"} 0',
            'sglang:queue_time_seconds_count{tp_rank="0",model_name="foreign"} 100',
            'sglang:queue_time_seconds_bucket{tp_rank="0",model_name="foreign",le="0.005"} 0'])
        result = loadgen.parse_queue_histogram(text, .005, model_id="model-a")
        self.assertEqual(result["count"], 8)
        self.assertEqual(result["within_tolerance_count"], 7)
        self.assertEqual(result["series"], 1)

    def test_metrics_missing_bucket_or_tp_rank_is_invalid(self):
        for text in ('', 'sglang:queue_time_seconds_count{tp_rank="0"} 2',
                     'sglang:queue_time_seconds_count 2\nsglang:queue_time_seconds_bucket{le="0.005"} 2',
                     'sglang:queue_time_seconds_count{tp_rank="0"} 2\nsglang:queue_time_seconds_bucket{tp_rank="0",le="0.01"} 2'):
            with self.subTest(text=text), self.assertRaises(loadgen.TelemetryError):
                loadgen.parse_queue_histogram(text, .005)

    def test_loads_freshness_uses_scheduler_timestamp(self):
        result = loadgen.parse_loads({"timestamp": "fresh-http-envelope", "loads": [
            {"dp_rank": 0, "timestamp": 90, "num_waiting_reqs": 0, "num_running_reqs": 1}]},
            now=100, max_age_s=2)
        self.assertFalse(result["all_fresh"])
        self.assertEqual(result["loads"][0]["age_s"], 10)
        with self.assertRaises(loadgen.TelemetryError):
            loadgen.parse_loads({"loads": [{"dp_rank": 0, "num_waiting_reqs": 0,
                                          "num_running_reqs": 0}]}, now=100, max_age_s=2)

    def test_unsorted_or_invalid_schedule_is_rejected(self):
        for requests in ([request(0, 2), request(1, 1)], [dict(request(), output_tokens=0)],
                         [dict(request(), input_ids=[])], [request(0, -1)]):
            with self.subTest(requests=requests), self.assertRaises(ValueError):
                loadgen.build_schedule([endpoint()], requests, 1)


class BacklogEvidence(unittest.TestCase):
    def snapshot(self, timestamp, waiting, running=8, capacity=8, fresh=True):
        return {"all_fresh": fresh, "loads": [{"timestamp": timestamp}],
                "num_waiting_reqs": waiting, "num_running_reqs": running,
                "max_running_requests": capacity}

    def test_repeated_poll_is_not_sustained_evidence(self):
        state = {}
        for _ in range(100):
            reasons = loadgen.update_backlog_evidence(state, self.snapshot(10, 1), grace_s=2)
            self.assertFalse(reasons)
        self.assertEqual(state["max_sustained_waiting_s"], 0)

    def test_positive_waiting_across_two_seconds_is_backlog(self):
        state = {}
        self.assertFalse(loadgen.update_backlog_evidence(state, self.snapshot(10, 1), grace_s=2))
        self.assertFalse(loadgen.update_backlog_evidence(state, self.snapshot(11, 1), grace_s=2))
        self.assertIn("sustained_waiting_backlog", loadgen.update_backlog_evidence(
            state, self.snapshot(12, 1), grace_s=2))

    def test_zero_waiting_resets_continuity(self):
        state = {}
        loadgen.update_backlog_evidence(state, self.snapshot(10, 1), grace_s=2)
        loadgen.update_backlog_evidence(state, self.snapshot(11, 0), grace_s=2)
        self.assertFalse(loadgen.update_backlog_evidence(state, self.snapshot(12, 1), grace_s=2))

    def test_growth_requires_three_distinct_full_slot_snapshots(self):
        state = {}
        for timestamp, waiting in ((10, 1), (10.1, 2)):
            self.assertFalse(loadgen.update_backlog_evidence(state, self.snapshot(timestamp, waiting), grace_s=2))
        self.assertIn("growing_waiting_backlog_at_full_slots", loadgen.update_backlog_evidence(
            state, self.snapshot(10.2, 3), grace_s=2))
        state = {}
        for timestamp, waiting in ((10, 1), (10.1, 2), (10.2, 3)):
            self.assertFalse(loadgen.update_backlog_evidence(state,
                self.snapshot(timestamp, waiting, running=7), grace_s=2))

    def test_stale_snapshots_do_not_prove_backlog(self):
        state = {}
        self.assertFalse(loadgen.update_backlog_evidence(state, self.snapshot(10, 1, fresh=False), grace_s=2))
        self.assertFalse(state)


class SSE(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_utf8_multiline_and_eof(self):
        class Content:
            async def iter_any(self):
                body = ': keepalive\r\ndata: {"text":"你",\r\ndata: "meta_info":{}}\r\n\r\ndata: [DONE]'.encode()
                for index in range(0, len(body), 2):
                    yield body[index:index + 2]
        events = [event async for event in loadgen.iter_sse(Content())]
        self.assertEqual(json.loads(events[0])["text"], "你")
        self.assertEqual(events[1], "[DONE]")

    async def test_oversized_event_rejected(self):
        class Content:
            async def iter_any(self):
                yield b"data: " + b"x" * 100
        with self.assertRaises(ValueError):
            _ = [event async for event in loadgen.iter_sse(Content(), max_event_bytes=20)]


class FakeServer:
    def __init__(self, node="a", *, behavior="ok", delay=.003):
        self.node, self.model_id = node, "model-" + node
        self.behavior, self.delay = behavior, delay
        self.payloads, self.arrivals = [], []
        self.count = self.within = 0
        self.running = 0

    async def start(self):
        app = web.Application()
        app.router.add_get("/v1/loads", self.loads)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/generate", self.generate)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.url = "http://127.0.0.1:" + str(self.site._server.sockets[0].getsockname()[1])
        return self

    def endpoint(self):
        return {"node": self.node, "url": self.url, "model_id": self.model_id}

    async def close(self):
        await self.runner.cleanup()

    async def loads(self, request):
        timestamp = time.time()
        if self.behavior == "stale_always" or self.behavior == "stale_idle" and not self.payloads:
            timestamp -= 100
        return web.json_response({"timestamp": "not-scheduler-time", "loads": [{"dp_rank": 0,
            "timestamp": timestamp, "num_waiting_reqs": 1 if self.behavior == "waiting" and self.payloads else 0,
            "num_running_reqs": self.running, "max_running_requests": 8,
            "queues": {"retracted": 0}}]})

    async def metrics(self, request):
        if self.behavior == "missing_metrics":
            return web.Response(text="# unavailable\n")
        return web.Response(text='\n'.join([
            f'sglang:queue_time_seconds_count{{tp_rank="0",pp_rank="0",model_name="{self.model_id}"}} {self.count}',
            f'sglang:queue_time_seconds_bucket{{tp_rank="0",pp_rank="0",model_name="{self.model_id}",le="0.005"}} {self.within}',
            f'sglang:queue_time_seconds_count{{tp_rank="1",pp_rank="0",model_name="{self.model_id}"}} 999',
            f'sglang:queue_time_seconds_bucket{{tp_rank="1",pp_rank="0",model_name="{self.model_id}",le="0.005"}} 0']))

    async def generate(self, request):
        payload = await request.json()
        self.payloads.append(payload)
        self.arrivals.append(time.time())
        self.count += 1
        self.within += self.behavior not in ("histogram_wait",)
        self.running += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        prompt = len(payload["input_ids"]) + (1 if self.behavior == "wrong_prompt" else 0)
        output = payload["sampling_params"]["max_new_tokens"]
        async def emit(count, *, finished=False):
            meta = {"completion_tokens": count, "prompt_tokens": prompt,
                    "num_retractions": 1 if self.behavior == "retracted" else 0,
                    "finish_reason": {"type": "length"} if finished else None}
            if self.behavior != "missing_queue":
                meta["queue_time"] = .02 if self.behavior == "request_wait" else .001
            event = {"text": "private-payload-never-persist", "meta_info": meta}
            body = ("data: " + json.dumps(event) + "\n\n").encode()
            # Every event is fragmented across writes, unrelated to token boundaries.
            await response.write(body[:17])
            await response.write(body[17:])
        try:
            await response.write(b': heartbeat\n\ndata: {"role":"assistant"}\n\n')
            await emit(0)
            await asyncio.sleep(self.delay)
            await emit(1)
            await asyncio.sleep(self.delay if self.behavior != "slow" else .25)
            final_count = output - 1 if self.behavior == "wrong_output" else output
            await emit(final_count)
            # Repeated final cumulative metadata is not an extra generated token.
            await asyncio.sleep(self.delay)
            await emit(final_count, finished=True)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except ConnectionResetError:
            pass
        finally:
            self.running -= 1
        return response


@unittest.skipIf(web is None, "aiohttp required for CPU-only HTTP fixture")
class Stages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.servers = []

    async def asyncTearDown(self):
        for server in self.servers:
            await server.close()
        self.tmp.cleanup()

    async def server(self, *args, **kwargs):
        server = await FakeServer(*args, **kwargs).start()
        self.servers.append(server)
        return server

    async def stage(self, server, requests=None, **kwargs):
        parameters = {"duration_s": .08, "preserve_offsets": True,
                      "telemetry_interval_s": .005, "max_dispatch_lag_s": .2}
        parameters.update(kwargs)
        return await loadgen.run_stage([server.endpoint()], requests or [request()], 10,
            Path(self.tmp.name) / "stage", **parameters)

    def rows(self):
        return [json.loads(line) for line in (Path(self.tmp.name) / "stage" / "requests.jsonl").read_text().splitlines()]

    async def test_stream_tokens_lengths_latency_and_private_artifacts(self):
        server = await self.server()
        summary = await self.stage(server)
        self.assertTrue(summary["valid"], summary["reasons"])
        row, = self.rows()
        self.assertEqual(row["completion_tokens"], 3)
        self.assertGreater(row["ttft_s"], 0)
        self.assertGreater(row["tpot_s"], 0)
        self.assertLess(row["tpot_s"], row["e2e_s"] / 2)
        self.assertEqual(row["server_queue_s"], .001)
        self.assertEqual(row["stream_termination"], "done")
        self.assertEqual(summary["successful_requests"], 1)
        payload = server.payloads[0]
        self.assertTrue(payload["sampling_params"]["ignore_eos"])
        self.assertEqual(payload["sampling_params"]["temperature"], 0)
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 3)
        for path in (Path(self.tmp.name) / "stage").iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            text = path.read_text()
            self.assertNotIn("private-payload-never-persist", text)
            self.assertNotIn("input_ids", text)

    async def test_progress_contract_has_live_counts_and_queue(self):
        server = await self.server()
        summary = await self.stage(server)
        progress = json.loads((Path(self.tmp.name) / "stage" / "progress.json").read_text())
        self.assertEqual(progress["dispatched_requests"], 1)
        self.assertEqual(progress["completed_requests"], 1)
        self.assertEqual(progress["successful_requests"], 1)
        self.assertEqual(progress["inflight"], 0)
        self.assertEqual(progress["queue_waiting"], {"a": 0})
        self.assertEqual(progress["max_queue"], {"a": 0})
        self.assertEqual(progress["status"], summary["status"])
        self.assertTrue(progress["valid_so_far"])
        self.assertIn("phase_start_epoch", progress)
        self.assertIn("timestamp", progress)

    async def test_round_robin_balances_requests_with_overlapping_open_loop(self):
        a, b = await self.server("a", behavior="slow"), await self.server("b", behavior="slow")
        summary = await loadgen.run_stage([a.endpoint(), b.endpoint()],
            [request(i, i * .015) for i in range(5)], 100,
            Path(self.tmp.name) / "stage", duration_s=.08, preserve_offsets=True,
            telemetry_interval_s=.01, max_dispatch_lag_s=.2)
        self.assertTrue(summary["valid"], summary["reasons"])
        self.assertEqual([len(a.payloads), len(b.payloads)], [3, 2])
        self.assertLess(a.arrivals[-1] - a.arrivals[0], .15)
        self.assertGreater(summary["elapsed_s"], .25)
        self.assertEqual(summary["per_node"]["a"]["offered_requests"], 3)

    async def test_missing_metrics_prevents_any_dispatch(self):
        server = await self.server(behavior="missing_metrics")
        result = await self.stage(server)
        self.assertFalse(result["valid"])
        self.assertFalse(server.payloads)
        self.assertTrue(any("telemetry_failed" in reason for reason in result["reasons"]))

    async def test_request_queue_is_exact_independent_evidence(self):
        server = await self.server(behavior="request_wait")
        result = await self.stage(server)
        self.assertFalse(result["valid"])
        self.assertTrue(any("server_queue_above_tolerance" in reason for reason in result["reasons"]))
        self.assertEqual(result["telemetry"]["a"]["max_above_tolerance_delta"], 0)

    async def test_histogram_queue_delta_is_independent_evidence(self):
        server = await self.server(behavior="histogram_wait")
        result = await self.stage(server)
        self.assertFalse(result["valid"])
        self.assertTrue(any("queue_histogram_above_tolerance" in reason for reason in result["reasons"]))
        self.assertEqual(result["telemetry"]["a"]["max_above_tolerance_delta"], 1)

    async def test_fresh_waiting_snapshot_invalidates_stage(self):
        server = await self.server(behavior="waiting")
        result = await self.stage(server)
        self.assertFalse(result["valid"])
        self.assertIn("queue_snapshot:a", result["reasons"])

    async def test_stale_idle_snapshot_does_not_prevent_traffic(self):
        server = await self.server(behavior="stale_idle")
        result = await self.stage(server)
        self.assertTrue(result["valid"], result["reasons"])
        self.assertEqual(len(server.payloads), 1)

    async def test_stale_only_telemetry_is_never_valid(self):
        server = await self.server(behavior="stale_always")
        result = await self.stage(server)
        self.assertFalse(result["valid"])
        self.assertIn("missing_fresh_load_evidence:a", result["reasons"])

    async def test_missing_exact_queue_or_wrong_lengths_invalid(self):
        for behavior, error in (("missing_queue", "missing_server_queue_time"),
                                ("wrong_prompt", "prompt_token_count_mismatch"),
                                ("wrong_output", "completion_token_count_mismatch")):
            with self.subTest(behavior=behavior):
                server = await self.server(behavior=behavior)
                result = await self.stage(server)
                self.assertFalse(result["valid"])
                self.assertTrue(any(error in reason for reason in result["reasons"]), result["reasons"])

    async def test_no_backlog_allows_and_reports_short_admission_waits(self):
        server = await self.server(behavior="request_wait")
        result = await self.stage(server, queue_policy="no_backlog")
        self.assertTrue(result["valid"], result["reasons"])
        self.assertEqual(result["latency_s"]["server_queue"]["p95"], .02)
        self.assertTrue(result["client_requests_drained"])
        self.assertEqual(result["queue_policy"], "no_backlog")

    async def test_no_backlog_records_histogram_excursions(self):
        server = await self.server(behavior="histogram_wait")
        result = await self.stage(server, queue_policy="no_backlog")
        self.assertTrue(result["valid"], result["reasons"])
        self.assertEqual(result["telemetry"]["a"]["max_above_tolerance_delta"], 1)

    async def test_no_backlog_rejects_continuous_waiting(self):
        server = await self.server(behavior="waiting")
        result = await self.stage(server, queue_policy="no_backlog", backlog_grace_s=.02)
        self.assertFalse(result["valid"])
        self.assertIn("sustained_waiting_backlog:a", result["reasons"])

    async def test_no_backlog_optional_absolute_wait_ceiling(self):
        server = await self.server(behavior="request_wait")
        result = await self.stage(server, queue_policy="no_backlog", max_request_queue_s=.01)
        self.assertFalse(result["valid"])
        self.assertTrue(any("server_queue_above_maximum" in reason for reason in result["reasons"]))

    async def test_retraction_invalidates_both_queue_policies(self):
        for policy in ("strict", "no_backlog"):
            with self.subTest(policy=policy):
                server = await self.server(behavior="retracted")
                result = await self.stage(server, queue_policy=policy)
                self.assertFalse(result["valid"])
                self.assertTrue(any("server_retractions" in reason for reason in result["reasons"]))

    async def test_safety_bound_cancels_instead_of_hiding_client_queue(self):
        server = await self.server(behavior="slow")
        result = await self.stage(server, [request(i) for i in range(5)], max_in_flight=1)
        self.assertFalse(result["valid"])
        self.assertIn("client_in_flight_safety_bound", result["reasons"])
        self.assertLessEqual(result["dispatched_requests"], 1)
        self.assertGreater(result["undispatched_requests"], 0)

    async def test_no_catch_up_burst_on_late_start(self):
        server = await self.server()
        result = await self.stage(server, [request(i, i * .01) for i in range(5)],
                                  start_at=time.time() - 1, max_dispatch_lag_s=.05)
        self.assertFalse(result["valid"])
        self.assertIn("client_dispatch_lag", result["reasons"])
        self.assertFalse(server.payloads)

    async def test_stop_during_future_start_is_prompt(self):
        server = await self.server()
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(.03, stop.set)
        before = time.monotonic()
        result = await self.stage(server, start_at=time.time() + 10, stop_check=stop.is_set)
        self.assertLess(time.monotonic() - before, .3)
        self.assertEqual(result["status"], "stopped")
        self.assertFalse(server.payloads)

    async def test_preemption_cancels_active_stream_and_saves_partial(self):
        server = await self.server(behavior="slow")
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(.06, stop.set)
        before = time.monotonic()
        result = await self.stage(server, duration_s=10, stop_check=stop.is_set)
        self.assertLess(time.monotonic() - before, .3)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["cancelled_requests"], 1)
        self.assertEqual(self.rows()[0]["status"], "cancelled")
        self.assertTrue((Path(self.tmp.name) / "stage" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
