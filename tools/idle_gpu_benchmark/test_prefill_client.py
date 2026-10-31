"""CPU-only HTTP tests for sequential prefill and cooperative cancellation."""
import asyncio
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location("idle_prefill_client", Path(__file__).with_name("prefill_client.py"))
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)
try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = web = None


class FakeServer:
    def __init__(self, behavior="ok", delay=.005):
        self.behavior, self.delay = behavior, delay
        self.payloads = []
        self.running = self.max_running = 0
        self.arrived = asyncio.Event()

    async def start(self):
        app = web.Application()
        app.router.add_post("/generate", self.generate)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.url = "http://127.0.0.1:" + str(self.site._server.sockets[0].getsockname()[1])
        return self

    def endpoint(self):
        return {"node": "065", "url": self.url, "model_id": "served-model"}

    async def close(self):
        await self.runner.cleanup()

    async def generate(self, request):
        self.payloads.append(await request.json())
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.arrived.set()
        if self.behavior == "http_error":
            self.running -= 1
            return web.Response(status=503, text="secret server body")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            # An initial metadata-only event must not start the TTFT clock.
            await response.write(b'data: {"meta_info":{"completion_tokens":0}}\n\n')
            await asyncio.sleep(self.delay)
            meta = {"prompt_tokens": 3, "completion_tokens": 1,
                    "finish_reason": {"type": "length"}}
            if self.behavior == "large_queue":
                meta.update(queue_time=123.5, num_retractions=17)
            if self.behavior == "invalid_optional":
                meta.update(queue_time="unavailable", num_retractions="not reported")
            if self.behavior == "wrong_prompt":
                meta["prompt_tokens"] = 4
            if self.behavior == "wrong_output":
                meta["completion_tokens"] = 2
            if self.behavior == "boolean_output":
                meta["completion_tokens"] = True
            if self.behavior == "missing_output":
                meta.pop("completion_tokens")
            if self.behavior == "server_error":
                event = {"error": "secret output and body"}
            else:
                event = {"text": "secret generated text", "meta_info": meta}
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            await response.write(b'data: [DONE]\n\n')
            await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.running -= 1
        return response


@unittest.skipIf(web is None, "aiohttp required")
class PrefillClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.servers = []
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await asyncio.gather(*(server.close() for server in self.servers))

    async def server(self, *args, **kwargs):
        server = await FakeServer(*args, **kwargs).start()
        self.servers.append(server)
        return server

    async def never_stop(self):
        return False

    def request(self, index=0):
        return {"sample_id": "source-" + str(index), "input_ids": [256, 257, 258],
                "planned_prompt_tokens": 3, "output_tokens": 300}

    async def send(self, server, **kwargs):
        return await client.send_prefill(self.session, server.endpoint(), self.request(),
                                         self.never_stop, **kwargs)

    async def test_sequential_await_has_no_overlap_or_rate_limit(self):
        server = await self.server(delay=.01)
        rows = []
        started = time.monotonic()
        for index in range(4):
            rows.append(await client.send_prefill(self.session, server.endpoint(), self.request(index), self.never_stop))
        self.assertEqual([row["status"] for row in rows], ["ok"] * 4)
        self.assertEqual(server.max_running, 1)
        self.assertEqual(len(server.payloads), 4)
        self.assertLess(time.monotonic() - started, .5)
        for payload in server.payloads:
            self.assertEqual(payload["sampling_params"], {
                "max_new_tokens": 1, "ignore_eos": True, "temperature": 0})
            self.assertTrue(payload["stream"])
        self.assertEqual(rows[0]["planned_completion_tokens"], 1)
        self.assertEqual(rows[0]["completion_tokens"], 1)
        self.assertIsNone(rows[0]["tpot_s"])
        self.assertIsNone(rows[0]["server_queue_s"])
        self.assertNotIn("input_ids", rows[0])
        self.assertNotIn("text", rows[0])
        self.assertNotIn("secret", json.dumps(rows))

    async def test_ttft_waits_for_first_actual_output_not_http_headers(self):
        server = await self.server(delay=.08)
        row = await self.send(server)
        self.assertEqual(row["status"], "ok")
        self.assertGreaterEqual(row["ttft_s"], .07)
        self.assertGreaterEqual(row["e2e_s"], row["ttft_s"])
        self.assertGreaterEqual(row["finished_epoch"], row["dispatch_epoch"])

    async def test_queue_and_retractions_never_abort(self):
        server = await self.server("large_queue")
        row = await self.send(server)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["server_queue_s"], 123.5)
        self.assertEqual(row["num_retractions"], 17)

    async def test_missing_or_malformed_optional_diagnostics_are_allowed(self):
        server = await self.server("invalid_optional")
        row = await self.send(server)
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["server_queue_s"])
        self.assertIsNone(row["num_retractions"])

    async def test_malformed_or_wrong_exact_counts_fail(self):
        for behavior, error in (("wrong_prompt", "prompt_token_count_mismatch"),
                                ("wrong_output", "completion_token_count_mismatch"),
                                ("boolean_output", "invalid_completion_token_count"),
                                ("missing_output", "completion_token_count_mismatch")):
            with self.subTest(behavior=behavior):
                server = await self.server(behavior)
                row = await self.send(server)
                self.assertEqual(row["status"], "error")
                self.assertEqual(row["error"], error)
                self.assertNotIn("secret", json.dumps(row))

    async def test_preempt_before_dispatch_sends_nothing(self):
        server = await self.server()
        async def stopped():
            return True
        row = await client.send_prefill(self.session, server.endpoint(), self.request(), stopped)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["error"], "stop_requested")
        self.assertIsNone(row["dispatch_epoch"])
        self.assertIsNone(row["e2e_s"])
        self.assertEqual(server.payloads, [])

    async def test_preempt_cancels_inflight_within_poll_bound(self):
        server = await self.server(delay=.3)
        stopped = asyncio.Event()
        async def stop_check():
            return stopped.is_set()
        pending = asyncio.create_task(client.send_prefill(
            self.session, server.endpoint(), self.request(), stop_check))
        await asyncio.wait_for(server.arrived.wait(), 1)
        began = time.monotonic()
        stopped.set()
        row = await asyncio.wait_for(pending, .2)
        self.assertLess(time.monotonic() - began, .15)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["completion_tokens"], 0)
        self.assertIsNone(row["ttft_s"])
        self.assertEqual(len(server.payloads), 1)

    async def test_checker_failure_fails_closed_without_inference(self):
        server = await self.server()
        async def failed():
            raise RuntimeError("secret guard receipt")
        row = await client.send_prefill(self.session, server.endpoint(), self.request(), failed)
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["error"], "stop_check_failed")
        self.assertEqual(server.payloads, [])
        self.assertNotIn("secret", json.dumps(row))

    async def test_external_cancel_returns_partial_row(self):
        server = await self.server(delay=.3)
        pending = asyncio.create_task(self.send(server))
        await asyncio.wait_for(server.arrived.wait(), 1)
        pending.cancel()
        row = await pending
        self.assertEqual(row["status"], "cancelled")
        self.assertIsNotNone(row["dispatch_epoch"])
        self.assertIsNone(row["ttft_s"])

    async def test_timeout_is_recorded_without_body(self):
        server = await self.server(delay=.2)
        row = await self.send(server, request_timeout_s=.02)
        self.assertEqual(row["status"], "error")
        self.assertIn("Timeout", row["error"])
        self.assertNotIn("secret", json.dumps(row))

    async def test_http_and_stream_errors_never_persist_response_text(self):
        for behavior, expected in (("http_error", "request_http_503"),
                                   ("server_error", "server_stream_error")):
            with self.subTest(behavior=behavior):
                server = await self.server(behavior)
                row = await self.send(server)
                self.assertEqual(row["status"], "error")
                self.assertEqual(row["error"], expected)
                self.assertNotIn("secret", json.dumps(row))

    async def test_invalid_inputs_never_send(self):
        server = await self.server()
        for request in (dict(self.request(), input_ids=[]),
                        dict(self.request(), input_ids=[True]),
                        dict(self.request(), planned_prompt_tokens=9)):
            row = await client.send_prefill(self.session, server.endpoint(), request, self.never_stop)
            self.assertEqual(row["status"], "error")
        self.assertEqual(server.payloads, [])


if __name__ == "__main__":
    unittest.main()
