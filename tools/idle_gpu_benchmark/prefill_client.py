"""One cancellable SGLang prefill request; callers send the next on completion.

Output is fixed to one token, so TPOT is undefined. TTFT and E2E include waiting.
Queue waits and retractions are diagnostics only and never gate this workload.
Neither token IDs nor request/output text are included in measurement rows.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Awaitable, Callable

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from .loadgen import iter_sse
except ImportError:
    from loadgen import iter_sse


class _MeasurementError(ValueError):
    """A fixed safe error code, never a server response or request body."""


async def send_prefill(session: Any, endpoint: dict, request: dict,
                       stop_check: Callable[[], Awaitable[bool]],
                       request_timeout_s: float = 600) -> dict:
    """Send one native request, returning status ok/error/cancelled and timings.

    request requires sample_id and input_ids. Optional prompt_tokens and
    planned_prompt_tokens must match the actual input length. Await calls
    sequentially for one in-flight request per instance without an RPS limit.

    stop_check is awaited immediately before POST and every 50 ms in flight.
    A truthy result or checker exception cancels promptly. External cancellation
    also returns a cancelled row so a worker can persist partial measurements.
    """
    loop = asyncio.get_running_loop()
    row = {
        "sample_id": str(request.get("sample_id", "")),
        "node": str(endpoint.get("node", "")),
        "planned_prompt_tokens": None, "planned_completion_tokens": 1,
        "prompt_tokens": None, "completion_tokens": 0,
        "ttft_s": None, "tpot_s": None, "e2e_s": None,
        "server_queue_s": None, "num_retractions": None,
        "dispatch_epoch": None, "finished_epoch": None, "status": "error",
    }
    dispatch_clock = first_token_clock = None
    token_count = 0
    stop_reason = "stop_requested"
    http_task = watcher_task = None

    async def should_stop() -> bool:
        nonlocal stop_reason
        try:
            return bool(await stop_check())
        except asyncio.CancelledError:
            raise
        except Exception:
            stop_reason = "stop_check_failed"
            return True

    async def perform() -> None:
        nonlocal dispatch_clock, first_token_clock, token_count
        # No await separates the final ownership check from constructing POST.
        if await should_stop():
            raise asyncio.CancelledError
        dispatch_clock = loop.time()
        row["dispatch_epoch"] = time.time()
        payload = {"input_ids": ids, "sampling_params": {
            "max_new_tokens": 1, "ignore_eos": True, "temperature": 0},
            "stream": True, "return_logprob": False}
        async with session.post(endpoint["url"].rstrip("/") + "/generate",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=request_timeout_s)) as response:
            row["http_status"] = response.status
            if response.status != 200:
                raise _MeasurementError("request_http_" + str(response.status))
            finish_reason = None
            async for data in iter_sse(response.content):
                if data.strip() == "[DONE]":
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise _MeasurementError("stream_event_not_object")
                if "error" in event:
                    raise _MeasurementError("server_stream_error")
                meta = event.get("meta_info")
                if not isinstance(meta, dict):
                    continue
                if "completion_tokens" in meta:
                    count = meta["completion_tokens"]
                    if (isinstance(count, bool) or not isinstance(count, int)
                            or count < token_count):
                        raise _MeasurementError("invalid_completion_token_count")
                    if count > token_count and first_token_clock is None:
                        first_token_clock = loop.time()
                    token_count = count
                if "prompt_tokens" in meta:
                    count = meta["prompt_tokens"]
                    if (isinstance(count, bool) or not isinstance(count, int)
                            or count != len(ids)):
                        raise _MeasurementError("prompt_token_count_mismatch")
                    row["prompt_tokens"] = count
                # Optional diagnostics never participate in request acceptance.
                if "queue_time" in meta:
                    value = meta["queue_time"]
                    if (not isinstance(value, bool) and isinstance(value, (int, float))
                            and math.isfinite(value) and value >= 0):
                        row["server_queue_s"] = max(row["server_queue_s"] or 0, value)
                if "num_retractions" in meta:
                    value = meta["num_retractions"]
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        row["num_retractions"] = max(row["num_retractions"] or 0, value)
                if meta.get("finish_reason") is not None:
                    finish_reason = meta["finish_reason"]
            if row["prompt_tokens"] != len(ids):
                raise _MeasurementError("prompt_token_count_mismatch")
            if token_count != 1 or first_token_clock is None:
                raise _MeasurementError("completion_token_count_mismatch")
            if isinstance(finish_reason, dict) and finish_reason.get("type") in ("abort", "error"):
                raise _MeasurementError("server_finish_aborted")
        row["status"] = "ok"

    async def watch() -> None:
        while True:
            await asyncio.sleep(.05)
            if await should_stop():
                return

    try:
        ids = request.get("input_ids")
        if (not isinstance(ids, list) or not ids
                or any(isinstance(token, bool) or not isinstance(token, int)
                       or token < 0 for token in ids)):
            raise _MeasurementError("invalid_input_ids")
        row["planned_prompt_tokens"] = len(ids)
        for name in ("prompt_tokens", "planned_prompt_tokens"):
            if name in request and (isinstance(request[name], bool)
                                    or not isinstance(request[name], int)
                                    or request[name] != len(ids)):
                raise _MeasurementError("planned_prompt_token_count_mismatch")
        if not endpoint.get("node") or not isinstance(endpoint.get("url"), str):
            raise _MeasurementError("invalid_endpoint")
        if (isinstance(request_timeout_s, bool)
                or not isinstance(request_timeout_s, (int, float))
                or not math.isfinite(request_timeout_s) or request_timeout_s <= 0):
            raise _MeasurementError("invalid_request_timeout")
        if aiohttp is None:
            raise _MeasurementError("aiohttp_unavailable")
        http_task = asyncio.create_task(perform())
        watcher_task = asyncio.create_task(watch())
        done, _ = await asyncio.wait((http_task, watcher_task), return_when=asyncio.FIRST_COMPLETED)
        if watcher_task in done:
            await watcher_task
            http_task.cancel()
        await http_task
    except asyncio.CancelledError:
        row["status"], row["error"] = "cancelled", stop_reason
    except Exception as exc:
        row["error"] = str(exc) if isinstance(exc, _MeasurementError) else type(exc).__name__
    finally:
        tasks = [task for task in (http_task, watcher_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        row["completion_tokens"] = token_count
        row["finished_epoch"] = time.time()
        if dispatch_clock is not None:
            row["e2e_s"] = loop.time() - dispatch_clock
        if first_token_clock is not None:
            row["ttft_s"] = first_token_clock - dispatch_clock
    return row
