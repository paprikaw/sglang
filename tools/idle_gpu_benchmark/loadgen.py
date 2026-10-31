"""Open-loop SGLang streaming benchmark client. No model processes are managed here.

``run_stage`` sends a fixed trace, never gates arrivals on completions, and writes
private request/telemetry JSONL plus a summary. Token IDs and generated text are
never persisted. A passing stage means no queue was observed inside the sampled
window and every exact per-request queue time was within the configured allowance;
it is not a mathematical guarantee that no transient queue ever existed.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, AsyncIterator, Callable

try:
    import aiohttp
except ImportError:  # Helpers remain importable with a lightweight system Python.
    aiohttp = None


class TelemetryError(ValueError):
    """Required evidence is absent, stale under load, or inconsistent."""


def percentiles(values: list[float]) -> dict[str, Any]:
    """Linear interpolated sample quantiles, in the same units as ``values``."""
    clean = sorted(float(value) for value in values if math.isfinite(value))
    if not clean:
        return {"count": 0, "mean": None, "min": None, "p50": None,
                "p90": None, "p95": None, "p99": None, "max": None}
    def quantile(p: float) -> float:
        position = (len(clean) - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)
    return {"count": len(clean), "mean": sum(clean) / len(clean),
            "min": clean[0], **{f"p{p}": quantile(p / 100) for p in (50, 90, 95, 99)},
            "max": clean[-1]}


def build_schedule(endpoints: list[dict], requests: list[dict], total_rps: float,
                   *, preserve_offsets: bool = False,
                   duration_s: float | None = None) -> tuple[list[dict], float]:
    """Build cyclic routing; equal timestamps preserve input order and burst size.

    By default the original timestamp gaps are scaled so N requests occupy a
    nominal N / target_rps window (the final request is at (N-1) / target_rps).
    For a trace with no positive span, all arrivals remain simultaneous. Workers
    receiving a coordinator's partition must set ``preserve_offsets=True``: local
    offsets then remain in the original *global* phase schedule.
    """
    if not endpoints or not requests:
        raise ValueError("at least one endpoint and request are required")
    if not math.isfinite(total_rps) or total_rps <= 0:
        raise ValueError("total_rps must be finite and positive")
    if len({str(endpoint.get("node")) for endpoint in endpoints}) != len(endpoints):
        raise ValueError("endpoint node identifiers must be unique")
    for endpoint in endpoints:
        if not all(endpoint.get(key) for key in ("node", "url", "model_id")):
            raise ValueError("each endpoint requires node, url, model_id")
    offsets = []
    for request in requests:
        offset = float(request["offset_s"])
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("offset_s must be finite and nonnegative")
        ids = request["input_ids"]
        if not isinstance(ids, list) or not ids or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("input_ids must be a nonempty list of nonnegative integers")
        output = request["output_tokens"]
        if isinstance(output, bool) or not isinstance(output, int) or output < 1:
            raise ValueError("output_tokens must be a positive integer")
        offsets.append(offset)
    if offsets != sorted(offsets):
        raise ValueError("requests must be ordered by offset_s")
    if preserve_offsets:
        scaled = offsets
    elif offsets[-1] > offsets[0]:
        factor = (len(requests) - 1) / (total_rps * (offsets[-1] - offsets[0]))
        scaled = [(offset - offsets[0]) * factor for offset in offsets]
    else:
        scaled = [0.0] * len(requests)
    window = float(duration_s) if duration_s is not None else max(
        len(requests) / total_rps, scaled[-1] + 1 / total_rps)
    if not math.isfinite(window) or window <= 0 or scaled[-1] > window:
        raise ValueError("duration_s must be positive and contain the entire dispatch schedule")
    return [{"index": index, "endpoint": endpoints[index % len(endpoints)],
             "request": request, "offset_s": offset}
            for index, (request, offset) in enumerate(zip(requests, scaled))], window


async def iter_sse(content: Any, *, max_event_bytes: int = 4 * 1024 * 1024) -> AsyncIterator[str]:
    """Decode SSE from arbitrarily fragmented bytes, including multiline data."""
    buffered = b""
    data_lines: list[bytes] = []
    event_bytes = 0
    async for chunk in content.iter_any():
        buffered += chunk
        if len(buffered) + event_bytes > max_event_bytes:
            raise ValueError("SSE event exceeds configured maximum")
        while b"\n" in buffered:
            line, buffered = buffered.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                if data_lines:
                    yield b"\n".join(data_lines).decode("utf-8")
                data_lines, event_bytes = [], 0
            elif line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
                event_bytes += len(value)
    if buffered:
        line = buffered.rstrip(b"\r")
        if line.startswith(b"data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(b" ") else value)
    if data_lines:
        yield b"\n".join(data_lines).decode("utf-8")


_METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_queue_histogram(text: str, tolerance_s: float, *, model_id: str | None = None) -> dict:
    """Read only TP rank zero, PP rank zero queue histograms, not sum across TP.

    The tolerance must have an exact histogram boundary. Choosing a looser bucket
    would silently accept unobserved waits. Multiple priority/DP series are summed
    only after each matching count and bucket has been verified.
    """
    series: dict[tuple, dict] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        if not (name.endswith("queue_time_seconds_count") or name.endswith("queue_time_seconds_bucket")):
            continue
        labels = {key: json.loads('"' + value + '"') for key, value in _LABEL_RE.findall(raw_labels or "")}
        if labels.get("tp_rank") != "0" or labels.get("pp_rank", "0") != "0":
            continue
        if model_id is not None and labels.get("model_name") != model_id:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise TelemetryError("nonfinite or negative queue histogram counter")
        key = tuple(sorted((label, val) for label, val in labels.items() if label != "le"))
        item = series.setdefault(key, {})
        if name.endswith("_count"):
            if "count" in item:
                raise TelemetryError("duplicate queue histogram count series")
            item["count"] = value
        elif math.isclose(float(labels.get("le", "nan")), tolerance_s, rel_tol=1e-9, abs_tol=1e-12):
            if "within_tolerance_count" in item:
                raise TelemetryError("duplicate queue histogram bucket series")
            item["within_tolerance_count"] = value
    if not series or any(set(item) != {"count", "within_tolerance_count"} for item in series.values()):
        raise TelemetryError("missing TP0 queue histogram count or exact tolerance bucket")
    if any(item["within_tolerance_count"] > item["count"] for item in series.values()):
        raise TelemetryError("queue histogram bucket exceeds count")
    return {"count": sum(item["count"] for item in series.values()),
            "within_tolerance_count": sum(item["within_tolerance_count"] for item in series.values()),
            "series": len(series), "tp_rank": 0, "tolerance_s": tolerance_s}


def parse_loads(data: Any, *, now: float, max_age_s: float) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("loads"), list) or not data["loads"]:
        raise TelemetryError("missing scheduler load snapshots")
    loads, ranks = [], set()
    for raw in data["loads"]:
        if not isinstance(raw, dict):
            raise TelemetryError("malformed scheduler snapshot")
        try:
            timestamp = float(raw["timestamp"])
            waiting, running = raw["num_waiting_reqs"], raw["num_running_reqs"]
            rank = int(raw["dp_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryError("scheduler timestamp or queue fields missing") from exc
        if rank in ranks:
            raise TelemetryError("duplicate DP snapshot")
        ranks.add(rank)
        if (not math.isfinite(timestamp) or timestamp <= 0 or timestamp > now + max_age_s
                or isinstance(waiting, bool) or not isinstance(waiting, int) or waiting < 0
                or isinstance(running, bool) or not isinstance(running, int) or running < 0):
            raise TelemetryError("invalid scheduler queue snapshot")
        age = max(0.0, now - timestamp)
        max_running = raw.get("max_running_requests", 0)
        retracted = (raw.get("queues") or {}).get("retracted", 0)
        if (isinstance(max_running, bool) or not isinstance(max_running, int) or max_running < 0
                or isinstance(retracted, bool) or not isinstance(retracted, int) or retracted < 0):
            raise TelemetryError("invalid scheduler capacity or retraction field")
        loads.append({"dp_rank": rank, "timestamp": timestamp, "age_s": age,
                      "fresh": age <= max_age_s, "num_waiting_reqs": waiting,
                      "num_running_reqs": running, "max_running_requests": max_running,
                      "retracted": retracted})
    return {"loads": loads, "all_fresh": all(item["fresh"] for item in loads),
            "num_waiting_reqs": sum(item["num_waiting_reqs"] for item in loads),
            "num_running_reqs": sum(item["num_running_reqs"] for item in loads),
            "max_running_requests": sum(item["max_running_requests"] for item in loads),
            "retracted": sum(item["retracted"] for item in loads)}


def update_backlog_evidence(state: dict, loads: dict, *, grace_s: float) -> list[str]:
    """Use distinct fresh scheduler snapshots to establish persistent/growing queue."""
    if not loads["all_fresh"]:
        return []
    snapshot_epoch = min(item["timestamp"] for item in loads["loads"])
    if snapshot_epoch <= state.get("last_backlog_snapshot_epoch", -1):
        return []
    state["last_backlog_snapshot_epoch"] = snapshot_epoch
    waiting, running, capacity = (loads[key] for key in
        ("num_waiting_reqs", "num_running_reqs", "max_running_requests"))
    if waiting == 0:
        state["waiting_started_epoch"] = None
        state["backlog_history"] = []
        return []
    if state.get("waiting_started_epoch") is None:
        state["waiting_started_epoch"] = snapshot_epoch
    sustained = max(0.0, snapshot_epoch - state["waiting_started_epoch"])
    state["max_sustained_waiting_s"] = max(state.get("max_sustained_waiting_s", 0.0), sustained)
    history = state.setdefault("backlog_history", [])
    history.append({"timestamp": snapshot_epoch, "waiting": waiting,
                    "slots_full": capacity > 0 and running >= capacity})
    del history[:-3]
    reasons = []
    if sustained >= grace_s:
        reasons.append("sustained_waiting_backlog")
    if (len(history) == 3 and all(item["slots_full"] for item in history)
            and history[-1]["waiting"] - history[0]["waiting"] >= 2
            and all(newer["waiting"] >= older["waiting"] for older, newer in zip(history, history[1:]))):
        reasons.append("growing_waiting_backlog_at_full_slots")
    return reasons


def _private_file(path: Path):
    descriptor = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _write_json(path: Path, data: dict):
    temporary = path.with_name(path.name + ".tmp")
    with _private_file(temporary) as handle:
        json.dump(data, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _record(handle: Any, data: dict):
    handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    handle.flush()


async def run_stage(endpoints: list[dict], requests: list[dict], total_rps: float,
                    stage_dir: Path, stop_check: Callable | None = None,
                    duration_s: float | None = None, *, start_at: float | None = None,
                    preserve_offsets: bool = False, queue_wait_tolerance_s: float = 0.005,
                    max_in_flight: int = 1024, request_timeout_s: float = 600.0,
                    telemetry_interval_s: float = 0.1, telemetry_max_age_s: float = 2.0,
                    max_dispatch_lag_s: float = 0.05, telemetry_timeout_s: float = 2.0,
                    queue_policy: str = "strict", backlog_grace_s: float = 2.0,
                    max_request_queue_s: float | None = None) -> dict:
    """Execute one fixed open-loop phase, promptly cancelling on stop/preemption.

    ``start_at`` is a future wall-clock epoch shared by distributed workers.
    ``preserve_offsets`` keeps a partition's global offsets verbatim. A True
    ``stop_check()`` means stop; it may be synchronous or awaitable. Native server
    ``meta_info`` must report exact planned prompt/completion lengths and queue
    time. The strict policy rejects every observed wait above the allowance;
    no_backlog reports brief waits and rejects persistent or growing queue.
    Requests, snapshots, and TP0 histogram deltas remain mandatory. Any missing
    evidence, failed request, excessive dispatch lag or safety bound invalidates
    the phase and prevents further arrivals; already dispatched requests cancel.
    """
    if aiohttp is None:
        raise RuntimeError("aiohttp is required; run with the model-serving Python environment")
    if queue_policy not in ("strict", "no_backlog"):
        raise ValueError("queue_policy must be strict or no_backlog")
    if not math.isfinite(backlog_grace_s) or backlog_grace_s <= 0:
        raise ValueError("backlog_grace_s must be finite and positive")
    if max_request_queue_s is not None and (not math.isfinite(max_request_queue_s) or max_request_queue_s <= 0):
        raise ValueError("max_request_queue_s must be finite and positive")
    if queue_wait_tolerance_s < 0 or not math.isfinite(queue_wait_tolerance_s):
        raise ValueError("queue_wait_tolerance_s must be finite and nonnegative")
    if max_in_flight < 1 or min(request_timeout_s, telemetry_interval_s, telemetry_max_age_s,
                               max_dispatch_lag_s, telemetry_timeout_s) <= 0:
        raise ValueError("client limits and intervals must be positive")
    schedule, window = build_schedule(endpoints, requests, total_rps,
                                     preserve_offsets=preserve_offsets, duration_s=duration_s)
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    loop = asyncio.get_running_loop()
    wall_origin, clock_origin = time.time(), loop.time()
    phase_wall = float(start_at) if start_at is not None else wall_origin
    phase_clock = clock_origin + phase_wall - wall_origin
    if not math.isfinite(phase_wall):
        raise ValueError("start_at must be a finite wall-clock epoch")
    stop_event, reasons, rows = asyncio.Event(), [], []
    external_stop = False
    dispatched_count, last_progress_clock = 0, -math.inf
    first_dispatch_clock: float | None = None
    request_tasks: set[asyncio.Task] = set()
    pending_by_node = Counter()
    telemetry: dict[str, dict] = {
        str(endpoint["node"]): {"samples": 0, "fresh_samples": 0, "fresh_waiting_samples": 0,
             "max_waiting_reqs": 0, "baseline": None, "last": None, "max_above_tolerance_delta": 0.0,
             "errors": [], "last_dp_timestamps": {}, "latest_waiting_reqs": None,
             "latest_snapshot_fresh": False, "waiting_started_epoch": None,
             "max_sustained_waiting_s": 0.0, "backlog_history": []}
        for endpoint in endpoints}
    request_handle = _private_file(directory / "requests.jsonl")
    telemetry_handle = _private_file(directory / "telemetry.jsonl")

    def invalidate(reason: str):
        if reason not in reasons:
            reasons.append(reason)
        stop_event.set()

    def publish_progress(*, force: bool = False, status: str | None = None):
        nonlocal last_progress_clock
        now = loop.time()
        if not force and now - last_progress_clock < 0.5:
            return
        last_progress_clock = now
        elapsed = max(0.0, now - phase_clock)
        successful = sum(row["status"] == "ok" for row in rows)
        _write_json(directory / "progress.json", {
            "schema_version": 1, "timestamp": time.time(),
            "status": status or ("stopping" if stop_event.is_set() else
                                  "waiting_start" if now < phase_clock else "running"),
            "phase_start_epoch": phase_wall, "target_rps": total_rps,
            "dispatch_window_s": window, "elapsed_s": elapsed,
            "offered_requests": len(schedule), "dispatched_requests": dispatched_count,
            "completed_requests": len(rows), "successful_requests": successful,
            "inflight": sum(pending_by_node.values()),
            "queue_waiting": {node: state["latest_waiting_reqs"] for node, state in telemetry.items()},
            "queue_sample_fresh": {node: state["latest_snapshot_fresh"] for node, state in telemetry.items()},
            "max_queue": {node: state["max_waiting_reqs"] for node, state in telemetry.items()},
            "queue_policy": queue_policy, "valid_so_far": not reasons,
            "reasons": list(reasons), "completed_rps": successful / elapsed if elapsed else 0.0})

    async def interruptible_until(deadline: float) -> bool:
        while not stop_event.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return True
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=min(0.05, remaining))
            except asyncio.TimeoutError:
                pass
        return False

    async def monitor_stop():
        nonlocal external_stop
        while not stop_event.is_set():
            if stop_check is not None:
                try:
                    answer = stop_check()
                    if inspect.isawaitable(answer):
                        answer = await answer
                    if answer:
                        external_stop = True
                        invalidate("stop_requested_or_preempted")
                        return
                except Exception as exc:
                    invalidate("stop_check_failed:" + type(exc).__name__)
                    return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

    async def collect_one(session: Any, endpoint: dict):
        node, base = str(endpoint["node"]), endpoint["url"].rstrip("/")
        state = telemetry[node]
        sample = {"node": node, "sample_epoch": time.time()}
        try:
            timeout = aiohttp.ClientTimeout(total=telemetry_timeout_s)
            async def get_loads():
                async with session.get(base + "/v1/loads", timeout=timeout) as response:
                    if response.status != 200:
                        raise TelemetryError("loads_http_" + str(response.status))
                    return await response.json()
            async def get_metrics():
                async with session.get(base + "/metrics", timeout=timeout) as response:
                    if response.status != 200:
                        raise TelemetryError("metrics_http_" + str(response.status))
                    return await response.text()
            raw_loads, raw_metrics = await asyncio.gather(get_loads(), get_metrics())
            loads = parse_loads(raw_loads, now=time.time(), max_age_s=telemetry_max_age_s)
            histogram = parse_queue_histogram(raw_metrics, queue_wait_tolerance_s,
                                              model_id=str(endpoint["model_id"]))
            state["samples"] += 1
            state["latest_waiting_reqs"] = loads["num_waiting_reqs"]
            state["latest_snapshot_fresh"] = loads["all_fresh"]
            if loads["all_fresh"]:
                state["fresh_samples"] += 1
                state["max_waiting_reqs"] = max(state["max_waiting_reqs"], loads["num_waiting_reqs"])
                if loads["num_waiting_reqs"]:
                    state["fresh_waiting_samples"] += 1
                    if queue_policy == "strict":
                        invalidate("queue_snapshot:" + node)
                if queue_policy == "no_backlog":
                    for reason in update_backlog_evidence(state, loads, grace_s=backlog_grace_s):
                        invalidate(reason + ":" + node)
            elif (pending_by_node[node] and first_dispatch_clock is not None
                  and loop.time() - first_dispatch_clock > telemetry_max_age_s):
                invalidate("stale_load_snapshot_under_traffic:" + node)
            for item in loads["loads"]:
                key = str(item["dp_rank"])
                previous = state["last_dp_timestamps"].get(key)
                if previous is not None and item["timestamp"] < previous:
                    raise TelemetryError("scheduler snapshot timestamp moved backwards")
                state["last_dp_timestamps"][key] = item["timestamp"]
            if state["last"] is not None and any(
                    histogram[key] < state["last"][key] - 1e-8
                    for key in ("count", "within_tolerance_count")):
                raise TelemetryError("queue histogram counter reset")
            if state["baseline"] is None:
                state["baseline"] = histogram
            state["last"] = histogram
            baseline = state["baseline"]
            count_delta = histogram["count"] - baseline["count"]
            within_delta = histogram["within_tolerance_count"] - baseline["within_tolerance_count"]
            above_delta = count_delta - within_delta
            if above_delta < -1e-8:
                raise TelemetryError("inconsistent queue histogram counter deltas")
            state["max_above_tolerance_delta"] = max(state["max_above_tolerance_delta"], above_delta)
            if above_delta > 1e-8 and queue_policy == "strict":
                invalidate("queue_histogram_above_tolerance:" + node)
            sample.update(loads=loads, histogram=histogram,
                          histogram_delta={"count": count_delta, "within_tolerance_count": within_delta,
                                           "above_tolerance_count": max(0.0, above_delta)})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Error types/context only: response bodies may contain private model text.
            error = str(exc) if isinstance(exc, TelemetryError) else type(exc).__name__
            state["errors"].append(error)
            sample["error"] = error
            invalidate("telemetry_failed:" + node + ":" + error)
        _record(telemetry_handle, sample)
        publish_progress()

    async def monitor_telemetry(session: Any):
        while not stop_event.is_set():
            await asyncio.gather(*(collect_one(session, endpoint) for endpoint in endpoints))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=telemetry_interval_s)
            except asyncio.TimeoutError:
                pass

    async def send_one(session: Any, entry: dict):
        nonlocal first_dispatch_clock
        endpoint, request = entry["endpoint"], entry["request"]
        node, dispatch_clock = str(endpoint["node"]), loop.time()
        if first_dispatch_clock is None:
            first_dispatch_clock = dispatch_clock
        pending_by_node[node] += 1
        row = {"index": entry["index"], "sample_id": str(request.get("sample_id", entry["index"])),
               "node": node, "planned_offset_s": entry["offset_s"],
               "dispatch_epoch": time.time(), "dispatch_lag_s": max(0.0, dispatch_clock - phase_clock - entry["offset_s"]),
               "planned_prompt_tokens": len(request["input_ids"]), "planned_completion_tokens": request["output_tokens"],
               "status": "error", "prompt_tokens": None, "completion_tokens": 0,
               "server_queue_s": None, "num_retractions": 0, "ttft_s": None, "tpot_s": None, "e2e_s": None}
        first_token_clock = last_token_clock = None
        token_count, terminated, finish_reason = 0, "eof", None
        try:
            if stop_event.is_set():
                raise asyncio.CancelledError
            if row["dispatch_lag_s"] > max_dispatch_lag_s:
                raise TelemetryError("client_dispatch_lag")
            payload = {"input_ids": request["input_ids"], "sampling_params": {
                "max_new_tokens": request["output_tokens"], "ignore_eos": True, "temperature": 0},
                "stream": True, "return_logprob": False}
            async with session.post(endpoint["url"].rstrip("/") + "/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=request_timeout_s)) as response:
                row["http_status"] = response.status
                if response.status != 200:
                    raise TelemetryError("request_http_" + str(response.status))
                async for data in iter_sse(response.content):
                    if data.strip() == "[DONE]":
                        terminated = "done"
                        break
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise TelemetryError("stream_event_not_object")
                    if "error" in event:
                        raise TelemetryError("server_stream_error")
                    meta = event.get("meta_info")
                    if not isinstance(meta, dict):
                        continue  # Keepalive/header events do not produce a token.
                    count = meta.get("completion_tokens")
                    if count is not None:
                        if isinstance(count, bool) or not isinstance(count, int) or count < token_count:
                            raise TelemetryError("nonmonotonic_completion_tokens")
                        if count > token_count:
                            now = loop.time()
                            if first_token_clock is None:
                                first_token_clock = now
                            last_token_clock = now
                            token_count = count
                    if "prompt_tokens" in meta:
                        prompt_tokens = meta["prompt_tokens"]
                        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens != len(request["input_ids"]):
                            raise TelemetryError("prompt_token_count_mismatch")
                        row["prompt_tokens"] = prompt_tokens
                    if "queue_time" in meta:
                        queue_time = float(meta["queue_time"])
                        if not math.isfinite(queue_time) or queue_time < 0:
                            raise TelemetryError("invalid_server_queue_time")
                        row["server_queue_s"] = max(row["server_queue_s"] or 0.0, queue_time)
                    if "num_retractions" in meta:
                        retractions = meta["num_retractions"]
                        if isinstance(retractions, bool) or not isinstance(retractions, int) or retractions < 0:
                            raise TelemetryError("invalid_server_retractions")
                        row["num_retractions"] = max(row["num_retractions"], retractions)
                        if retractions:
                            raise TelemetryError("server_retractions")
                    if meta.get("finish_reason") is not None:
                        finish_reason = meta["finish_reason"]
                row["e2e_s"] = loop.time() - dispatch_clock
            row["stream_termination"] = terminated
            row["completion_tokens"] = token_count
            if first_token_clock is None:
                raise TelemetryError("stream_has_no_output_tokens")
            if row["prompt_tokens"] != len(request["input_ids"]):
                raise TelemetryError("prompt_token_count_mismatch")
            if token_count != request["output_tokens"]:
                raise TelemetryError("completion_token_count_mismatch")
            if row["server_queue_s"] is None:
                raise TelemetryError("missing_server_queue_time")
            if queue_policy == "strict" and row["server_queue_s"] > queue_wait_tolerance_s + 1e-12:
                raise TelemetryError("server_queue_above_tolerance")
            if max_request_queue_s is not None and row["server_queue_s"] > max_request_queue_s + 1e-12:
                raise TelemetryError("server_queue_above_maximum")
            if isinstance(finish_reason, dict) and finish_reason.get("type") in ("abort", "error"):
                raise TelemetryError("server_finish_aborted")
            row["ttft_s"] = first_token_clock - dispatch_clock
            row["tpot_s"] = ((last_token_clock - first_token_clock) / (token_count - 1)
                             if token_count > 1 else None)
            row["status"] = "ok"
        except asyncio.CancelledError:
            row["status"], row["error"] = "cancelled", "stage_stopped"
            raise
        except Exception as exc:
            row["error"] = str(exc) if isinstance(exc, TelemetryError) else type(exc).__name__
            invalidate("request_failed:" + node + ":" + row["error"])
        finally:
            row["completion_tokens"] = token_count
            if row["e2e_s"] is None:
                row["e2e_s"] = loop.time() - dispatch_clock
            if first_token_clock is not None:
                row["ttft_s"] = first_token_clock - dispatch_clock
                row["tpot_s"] = ((last_token_clock - first_token_clock) / (token_count - 1)
                                 if token_count > 1 else None)
            row["finished_epoch"] = time.time()
            rows.append(row)
            _record(request_handle, row)
            pending_by_node[node] -= 1

    publish_progress(force=True)
    monitor_tasks = []
    was_cancelled = False
    try:
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector, trust_env=False) as session:
            monitor_tasks.append(asyncio.create_task(monitor_stop()))
            # Establish both independent telemetry baselines before any traffic.
            await asyncio.gather(*(collect_one(session, endpoint) for endpoint in endpoints))
            if start_at is None:
                # Baseline HTTP setup is preparation, not missed scheduled traffic.
                phase_clock, phase_wall = loop.time() + 0.01, time.time() + 0.01
            monitor_tasks.append(asyncio.create_task(monitor_telemetry(session)))
            for entry in schedule:
                if not await interruptible_until(phase_clock + entry["offset_s"]):
                    break
                if len(request_tasks) >= max_in_flight:
                    invalidate("client_in_flight_safety_bound")
                    break
                lag = loop.time() - phase_clock - entry["offset_s"]
                if lag > max_dispatch_lag_s:
                    invalidate("client_dispatch_lag")
                    break  # Never bunch overdue requests into a catch-up burst.
                task = asyncio.create_task(send_one(session, entry))
                dispatched_count += 1
                request_tasks.add(task)
                task.add_done_callback(request_tasks.discard)
                # Yield only to actually dispatch; no response/concurrency semaphore.
                await asyncio.sleep(0)
            if not stop_event.is_set():
                await interruptible_until(phase_clock + window)
            while request_tasks and not stop_event.is_set():
                await asyncio.wait(request_tasks, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
            if stop_event.is_set():
                for task in list(request_tasks):
                    task.cancel()
                await asyncio.gather(*list(request_tasks), return_exceptions=True)
            for task in monitor_tasks:
                task.cancel()
            await asyncio.gather(*monitor_tasks, return_exceptions=True)
            monitor_tasks.clear()
            # Final metric snapshot includes the last completed request's counter.
            if not external_stop:
                await asyncio.gather(*(collect_one(session, endpoint) for endpoint in endpoints))
    except asyncio.CancelledError:
        was_cancelled = True
        external_stop = True
        invalidate("stage_task_cancelled")
    except Exception as exc:
        invalidate("stage_exception:" + type(exc).__name__)
    finally:
        for task in list(request_tasks) + monitor_tasks:
            task.cancel()
        await asyncio.gather(*list(request_tasks), *monitor_tasks, return_exceptions=True)
        request_handle.close()
        telemetry_handle.close()

    elapsed = max(0.0, loop.time() - phase_clock)
    successes = [row for row in rows if row["status"] == "ok"]
    for endpoint in endpoints:
        node, state = str(endpoint["node"]), telemetry[str(endpoint["node"])]
        if not state["samples"] or not state["fresh_samples"]:
            if "missing_fresh_load_evidence:" + node not in reasons:
                reasons.append("missing_fresh_load_evidence:" + node)
        if state["baseline"] is None or state["last"] is None:
            reasons.append("missing_histogram_evidence:" + node)
        elif state["last"]["count"] - state["baseline"]["count"] + 1e-8 < sum(
                row["node"] == node for row in successes):
            reasons.append("insufficient_histogram_request_count:" + node)
    if len(rows) != len(schedule) and not reasons:
        reasons.append("dispatch_schedule_incomplete")
    def aggregate(records: list[dict], planned_count: int) -> dict:
        successful = [row for row in records if row["status"] == "ok"]
        return {"offered_requests": planned_count, "dispatched_requests": len(records),
                "successful_requests": len(successful),
                "failed_requests": sum(row["status"] == "error" for row in records),
                "cancelled_requests": sum(row["status"] == "cancelled" for row in records),
                "undispatched_requests": planned_count - len(records),
                "offered_rps": planned_count / window,
                "dispatched_rps": len(records) / window,
                "completed_rps": len(successful) / elapsed if elapsed else 0.0,
                "latency_s": {metric: percentiles([row[metric + "_s"] for row in successful
                                                    if row[metric + "_s"] is not None])
                              for metric in ("ttft", "tpot", "e2e", "server_queue")},
                "dispatch_lag_s": percentiles([row["dispatch_lag_s"] for row in records])}
    summary = {"schema_version": 1, "valid": not reasons,
               "status": "stopped" if external_stop else ("invalid" if reasons else "complete"),
               "reasons": reasons, "target_total_rps": total_rps,
               "phase_start_epoch": phase_wall, "finished_epoch": time.time(),
               "dispatch_window_s": window, "elapsed_s": elapsed,
               "preserve_offsets": preserve_offsets,
               "queue_wait_tolerance_s": queue_wait_tolerance_s,
               "queue_policy": queue_policy, "backlog_grace_s": backlog_grace_s,
               "max_request_queue_s": max_request_queue_s,
               "client_requests_drained": len(rows) == len(schedule) and all(row["status"] == "ok" for row in rows),
               "telemetry_interval_s": telemetry_interval_s,
               "telemetry_max_age_s": telemetry_max_age_s,
               "max_dispatch_lag_s": max_dispatch_lag_s,
               "max_in_flight": max_in_flight,
               "queue_interpretation": (
                   "No queue observed in this test window: all exact request queue_time values <= allowance, fresh waiting samples zero, TP0 histogram has zero above-allowance delta. Not a guarantee about unseen transient queues."
                   if queue_policy == "strict" else
                   "No sustained or growing backlog observed in this test window; brief admission waits are allowed and reported. Distinct fresh snapshots must show neither continuously positive waiting for backlog_grace_s nor monotonic queue growth >=2 over 3 snapshots with all running slots full. Every planned request must finish; retractions and missing evidence invalidate. Not a guarantee about unseen queues."),
               **aggregate(rows, len(schedule)),
               "per_node": {str(endpoint["node"]): aggregate(
                   [row for row in rows if row["node"] == str(endpoint["node"])],
                   sum(str(entry["endpoint"]["node"]) == str(endpoint["node"]) for entry in schedule))
                   for endpoint in endpoints},
               "telemetry": telemetry}
    _write_json(directory / "summary.json", summary)
    publish_progress(force=True, status=summary["status"])
    if was_cancelled:
        raise asyncio.CancelledError
    return summary
