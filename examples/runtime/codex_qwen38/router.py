"""Loopback Codex router: local Qwen, unchanged official upstream for other models.

Credentials and prompt bodies are never logged. Only explicitly configured
upstreams are reachable. Qwen receives only the deployment's own API key.
"""
import asyncio
import base64
import copy
import hmac
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import zstandard
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("QWEN_MODEL", "orcarouter/Qwen3.8-27B-Uncensored")
LOCAL = os.environ.get("QWEN_LOCAL_URL", "http://127.0.0.1:30038")
LOCAL_KEY = os.environ.get("QWEN_LOCAL_API_KEY")
if not LOCAL_KEY:
    key_path = Path(os.environ.get("QWEN_LOCAL_API_KEY_FILE", ROOT / "api.key"))
    LOCAL_KEY = key_path.read_text().strip()
AUTH_PATH = Path(os.environ.get("CODEX_AUTH_PATH", Path.home() / ".codex" / "auth.json"))
HOP = {"host", "connection", "upgrade", "content-length", "transfer-encoding",
       "keep-alive", "proxy-authorization", "proxy-authenticate", "te", "trailer"}
QWEN_REASONING_EFFORT = os.environ.get("QWEN_REASONING_EFFORT", "low")
QWEN_TOOL_REASONING_EFFORT = os.environ.get("QWEN_TOOL_REASONING_EFFORT", "none")
QWEN_MAX_TOOL_OUTPUT_TOKENS = int(os.environ.get("QWEN_MAX_TOOL_OUTPUT_TOKENS", "2048"))
QWEN_MAX_FINAL_OUTPUT_TOKENS = int(os.environ.get("QWEN_MAX_FINAL_OUTPUT_TOKENS", "4096"))
LOCAL_COMPACTION_PREFIX = "local-qwen-v1:"
LOCAL_COMPACTION_PROMPT = """Create a compact handoff summary of the conversation so far.
Preserve the user's objective, constraints, decisions, files inspected or changed,
commands and test results, current state, unresolved errors, and exact next steps.
Do not call tools. Return only the handoff summary."""


def owned_token(headers):
    token = headers.get("authorization", "").removeprefix("Bearer ")
    try:
        auth = json.loads(AUTH_PATH.read_text())
    except (OSError, ValueError):
        return False
    valid = [LOCAL_KEY, auth.get("OPENAI_API_KEY"), auth.get("tokens", {}).get("access_token")]
    return bool(token) and any(isinstance(v, str) and hmac.compare_digest(token, v) for v in valid)


def upstream(headers):
    # Preserve the authentication surface selected by Codex, not by model name.
    bearer = headers.get("authorization", "").removeprefix("Bearer ")
    if bearer.startswith("sk-"):
        return "https://api.openai.com/v1"
    return "https://chatgpt.com/backend-api/codex"


def filtered_headers(headers, websocket=False):
    return {k: v for k, v in headers.items()
            if k.lower() not in HOP
            and not (websocket and k.lower().startswith("sec-websocket-"))}


def parse_body(raw, headers):
    if not raw:
        return {}
    if headers.get("content-encoding", "").lower() == "zstd":
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
            raw = reader.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("Request exceeds router's 32 MiB decoded limit")
    return json.loads(raw)


def request_shape(body):
    """Return useful request metadata without retaining prompt or tool content."""
    value = body.get("input")
    if isinstance(value, str):
        input_meta = {"kind": "string", "chars": len(value)}
    elif isinstance(value, list):
        input_meta = {
            "kind": "list",
            "items": len(value),
            "types": sorted({item.get("type", "") for item in value if isinstance(item, dict)}),
            "roles": sorted({item.get("role", "") for item in value if isinstance(item, dict)}),
        }
    else:
        input_meta = {"kind": type(value).__name__}
    return {
        "input": input_meta,
        "tools": len(body.get("tools", [])),
        "previous_response": bool(body.get("previous_response_id")),
        "requested_effort": (body.get("reasoning") or {}).get("effort")
        if isinstance(body.get("reasoning"), dict) else None,
    }


def backend_error_message(raw, status_code):
    try:
        value = json.loads(raw)
        error = value.get("error", value) if isinstance(value, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message:
            return message[:1000]
    except (TypeError, ValueError):
        pass
    return f"Local model request failed with HTTP {status_code}"


def encode_local_compaction(summary):
    return LOCAL_COMPACTION_PREFIX + base64.urlsafe_b64encode(summary.encode()).decode()


def decode_local_compaction(value):
    if not isinstance(value, str) or not value.startswith(LOCAL_COMPACTION_PREFIX):
        return None
    try:
        return base64.urlsafe_b64decode(value.removeprefix(LOCAL_COMPACTION_PREFIX)).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def response_text(response):
    chunks = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                text = content.get("text")
                if content.get("type") in ("output_text", "text") and isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def limited_output_tokens(requested, limit):
    """Honor a smaller client limit while bounding local runaway generations."""
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = limit
    return min(max(requested, 1), limit)


def local_compaction_events(adapter, completed):
    response = completed.get("response", {})
    summary = response_text(response)
    if not summary:
        return [{"type": "error", "code": "local_compaction_empty",
                 "message": "Local model returned an empty compaction summary",
                 "param": None, "sequence_number": adapter.sequence}]
    item = {
        "type": "compaction",
        "id": "cmp_local_" + uuid.uuid4().hex,
        "encrypted_content": encode_local_compaction(summary),
    }
    done = {"type": "response.output_item.done", "output_index": 0, "item": item,
            "sequence_number": adapter.sequence}
    final = copy.deepcopy(completed)
    final["sequence_number"] = adapter.sequence + 1
    final["response"]["output"] = [item]
    final["response"]["status"] = "completed"
    return [done, final]


class ToolAdapter:
    """Map Codex namespace/freeform tools onto SGLang's function-tool protocol."""
    def __init__(self, body):
        self.body = copy.deepcopy(body)
        self.by_wire = {}
        self.by_client = {}
        self.items = {}
        self.sequence = 0
        self.compaction_requested = False
        tools = []

        def add(tool, namespace=None):
            if tool.get("type") == "namespace":
                for nested in tool.get("tools", []):
                    add(nested, tool["name"])
                return
            if tool.get("type") not in ("function", "custom"):
                return
            wire = f"codex_tool_{len(tools)}"
            meta = {"name": tool["name"], "namespace": namespace, "custom": tool["type"] == "custom"}
            self.by_wire[wire] = meta
            self.by_client[(namespace, tool["name"])] = wire
            mapped = {"type": "function", "name": wire, "description": tool.get("description", ""), "strict": False}
            if meta["custom"]:
                mapped["description"] += "\nPass the complete raw tool input as the input string."
                mapped["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"], "additionalProperties": False}
            else:
                mapped["parameters"] = tool.get("parameters", {"type": "object", "properties": {}})
            tools.append(mapped)

        for tool in body.get("tools", []):
            add(tool)
        self.body["tools"] = tools
        self.body["parallel_tool_calls"] = False
        self.body["store"] = False
        requested_max_output_tokens = self.body.get("max_output_tokens")
        output_limit = (QWEN_MAX_TOOL_OUTPUT_TOKENS if tools
                        else QWEN_MAX_FINAL_OUTPUT_TOKENS)
        self.body["max_output_tokens"] = limited_output_tokens(
            requested_max_output_tokens, output_limit)
        reasoning = self.body.get("reasoning")
        if not isinstance(reasoning, dict):
            reasoning = {}
        reasoning_effort = (QWEN_TOOL_REASONING_EFFORT if tools
                            else QWEN_REASONING_EFFORT)
        self.body["reasoning"] = {**reasoning, "effort": reasoning_effort}
        self.body.pop("previous_response_id", None)
        choice = self.body.get("tool_choice")
        if isinstance(choice, dict) and "name" in choice:
            wire = self.by_client.get((choice.get("namespace"), choice["name"]))
            if wire:
                self.body["tool_choice"] = {"type": "function", "name": wire}
        if isinstance(self.body.get("input"), list):
            translated_input = []
            for item in self.body["input"]:
                kind = item.get("type")
                if kind == "compaction_trigger":
                    self.compaction_requested = True
                    continue
                if kind in ("compaction", "context_compaction"):
                    summary = decode_local_compaction(item.get("encrypted_content"))
                    if summary:
                        translated_input.append({
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text":
                                         "Local context handoff:\n" + summary}],
                        })
                    continue
                if kind in ("function_call", "custom_tool_call"):
                    wire = self.by_client.get((item.get("namespace"), item.get("name")))
                    if wire:
                        item["name"] = wire
                    item.pop("namespace", None)
                    if kind == "custom_tool_call":
                        item["type"] = "function_call"
                        item["arguments"] = json.dumps({"input": item.pop("input", "")})
                elif kind == "custom_tool_call_output":
                    item["type"] = "function_call_output"
                translated_input.append(item)
            if self.compaction_requested:
                translated_input.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": LOCAL_COMPACTION_PROMPT}],
                })
                self.body["tools"] = []
                self.body["tool_choice"] = "none"
                self.body["max_output_tokens"] = limited_output_tokens(
                    requested_max_output_tokens, QWEN_MAX_FINAL_OUTPUT_TOKENS)
            self.body["input"] = translated_input

    def item(self, item):
        item = copy.deepcopy(item)
        if item.get("type") != "function_call":
            return item
        meta = self.by_wire.get(item.get("name"))
        if not meta:
            return item
        self.items[item.get("id")] = meta
        item["name"] = meta["name"]
        if meta["namespace"]:
            item["namespace"] = meta["namespace"]
        if meta["custom"]:
            item["type"] = "custom_tool_call"
            arguments = item.pop("arguments", "")
            item["input"] = json.loads(arguments).get("input", "") if arguments else ""
        return item

    def response(self, response):
        response = copy.deepcopy(response)
        response["output"] = [self.item(item) for item in response.get("output", [])]
        return response

    def events(self, event):
        event = copy.deepcopy(event)
        kind = event.get("type", "")
        if "item" in event:
            event["item"] = self.item(event["item"])
        if "response" in event:
            event["response"] = self.response(event["response"])
        meta = self.items.get(event.get("item_id"), {})
        if meta.get("custom") and kind == "response.function_call_arguments.delta":
            return []  # JSON fragments are not raw freeform input.
        if meta.get("custom") and kind == "response.function_call_arguments.done":
            raw = json.loads(event["arguments"]).get("input", "")
            base = {"item_id": event["item_id"], "output_index": event["output_index"]}
            output = [{**base, "type": "response.custom_tool_call_input.delta", "delta": raw},
                      {**base, "type": "response.custom_tool_call_input.done", "input": raw}]
        else:
            output = [event]
        for value in output:
            value["sequence_number"] = self.sequence
            self.sequence += 1
        return output


@asynccontextmanager
async def lifespan(app):
    app.state.local = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), trust_env=False)
    app.state.remote = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20))
    yield
    await app.state.local.aclose()
    await app.state.remote.aclose()


app = FastAPI(lifespan=lifespan)


async def qwen_events(adapter):
    adapter.body["stream"] = True
    headers = {"Authorization": "Bearer " + LOCAL_KEY}
    started = time.monotonic()
    print(json.dumps({"event": "qwen_request_start", **request_shape(adapter.body)}), flush=True)
    async with app.state.local.stream("POST", LOCAL + "/v1/responses", json=adapter.body, headers=headers) as response:
        headers_elapsed = round(time.monotonic() - started, 3)
        if response.status_code != 200:
            raw = (await response.aread()).decode(errors="replace")
            message = backend_error_message(raw, response.status_code)
            print(json.dumps({"event": "qwen_request_failed", "status": response.status_code,
                              "headers_seconds": headers_elapsed}), flush=True)
            yield {"type": "error", "code": "local_model_error", "message": message,
                   "param": None, "sequence_number": adapter.sequence}
            adapter.sequence += 1
            return
        print(json.dumps({"event": "qwen_response_headers", "status": response.status_code,
                          "seconds": headers_elapsed}), flush=True)
        completed = None
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            raw_event = json.loads(line[6:])
            if adapter.compaction_requested:
                if raw_event.get("type") == "response.completed":
                    completed = raw_event
                continue
            for event in adapter.events(raw_event):
                yield event
        if adapter.compaction_requested:
            if completed is None:
                yield {"type": "error", "code": "local_compaction_incomplete",
                       "message": "Local compaction stream ended before response.completed",
                       "param": None, "sequence_number": adapter.sequence}
                return
            for event in local_compaction_events(adapter, completed):
                yield event


@app.get("/health")
async def health():
    return {"router": "ok"}


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def route(path: str, request: Request):
    if not owned_token(request.headers):
        return JSONResponse({"error": {"message": "Unauthorized", "type": "authentication_error"}}, status_code=401)
    raw = await request.body()
    try:
        body = parse_body(raw, request.headers)
    except (ValueError, zstandard.ZstdError):
        return JSONResponse({"error": {"message": "Invalid JSON or compression"}}, status_code=400)
    suffix = "/" + path.removeprefix("v1/")
    model = body.get("model")
    local = model == MODEL
    print(json.dumps({"transport": "http", "path": suffix, "model": model, "route": "qwen" if local else "openai", "tool_types": sorted({t.get("type", "") for t in body.get("tools", [])})}), flush=True)
    if local and suffix == "/responses":
        adapter = ToolAdapter(body)
        if body.get("stream"):
            async def events():
                async for event in qwen_events(adapter):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        response = await app.state.local.post(LOCAL + "/v1/responses", json=adapter.body, headers={"Authorization": "Bearer " + LOCAL_KEY})
        return JSONResponse(adapter.response(response.json()) if response.status_code == 200 else response.json(), status_code=response.status_code)
    if local:
        return JSONResponse({"error": {"message": "This local-model route only supports /responses"}}, status_code=400)
    if request.headers.get("authorization") == "Bearer " + LOCAL_KEY:
        return JSONResponse({"error": {"message": "Local credential cannot access OpenAI"}}, status_code=403)
    url = upstream(request.headers) + suffix
    if request.url.query:
        url += "?" + request.url.query
    outgoing = app.state.remote.build_request(request.method, url, headers=filtered_headers(request.headers), content=raw)
    response = await app.state.remote.send(outgoing, stream=True)
    headers = filtered_headers(response.headers)
    return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=headers, background=BackgroundTask(response.aclose))


@app.websocket("/{path:path}")
async def websocket_route(path: str, socket: WebSocket):
    if not owned_token(socket.headers):
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        first = await socket.receive_text()
        value = json.loads(first)
        payload = value.get("response", value)
        model = payload.get("model")
        print(json.dumps({"transport": "websocket", "path": path, "model": model, "route": "qwen" if model == MODEL else "openai"}), flush=True)
        if model == MODEL:
            history = []
            while True:
                body = dict(payload)
                body.pop("type", None)
                if body.get("previous_response_id") and isinstance(body.get("input"), list):
                    body["input"] = history + body["input"]
                adapter = ToolAdapter(body)
                async for event in qwen_events(adapter):
                    if event.get("type") == "response.completed":
                        incoming = body.get("input", [])
                        history = (incoming if isinstance(incoming, list) else [{"role": "user", "content": incoming}]) + event["response"]["output"]
                    await socket.send_json(event)
                    if event.get("type") in ("error", "response.failed"):
                        await socket.close(code=1011)
                        return
                value = json.loads(await socket.receive_text())
                payload = value.get("response", value)
        else:
            if socket.headers.get("authorization") == "Bearer " + LOCAL_KEY:
                await socket.close(code=1008)
                return
            url = upstream(socket.headers).replace("https://", "wss://", 1) + "/" + path.removeprefix("v1/")
            if socket.url.query:
                url += "?" + socket.url.query
            async with connect(url, additional_headers=filtered_headers(socket.headers, websocket=True), max_size=32 * 1024 * 1024) as remote:
                await remote.send(first)

                async def to_remote():
                    while True:
                        message = await socket.receive()
                        if message["type"] == "websocket.disconnect":
                            break
                        await remote.send(message.get("text") if message.get("text") is not None else message["bytes"])

                async def to_client():
                    async for message in remote:
                        if isinstance(message, str):
                            await socket.send_text(message)
                        else:
                            await socket.send_bytes(message)

                tasks = [asyncio.create_task(to_remote()), asyncio.create_task(to_client())]
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Log exception type only: upstream errors can contain headers/URLs.
        print(json.dumps({"websocket_error": type(exc).__name__}), flush=True)
        try:
            await socket.close(code=1011)
        except RuntimeError:
            pass


if __name__ == "__main__":
    (ROOT / "router.pid").write_text(str(os.getpid()) + "\n")
    try:
        uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("QWEN_ROUTER_PORT", "31038")),
                    log_level="warning", access_log=False)
    finally:
        (ROOT / "router.pid").unlink(missing_ok=True)
