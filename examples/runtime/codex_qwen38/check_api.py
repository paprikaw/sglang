"""Bounded live checks of the deployed model, SSE, and function-call round trips."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:30038"
MODEL = "orcarouter/Qwen3.8-27B-Uncensored"
KEY = (ROOT / "api.key").read_text().strip()


def request(path, body=None, timeout=180):
    headers = {"Authorization": "Bearer " + KEY}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from None


def main():
    report = {}
    report["models"] = request("/v1/models")
    assert any(m["id"] == MODEL for m in report["models"]["data"])
    print("model list OK", flush=True)
    chat = request("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "只回答四个字：部署成功"}],
        "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    report["chat"] = chat
    assert "部署成功" in chat["choices"][0]["message"]["content"]
    print("chat completion OK", flush=True)
    common = {"model": MODEL, "temperature": 0, "max_output_tokens": 256,
              "store": False, "chat_template_kwargs": {"enable_thinking": False}}
    body = {**common, "input": "Reply exactly: SGLANG_RESPONSES_OK", "stream": True}
    req = urllib.request.Request(BASE + "/v1/responses", data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    report["sse_events"] = events
    assert any(e["type"] == "response.completed" for e in events)
    assert "SGLANG_RESPONSES_OK" in "".join(e.get("delta", "") for e in events if e["type"] == "response.output_text.delta")
    print("Responses SSE OK", flush=True)
    tool = {"type": "function", "name": "add_numbers", "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}
    prompt = {"role": "user", "content": "Use add_numbers to add 17 and 25. You must call the tool."}
    call = request("/v1/responses", {**common, "input": [prompt], "tools": [tool], "tool_choice": "required"})
    report["function_call"] = call
    calls = [o for o in call["output"] if o["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "add_numbers"
    assert json.loads(calls[0]["arguments"]) == {"a": 17, "b": 25}
    final = request("/v1/responses", {**common, "tools": [tool], "input": [prompt, *call["output"],
        {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "42"}]})
    report["function_result"] = final
    assert "42" in json.dumps(final["output"])
    print("function-call round trip OK", flush=True)
    report["verified_at_unix"] = time.time()
    (ROOT / "checks/api-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
