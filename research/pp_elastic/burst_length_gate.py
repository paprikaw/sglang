"""Check P/D capacity using the longest observed AgentX request shape."""
import json


def measure(run, p, d, key, replay, prior):
    rows = [json.loads(line) for line in (prior / "replay/routing.jsonl").read_text().splitlines()]
    longest = max(rows, key=lambda row: row["usage"]["prompt_tokens"])["usage"]
    length = longest["prompt_tokens"]
    output = longest["completion_tokens"]
    trace = json.loads((prior / "data/traces.jsonl").read_text())

    def turns(requests):
        for turn in requests:
            if "in" in turn:
                yield turn
            yield from turns(turn.get("requests", []))

    source = max(turns(trace["requests"]), key=lambda turn: turn["in"])
    turn = dict(source, input_tokens=length, source_outer_idx=0)
    tokens = replay.token_ids(trace, turn, "cold-long-shape-qualification", [993, 1479, 3214], {})
    result = replay.request_one(p, d, key, tokens, output, 0, True, request_timeout_s=240)
    if result["meta_info"].get("prompt_tokens") != length or result["output_tokens"] != output:
        raise RuntimeError("Longest input or output truncated")
    return dict(passed=True, input_tokens=length, output_tokens=output, prefill_prompt_tokens=result[
        "prefill_meta"].get("prompt_tokens"), ttft_ms=result["ttft_ms"], e2e_ms=result["e2e_ms"],
        scope="Cold length-matched handoff, synthetic token IDs from source block hashes; capacity check only.")
