"""Compare live Qwen3.5 autoscaling output against fixed-topology continuations.

The caller owns the server, booking/preflight checks and launch configuration.
This client never invokes manual reconfiguration. Configure scale-up at >=3
requests, scale-down at <=1, two operations, and at least four request slots.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with _HTTP.open(req, timeout=timeout) as response:
        result = json.load(response)
    if isinstance(result, dict) and (
        result.get("error") or result.get("success") is False
    ):
        raise RuntimeError(f"{path}: {result}")
    return result


def snapshot(url):
    info = request(url, "/server_info", timeout=600)
    states = [s["elastic_pp_state"] for s in info["internal_states"]]
    signatures = {(s["generation"], tuple(s["layer_counts"])) for s in states}
    if len(signatures) != 1:
        raise RuntimeError(f"inconsistent scheduler topologies: {signatures}")
    generation, partition = signatures.pop()
    if not all(s["source_usable"] for s in states):
        raise RuntimeError(f"unusable source generation: {states}")
    if any(s["migration_phase"] == "failed" for s in states):
        raise RuntimeError(f"migration failed; inspect the server logs: {states}")
    return {
        "time": time.monotonic(),
        "generation": generation,
        "partition": list(partition),
        "states": states,
    }


def generate(url, prompt, tokens):
    start = time.monotonic()
    result = request(
        url,
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
        },
    )
    if result["meta_info"]["completion_tokens"] != tokens:
        raise RuntimeError(f"incomplete generation: {result}")
    return {"start": start, "end": time.monotonic(), "response": result}


def output_key(result):
    response = result["response"]
    if not response.get("output_ids"):
        raise RuntimeError("server must return output_ids for exact token comparison")
    return response["output_ids"]


def run(url, output, initial, expanded):
    report = {
        "status": "running",
        "manual_reconfigure_calls": 0,
        "initial_partition": initial,
        "expanded_partition": expanded,
        "controls": [],
        "live": [],
        "timeline": [],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    prompts = [
        "<|im_start|>user\nWrite a long numbered explanation of how a CPU executes "
        f"instructions, with at least 100 steps. Example number {i}. Do not stop early."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        for i in range(4)
    ]
    lengths = [64, 64, 64, 256]
    report["workload"] = [
        {"prompt": p, "output_tokens": n} for p, n in zip(prompts, lengths)
    ]
    try:
        first = snapshot(url)
        if first["generation"] != 0 or first["partition"] != initial:
            raise RuntimeError(f"wrong initial topology: {first}")
        for i, (prompt, count) in enumerate(zip(prompts, lengths)):
            report["controls"].append(generate(url, prompt, count))
            save()
            print(f"Fixed-topology control {i + 1}/4 complete", flush=True)
        if snapshot(url)["generation"] != 0:
            raise RuntimeError(
                "autoscaler changed topology during single-request controls"
            )

        seen = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate, url, p, n) for p, n in zip(prompts, lengths)
            ]
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                state = snapshot(url)
                report["timeline"].append(state)
                generation = state["generation"]
                if generation not in seen:
                    seen[generation] = state
                    print(
                        f"Observed generation {generation}: {state['partition']}",
                        flush=True,
                    )
                completed_live = {}
                for i, future in enumerate(futures):
                    if future.done():
                        completed_live[i] = future.result()
                report["completed_live"] = completed_live
                save()
                if all(f.done() for f in futures) and generation == 2:
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError("automatic PP4 -> PP8 -> PP4 did not complete")
            report["live"] = [f.result() for f in futures]
        if 1 not in seen or seen[1]["partition"] != expanded:
            raise RuntimeError("expanded topology was not observed")
        if 2 not in seen or seen[2]["partition"] != initial:
            raise RuntimeError("return to the initial topology was not observed")
        report["crossed_scale_up"] = [
            i
            for i, r in enumerate(report["live"])
            if r["start"] < seen[1]["time"] < r["end"]
        ]
        report["crossed_scale_down"] = [
            i
            for i, r in enumerate(report["live"])
            if r["start"] < seen[2]["time"] < r["end"]
        ]
        if not report["crossed_scale_up"] or not report["crossed_scale_down"]:
            raise RuntimeError("both transitions must retain at least one live request")
        report["exact_output_matches"] = [
            output_key(before) == output_key(after)
            for before, after in zip(report["controls"], report["live"])
        ]
        if not all(report["exact_output_matches"]):
            raise RuntimeError("live continuations differ from fixed-topology controls")
        report["after"] = generate(url, prompts[0], lengths[0])
        if output_key(report["after"]) != output_key(report["controls"][0]):
            raise RuntimeError("output changed after the complete scale cycle")
        report["final_state"] = snapshot(url)
        report["status"] = "completed"
        print(
            "Automatic scale-up/down and exact continuation checks passed", flush=True
        )
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", default="15,0,15,0,15,0,15,0")
    parser.add_argument("--expanded", default="8,7,8,7,8,7,8,7")
    args = parser.parse_args()
    run(
        args.base_url,
        args.output,
        [int(x) for x in args.initial.split(",")],
        [int(x) for x in args.expanded.split(",")],
    )
