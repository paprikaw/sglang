"""Bounded real-weight qualification before the concurrent AgentX pair."""

import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path


def exercise(run, root, replay, p, d, key, info, health, save):
    import hicache_transition_smoke as cache_probe

    from sglang.srt.elastic_pp.pd_reactive import (
        Observation,
        Parameters,
        ReactivePDPolicy,
    )

    result = dict(passed=False, samples=[], transitions=[])
    original_stream = replay.stream

    def traced_stream(*args, **kwargs):
        events = []
        kwargs["event_sink"] = events
        response = original_stream(*args, **kwargs)
        response["token_events"] = events
        response["token_event_started"] = response["started"]
        return response

    replay.stream = traced_stream

    def observe():
        health()
        state = info("d0")
        rows = state["internal_states"]
        result["samples"].append(
            dict(
                unix=time.time(),
                running=[r["e2e_runtime"]["running"] for r in rows],
                kv=[r.get("e2e_kv_capacity") for r in rows],
                waiting=[r.get("e2e_kv_wait") for r in rows],
                pd_pending=[r["e2e_runtime"]["pd_pending"] for r in rows],
            )
        )
        save(run / "concurrent_smoke.partial.json", result)
        return state

    def transition(counts, generation):
        before = observe()
        deadline = time.monotonic() + 60
        while any(r["e2e_runtime"]["pd_pending"] for r in before["internal_states"]):
            if time.monotonic() > deadline:
                raise TimeoutError("published P/D writers did not drain")
            time.sleep(0.1)
            before = observe()
        started = time.time()
        answer = replay.http(
            d["host"],
            d["port"],
            "/reconfigure_elastic_pp",
            key,
            dict(layer_counts=counts, generation=generation),
            120,
        )
        result["transitions"].append(
            dict(started=started, finished=time.time(), answer=answer)
        )
        if not answer.get("success"):
            raise RuntimeError("concurrent migration rejected: " + str(answer))
        deadline = time.monotonic() + 60
        while True:
            after = observe()
            if all(
                r.get("transition_metrics", {}).get("phase") == "done"
                for r in after["internal_states"]
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("concurrent migration cleanup")
            time.sleep(0.1)
        return after

    prompts = [
        [1000 + ((i * 31 + j * 937) % 20000) for i in range(4608)] for j in range(2)
    ]
    try:
        short_prompts = [tokens[:2048] for tokens in prompts]
        short_references = [
            replay.request_one(p, d, key, tokens, 128, j, True, request_timeout_s=120)
            for j, tokens in enumerate(short_prompts)
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            short_futures = [
                executor.submit(
                    replay.request_one,
                    p,
                    d,
                    key,
                    tokens,
                    128,
                    j,
                    True,
                    request_timeout_s=120,
                )
                for j, tokens in enumerate(short_prompts)
            ]
            while not all(f.done() for f in short_futures):
                observe()
                time.sleep(0.1)
            result["static_pp2_concurrent"] = [f.result() for f in short_futures]
        result["static_pp2_concurrent_matches"] = [
            a["output_ids"] == b["output_ids"]
            for a, b in zip(short_references, result["static_pp2_concurrent"])
        ]
        result["static_pp2_max_running"] = max(
            min(s["running"]) for s in result["samples"]
        )
        # Priming and references preserve HiCache; no flush-between-request path.
        references = []
        for j, tokens in enumerate(prompts):
            replay.request_one(p, d, key, tokens, 64, j, True, request_timeout_s=120)
            references.append(
                replay.request_one(
                    p, d, key, tokens, 512, j, True, request_timeout_s=120
                )
            )
        result["references"] = references
        repeats = [
            replay.request_one(p, d, key, tokens, 512, j, True, request_timeout_s=120)
            for j, tokens in enumerate(prompts)
        ]
        result["serial_repeats"] = repeats
        result["serial_repeat_matches"] = [
            a["output_ids"] == b["output_ids"] for a, b in zip(references, repeats)
        ]
        initial = observe()
        policy = ReactivePDPolicy("inplace", Parameters(**root["policy"]))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    replay.request_one,
                    p,
                    d,
                    key,
                    tokens,
                    512,
                    j,
                    True,
                    request_timeout_s=240,
                )
                for j, tokens in enumerate(prompts)
            ]
            deadline = time.monotonic() + 90
            while True:
                state = observe()
                now = time.monotonic()
                waits = [r.get("e2e_kv_wait", {}) for r in state["internal_states"]]
                blocked = any(r.get("blocked", False) for r in waits)
                decision = policy.observe(
                    [
                        Observation(
                            "decode",
                            now,
                            capacity_blocked=blocked,
                            capacity_wait_s=max(
                                (r.get("oldest_wait_s", 0) for r in waits), default=0
                            ),
                        )
                    ],
                    now=now,
                    occupied_gpus=4,
                    free_by_role={"decode": 1},
                )
                if decision:
                    result["trigger"] = dict(
                        decision=asdict(decision), native_waits=waits
                    )
                    transition([16, 16, 16], 1)
                    break
                if all(f.done() for f in futures):
                    raise RuntimeError(
                        "no true KV capacity trigger before requests ended"
                    )
                if now > deadline:
                    raise TimeoutError("no progressing capacity trigger")
                time.sleep(0.25)
            while not all(f.done() for f in futures):
                observe()
                if time.monotonic() > deadline + 120:
                    raise TimeoutError("concurrent requests did not complete")
                time.sleep(0.25)
            result["responses"] = [f.result() for f in futures]
        result["outputs_match"] = [
            a["output_ids"] == b["output_ids"]
            for a, b in zip(references, result["responses"])
        ]
        result["max_running_after_grow"] = max(
            (
                min(s["running"])
                for s in result["samples"]
                if all(k and k["capacity"] == 12288 for k in s["kv"])
            ),
            default=0,
        )
        if result["max_running_after_grow"] < 2:
            raise AssertionError("expanded KV did not execute two concurrent requests")
        # Preserve the strict check, but collect the static target-layout
        # control and shrink/restore evidence before deciding qualification.
        with ThreadPoolExecutor(max_workers=2) as executor:
            result["static_pp3_concurrent"] = list(
                executor.map(
                    lambda item: replay.request_one(
                        p, d, key, item[1], 512, item[0], True, request_timeout_s=120
                    ),
                    enumerate(prompts),
                )
            )
        mapped = lambda state: sum(
            r["e2e_kv_capacity"]["physical"]["mapped_memory_bytes"]
            for r in state["internal_states"]
        )
        grown = observe()
        if mapped(grown) <= mapped(initial):
            raise AssertionError("additional KV capacity has no physical backing")

        def http(path, body=None):
            return replay.http(d["host"], d["port"], path, key, body, 30)

        result["demotion"] = cache_probe.demote_cpu_prefixes(http)
        final = transition([24, 0, 24], 2)
        if any(
            r["e2e_kv_capacity"]["capacity"] != 8192 for r in final["internal_states"]
        ):
            raise AssertionError("concurrent path did not shrink KV capacity")
        retired = cache_probe.rank_states(final["internal_states"])[1]
        if retired["e2e_kv_capacity"]["physical"]["mapped_memory_bytes"] != 0 or mapped(
            final
        ) >= mapped(grown):
            raise AssertionError("shrink did not physically return KV backing")
        result["physical_kv_bytes"] = dict(
            initial=mapped(initial), grown=mapped(grown), shrunk=mapped(final)
        )
        restored_at = time.time()
        restored = replay.request_one(
            p, d, key, prompts[0], 32, 0, True, request_timeout_s=120
        )
        result["restore_match"] = (
            restored["output_ids"] == references[0]["output_ids"][:32]
        )
        result["final_state"] = info("d0")
        if not result["restore_match"]:
            raise AssertionError("HiCache restore after shrink changed output")
        result["cpu_restores"] = cache_probe.fresh_prefix_restores(
            result["final_state"]["internal_states"], restored_at, (0, 2), 512
        )
        import pd_elastic_smoke as q

        h, PYTHON = q.h, q.PYTHON
        command = [
            PYTHON,
            str(Path(__file__).with_name("run.py")),
            "--reuse-probe",
            "--run",
            str(run),
        ]
        with (run / "gpu_reuse.log").open("wb") as log:
            process = subprocess.Popen(
                ["ssh", "-T", "-o", "BatchMode=yes", d["host"], shlex.join(command)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                process.stdin.write(
                    (
                        json.dumps(
                            dict(environment={k: os.environ[k] for k in h.GUARD_KEYS})
                        )
                        + "\n"
                    ).encode()
                )
                process.stdin.close()
                deadline = time.monotonic() + 60
                while not (run / "gpu_reuse_started.json").exists():
                    health()
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("independent GPU reuse probe failed")
                    time.sleep(0.1)
                result["during_reuse"] = replay.request_one(
                    p, d, key, prompts[0], 32, 0, True, request_timeout_s=120
                )
                if process.wait(timeout=60) != 0:
                    raise RuntimeError("GPU reuse probe exited unsuccessfully")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result["gpu_reuse"] = json.loads((run / "gpu_reuse_started.json").read_text())
        if not all(result["serial_repeat_matches"]) or not all(result["outputs_match"]):
            raise AssertionError(
                "long-output static/concurrent equivalence failed; diagnostic controls preserved"
            )
        result["passed"] = True
        save(
            run / "completed.json",
            dict(
                passed=True,
                qualification="concurrent_kv_capacity",
                capacity_frontier=[8192, 12288, 8192],
                max_running_after_grow=result["max_running_after_grow"],
            ),
        )
        return result
    finally:
        replay.stream = original_stream
        save(run / "concurrent_smoke.json", result)
