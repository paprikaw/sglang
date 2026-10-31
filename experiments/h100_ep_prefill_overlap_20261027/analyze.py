"""Audit saved cohorts and inspect actual communication/GEMM intersections."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "h100_prefill_ep_pp_optimized_20261027"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a = load("overlap_original_analysis", PREVIOUS / "analyze.py")
read, sha, save = a.read, a.sha, a.save


def merged(spans):
    result = []
    for low, high in sorted(spans):
        if not result or low > result[-1][1]: result.append([low, high])
        else: result[-1][1] = max(high, result[-1][1])
    return result


def intersection(left, right):
    i = j = 0
    total = 0.
    while i < len(left) and j < len(right):
        total += max(0., min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]: i += 1
        else: j += 1
    return total


def overlap(path, tools):
    with gzip.open(path, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    ann = [e for e in events if e.get("cat") == "user_annotation"]
    si = tools.IntervalIndex([e for e in ann if e["name"].startswith("step[")])
    layers = [e for e in ann if re.fullmatch(r"component\.layer\.\d+", e["name"])
              and (s := si.find(e["ts"], e.get("tid"))) and ("EXTEND" in s["name"] or "PREFILL" in s["name"])]
    assert layers and len(layers) % 40 == 0
    li = tools.IntervalIndex(layers)
    ci = tools.IntervalIndex([e for e in ann if e["name"].startswith("component.")
                             and not e["name"].startswith("component.layer.")])
    runtime = {e.get("args", {}).get("correlation"): e for e in events
               if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("args", {}).get("correlation") is not None}
    spans, streams = defaultdict(list), defaultdict(set)
    for event in events:
        if event.get("cat") != "kernel": continue
        launch = runtime.get(event.get("args", {}).get("correlation"))
        if not launch or not li.find(launch["ts"], launch.get("tid")): continue
        scope = ci.find(launch["ts"], launch.get("tid"))
        label = tools.label(scope) if scope else "other"
        if label == "expert.routed" and (("deep_gemm::" in event["name"] and "gemm" in event["name"].lower()) or "fused_moe_kernel" in event["name"]):
            bucket = "expert_gemm"
        elif label in ("communication.dispatch", "communication.combine"):
            bucket = "communication"
        elif label in ("attention.full", "attention.linear"):
            bucket = "attention"
        else: continue
        spans[bucket].append((event["ts"], event["ts"] + event["dur"]))
        streams[bucket].add(event.get("args", {}).get("stream"))
    unions = {key: merged(value) for key, value in spans.items()}
    hit = intersection(unions["communication"], unions["expert_gemm"])
    comm = sum(high-low for low, high in unions["communication"])
    return dict(actual_layer_calls=len(layers), comm_gemm_overlap_us_per_layer=hit / len(layers),
                communication_overlapped_pct=hit / comm * 100,
                communication_attention_overlap_us_per_layer=intersection(unions["communication"], unions["attention"]) / len(layers),
                streams={key: sorted(value, key=str) for key, value in streams.items()})


def trace_task(task):
    descriptor, arm = task
    assert sha(descriptor["path"]) == descriptor["sha256"]
    method = sha(Path(__file__)) + sha(PREVIOUS / "analyze.py") + sha(ROOT.parent / "pp_ep_steady_decode_20261026/analyze.py")
    cache = ROOT / "analysis/profile_cache" / hashlib.sha256(method.encode()).hexdigest()[:16] / (descriptor["sha256"] + ".json")
    if cache.exists():
        result = read(cache)
        assert result["arm"] == arm and result["trace_sha256"] == descriptor["sha256"]
        return result
    tools = load("overlap_gpu_worker_tools", ROOT.parent / "pp_ep_steady_decode_20261026/analyze.py")
    component, details = a.profile(Path(descriptor["path"]), arm, tools)
    result = dict(arm=arm, trace_sha256=descriptor["sha256"], method_sha256=hashlib.sha256(method.encode()).hexdigest(),
                  component=component, details=details,
                  temporal=overlap(Path(descriptor["path"]), tools) if arm == "ep2" else {})
    save(cache, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--skip-profiles", action="store_true")
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    run = args.run.absolute()
    cfg = read(run / "config.json")
    progress = read(run / "task/progress.json")
    if not args.partial:
        assert progress["stage"] == "complete" and progress["valid_points"] == cfg["expected_points"], progress
        assert read(run / "task/cleanup.json")["passed"]
        post = [read(p) for p in (run / "dashboard_guard/checks").glob("*-post.json")]
        assert len(post) == 1 and post[0]["passed"] and post[0]["cleanup_verified"]
        assert '"gpu_registry_state": "released"' in (run / "supervisor.log").read_text()
        assert read(run / "cpu_qualification.json")["passed"] and read(run / "overlap_cpu_qualification.json")["passed"]
    for descriptor in read(run / "dashboard_spec.json")["frozen_inputs"]:
        assert sha(descriptor["path"]) == descriptor["sha256"], descriptor
    if not args.skip_profiles:
        tasks = []
        for arm in cfg["arm_order"]:
            for path in (run / arm / "workloads").glob("*/traces/*/manifest.json"):
                tasks += [(descriptor, arm) for descriptor in read(path)["files"]]
        if args.workers == 1:
            for task in tasks: trace_task(task)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
                list(pool.map(trace_task, tasks))
    visibility = load("overlap_visibility", Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py"))
    metrics, components, ranks, sources, outputs = [], [], [], [], {}
    for case in cfg["workloads"]:
        protocols = []
        for arm in cfg["arm_order"]:
            folder = run / arm / "workloads" / case["id"]
            if not (folder / "protocol.json").exists(): continue
            protocols.append(read(folder / "protocol.json")["input_sha256"])
            if not args.partial: assert read(folder / "completed.json")["passed"]
            for policy in (cfg["ep_overlap_policies"] if arm == "ep2" else [0, 1]):
                path = folder / f"metrics.prefill.p{policy}.json"
                if not path.exists(): continue
                value, _ = visibility.wait_for_stable_json(path, timeout_s=25)
                groups = value["rounds"]
                if len(groups) != cfg["repeats"]: continue
                assert value["valid"] and not value["profiler_enabled"]
                flat = []
                rates = []
                for group in groups:
                    assert len(group["requests"]) == case["requests"]
                    assert abs(group["input_tokens_per_s"] - case["requests"] * case["new_input"] / group["makespan_s"]) < 1e-6
                    rates.append(group["input_tokens_per_s"])
                    flat += group["requests"]
                    if arm == "ep2":
                        receipts = group["overlap_receipts"]
                        assert len(receipts) == 2 and all(r["drained"] and r["policy"] == policy and r["qualification_passed"] for r in receipts)
                assert all((r.get("cached_tokens") or 0) == 0 and r["prompt_tokens"] == case["new_input"] and r["output_tokens"] == 1 for r in flat)
                common = dict(workload=case["id"], arm=arm, policy=policy, requests=case["requests"], new_input=case["new_input"])
                metrics.append(dict(**common, input_tok_s=st.mean(rates), stdev_tok_s=st.stdev(rates),
                                    min_tok_s=min(rates), max_tok_s=max(rates), repeats=len(groups),
                                    e2e_ms=st.mean(r["e2e_s"] for r in flat)*1000,
                                    cohort_makespan_ms=st.mean(g["makespan_s"] for g in groups)*1000))
                outputs[(case["id"], arm, policy)] = [[r["output_ids"] for r in g["requests"]] for g in groups]
                sources.append(dict(path=str(path), sha256=sha(path), kind="formal_metrics"))
                manifest = folder / "traces" / f"p{policy}" / "manifest.json"
                if args.skip_profiles or not manifest.exists(): continue
                files = read(manifest)["files"]
                assert len(files) == 2
                local = []
                for descriptor in files:
                    assert sha(descriptor["path"]) == descriptor["sha256"]
                    cached = trace_task((descriptor, arm))
                    measured, details = cached["component"], cached["details"]
                    row = dict(**common, trace=descriptor, **measured, profile_details=details)
                    row.update(cached["temporal"])
                    ranks.append(row)
                    local.append(row)
                    sources.append(dict(**descriptor, kind="independent_gpu_profile"))
                averaged = dict(common)
                for key in local[0]:
                    if key.endswith(("_ms_card", "_us_per_layer", "_pct")):
                        averaged[key] = st.mean(row[key] for row in local)
                components.append(averaged)
        assert all(p == protocols[0] for p in protocols)
    agreement = []
    for case in cfg["workloads"]:
        reference = outputs.get((case["id"], "ep2", 0))
        if not reference: continue
        for (name, arm, policy), rounds in outputs.items():
            if name != case["id"]: continue
            agreement.append(dict(workload=name, arm=arm, policy=policy,
                                  native_ep_first_token_equal=all(row == reference[0] for row in rounds)))
    if not args.partial:
        assert len(metrics) == cfg["expected_points"]
        if not args.skip_profiles: assert len(ranks) == cfg["expected_points"] * 2
        qual = [read(p) for p in (run / "ep2/overlap_qualification").glob("rank*.json")]
        assert len(qual) == 2 and all(len(rank) == 40 and all(row["passed"] for row in rank.values()) for rank in qual)
    out = ROOT / "analysis" / run.name
    result = dict(run=str(run), config=cfg, stage=progress["stage"], partial=args.partial,
                  metrics=metrics, components=components, per_rank_components=ranks,
                  output_agreement=agreement, source_descriptors=sources,
                  run_audit=dict(frozen_hashes_verified=True, guard_and_cleanup_verified=not args.partial))
    save(out / "results.json", result)
    a.table(out / "metrics.csv", metrics)
    a.table(out / "components.csv", components)
    print(json.dumps(dict(run=run.name, metrics=metrics, components=components, agreement=agreement), indent=2))


if __name__ == "__main__": main()
