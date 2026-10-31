"""Recompute cold-Prefill metrics and attribute actual GPU kernels to Prefill layers."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def table(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader(); w.writerows(rows)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def profile(path, arm, tools):
    with gzip.open(path, "rt") as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    ann = [e for e in events if e.get("cat") == "user_annotation"]
    steps = [e for e in ann if e["name"].startswith("step[")]
    si = tools.IntervalIndex(steps)
    layers = []
    for e in ann:
        if re.fullmatch(r"component\.layer\.\d+", e["name"]):
            step = si.find(e["ts"], e.get("tid"))
            if step and ("EXTEND" in step["name"] or "PREFILL" in step["name"]):
                layers.append(e)
    assert layers, (path, "No actual Prefill layer annotations")
    li = tools.IntervalIndex(layers)
    ci = tools.IntervalIndex([e for e in ann if e["name"].startswith("component.")
                             and not e["name"].startswith("component.layer.")])
    runtime = {e.get("args", {}).get("correlation"): e for e in events
               if e.get("cat") in ("cuda_runtime", "cuda_driver")
               and e.get("args", {}).get("correlation") is not None}
    costs = defaultdict(lambda: defaultdict(list))
    counts, names, excluded = Counter(), Counter(), Counter()
    for e in events:
        if e.get("cat") != "kernel":
            continue
        launch = runtime.get(e.get("args", {}).get("correlation"))
        if launch is None:
            excluded["without_host_launch"] += 1
            continue
        layer = li.find(launch["ts"], launch.get("tid"))
        if layer is None:
            excluded["outside_prefill_layer"] += 1
            continue
        scope = ci.find(launch["ts"], launch.get("tid"))
        label = tools.label(scope) if scope else "other"
        name = e["name"]
        # Normal Prefill uses grouped-contiguous GemmType1; Decode uses type2.
        # Require routed-Expert ancestry so shared/Attention GEMMs are excluded.
        if label == "expert.routed" and (("deep_gemm::" in name and "gemm" in name.lower())
                                          or "fused_moe_kernel" in name):
            bucket = "expert_gemm"
            names[name] += 1
        elif label in ("communication.dispatch", "communication.combine"):
            bucket = label.split(".")[-1]
        elif label in ("attention.full", "attention.linear"):
            bucket = "attention_module"
        elif label == "expert.routed":
            bucket = "expert_non_gemm"
        elif label == "expert.shared":
            bucket = "shared_expert"
        else:
            bucket = "other"
        costs[id(layer)][bucket].append((e["ts"], e["ts"] + e["dur"]))
        counts[bucket] += 1
    totals = defaultdict(float)
    for per_layer in costs.values():
        for key, spans in per_layer.items():
            totals[key] += tools.union_us(spans) / 1000
    assert totals["expert_gemm"] > 0, (path, counts)
    if arm == "ep2":
        assert totals["dispatch"] > 0 and totals["combine"] > 0, (path, totals)
    else:
        assert not totals["dispatch"] and not totals["combine"], (path, totals)
    selected_steps = {id(si.find(e["ts"], e.get("tid"))): si.find(e["ts"], e.get("tid")) for e in layers}
    tokens = [int(re.search(r"toks=(\d+)", e["name"])[1]) for e in selected_steps.values()]
    layers_per_forward = 40 if arm == "ep2" else 20
    assert len(layers) == len(selected_steps) * layers_per_forward, (path, len(layers), len(selected_steps))
    result = dict(actual_layer_calls=len(layers), actual_prefill_forwards=len(selected_steps),
        local_source_tokens_min=min(tokens), local_source_tokens_max=max(tokens),
        local_source_tokens_mean=st.mean(tokens), local_source_tokens_total=sum(tokens),
        expert_gemm_kernels=counts["expert_gemm"],
        unit="GPU kernel ms/card/cold Prefill cohort; us/card/actual Prefill layer call",
        scope="Prefill-only GPU kernels correlated to host launch inside layer annotations; Attention includes projections")
    for key in ("expert_gemm", "dispatch", "combine", "attention_module", "expert_non_gemm", "shared_expert", "other"):
        result[key + "_ms_card"] = totals[key]
        result[key + "_us_per_layer"] = totals[key] * 1000 / len(layers)
    result["dispatch_combine_ms_card"] = totals["dispatch"] + totals["combine"]
    result["dispatch_combine_us_per_layer"] = result["dispatch_combine_ms_card"] * 1000 / len(layers)
    detail = dict(step_labels=[e["name"] for e in sorted(selected_steps.values(), key=lambda e:e["ts"])],
                  expert_gemm_kernel_counts=dict(names), excluded_kernel_counts=dict(excluded))
    return result, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--skip-profiles", action="store_true")
    args = ap.parse_args()
    run = args.run.absolute()
    out = ROOT / "analysis" / run.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = read(run / "config.json")
    progress = read(run / "task/progress.json")
    audit = dict(stage=progress["stage"], partial=args.partial)
    if not args.partial:
        assert progress["stage"] == "complete" and progress["valid_points"] == cfg["expected_points"], progress
        assert read(run / "task/cleanup.json")["passed"]
        checks = [read(p) for p in (run / "dashboard_guard/checks").glob("*-post.json")]
        assert len(checks) == 1 and checks[0]["passed"] and checks[0]["cleanup_verified"], checks
        records = []
        for line in (run / "supervisor.log").read_text().splitlines():
            try: r = json.loads(line)
            except ValueError: continue
            if isinstance(r, dict) and "gpu_registry_state" in r: records.append(r)
        assert records and records[-1]["gpu_registry_state"] == "released" and records[-1]["postcheck"] == "passed"
        for d in read(run / "dashboard_spec.json")["frozen_inputs"]:
            assert sha(d["path"]) == d["sha256"], d
        assert read(run / "cpu_qualification.json")["passed"]
        audit.update(guard_postcheck="passed", cleanup_verified=True, gpu_registry_state="released",
                     all_frozen_hashes_verified=True)
    tools = load("prefill_gpu_tools", ROOT.parent / "pp_ep_steady_decode_20261026/analyze.py")
    vis = load("prefill_visibility", Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py"))
    metrics, cohorts, requests, profiles, details, sources, receipts = [], [], [], [], [], [], []
    protocols, outputs, numerical = {}, {}, {}
    for arm in cfg["arm_order"]:
        if not args.partial:
            assert read(run / arm / "cleanup.json")["cleanup_verified"]
            assert read(run / arm / "completed.json")["passed"]
        for case in cfg["workloads"]:
            folder = run / arm / "workloads" / case["id"]
            if not (folder / "protocol.json").exists(): continue
            common = dict(workload=case["id"], arm=arm, global_requests=case["requests"],
                          new_input=case["new_input"], cached_context=0, family=case["family"])
            protocols[(arm, case["id"])] = read(folder / "protocol.json")
            validation = folder / "validation.prefill_logprobs.json"
            if validation.exists():
                value, receipt = vis.wait_for_stable_json(validation, timeout_s=25)
                receipts.append(receipt)
                assert not value["performance_sample"]
                numerical[(arm,case["id"])] = value
                sources.append(dict(path=str(validation),sha256=sha(validation),kind="untimed_first_token_logprob_validation"))
            if not args.partial:
                assert read(folder / "completed.json")["passed"]
                if arm == "pp2": assert read(folder / "output_agreement.json")["passed"]
            for p in ([0, 1] if arm == "pp2" else [0]):
                path = folder / f"metrics.prefill.p{p}.json"
                if not path.exists(): continue
                data, receipt = vis.wait_for_stable_json(path, timeout_s=25)
                receipts.append(receipt)
                if args.partial and len(data["rounds"]) < cfg["repeats"]: continue
                assert data["valid"] and not data["profiler_enabled"] and len(data["rounds"]) == cfg["repeats"]
                rates, flat = [], []
                for index, group in enumerate(data["rounds"]):
                    assert len(group["requests"]) == case["requests"]
                    expected = case["requests"] * case["new_input"]
                    rate = expected / group["makespan_s"]
                    assert abs(rate - group["input_tokens_per_s"]) < 1e-6
                    rates.append(rate)
                    cohorts.append(dict(**common, policy=p, cohort=index,
                        throughput_tok_s=rate, makespan_ms=group["makespan_s"] * 1000))
                    if arm == "pp2":
                        rec = group["scheduling_receipt"]
                        assert rec["policy"] == p
                        d = rec["early_proxy_deltas"]
                        assert (p == 1 and d[0] > 0 and d[1] == 0) or (p == 0 and not any(d)), rec
                        assert all(not s["ahead_produced"] and not s["fast_sent"] and not s["early_proxy_pending"]
                                   for s in rec["states"])
                    for j, r in enumerate(group["requests"]):
                        assert (r.get("cached_tokens") or 0) == 0 and r["prompt_tokens"] == case["new_input"]
                        assert r["output_tokens"] == 1 and len(r["output_ids"]) == 1
                        assert r["tpot_ms"] is None
                        flat.append(r)
                        requests.append(dict(**common, policy=p, cohort=index, request=j,
                            ttft_ms=r["ttft_s"] * 1000, e2e_ms=r["e2e_s"] * 1000,
                            prompt_tokens=r["prompt_tokens"], cached_tokens=0, output_tokens=1))
                outputs[(arm,case["id"],p)] = [r["output_ids"] for r in data["rounds"][0]["requests"]]
                metrics.append(dict(**common, policy=p, repeats=len(rates), throughput_tok_s=st.mean(rates),
                    throughput_stdev=st.stdev(rates), throughput_min=min(rates), throughput_max=max(rates),
                    mean_makespan_ms=st.mean(g["makespan_s"] for g in data["rounds"]) * 1000,
                    ttft_ms=st.mean(r["ttft_s"] for r in flat) * 1000,
                    e2e_ms=st.mean(r["e2e_s"] for r in flat) * 1000,
                    source=str(path), sha256=sha(path)))
                sources.append(dict(path=str(path), sha256=sha(path), kind="formal_metrics"))
                manifest = folder / "traces" / f"p{p}" / "manifest.json"
                if args.skip_profiles or (args.partial and not manifest.exists()): continue
                data, receipt = vis.wait_for_stable_json(manifest, timeout_s=25)
                receipts.append(receipt)
                assert len(data["files"]) == 2
                for rank, descriptor in enumerate(data["files"]):
                    path = Path(descriptor["path"])
                    assert sha(path) == descriptor["sha256"], path
                    component, detail = profile(path, arm, tools)
                    profiles.append(dict(**common, policy=p, rank_trace=path.name, **component))
                    details.append(dict(**common, policy=p, rank_trace=path.name, **detail))
                    sources.append(dict(**descriptor, kind="prefill_trace"))
    components = []
    for m in metrics:
        chosen = [r for r in profiles if (r["workload"],r["arm"],r["policy"]) == (m["workload"],m["arm"],m["policy"])]
        if not chosen: continue
        assert len(chosen) == 2
        row = {k:m[k] for k in ("workload","arm","policy","global_requests","new_input","family")}
        row.update(rank_profiles=2, scope=chosen[0]["scope"], unit=chosen[0]["unit"])
        for k in chosen[0]:
            if k.endswith(("_ms_card","_us_per_layer")) or k.startswith("local_source_tokens_") or k in ("actual_layer_calls","actual_prefill_forwards"):
                row[k] = st.mean(r[k] for r in chosen)
        components.append(row)
    comparisons = []
    for case in cfg["workloads"]:
        selected = {(r["arm"],r["policy"]):r for r in metrics if r["workload"] == case["id"]}
        if len(selected) != 3: continue
        ep, pp, opt = [selected[k] for k in (("ep2",0),("pp2",0),("pp2",1))]
        assert protocols[("ep2",case["id"])]["input_sha256"] == protocols[("pp2",case["id"])]["input_sha256"]
        comparisons.append(dict(workload=case["id"], global_requests=case["requests"], new_input=case["new_input"],
            family=case["family"], ep_tok_s=ep["throughput_tok_s"], pp_native_tok_s=pp["throughput_tok_s"],
            pp_early_tok_s=opt["throughput_tok_s"],
            early_gain_pct=(opt["throughput_tok_s"] / pp["throughput_tok_s"] - 1) * 100,
            pp_early_vs_ep_pct=(opt["throughput_tok_s"] / ep["throughput_tok_s"] - 1) * 100,
            ep_ttft_ms=ep["ttft_ms"], pp_early_ttft_ms=opt["ttft_ms"],
            cross_layout_first_token_equal=outputs[("ep2",case["id"],0)] == outputs[("pp2",case["id"],1)]))
    numerical_summary, numerical_details = [], []
    for case in cfg["workloads"]:
        if not all((arm,case["id"]) in numerical for arm in ("ep2","pp2")): continue
        ep = numerical[("ep2",case["id"])]["by_policy"]["0"]
        pp = numerical[("pp2",case["id"])]["by_policy"]
        for label, reference, candidate in (("EP_vs_PP_native",ep,pp["0"]),
                                           ("PP_native_vs_early",pp["0"],pp["1"])):
            changed, deltas, margins = [], [], []
            for repeat,(left,right) in enumerate(zip(reference,candidate)):
                for i,(a,b) in enumerate(zip(left["requests"],right["requests"])):
                    def top(row):
                        values=row["validation_logits"]["output_top_logprobs"]
                        assert len(values)==1 and len(values[0])==10, values
                        return {int(x[1]):float(x[0]) for x in values[0]}
                    ta,tb=top(a),top(b)
                    shared=set(ta)&set(tb)
                    deltas.extend(abs(ta[t]-tb[t]) for t in shared)
                    aa,bb=a["output_ids"][0],b["output_ids"][0]
                    if aa != bb:
                        comparable=aa in tb and bb in ta
                        row=dict(workload=case["id"],comparison=label,repeat=repeat,request=i,
                                 reference_token=aa,candidate_token=bb,both_predictions_in_both_top10=comparable)
                        if comparable:
                            row.update(reference_logprob_margin=ta[aa]-ta[bb],candidate_logprob_margin=tb[bb]-tb[aa],
                                reference_probability_margin=math.exp(ta[aa])-math.exp(ta[bb]),
                                candidate_probability_margin=math.exp(tb[bb])-math.exp(tb[aa]))
                            margins += [row["reference_logprob_margin"],row["candidate_logprob_margin"]]
                        changed.append(row)
            numerical_details.extend(changed)
            numerical_summary.append(dict(workload=case["id"],comparison=label,request_comparisons=case["requests"]*len(reference),
                changed_predictions=len(changed),all_changes_within_both_top10=all(r["both_predictions_in_both_top10"] for r in changed),
                max_changed_prediction_logprob_margin=max(margins) if margins else None,
                max_abs_shared_top10_logprob_delta=max(deltas),mean_abs_shared_top10_logprob_delta=st.mean(deltas)))
    if not args.partial:
        assert len(metrics) == cfg["expected_points"] and len(comparisons) == len(cfg["workloads"])
        if not args.skip_profiles: assert len(profiles) == cfg["expected_points"] * 2
    result = dict(run=str(run), config=cfg, partial=args.partial, profiles_skipped=args.skip_profiles,
        run_audit=audit, metrics=metrics, cohorts=cohorts, requests=requests, components=components,
        components_per_rank=profiles, profile_details=details, comparisons=comparisons, source_descriptors=sources,
        numerical_summary=numerical_summary,numerical_details=numerical_details)
    for name, rows in (("metrics",metrics),("cohorts",cohorts),("requests",requests),("components",components),
                       ("components_per_rank",profiles),("comparisons",comparisons),
                       ("numerical_summary",numerical_summary),("numerical_details",numerical_details)):
        table(out / f"{name}.csv", rows)
    save(out / "results.json", result)
    save(out / "visibility_receipts.json", receipts)
    print(json.dumps(dict(metrics=len(metrics), profiles=len(profiles), comparisons=comparisons), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
