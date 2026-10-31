"""Recompute workload metrics, real routing and independent GPU profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    args = parser.parse_args()
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    fragment = module("workload_fragment_analysis", ROOT.parent / "h100_pp_expert_fragmentation_20261026/analyze.py")
    tools = fragment.load_analysis()
    previous = tools.graph_bucket
    tools.graph_bucket = lambda name: ("expert.layout_adapter" if name == "_metadata" or name.startswith("_metadata<")
                                      else previous(name))
    probabilities = module("workload_probability_analysis", ROOT.parent / "h100_pp_expert_fragmentation_20261026/validate_probabilities.py")
    vis_path = Path("/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py")
    visibility = module("workload_visibility", vis_path)
    metrics, cohorts, requests, components, details, routes, micro, audits, sources = [], [], [], [], [], [], [], [], []
    validation = {}
    qualifications = []
    protocols = {}
    metric_paths = list((ROOT / "runs").glob("*/*/workloads/*/metrics.*.json"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(lambda p: visibility.wait_for_stable_json(p, timeout_s=20)[1], metric_paths))
    save(out / "visibility_receipts.json", receipts)
    for run in sorted((ROOT / "runs").iterdir()):
        if not (run / "config.json").exists():
            continue
        cfg = read(run / "config.json")
        progress = read(run / "task/progress.json")
        audit = dict(run=run.name, stage=progress["stage"], launched=progress["stage"] != "queued")
        if not args.partial:
            assert progress["stage"] == "complete", audit
            cleanup = read(run / "task/cleanup.json")
            assert cleanup["passed"] and cleanup["model_processes_exited"], cleanup
            postchecks = list((run / "dashboard_guard/checks").glob("*-post.json"))
            assert len(postchecks) == 1
            postcheck = read(postchecks[0])
            assert postcheck["cleanup_verified"], postcheck
            # Also require the supervisor's final gate and released lease.
            final = []
            for line in (run / "supervisor.log").read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and "gpu_registry_state" in record:
                    final.append(record)
            assert final and final[-1]["gpu_registry_state"] == "released" and final[-1]["postcheck"] == "passed", final
            audit.update(cleanup_verified=True, guard_postcheck="passed", gpu_registry_state="released")
            for descriptor in read(run / "dashboard_spec.json")["frozen_inputs"]:
                assert sha(Path(descriptor["path"])) == descriptor["sha256"], descriptor
            audit["all_frozen_hashes_verified"] = True
        audits.append(audit)
        for arm in cfg["arm_order"]:
            if not args.partial:
                assert read(run / arm / "cleanup.json")["cleanup_verified"]
                assert read(run / arm / "completed.json")["passed"]
            if arm == "ep2":
                for path in sorted((run / arm / "ep_kernel_qualification").glob("*.json")):
                    value = read(path)
                    assert value["passed"] and all(t["finite"] and t["relative_l2"] <= .015 and t["metadata_coverage"] for t in value["tests"])
                    assert {t.get("logical_tokens") for t in value["tests"]} >= {4, 8, 16}
                    qualifications.append(dict(run=run.name, source=str(path), **value))
            for case in cfg["workloads"]:
                root = run / arm / "workloads" / case["id"]
                common = dict(run=run.name, mode=cfg["expert_mode"], arm=arm,
                              workload=case["id"], family=case["family"],
                              global_requests=case["requests"], target_local_tokens=case["requests"] // 2)
                if (root / "protocol.json").exists():
                    protocol = read(root / "protocol.json")
                    protocols[(run.name,arm,case["id"])] = protocol
                    if arm == "pp2":
                        info = read(root / "server_info.json")
                        values = [r["pp_max_micro_batch_size"] for r in info["internal_states"]
                                  if "pp_max_micro_batch_size" in r]
                        assert values and set(values) == {common["target_local_tokens"]}, (root, values)
                if not args.partial:
                    assert read(root / "completed.json")["passed"]
                for phase in ("prefill", "decode"):
                    path = root / f"metrics.{phase}.json"
                    if not path.exists():
                        continue
                    value = read(path)
                    assert value["valid"] and not value["profiler_enabled"]
                    assert len(value["rounds"]) == cfg["repeats"]
                    expected_output = cfg["output_tokens"] if phase == "decode" else 1
                    expected_cache = cfg["prefix_tokens"] if phase == "decode" else 0
                    flat, rates = [], []
                    for i, group in enumerate(value["rounds"]):
                        assert len(group["requests"]) == case["requests"]
                        for j, row in enumerate(group["requests"]):
                            assert row["output_tokens"] == expected_output and (row.get("cached_tokens") or 0) == expected_cache
                            assert row["prompt_tokens"] == cfg["prefix_tokens"] + (phase == "decode")
                            if phase == "decode":
                                recomputed = (row["e2e_s"] - row["ttft_s"]) * 1000 / (expected_output - 1)
                                assert abs(recomputed - row["tpot_ms"]) < 1e-5
                                counts = {e["completion_tokens"] for e in row["events"]}
                                assert counts == set(range(1,expected_output+1)), (path,i,j,counts)
                            flat.append(row)
                            requests.append(dict(**common, phase=phase, cohort=i, request=j,
                                cached_tokens=expected_cache, prompt_tokens=row["prompt_tokens"],
                                output_tokens=expected_output, e2e_ms=row["e2e_s"]*1000,
                                ttft_ms=row["ttft_s"]*1000, tpot_ms=row.get("tpot_ms")))
                        total = case["requests"] * (cfg["output_tokens"] if phase == "decode" else cfg["prefix_tokens"])
                        rate = total / group["makespan_s"]
                        rates.append(rate)
                        cohorts.append(dict(**common, phase=phase, cohort=i, throughput_tok_s=rate,
                                            makespan_ms=group["makespan_s"]*1000))
                    metrics.append(dict(**common, phase=phase, repeats=len(rates),
                        throughput_tok_s=st.mean(rates), throughput_stdev=st.stdev(rates),
                        throughput_median=st.median(rates),
                        e2e_ms=st.mean(r["e2e_s"] for r in flat)*1000,
                        ttft_ms=st.mean(r["ttft_s"] for r in flat)*1000,
                        tpot_ms=st.mean(r["tpot_ms"] for r in flat) if phase == "decode" else None,
                        cached_context_per_request=expected_cache,
                        new_input_per_request=1 if phase == "decode" else cfg["prefix_tokens"],
                        output_per_request=expected_output, source=str(path), sha256=sha(path)))
                    sources.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
                    manifest_path = root / "traces" / phase / "manifest.json"
                    if args.skip_profiles or (args.partial and not manifest_path.exists()):
                        continue
                    manifest = read(manifest_path)
                    assert len(manifest["files"]) == 2
                    for descriptor in manifest["files"]:
                        trace = Path(descriptor["path"])
                        assert sha(trace) == descriptor["sha256"], trace
                        sources.append(dict(**descriptor,kind="trace"))
                        if phase == "decode":
                            target = common["target_local_tokens"]
                            tools.steady_step = lambda event, n=target: event is not None and f"step[DECODE bs={n} " in event["name"]
                            component, detail = tools.trace_components(trace, arm, "native", "H100", cfg["selected_forwards"])
                            component["unit"] = f"ms/rank/actual-layer-call; local Decode batch {target}"
                            details.append({**common, "phase": phase, **detail})
                        else:
                            component = fragment.prefill_components(trace,arm,tools)
                        components.append(dict(**component, **{k:v for k,v in common.items() if k not in component},phase=phase))
                for path in sorted((root / "routing").glob("*.json")):
                    value = read(path)
                    if path.name.startswith("microbench"):
                        for candidate, error in value["numerical"].items():
                            assert error["passed"], (path,candidate,error)
                        micro.append(dict(**common,source=str(path),**value))
                        continue
                    if arm == "ep2":
                        numerical = value["numerical"]
                        assert numerical["finite"] and numerical["relative_l2"] <= .015 and numerical["metadata_coverage"]
                    else:
                        numerical = {}
                    routes.append(dict(**common, rank=value.get("rank",value.get("pp_rank")),
                        layer=value["layer_id"], local_input_tokens=value.get("local_input_tokens",value.get("tokens")),
                        assignments=value["assignments"],active_experts=value["active_experts"],
                        padded_rows_16=value["padded_rows"]["16"],padded_rows_128=value["padded_rows"]["128"],
                        relative_l2=numerical.get("relative_l2"),source=str(path)))
                path = root / "validation.probabilities.json"
                if path.exists():
                    validation[(run.name,arm,case["id"])] = probabilities.samples(path)
    if not args.partial:
        for case in {key[2] for key in protocols}:
            fingerprints = [p["prompt_sha256"] for (run,arm,workload),p in protocols.items() if workload==case]
            assert len(fingerprints)==4 and all(f==fingerprints[0] for f in fingerprints), case
    routing_summary = []
    for key in sorted({(r["run"],r["arm"],r["workload"]) for r in routes}):
        chosen = [r for r in routes if (r["run"],r["arm"],r["workload"]) == key
                  and r["local_input_tokens"] == r["target_local_tokens"]]
        if not chosen:
            continue
        sample = chosen[0]
        routing_summary.append(dict(run=key[0],arm=key[1],workload=key[2],mode=sample["mode"],
            global_requests=sample["global_requests"],local_tokens=sample["target_local_tokens"],
            layer_samples=len(chosen),active_experts_mean=st.mean(r["active_experts"] for r in chosen),
            active_experts_min=min(r["active_experts"] for r in chosen),active_experts_max=max(r["active_experts"] for r in chosen),
            assignments_mean=st.mean(r["assignments"] for r in chosen),
            assignments_per_active_expert=st.mean(r["assignments"]/r["active_experts"] for r in chosen if r["active_experts"]),
            padded16_mean=st.mean(r["padded_rows_16"] for r in chosen)))
    probability_summary, probability_positions = [], []
    for arm in ("ep2","pp2"):
        for workload in sorted({r["workload"] for r in metrics}):
            reference = validation.get(("native_r1",arm,workload))
            if reference is None:
                continue
            for (run_name, candidate_arm, candidate_workload), rounds in validation.items():
                if (candidate_arm,candidate_workload) != (arm,workload):
                    continue
                for i, sample in enumerate(rounds):
                    if run_name == "native_r1" and i == 0:
                        continue
                    summary, positions = probabilities.compare(reference[0],sample,f"{workload}.{arm}.{run_name}.r{i}")
                    probability_summary.append(summary)
                    probability_positions.extend(positions)
    table(out / "metrics.csv",metrics)
    table(out / "cohorts.csv",cohorts)
    table(out / "requests.csv",requests)
    table(out / "components_per_rank.csv",components)
    table(out / "routing.csv",routes)
    table(out / "routing_summary.csv",routing_summary)
    table(out / "probability_positions.csv",probability_positions)
    result = dict(partial=args.partial,profiles_skipped=args.skip_profiles,
        metrics=metrics,components=components,routing_summary=routing_summary,
        probability_summary=probability_summary,run_audits=audits,
        source_descriptors=sources,qualification_files=len(qualifications))
    save(out / "results.json", result)
    save(out / "profile_details.json", details)
    save(out / "microbenchmarks.json", micro)
    save(out / "ep_qualifications.json",qualifications)
    print(json.dumps(dict(partial=args.partial,metrics=len(metrics),components=len(components),
                          routing_rows=len(routes),probability_comparisons=len(probability_summary),
                          audits=audits),indent=2))


if __name__ == "__main__":
    main()
