"""Read-only, bounded qualification of saved uninstrumented PP microbatch A/B.

Reuses the independently produced witness analysis: no torch import, GPU work,
large payload re-read, or edits to frozen inputs and raw experiment artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

COHORTS = ("timed", "repeat", "witness", "witness-repeat")
WITNESS = COHORTS[2:]
RANKS = set(range(8))
TAIL = {"4", "5", "6", "7"}
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_proof(value, label):
    require(value.get("passed") is True, label + ": failed")
    ranks = value.get("ranks", [])
    require(len(ranks) == 8 and {row["rank"] for row in ranks} == RANKS,
            label + ": missing or duplicated rank")


def exact_outputs(value, label):
    require(value.get("status") == "valid" and value.get("request_count") == 8
            and value.get("exact_token_id_requests") == 8
            and value.get("exact_text_requests") == 8
            and value.get("exact_ids_and_text") is True, label + ": output mismatch")
    rows = value.get("requests", [])
    require(len(rows) == 8 and {row["request_index"] for row in rows} == RANKS,
            label + ": incomplete request equality")
    require(all(row.get("token_ids_equal") is True and row.get("text_equal") is True
                and row.get("lengths") == [64, 64] for row in rows), label + ": non-64 output")


def exact_comparison(value, label):
    require(value.get("status") == "valid", label + ": invalid comparison")
    for key, count, classification in (
        ("batch_summary", 32, "identical_logit_and_acceptance_witness"),
        ("request_summary", 128, "identical_request_logit_and_acceptance_witness"),
    ):
        summary = value.get(key, {})
        require(summary.get("matched") == count and summary.get("unmatched_left") == 0
                and summary.get("unmatched_right") == 0
                and summary.get("classifications") == {classification: count},
                label + ": nonidentical/missing witness observations")
    require(value["request_summary"].get("first_nonidentical_observation_per_request") == {},
            label + ": first divergence present")


def oracle(value, label):
    require(value.get("status") == "valid" and value.get("snapshot_count") == 32
            and value.get("counts") == {"pass": 32, "fail": 0, "inapplicable": 0}
            and value.get("all_observed_snapshots_match_cpu_reference") is True,
            label + ": incomplete/failed CPU oracle")
    snapshots = value.get("snapshots", [])
    require(len(snapshots) == 32, label + ": wrong snapshot count")
    for row in snapshots:
        ref = row.get("oracle", {})
        require(row.get("oracle_status") == "pass" and ref.get("passed") is True
                and ref.get("reference_applicable") is True
                and ref.get("accepted_indices_equal") is True,
                label + ": oracle missing exact accepted-index agreement")
    proof = value.get("payload_proof", {})
    require(proof.get("rows") == 32 and proof.get("request_indices") == list(range(8))
            and proof.get("slots") == [0, 1]
            and proof.get("synchronized_only_at_idle") is True
            and proof.get("parsed_as_weights_only_torch_payload") is True
            and proof.get("performance_measurement") is False,
            label + ": incomplete idle-flush payload validation")


def verify_cached_analysis_sources(report, report_path):
    """Hash small source/JSON receipts; reuse saved weight-only .pt validation.

    Large payloads must retain the size and pre-report modification time from
    the already validated report. This is artifact freshness, not an adversarial
    tamper-proof substitute for rerunning its saved SHA256/weights-only analysis.
    """
    seen = set()
    payloads = set()
    def walk(item):
        if isinstance(item, dict):
            if "path" in item and "sha256" in item:
                path = Path(item["path"])
                if path.suffix == ".pt":
                    if str(path) not in payloads:
                        stat = path.stat()
                        require(stat.st_size == item["size_bytes"], "Witness payload size changed: " + str(path))
                        require(stat.st_mtime_ns <= Path(report_path).stat().st_mtime_ns,
                                "Witness payload newer than cached analysis: " + str(path))
                        payloads.add(str(path))
                elif str(path) not in seen:
                    require(sha(path) == item["sha256"], "Cached analysis source changed: " + str(path))
                    seen.add(str(path))
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)
    walk(report)
    require(len(payloads) == 16, "Expected exactly 16 validated tail-rank witness payloads")
    return dict(small_source_sha256_verified=len(seen), reused_validated_large_payloads=len(payloads),
                large_payloads_rehashed=False, large_payloads_reparsed=False,
                cached_analysis_sha256=sha(report_path))


def guard_proof(run, cfg):
    posts = sorted((run / "dashboard_guard/checks").glob("*-post.json"))
    require(len(posts) == 1, "Expected exactly one final Guard post: " + str(run))
    post = read_json(posts[0])
    require(post.get("phase") == "post" and post.get("experiment_id") == cfg["id"]
            and post.get("state") == "complete" and post.get("valid") == post.get("expected") == 1
            and post.get("exit_code") == 0
            and all(post.get(key) is True for key in
                    ("passed", "cleanup_verified", "scientific_success")), "Guard final post failed")
    require(post.get("dashboard", {}).get("spec_sha256") == sha(run / "dashboard_spec.json"),
            "Guard checked different frozen spec")
    cache = post.get("model_cache", {})
    require(cache.get("passed") is True and cache.get("policy") == "required"
            and cache.get("provider") == "sglang_canonical_full_weight_v1", "Required canonical full-bank proof missing")
    task = read_json(run / "task/results.json")
    cleanup = read_json(run / "task/cleanup.json")
    require(task.get("passed") is True and task.get("valid_points") == task.get("expected") == 1,
            "Task single point did not pass")
    require(len(task.get("arms", [])) == 1
            and all(task["arms"][0].get(key) is True for key in ("passed", "cleanup_verified")),
            "Task arm cleanup failed")
    require(all(cleanup.get(key) is True for key in
                ("passed", "gpu_idle", "model_processes_exited", "gpu_resources_released")),
            "Owned GPU/process cleanup failed")
    return dict(passed=True, guard_post=dict(path=str(posts[0]), sha256=sha(posts[0])),
                cleanup=dict(path=str(run / "task/cleanup.json"), sha256=sha(run / "task/cleanup.json")))


def point_gate(point, analyzed, mode, cohort):
    label = mode + "/" + cohort
    require(analyzed.get("status") == "valid" and analyzed.get("actual_outputs_complete") is True
            and analyzed.get("expected_tokens") == 64, label + ": incomplete actual output")
    if cohort in ("timed", "repeat"):
        require(point.get("microbatch_mode") == mode and point.get("request_count") == 8
            and point.get("prefix_tokens_per_request") == 4096
            and point.get("actual_cached_tokens_per_request") == 4032
            and point.get("first_forward_new_tokens_per_request") == 65
            and point.get("query_tokens_per_request") == 1
            and point.get("output_tokens_per_request") == 64
            and point.get("synthetic_kv") is False
            and point.get("old_context_constructed_by_model") is True
            and point.get("hicache_h2d_timed") is True
            and point.get("weights_loading_timed") is False
            and point.get("atomic_http_batch") is True,
                label + ": semantic/cache/workload boundary mismatch")
    else:
        require(point.get("performance_measurement") is False and point.get("cohort") == cohort,
                label + ": witness must be separate and untimed")
    rank_proof(point["actual_graph_proof"], label + "/Graph")
    for row in point["actual_graph_proof"]["ranks"]:
        require(row.get("target_verify_forwards", 0) > 0
                and row.get("actual_native_replays", {}).get("target", 0) > 0,
                label + ": no native Verify Graph replay")
        if row["rank"] >= 4:
            require(row.get("draft_calls", 0) > 0
                    and row.get("actual_native_replays", {}).get("draft", 0) > 0,
                    label + ": no tail native draft Graph replay")
    schedule = analyzed["schedule_audit"]
    rank_proof(schedule, label + "/schedule")
    require(schedule.get("identical_ordered_composition_at_each_microstep") is True
            and schedule.get("request_slot_migration") is False
            and schedule.get("observed_slots") == [0, 1], label + ": slot composition mismatch")
    scheduler = point["actual_microbatch_scheduler_proof"]
    require(scheduler.get("passed") is True and scheduler.get("mode") == mode
            and scheduler.get("actual_slots") == 2 and scheduler.get("max_requests_per_slot") == 4,
            label + ": mode/atomic scheduler proof missing")
    rank_proof(scheduler["deterministic_runtime"], label + "/deterministic")
    for key in ("humming_before", "humming_after"):
        rank_proof(analyzed[key], label + "/" + key)
        require(all(row.get("passed") is True and set(row.get("required_gemm_types", []))
                    == {"dense", "grouped_masked"} for row in analyzed[key]["ranks"]),
                label + ": loaded Humming invariants incomplete")
    for key in ("native_cpu_ready_gpu_empty", "all_rank_after"):
        receipts = point[key]
        require(len(receipts) == 8 and {row["rank"] for row in receipts} == RANKS,
                label + ": incomplete actual rank cache receipts")
        for row in receipts:
            require(row.get("passed") is True and row.get("native_hicache_quiescent") is True
                    and all(value == 0 for value in row["pending_counts"].values())
                    and row.get("source_pool_reads") == row.get("manual_host_kv_writes") == 0
                    and row.get("synthetic_kv_forbidden") is True, label + ": cache is not native/idle")
            require(not any(row.get("profiler", {}).get(k, False) for k in ("installed", "armed", "active")),
                    label + ": CUPTI installed or active")
            if key == "native_cpu_ready_gpu_empty":
                matches = row["matches"]
                require(len(matches) == 8 and {x["request_index"] for x in matches} == RANKS
                        and all(x["cpu_tokens"] == 4032 and x["gpu_tokens"] == 0 for x in matches),
                        label + ": not cold native HiCache P4032")
    if cohort in ("timed", "repeat"):
        require(point.get("cuda_profiler_off_during_production") is True, label + ": production profiler on")
        for key in ("zero_witness_before", "zero_witness_after"):
            proof = analyzed[key]
            rank_proof(proof, label + "/" + key)
            require(proof.get("witness_clones") == proof.get("witness_readbacks")
                    == proof.get("witness_flushes") == 0 and proof.get("no_cupti") is True,
                    label + ": timed witness activity")
            require(all(all(row.get(k) == 0 for k in
                        ("pending_snapshots", "pending_gpu_bytes", "total_selected_snapshots", "flushed_cohorts"))
                        for row in proof["ranks"]), label + ": nonzero timed witness counters")
    requests = point["requests"]
    require(len(requests) == 8 and {row["request_index"] for row in requests} == RANKS,
            label + ": incomplete request records")
    for row in requests:
        require(len(row["output_ids"]) == row["completion_tokens"] == 64
                and isinstance(row["output_text"], str)
                and row["meta"]["cached_tokens_details"] == {"device": 0, "host": 4032}
                and row["meta"].get("num_retractions") == 0, label + ": actual cache/output mismatch")


def audit_analysis_arm(arm):
    for cohort in WITNESS:
        records = arm["oracle_by_cohort_rank"][cohort]
        require(set(records) == TAIL, "Incomplete oracle tail ranks")
        for rank, record in records.items():
            oracle(record, cohort + "/" + rank)
        cross = arm["cross_tp_by_cohort"][cohort]
        require(set(cross) == {"4_vs_5", "4_vs_6", "4_vs_7"}, "Incomplete cross-TP comparisons")
        for name, value in cross.items():
            exact_comparison(value, cohort + "/" + name)
    repeat = arm["within_repeat_by_rank"]
    require(set(repeat) == TAIL, "Incomplete within-arm witness repeat comparisons")
    for rank, value in repeat.items():
        exact_comparison(value, "repeat/" + rank)
    expected = {"timed_vs_repeat", "witness_vs_witness-repeat", "timed_vs_witness", "repeat_vs_witness-repeat"}
    require(set(arm["output_exactness"]) == expected, "Incomplete within-arm output comparisons")
    for name, value in arm["output_exactness"].items():
        exact_outputs(value, name)


def stats(values):
    require(len(values) == 8 and all(math.isfinite(x) and x > 0 for x in values), "Invalid latency samples")
    return dict(mean=statistics.mean(values), median=statistics.median(values), p99=max(values),
                sample_count=8, p99_estimator="nearest_rank_N8_equals_sample_maximum")


def metrics(point, mode, cohort):
    row = dict(configuration=mode, cohort=cohort, wall_s=point["wall_s"],
               throughput_tok_s=point["throughput_tok_s"])
    require(math.isclose(point["throughput_tok_s"], 512 / point["wall_s"], rel_tol=1e-10),
            "Global throughput not equal to actual 8x64/wall")
    for key in ("ttft_s", "tpot_s", "end_to_end_s", "spec_accept_rate"):
        values = [request["meta"][key] if key == "spec_accept_rate" else request[key]
                  for request in point["requests"]]
        row[key] = stats(values)
        for estimate in ("mean", "median", "p99"):
            require(math.isclose(row[key][estimate], point[key][estimate], rel_tol=1e-10),
                    "Recorded metric disagrees with raw requests: " + key)
    for request in point["requests"]:
        require(math.isclose(request["tpot_s"],
                            (request["end_to_end_s"] - request["ttft_s"]) / 63, rel_tol=1e-10),
                "TPOT boundary changed")
    return row


def qualify(baseline, candidate, report_path):
    report = read_json(report_path)
    require(report.get("schema") == "pp-eagle-offline-first-divergence-analysis-v1"
            and report.get("cpu_only") is True and report.get("runtime_or_artifacts_modified") is False
            and report.get("prefix_tokens_per_request") == 4096,
            "Wrong or mutable analyzer report")
    source_proof = verify_cached_analysis_sources(report, report_path)
    rows, configs, guards = [], {}, {}
    points = {}
    for mode, run, name in (("baseline", baseline, "baseline"), ("candidate", candidate, "candidate")):
        run = Path(run).resolve()
        arm = report[name]
        require(Path(arm["run"]).resolve() == run, "Analyzer run identity mismatch")
        cfg = read_json(run / "config.json")
        require(cfg["mode"] == mode and cfg["production_output_tokens"] == 64
                and cfg["prefix_lengths"] == [4096] and cfg["deterministic_inference"] is True
                and cfg["prefill_truncation_alignment"] == 1024 and cfg["profile_enabled"] is False
                and cfg["production_repeats"] == cfg["untimed_witness_cohorts"] == 2,
                "Configuration protocol mismatch")
        configs[mode] = cfg
        guards[mode] = guard_proof(run, cfg)
        audit_analysis_arm(arm)
        require(set(arm["result_artifacts"]) == set(COHORTS), "Missing cohort result")
        for cohort in COHORTS:
            analyzed = arm["result_artifacts"][cohort]
            point = read_json(analyzed["source"]["path"])
            point_gate(point, analyzed, mode, cohort)
            points[mode, cohort] = point
            if cohort in ("timed", "repeat"):
                rows.append(metrics(point, mode, cohort))
    for key in ("engine_commit", "engine_root", "nodes", "jobs", "layouts", "prefix_lengths",
                "production_output_tokens", "semantic_inputs", "semantic_source_commit",
                "deterministic_inference", "prefill_truncation_alignment"):
        require(configs["baseline"][key] == configs["candidate"][key], "Unmatched configuration: " + key)
    manifests = [point["semantic_input_manifest"] for (mode, cohort), point in points.items()
                 if cohort in ("timed", "repeat")]
    require(all(value == manifests[0] for value in manifests), "Actual semantic prompt manifests differ")
    cross = report["cross_arm"]
    require(cross.get("status") == "evaluated"
            and set(cross["output_exactness"]) == set(COHORTS)
            and set(cross["witness_by_cohort_rank"]) == set(WITNESS), "Incomplete paired analysis")
    for cohort, value in cross["output_exactness"].items():
        exact_outputs(value, "cross-arm/" + cohort)
    for cohort, records in cross["witness_by_cohort_rank"].items():
        require(set(records) == TAIL, "Incomplete cross-arm witness tail ranks")
        for rank, value in records.items():
            exact_comparison(value, "cross-arm/" + cohort + "/" + rank)
    speedups = {}
    for cohort in ("timed", "repeat"):
        a = next(x for x in rows if x["configuration"] == "baseline" and x["cohort"] == cohort)
        b = next(x for x in rows if x["configuration"] == "candidate" and x["cohort"] == cohort)
        speedups[cohort] = dict(throughput_ratio=b["throughput_tok_s"] / a["throughput_tok_s"],
            tpot_mean_ratio=a["tpot_s"]["mean"] / b["tpot_s"]["mean"],
            tpot_mean_reduction_fraction=1-b["tpot_s"]["mean"]/a["tpot_s"]["mean"])
    return dict(schema="pp-eagle-bounded-pair-qualification-v1", bounded_workload_qualified=True,
        performance_comparison_claim_valid_for_this_bounded_workload=True,
        general_correctness_proven=False, model_quality_proven=False,
        non_speculative_target_parity_proven=False, actual_gpu_kv_value_equality_proven=False,
        default_microbatch_flag_changed=False, raw_runtime_validity_flags_modified=False,
        engine_commit=configs["baseline"]["engine_commit"],
        baseline=str(Path(baseline).resolve()), candidate=str(Path(candidate).resolve()),
        cached_witness_analysis=dict(path=str(Path(report_path).resolve()), **source_proof),
        qualifier_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        guard_and_cleanup=guards, metrics=rows, paired_speedups=speedups,
        protocol=dict(model="GLM-4.7-AWQ", layout="PP2 x Attention TP4 / Expert EP4", gpu="8 x A100 80GB",
            nodes=configs["baseline"]["nodes"], semantic_prefix_tokens_per_request=4096,
            native_hicache_host_tokens_per_request=4032, gpu_cached_tokens_before_timing=0,
            first_forward_new_tokens_per_request=65, query_tokens=1, output_tokens=64,
            requests=8, fixed_slots=2, initial_requests_per_slot=4,
            timed_cohorts=["timed", "repeat"], separate_untimed_cohorts=list(WITNESS),
            witness_cpu_oracle_snapshots_per_arm=256,
            native_deterministic=True, humming_batch_invariant=True, humming_stream_k=False,
            cupti_enabled=False, timings_include_native_hicache_h2d=True,
            timings_exclude_model_load_semantic_prefill_warmup_and_witness=True),
        limitations=report["limitations"] + ["Only eight fixed prompts, P4096, B8 and 64 generated tokens are qualified",
            "Two sequential production cohorts are not an independent-run confidence interval",
            "P99 is the maximum of eight requests, not a population tail-latency estimate",
            "Loaded Humming kernel metadata is not a per-launch trace",
            "No default scheduling flag was changed; broader model/workload correctness is not established"])


def write_report(value, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "qualification.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (out / "metrics.csv").open("x", newline="") as handle:
        fields = ["configuration", "cohort", "metric", "unit", "mean", "median", "p99", "sample_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in value["metrics"]:
            for key, unit, scale in (("ttft_s", "s", 1), ("tpot_s", "ms", 1000),
                                     ("end_to_end_s", "s", 1), ("spec_accept_rate", "%", 100)):
                writer.writerow(dict(configuration=row["configuration"], cohort=row["cohort"], metric=key,
                    unit=unit, sample_count=8, **{x: row[key][x]*scale for x in ("mean", "median", "p99")}))
            for key, unit in (("throughput_tok_s", "output tok/s"), ("wall_s", "s")):
                writer.writerow(dict(configuration=row["configuration"], cohort=row["cohort"], metric=key,
                                     unit=unit, mean=row[key], sample_count=1))
    lines = ["# PP2 × EP4: bounded semantic microbatch comparison", "",
        "Qualified only for this measured workload; no general correctness or model-quality claim.", "",
        "GLM-4.7-AWQ; 065/167; 8×A100 80GB; Attention TP4, Expert EP4, PP2; two fixed B4 slots.",
        "B8; semantic P4096 + Q1 → 64 tokens. Each cohort starts with native HiCache CPU P4032 and GPU P0; first forward has 65 tokens.",
        "Timing includes HiCache H2D, excludes model loading, semantic Prefill and warmup. Two timed cohorts run before the two separate untimed witness cohorts; zero witness clones/readbacks and no CUPTI during timing.", "",
        "| Configuration | Cohort | TTFT mean / median / P99 (s) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (s) | output tok/s |", "|---|---|---:|---:|---:|---:|"]
    for row in value["metrics"]:
        cells = [" / ".join(f"{row[key][estimate]*scale:.3f}" for estimate in ("mean", "median", "p99"))
                 for key, scale in (("ttft_s", 1), ("tpot_s", 1000), ("end_to_end_s", 1))]
        lines.append(f'| {row["configuration"]} | {row["cohort"]} | ' + " | ".join(cells) + f' | {row["throughput_tok_s"]:.3f} |')
    lines += ["", "P99 is nearest-rank over eight requests (the sample maximum). Two cohorts do not establish run-to-run statistical confidence.", "",
        "All run Guard posts and owned GPU cleanup passed; all eight ranks have native Graph, atomic schedule, deterministic and loaded-Humming invariant receipts. All saved 64-token outputs/text agree across both arms and all four cohorts. Every bounded witness oracle and exact repeat/cross-TP/cross-arm comparison passed.", "",
        "Humming receipts prove registered dense/grouped-masked cubin invariants, not each actual launch. Witnesses cover only the first 16 Verifies per slot and do not independently prove target logits or actual KV-value equality. The default microbatch flag remains unchanged.", ""]
    for cohort, row in value["paired_speedups"].items():
        lines.append(f'{cohort}: output throughput ×{row["throughput_ratio"]:.3f}; mean TPOT ×{row["tpot_mean_ratio"]:.3f} speedup ({row["tpot_mean_reduction_fraction"]*100:.1f}% reduction).')
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--witness-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(not any(out.is_relative_to(path.resolve()) for path in
                    (args.baseline, args.candidate, ROOT / "runtime")), "Output cannot alter frozen artifacts")
    value = qualify(args.baseline, args.candidate, args.witness_report)
    write_report(value, out)
    print(json.dumps(dict(report=str(out / "qualification.json"), csv=str(out / "metrics.csv"),
                          bounded_workload_qualified=True)))


if __name__ == "__main__":
    main()
