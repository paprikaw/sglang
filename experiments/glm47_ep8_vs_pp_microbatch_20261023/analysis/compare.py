#!/usr/bin/env python3
"""Offline, fail-closed comparison of the bounded semantic MTP Decode pair.

This module never imports the serving runtime, contacts nodes, or opens tensor
payloads. Raw measurements remain in the report when a qualification gate fails.
P99 is nearest-rank over eight requests, not a population tail estimate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ENGINE_COMMIT = "9e48f876a8eb2d7460317bc713bfa8680d78aea5"
SEMANTIC_COMMIT = "f415858311c0d7646b368b30b64b371dca3427e4"
ROOT = Path(__file__).resolve().parents[1]
PP_ROOT = ROOT.parent / "glm47_pp_microbatch_witness_run_20261023"
DEFAULT_PP = PP_ROOT / "runs/witness_micro_r1"
DEFAULT_EP8 = ROOT / "runs/ep8_r1"
DEFAULT_QUALIFICATION = PP_ROOT / "analysis/qualified_pair_r1/qualification.json"
MATCHED_POINT_FIELDS = (
    "workload", "request_count", "engine_commit", "prefix_tokens_per_request",
    "total_context_tokens", "actual_cached_tokens_per_request",
    "prefix_tail_recomputed_tokens", "first_forward_new_tokens_per_request",
    "query_tokens_per_request", "output_tokens_per_request", "throughput_kind",
    "weights_loading_timed", "hicache_h2d_timed", "persistent_source_kv_bank",
    "old_context_constructed_by_model", "synthetic_kv", "atomic_http_batch",
)
EXPECTED_POINT = dict(
    workload="decode", request_count=8, engine_commit=ENGINE_COMMIT,
    prefix_tokens_per_request=4096, total_context_tokens=32768,
    actual_cached_tokens_per_request=4032, prefix_tail_recomputed_tokens=64,
    first_forward_new_tokens_per_request=65, query_tokens_per_request=1,
    output_tokens_per_request=64, throughput_kind="output_tokens",
    weights_loading_timed=False, hicache_h2d_timed=True,
    persistent_source_kv_bank=False, old_context_constructed_by_model=True,
    synthetic_kv=False, atomic_http_batch=True, profile_only_performance_not_production_measurement=False,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, sources: list) -> dict:
    raw = path.read_bytes()
    sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def check(ok: bool, description: str, errors: list[str]) -> None:
    if not ok:
        errors.append(description)


def all_ranks(rows: list, label: str, errors: list[str]) -> None:
    ranks = [r.get("rank") for r in rows]
    check(len(rows) == 8 and sorted(ranks, key=str) == list(range(8)),
          f"{label}: expected each world rank 0..7 exactly once", errors)


def close(a, b) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def stats(values: list[float]) -> dict:
    if not values or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("finite, nonempty numeric sample required")
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p99": sorted(values)[math.ceil(0.99 * len(values)) - 1],
            "sample_count": len(values),
            "p99_estimator": f"nearest_rank_N{len(values)}"}


def output_parity(left: dict, right: dict) -> dict:
    a = {r["request_index"]: r for r in left.get("requests", [])}
    b = {r["request_index"]: r for r in right.get("requests", [])}
    rows = []
    for index in sorted(a.keys() | b.keys()):
        x, y = a.get(index, {}), b.get(index, {})
        xi, yi = x.get("output_ids"), y.get("output_ids")
        same = xi is not None and yi is not None and xi == yi
        first = None
        if not same and xi is not None and yi is not None:
            first = next((i for i, (u, v) in enumerate(zip(xi, yi)) if u != v), min(len(xi), len(yi)))
        rows.append({"request_index": index, "token_ids_equal": same,
                     "text_equal": "output_text" in x and "output_text" in y and x["output_text"] == y["output_text"],
                     "first_different_output_position_zero_based": first})
    return {"passed": len(rows) == 8 and all(r["token_ids_equal"] and r["text_equal"] for r in rows),
            "requests": rows, "exact_logits_compared": False,
            "mismatch_interpretation": "Diagnostic only; no numerical-branching or KV-correctness explanation is assumed"}


def summarize_point(point: dict, errors: list[str], label: str) -> dict:
    """Recompute every metric from requests, validating the streaming boundary."""
    rows = point.get("requests", [])
    check(len(rows) == 8 and sorted(r.get("request_index", -1) for r in rows) == list(range(8)),
          f"{label}: exact eight request indices required", errors)
    for row in rows:
        ix = row.get("request_index")
        events = row.get("token_events", [])
        n = row.get("completion_tokens")
        check(n == 64 and len(row.get("output_ids", [])) == 64,
              f"{label}/request{ix}: expected 64 generated token IDs", errors)
        check(bool(events) and events[0][0] == 1 and row.get("first_stream_chunk_tokens") == 1,
              f"{label}/request{ix}: first stream event must expose exactly token 1", errors)
        if not events:
            continue
        check(events[-1][0] == n and all(a[0] < b[0] and a[1] <= b[1] for a, b in zip(events, events[1:])),
              f"{label}/request{ix}: non-monotone or incomplete stream", errors)
        check(close(row.get("ttft_s"), events[0][1]) and close(row.get("end_to_end_s"), events[-1][1])
              and close(row.get("tpot_s"), (events[-1][1] - events[0][1]) / 63),
              f"{label}/request{ix}: client TTFT/TPOT/E2E disagrees with stream", errors)
        meta = row.get("meta", {})
        cached = meta.get("cached_tokens_details") or {}
        check(meta.get("cached_tokens") == 4032 and cached.get("device") == 0 and cached.get("host") == 4032,
              f"{label}/request{ix}: must load 4032 native host cached tokens and zero device prefix tokens", errors)
        check(meta.get("prompt_tokens") == 4097 and row.get("prefix_tokens") == 4096 and row.get("query_tokens") == 1,
              f"{label}/request{ix}: semantic P/Q mismatch", errors)
    result = {"request_count": len(rows), "wall_s": point.get("wall_s"),
              "throughput_kind": "output_tokens", "metrics_source": "recomputed from client request rows"}
    for key in ("ttft_s", "tpot_s", "end_to_end_s"):
        measured = stats([r[key] for r in rows])
        result[key] = measured
        check(all(close(point.get(key, {}).get(s), measured[s]) for s in ("mean", "median", "p99")),
              f"{label}: aggregate {key} differs from request rows", errors)
    wall = point.get("wall_s", 0)
    check(isinstance(wall, (float, int)) and wall > 0 and wall >= max(r["end_to_end_s"] for r in rows),
          f"{label}: cohort wall time must cover all completions", errors)
    result["throughput_tok_s"] = sum(r["completion_tokens"] for r in rows) / wall
    check(close(point.get("throughput_tok_s"), result["throughput_tok_s"]), f"{label}: throughput mismatch", errors)
    for key in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct"):
        result[key] = stats([r["meta"][key] for r in rows])
        check(all(close(point.get(key, {}).get(s), result[key][s]) for s in ("mean", "median", "p99")),
              f"{label}: aggregate {key} differs from request metadata", errors)
    result["mtp_totals"] = {key: (sum(r["meta"][key] for r in rows) if all(key in r["meta"] for r in rows) else None)
                            for key in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")}
    result["mtp_per_request"] = [{"request_index": r["request_index"],
                                  **{key: r["meta"].get(key) for key in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct")}}
                                 for r in sorted(rows, key=lambda x: x["request_index"])]
    groups = [[r["ttft_s"] for r in rows if r["request_index"] // 4 == dp] for dp in range(2)]
    result["logical_group_ttft"] = {"group0_request_indices": [0, 1, 2, 3], "group1_request_indices": [4, 5, 6, 7],
        "group0_s": stats(groups[0]), "group1_s": stats(groups[1]),
        "group1_minus_group0_mean_s": statistics.mean(groups[1]) - statistics.mean(groups[0]),
        "clock_boundary": "same client HTTP batch submission clock"}
    return result


def validate_semantics(cfg: dict, point: dict, sources: list, errors: list[str]) -> dict:
    manifest = point.get("semantic_input_manifest", {})
    check(read_json(Path(cfg["semantic_inputs"]) / "manifest.json", sources) == manifest,
          "saved semantic corpus manifest differs from point manifest", errors)
    check(manifest.get("source_commit") == SEMANTIC_COMMIT == cfg.get("semantic_source_commit"),
          "semantic source revision mismatch", errors)
    check(manifest.get("input_ids_are_semantic") is True and manifest.get("kv_requires_actual_model_prefill") is True
          and manifest.get("kv_tensors_constructed") is False, "actual semantic model Prefill required", errors)
    check(manifest.get("raw_prompt_tokens") == 4097 and manifest.get("raw_prefix_tokens") == 4096
          and manifest.get("expected_native_bigram_cached_tokens") == 4032 and manifest.get("request_count") == 8,
          "semantic manifest P/Q/cache size mismatch", errors)
    entries = manifest.get("requests", [])
    check(len(entries) == 8 and sorted(r.get("request_index", -1) for r in entries) == list(range(8)),
          "semantic manifest requires all eight distinct request indices", errors)
    identities = []
    for row in entries:
        path = Path(cfg["semantic_inputs"]) / row["file"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        sources.append({"path": str(path.resolve()), "sha256": digest})
        check(len(raw) == 4097 * 4 and row.get("tokens") == 4097 and digest == row.get("sha256"),
              f"semantic bytes do not match manifest: request {row.get('request_index')}", errors)
        identities.append({"request_index": row["request_index"], "sha256": digest,
                           "source_files": row.get("source_files", [])})
    unique_source_files = {s["path"]: s["source_sha256"] for row in entries for s in row.get("source_files", [])}
    check(bool(unique_source_files), "semantic source provenance is missing", errors)
    for source_path, expected in unique_source_files.items():
        path = Path(source_path)
        digest = sha256(path)
        sources.append({"path": str(path.resolve()), "sha256": digest})
        check(digest == expected, f"semantic source bytes changed: {source_path}", errors)
    return {"requests": sorted(identities, key=lambda x: x["request_index"]),
            "tokenizer_config_sha256": manifest.get("tokenizer_config_sha256"),
            "model_path": manifest.get("model_path"), "source_commit": manifest.get("source_commit")}


def validate_receipts(point: dict, layout: str, errors: list[str]) -> None:
    for key in ("native_cpu_ready_gpu_empty", "semantic_ready_receipts", "all_rank_after"):
        receipts = point.get(key, [])
        all_ranks(receipts, key, errors)
        for receipt in receipts:
            rank = receipt.get("rank")
            label = f"{key}/rank{rank}"
            check(receipt.get("passed") is True and receipt.get("native_hicache_quiescent") is True
                  and not any(receipt.get("pending_counts", {"missing": 1}).values()), f"{label}: HiCache not quiescent", errors)
            check(receipt.get("source_pool_reads") == 0 and receipt.get("manual_host_kv_writes") == 0
                  and receipt.get("synthetic_kv_forbidden") is True, f"{label}: unexpected KV construction/replay", errors)
            profiler = receipt.get("profiler", {})
            check(all(profiler.get(x) is False for x in ("installed", "armed", "active")),
                  f"{label}: production profiler hooks active", errors)
            graph_probe = receipt.get("graph", {})
            check(graph_probe.get("installed") is True
                  and all(graph_probe.get(k) == 0 for k in ("gpu_synchronizations_added", "cuda_events_added", "device_tensor_readbacks_added")),
                  f"{label}: Graph proof added timed device instrumentation", errors)
            witness = receipt.get("witness", {})
            if witness:
                check(all(witness.get(k) == 0 for k in ("pending_snapshots", "pending_gpu_bytes", "total_selected_snapshots", "flushed_cohorts")),
                      f"{label}: witness affected production cohort", errors)
            d = receipt.get("deterministic", {})
            expected = dict(source="initialized_scheduler_rank", resolved_enabled=True, declared_enabled=True,
                            resolved_attention_backend="triton", actual_target_backend="TritonAttnBackend",
                            actual_target_deterministic=True, nccl_algo="allreduce:tree", nccl_min_nchannels="8",
                            nccl_max_nchannels="8", actual_truncation_align_size=1024,
                            actual_chunked_prefill_size=1024, effective_max_prefill_tokens=1024)
            check(all(d.get(k) == v for k, v in expected.items()), f"{label}: initialized deterministic/Triton configuration mismatch", errors)
            humming = receipt.get("humming", {})
            kernels = humming.get("kernels", [])
            check(receipt.get("humming_validation", {}).get("passed") is True
                  and {"dense", "grouped_masked"}.issubset({k.get("gemm_type") for k in kernels})
                  and all(k.get("use_batch_invariant") is True and k.get("use_stream_k") is False and k.get("sm_version") == 80 for k in kernels),
                  f"{label}: loaded Humming invariant metadata mismatch", errors)
            if key == "native_cpu_ready_gpu_empty":
                matches = receipt.get("matches", [])
                expected_indices = list(range(8)) if layout == "pp2ep4" else list(range(4 * (rank // 4), 4 * (rank // 4 + 1)))
                check(sorted(m.get("request_index", -1) for m in matches) == expected_indices
                      and all(m.get("gpu_tokens") == 0 and m.get("cpu_tokens") == 4032 for m in matches),
                      f"{label}: actual CPU-ready/GPU-empty matches differ from local admission", errors)
    graph = point.get("actual_graph_proof", {})
    all_ranks(graph.get("ranks", []), "actual_graph_proof", errors)
    check(graph.get("passed") is True, "actual Graph replay proof failed", errors)
    for r in graph.get("ranks", []):
        check(r.get("target_verify_forwards", 0) > 0
              and r.get("actual_native_replays", {}).get("target") == r.get("target_verify_forwards"),
              f"rank{r.get('rank')}: not every target Verify used native Graph replay", errors)
        if layout == "ep8" or r.get("rank", -1) >= 4:
            check(r.get("draft_calls", 0) > 0 and all(r.get("actual_native_replays", {}).get(k, 0) > 0 for k in ("draft", "draft_extend")),
                  f"rank{r.get('rank')}: actual draft Graph evidence missing", errors)


def validate_scheduler(point: dict, layout: str, errors: list[str]) -> dict:
    if layout == "pp2ep4":
        proof = point.get("actual_microbatch_scheduler_proof", {})
        check(proof.get("passed") is True and proof.get("actual_slots") == 2 and proof.get("max_requests_per_slot") == 4
              and proof.get("feedback_group") == "isolated", "PP: expected initialized two B4 slots with isolated feedback", errors)
        ranks = proof.get("actual_slot_request_composition", [])
        all_ranks(ranks, "PP slot composition", errors)
        for rank in ranks:
            expected = [[r["rid"] for r in point["requests"] if r["request_index"] in range(i * 4, (i + 1) * 4)] for i in range(2)]
            actual = [r.get("request_ids", []) for r in sorted(rank.get("slot_request_groups", []), key=lambda x: x["slot_id"])]
            # Initial physical ring-slot numbering may globally rotate. The
            # two ordered logical request groups, not slot 0's label, are fixed.
            check(sorted(map(tuple, actual)) == sorted(map(tuple, expected)),
                  f"PP/rank{rank.get('rank')}: actual slot payload differs", errors)
        return {"kind": "two PP slots", "groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "proof": "actual_microbatch_scheduler_proof", "passed": proof.get("passed") is True}
    proof = point.get("actual_dp_scheduler_proof", {})
    check(proof.get("passed") is True and proof.get("attention_tp") == 4 and proof.get("attention_dp") == 2
          and proof.get("expert_ep") == 8 and proof.get("pipeline_stages") == 1,
          "EP8 actual TP4/DP2/EP8/PP1 scheduler proof failed", errors)
    check(proof.get("stock_colocated_input_blocker_enabled") is False
          and all(proof.get(k) == 0 for k in ("added_collectives", "device_reads", "gpu_synchronizations")),
          "EP8 admission control added collectives/device synchronization", errors)
    ranks = proof.get("ranks", [])
    all_ranks(ranks, "EP8 DP scheduler proof", errors)
    for row in ranks:
        rank = row.get("rank", -1)
        dp = rank // 4
        indices = list(range(dp * 4, (dp + 1) * 4))
        ids = [r["rid"] for r in point["requests"] if r["request_index"] in indices]
        admission, first = row.get("atomic_admission", {}), row.get("first_forward", {})
        check(row.get("dp_rank") == dp and row.get("request_indices") == indices
              and row.get("request_dp_migration") is False
              and row.get("tp_peers_have_identical_ordered_submissions") is True,
              f"EP8/rank{rank}: local DP identities/order/migration invalid", errors)
        check(admission.get("request_indices") == indices and admission.get("request_ids") == ids
              and admission.get("request_count") == 4 and admission.get("dp_rank") == dp
              and admission.get("atomic_local_release") is True
              and admission.get("observed_at_scheduler_not_inferred_from_http") is True
              and all(admission.get(k) == 0 for k in ("added_collectives", "device_reads", "gpu_synchronizations")),
              f"EP8/rank{rank}: actual local B4 release invalid", errors)
        check(first.get("batch_size") == 4 and first.get("request_ids") == ids
              and first.get("extend_num_tokens") == 260 and row.get("maximum_local_batch_size") == 4
              and row.get("local_submission_count", 0) >= 2,
              f"EP8/rank{rank}: expected first local B4 with 65 tokens/request", errors)
    first_times = {r["rank"]: r.get("first_forward", {}).get("timestamp") for r in ranks}
    leaders = {dp: first_times.get(dp * 4) for dp in range(2)}
    gap = None if any(v is None for v in leaders.values()) else leaders[1] - leaders[0]
    return {"kind": "two Attention DP groups with local B4 admission", "groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "proof": "actual_dp_scheduler_proof", "passed": proof.get("passed") is True,
            "first_forward_host_enqueue": {"leader_rank0_epoch_s": leaders[0], "leader_rank4_epoch_s": leaders[1],
                "rank4_minus_rank0_observed_s": gap,
                "clock_boundary": "CPU pre-forward time.time() on different nodes; clock-offset not independently calibrated, not GPU start-time difference"},
            "runtime_proof": proof}


def load_arm(run: Path, layout: str) -> dict:
    errors, sources = [], []
    cfg = read_json(run / "config.json", sources)
    task = read_json(run / "task/results.json", sources)
    cleanup = read_json(run / "task/cleanup.json", sources)
    arm = read_json(run / layout / "arm_result.json", sources)
    profile = read_json(run / "mtp_profile_qualification.json", sources)
    posts = sorted((run / "dashboard_guard/checks").glob("*-post.json"))
    if not posts:
        raise FileNotFoundError(f"Guard post-check not saved yet: {run}")
    post = read_json(posts[-1], sources)
    check(cfg.get("engine_commit") == ENGINE_COMMIT, f"{layout}: engine revision mismatch", errors)
    check(cfg.get("deterministic_inference") is True and cfg.get("profile_enabled") is False,
          f"{layout}: deterministic/unprofiled protocol required", errors)
    check(task.get("passed") is True and task.get("valid_points") == 1 and arm.get("passed") is True
          and arm.get("cleanup_verified") is True, f"{layout}: incomplete arm/task", errors)
    check(all(cleanup.get(k) is True for k in ("passed", "gpu_idle", "model_processes_exited", "gpu_resources_released")),
          f"{layout}: cleanup evidence failed", errors)
    check(profile.get("passed") is True and profile.get("same_complete_bank") is True
          and profile.get("disk_fallback") is False and profile.get("guard_required_checks_unchanged") is True,
          f"{layout}: canonical complete-bank/Guard qualification required", errors)
    check(post.get("passed") is True and post.get("phase") == "post" and post.get("state") == "complete"
          and post.get("cleanup_verified") is True and post.get("scientific_success") is True,
          f"{layout}: final Guard post-check not scientifically complete", errors)
    cache = post.get("model_cache", {})
    check(cache.get("passed") is True and cache.get("policy") == "required"
          and cache.get("provider") == "sglang_canonical_full_weight_v1",
          f"{layout}: final Guard full-weight bank proof missing", errors)
    bank_identities = sorted({(n.get("node"), n.get("group_key"), n.get("model_identity_sha256"))
                              for n in cache.get("nodes", [])})
    check(sorted({r[0] for r in bank_identities}) == sorted(cfg.get("nodes", []))
          and all(r[1] and r[2] for r in bank_identities), f"{layout}: full-weight bank node/identity mismatch", errors)
    points, summaries, semantic = {}, {}, None
    for cohort, filename in (("timed", "p4096_decode_results.json"), ("repeat", "p4096_decode_repeat_results.json")):
        point = read_json(run / layout / filename, sources)
        points[cohort] = point
        check(point.get("layout") == layout and point.get("cohort") == cohort, f"{layout}/{cohort}: identity mismatch", errors)
        for k, v in EXPECTED_POINT.items():
            check(type(point.get(k)) is type(v) and point.get(k) == v, f"{layout}/{cohort}: protocol mismatch {k}", errors)
        prep = point.get("preparation", {})
        check(prep.get("actual_model_prefill") is True and prep.get("synthetic_kv") is False
              and prep.get("excluded_from_decode_timing") is True and prep.get("checkpoint_replay") is False,
              f"{layout}/{cohort}: semantic preparation timing mismatch", errors)
        check(point.get("cuda_profiler_off_during_production") is True
              and point.get("startup_calibration_and_dormant_cpu_hooks_present") is False,
              f"{layout}/{cohort}: profiled/dormant-hook timing not allowed", errors)
        current = validate_semantics(cfg, point, sources, errors)
        if semantic is not None:
            check(semantic == current, f"{layout}: repeat semantic identities changed", errors)
        semantic = current
        validate_receipts(point, layout, errors)
        summaries[cohort] = summarize_point(point, errors, f"{layout}/{cohort}")
        summaries[cohort]["scheduler"] = validate_scheduler(point, layout, errors)
    repeat = output_parity(points["timed"], points["repeat"])
    check(repeat["passed"], f"{layout}: exact within-process output-ID/text repeat failed", errors)
    check(points["timed"].get("repeatability", {}).get("passed") is True, f"{layout}: embedded repeatability failed", errors)
    routes = [read_json(run / layout / f"route_rank{i}.json", sources) for i in range(8)]
    all_ranks(routes, f"{layout} DeepEP route", errors)
    for route in routes:
        check(route.get("passed") is True and route.get("low_latency") is True and route.get("nvlink_pair_only") is True
              and route.get("pcie_threshold") == 0, f"{layout}: expected pair NVLink/RDMA-outside-pair route", errors)
    route_identity = [{k: r.get(k) for k in ("rank", "sha256", "low_latency", "nvlink_pair_only", "pcie_threshold", "route_policy")} for r in routes]
    return {"run": str(run.resolve()), "layout": layout, "config": cfg, "points": points,
            "summaries": summaries, "semantic_identity": semantic, "repeat_parity": repeat,
            "routes": route_identity, "bank_identities": bank_identities, "errors": errors, "sources": sources}


def compare(pp_run: Path, ep8_run: Path, qualification: Path) -> dict:
    pp, ep8 = load_arm(pp_run, "pp2ep4"), load_arm(ep8_run, "ep8")
    errors = [*pp["errors"], *ep8["errors"]]
    sources = [*pp["sources"], *ep8["sources"]]
    qualified_pp = read_json(qualification, sources)
    check(qualified_pp.get("bounded_workload_qualified") is True
          and qualified_pp.get("performance_comparison_claim_valid_for_this_bounded_workload") is True
          and qualified_pp.get("engine_commit") == ENGINE_COMMIT
          and Path(qualified_pp.get("candidate", "/missing")).resolve() == pp_run.resolve(),
          "saved PP bounded qualification does not match this candidate", errors)
    check(pp["config"].get("nodes") == ep8["config"].get("nodes") and pp["config"].get("jobs") == ep8["config"].get("jobs"),
          "comparison must use the same nodes and bookings", errors)
    check(pp["semantic_identity"] == ep8["semantic_identity"], "cross-topology semantic token bytes/provenance mismatch", errors)
    check(pp["routes"] == ep8["routes"], "cross-topology DeepEP binary/route policy mismatch", errors)
    check(pp["bank_identities"] == ep8["bank_identities"], "cross-topology complete-weight bank identities mismatch", errors)
    cross, ratios, equivalence_errors = {}, {}, []
    for cohort in ("timed", "repeat"):
        a, b = pp["points"][cohort], ep8["points"][cohort]
        check(all(a.get(k) == b.get(k) for k in MATCHED_POINT_FIELDS), f"{cohort}: unmatched timing/workload protocol", errors)
        cross[cohort] = output_parity(a, b)
        check(cross[cohort]["passed"], f"{cohort}: exact cross-topology output-ID/text parity failed; not an equal-token-history overhead comparison", equivalence_errors)
        pa, ea = pp["summaries"][cohort], ep8["summaries"][cohort]
        ratios[cohort] = {"pp_over_ep8_throughput": pa["throughput_tok_s"] / ea["throughput_tok_s"],
                          "pp_over_ep8_mean_tpot": pa["tpot_s"]["mean"] / ea["tpot_s"]["mean"],
                          "pp_over_ep8_mean_ttft": pa["ttft_s"]["mean"] / ea["ttft_s"]["mean"],
                          "pp_over_ep8_mean_e2e": pa["end_to_end_s"]["mean"] / ea["end_to_end_s"]["mean"],
                          "pp_over_ep8_total_verifies": pa["mtp_totals"]["spec_verify_ct"] / ea["mtp_totals"]["spec_verify_ct"],
                          "pp_over_ep8_mean_accept_length": pa["spec_accept_length"]["mean"] / ea["spec_accept_length"]["mean"]}
    completed = not errors
    errors.extend(equivalence_errors)
    status = "QUALIFIED_BOUNDED_COMPARISON" if not errors else ("MEASURED_SAME_INPUTS_NOT_TOKEN_EQUIVALENT" if completed else "DIAGNOSTIC_PROTOCOL_GATE_FAILED")
    return {"schema": "glm47-ep8-vs-optimized-pp-bounded-decode-v1", "qualified": not errors,
            "status": status, "completed_native_measured": completed,
            "strict_cross_topology_equivalence": not equivalence_errors,
            "comparison_scope": "same semantic inputs and controlled native configurations; not pure equal-token-history PP overhead" if equivalence_errors else "bounded exact-output-equivalent native configurations",
            "errors": errors, "measurements_retained_on_gate_failure": True,
            "engine_commit": ENGINE_COMMIT, "protocol": EXPECTED_POINT,
            "topologies": {"ep8": "Attention TP4 x DP2; Expert EP8, PP1",
                           "pp2ep4": "Attention TP4; Expert EP4 per PP stage; PP2, two fixed B4 slots"},
            "arms": {a["layout"]: {k: a[k] for k in ("run", "summaries", "repeat_parity", "routes", "bank_identities")} for a in (ep8, pp)},
            "cross_topology_output_parity": cross, "ratios": ratios,
            "limitations": ["Bounded P4096 B8 Q1 to 64 semantic MTP Decode only; not general model-quality proof",
                            "Actual native HiCache prefix hit is 4032; 64 prefix tokens are recomputed, so first forward processes 65 new tokens",
                            "TTFT includes native HiCache H2D, queueing and first-forward work; model loading, semantic Prefill, warmup and witness are excluded",
                            "TPOT = (last client token time - first client token time) / 63, not mean Verify-step latency",
                            "P99 is maximum of eight requests; two sequential cohorts in one server are not independent-run confidence intervals",
                            "PP has separately qualified bounded untimed witness evidence; EP8 has no new untimed CPU oracle cohort",
                            "Humming loaded-kernel invariant metadata is not a per-launch trace",
                            "Exact output parity does not prove exact logits, GPU KV values, or non-speculative target parity",
                            "MTP acceptance/Verify counts are reported, not required equal; a timing difference alone is not communication-only attribution",
                            "Transport is NVLink within pairs and RDMA outside pairs, not Copy Engine PCIe optimization",
                            "EP8 uses explicit per-RID DP routing and local B4 admission buffering; PP uses two fixed B4 slots"],
            "sources": sources}


def markdown(result: dict) -> str:
    out = ["# GLM-4.7-AWQ: EP8 versus optimized PP2×EP4", "", f"Status: **{result['status']}**", "",
           f"Completed native measurements: **{result['completed_native_measured']}**. Strict cross-topology token equivalence: **{result['strict_cross_topology_equivalence']}**.", "",
           "P4096 · B8 · Q1 → 64 output tokens · semantic MTP · 8×A100 · same nodes. "
           "Native CPU HiCache starts with 4032 token hits/request and GPU prefix empty; the first forward includes 64 recomputed prefix tokens.", "",
           "| Cohort | Configuration | Output tok/s | TTFT mean / median / P99 (s) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (s) |",
           "|---|---|---:|---:|---:|---:|"]
    for cohort in ("timed", "repeat"):
        for arm, label in (("ep8", "EP8"), ("pp2ep4", "PP2×EP4 optimized")):
            p = result["arms"][arm]["summaries"][cohort]
            cells = []
            for k, scale in (("ttft_s", 1), ("tpot_s", 1000), ("end_to_end_s", 1)):
                cells.append(" / ".join(f"{p[k][s] * scale:.3f}" for s in ("mean", "median", "p99")))
            out.append(f"| {cohort} | {label} | {p['throughput_tok_s']:.2f} | " + " | ".join(cells) + " |")
    out.extend(["", "## MTP", "", "| Cohort | Configuration | Accept length mean / median / P99 | Verify count mean / median / P99 | Total Verifies |", "|---|---|---:|---:|---:|"])
    for cohort in ("timed", "repeat"):
        for arm in ("ep8", "pp2ep4"):
            p = result["arms"][arm]["summaries"][cohort]
            a, v = (" / ".join(f"{p[k][s]:.3f}" for s in ("mean", "median", "p99")) for k in ("spec_accept_length", "spec_verify_ct"))
            out.append(f"| {cohort} | {arm} | {a} | {v} | {p['mtp_totals']['spec_verify_ct']} |")
    out += ["", "## Qualification", "", f"Cross-topology output-ID/text parity: {json.dumps({k:v['passed'] for k,v in result['cross_topology_output_parity'].items()})}."]
    out += ["", "| Cohort | EP8 group1 − group0 mean TTFT (ms, common client clock) | EP8 rank4 − rank0 host first-forward timestamps (ms, uncalibrated node clocks) |",
            "|---|---:|---:|"]
    for cohort in ("timed", "repeat"):
        p = result["arms"]["ep8"]["summaries"][cohort]
        gap = p["scheduler"]["first_forward_host_enqueue"]["rank4_minus_rank0_observed_s"]
        observed = "unavailable" if gap is None else f"{gap * 1000:.3f}"
        out.append(f"| {cohort} | {p['logical_group_ttft']['group1_minus_group0_mean_s'] * 1000:.3f} | {observed} |")
    if result["errors"]:
        if result["completed_native_measured"]:
            out += ["", "Both completed measurements are valid as same-input configuration performance. "
                    "Different output histories and MTP acceptance mean this is not a pure equal-history PP/communication overhead comparison.", ""]
        else:
            out += ["", "Measurements are preserved for diagnosis; at least one protocol gate failed.", ""]
        out += [f"- {e}" for e in result["errors"]]
    out += ["", "## Boundaries", ""] + [f"- {s}" for s in result["limitations"]]
    out += ["", "Full per-request MTP metrics and source SHA256 receipts are in `comparison.json`.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp-run", type=Path, default=DEFAULT_PP)
    parser.add_argument("--ep8-run", type=Path, default=DEFAULT_EP8)
    parser.add_argument("--pp-qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/paired_r1")
    args = parser.parse_args(argv)
    try:
        result = compare(args.pp_run, args.ep8_run, args.pp_qualification)
    except (FileNotFoundError, KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        # Never manufacture partial metrics or turn absent runs into zeros.
        print(json.dumps({"status": "NOT_READY_OR_SCHEMA_INVALID", "error": str(exc)}))
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "RESULTS.md").write_text(markdown(result))
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        fields = ["cohort", "configuration", "throughput_tok_s", "wall_s"]
        fields += [f"{key}_{s}" for key in ("ttft_s", "tpot_s", "end_to_end_s", "spec_accept_length", "spec_accept_rate", "spec_verify_ct") for s in ("mean", "median", "p99")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort in ("timed", "repeat"):
            for arm in ("ep8", "pp2ep4"):
                p = result["arms"][arm]["summaries"][cohort]
                row = dict(cohort=cohort, configuration=arm, throughput_tok_s=p["throughput_tok_s"], wall_s=p["wall_s"])
                row.update({f"{key}_{s}": p[key][s] for key in ("ttft_s", "tpot_s", "end_to_end_s", "spec_accept_length", "spec_accept_rate", "spec_verify_ct") for s in ("mean", "median", "p99")})
                writer.writerow(row)
    print(json.dumps({"status": result["status"], "qualified": result["qualified"], "out": str(args.out), "errors": result["errors"]}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
