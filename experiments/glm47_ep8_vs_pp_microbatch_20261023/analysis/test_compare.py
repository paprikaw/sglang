"""Small CPU-only fixtures; no model imports, node access, or tensor payloads."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

SPEC = importlib.util.spec_from_file_location("offline_compare", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mutate(path, func):
    value = json.loads(path.read_text())
    func(value)
    write(path, value)


def manifest(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = tmp_path / "source.py"
    source.write_text("def schedule(requests):\n    return requests\n")
    rows = []
    for i in range(8):
        name = f"request{i}.u32le"
        raw = struct.pack("<4097I", *([i + 1] * 4097))
        (corpus / name).write_bytes(raw)
        rows.append(dict(request_index=i, file=name, tokens=4097,
                         sha256=hashlib.sha256(raw).hexdigest(),
                         source_files=[dict(path=str(source), source_sha256=compare.sha256(source))]))
    m = dict(source_commit=compare.SEMANTIC_COMMIT, input_ids_are_semantic=True,
             kv_requires_actual_model_prefill=True, kv_tensors_constructed=False,
             raw_prompt_tokens=4097, raw_prefix_tokens=4096,
             expected_native_bigram_cached_tokens=4032, request_count=8,
             requests=rows, tokenizer_config_sha256="a" * 64, model_path="/model/GLM-4.7-AWQ")
    write(corpus / "manifest.json", m)
    return corpus, m


def receipt(rank, layout):
    indices = range(8) if layout == "pp2ep4" else range(rank // 4 * 4, (rank // 4 + 1) * 4)
    return dict(passed=True, rank=rank, native_hicache_quiescent=True, pending_counts={"load": 0, "write": 0},
                source_pool_reads=0, manual_host_kv_writes=0, synthetic_kv_forbidden=True,
                profiler=dict(installed=False, armed=False, active=False),
                graph=dict(installed=True, gpu_synchronizations_added=0, cuda_events_added=0, device_tensor_readbacks_added=0),
                witness=dict(pending_snapshots=0, pending_gpu_bytes=0, total_selected_snapshots=0, flushed_cohorts=0),
                deterministic=dict(source="initialized_scheduler_rank", resolved_enabled=True, declared_enabled=True,
                                   resolved_attention_backend="triton", actual_target_backend="TritonAttnBackend",
                                   actual_target_deterministic=True, nccl_algo="allreduce:tree", nccl_min_nchannels="8",
                                   nccl_max_nchannels="8", actual_truncation_align_size=1024,
                                   actual_chunked_prefill_size=1024, effective_max_prefill_tokens=1024),
                humming_validation=dict(passed=True),
                humming=dict(kernels=[dict(gemm_type=k, use_batch_invariant=True, use_stream_k=False, sm_version=80)
                                      for k in ("dense", "grouped_masked")]),
                matches=[dict(request_index=i, gpu_tokens=0, cpu_tokens=4032) for i in indices])


def point(layout, cohort, m):
    rows = []
    for i in range(8):
        ttft, tpot = 0.2 + i * 0.01, (0.04 if layout == "pp2ep4" else 0.06) + i * 0.001
        e2e = ttft + tpot * 63
        rows.append(dict(request_index=i, rid=f"{cohort}-decode-4096-{i}", completion_tokens=64,
                         output_ids=list(range(64)), output_text="same semantic output",
                         token_events=[[1, ttft], [64, e2e]], first_stream_chunk_tokens=1,
                         ttft_s=ttft, tpot_s=tpot, end_to_end_s=e2e, prefix_tokens=4096, query_tokens=1,
                         meta=dict(cached_tokens=4032, cached_tokens_details=dict(device=0, host=4032),
                                   prompt_tokens=4097, spec_accept_length=2.5, spec_accept_rate=0.75,
                                   spec_verify_ct=26, spec_num_correct_drafts=38, spec_num_proposed_drafts=52)))
    p = dict(compare.EXPECTED_POINT, layout=layout, cohort=cohort, requests=rows,
             preparation=dict(actual_model_prefill=True, synthetic_kv=False, excluded_from_decode_timing=True,
                              checkpoint_replay=False), semantic_input_manifest=copy.deepcopy(m),
             cuda_profiler_off_during_production=True, startup_calibration_and_dormant_cpu_hooks_present=False,
             repeatability=dict(passed=True))
    p["wall_s"] = max(r["end_to_end_s"] for r in rows) + 0.01
    p["throughput_tok_s"] = 512 / p["wall_s"]
    for key in ("ttft_s", "tpot_s", "end_to_end_s"):
        p[key] = compare.stats([r[key] for r in rows])
    for key in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct"):
        p[key] = compare.stats([r["meta"][key] for r in rows])
    for key in ("native_cpu_ready_gpu_empty", "semantic_ready_receipts", "all_rank_after"):
        p[key] = [receipt(i, layout) for i in range(8)]
    p["actual_graph_proof"] = dict(passed=True, ranks=[dict(rank=i, target_verify_forwards=26,
        actual_native_replays=dict(target=26, draft=26, draft_extend=26), draft_calls=26) for i in range(8)])
    if layout == "pp2ep4":
        p["actual_microbatch_scheduler_proof"] = dict(passed=True, actual_slots=2, max_requests_per_slot=4,
            feedback_group="isolated", actual_slot_request_composition=[dict(rank=i,
                slot_request_groups=[dict(slot_id=j, request_ids=[r["rid"] for r in rows[j*4:(j+1)*4]]) for j in range(2)])
                for i in range(8)])
    else:
        proof_rows = []
        for rank in range(8):
            dp = rank // 4
            indices = list(range(dp * 4, (dp + 1) * 4))
            ids = [rows[i]["rid"] for i in indices]
            proof_rows.append(dict(rank=rank, dp_rank=dp, request_indices=indices,
                request_dp_migration=False, tp_peers_have_identical_ordered_submissions=True,
                atomic_admission=dict(request_indices=indices, request_ids=ids, request_count=4, dp_rank=dp,
                    atomic_local_release=True, observed_at_scheduler_not_inferred_from_http=True,
                    added_collectives=0, device_reads=0, gpu_synchronizations=0),
                first_forward=dict(batch_size=4, request_ids=ids, extend_num_tokens=260, timestamp=10 + dp * 0.01),
                maximum_local_batch_size=4, local_submission_count=26))
        p["actual_dp_scheduler_proof"] = dict(passed=True, attention_tp=4, attention_dp=2, expert_ep=8,
            pipeline_stages=1, stock_colocated_input_blocker_enabled=False,
            added_collectives=0, device_reads=0, gpu_synchronizations=0, ranks=proof_rows)
    return p


@pytest.fixture
def pair(tmp_path):
    corpus, m = manifest(tmp_path)
    runs = []
    for layout in ("pp2ep4", "ep8"):
        run = tmp_path / layout
        runs.append(run)
        write(run / "config.json", dict(engine_commit=compare.ENGINE_COMMIT,
            semantic_source_commit=compare.SEMANTIC_COMMIT, semantic_inputs=str(corpus),
            deterministic_inference=True, profile_enabled=False,
            nodes=["node0", "node1"], jobs={"node0": "1", "node1": "2"}))
        write(run / "task/results.json", dict(passed=True, valid_points=1))
        write(run / "task/cleanup.json", dict(passed=True, gpu_idle=True, model_processes_exited=True, gpu_resources_released=True))
        write(run / layout / "arm_result.json", dict(passed=True, cleanup_verified=True))
        write(run / "mtp_profile_qualification.json", dict(passed=True, same_complete_bank=True,
            disk_fallback=False, guard_required_checks_unchanged=True))
        write(run / "dashboard_guard/checks/123-post.json", dict(passed=True, phase="post", state="complete",
            cleanup_verified=True, scientific_success=True,
            model_cache=dict(passed=True, policy="required", provider="sglang_canonical_full_weight_v1",
                nodes=[dict(node=n, group_key="bank", model_identity_sha256="c" * 64) for n in ("node0", "node1")])))
        for cohort, name in (("timed", "p4096_decode_results.json"), ("repeat", "p4096_decode_repeat_results.json")):
            write(run / layout / name, point(layout, cohort, m))
        for i in range(8):
            write(run / layout / f"route_rank{i}.json", dict(passed=True, rank=i, low_latency=True,
                nvlink_pair_only=True, pcie_threshold=0, sha256="b" * 64,
                route_policy="NVLink within pairs, RDMA outside pairs; policy proof, not packet trace"))
    qualification = tmp_path / "qualification.json"
    write(qualification, dict(bounded_workload_qualified=True,
        performance_comparison_claim_valid_for_this_bounded_workload=True,
        engine_commit=compare.ENGINE_COMMIT, candidate=str(runs[0])))
    return *runs, qualification


def test_qualified_pair_recomputes_metrics_and_keeps_cohorts(pair):
    result = compare.compare(*pair)
    assert result["qualified"]
    assert result["cross_topology_output_parity"]["timed"]["passed"]
    p = result["arms"]["pp2ep4"]["summaries"]["repeat"]
    assert p["tpot_s"]["p99"] == pytest.approx(0.047)
    assert p["tpot_s"]["sample_count"] == 8
    assert p["mtp_totals"]["spec_verify_ct"] == 208
    assert len(p["mtp_per_request"]) == 8
    assert result["ratios"]["timed"]["pp_over_ep8_throughput"] > 1
    assert result["arms"]["ep8"]["summaries"]["timed"]["scheduler"]["first_forward_host_enqueue"]["rank4_minus_rank0_observed_s"] == pytest.approx(0.01)
    assert p["logical_group_ttft"]["group1_minus_group0_mean_s"] == pytest.approx(0.04)
    assert "4032" in compare.markdown(result)


def test_exact_source_bytes_revalidated(pair):
    source = pair[0].parent / "source.py"
    source.write_text("changed")
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert any("semantic source bytes changed" in e for e in result["errors"])


def test_pp_physical_slot_rotation_preserves_logical_groups(pair):
    for name in ("p4096_decode_results.json", "p4096_decode_repeat_results.json"):
        def swap(p):
            for row in p["actual_microbatch_scheduler_proof"]["actual_slot_request_composition"]:
                for group in row["slot_request_groups"]:
                    group["slot_id"] = 1 - group["slot_id"]
        mutate(pair[0] / "pp2ep4" / name, swap)
    result = compare.compare(*pair)
    assert result["qualified"]


def test_corpus_bytes_revalidated_not_just_declared_hash(pair):
    corpus = pair[0].parent / "corpus/request0.u32le"
    corpus.write_bytes(b"x" * (4097 * 4))
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert any("semantic bytes do not match" in e for e in result["errors"])


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p.update(hicache_h2d_timed=False), "hicache_h2d_timed"),
    (lambda p: p["native_cpu_ready_gpu_empty"][0]["matches"][0].update(gpu_tokens=1), "CPU-ready/GPU-empty"),
    (lambda p: p["actual_graph_proof"]["ranks"].pop(), "each world rank"),
    (lambda p: p["actual_graph_proof"]["ranks"][0]["actual_native_replays"].update(target=25), "native Graph"),
    (lambda p: p["all_rank_after"][0]["graph"].update(device_tensor_readbacks_added=1), "timed device instrumentation"),
    (lambda p: p["all_rank_after"][0]["deterministic"].update(actual_chunked_prefill_size=2048), "configuration mismatch"),
    (lambda p: p["all_rank_after"][0]["humming"]["kernels"][0].update(use_stream_k=True), "Humming invariant"),
    (lambda p: p["actual_dp_scheduler_proof"]["ranks"][0]["first_forward"].update(batch_size=3), "first local B4"),
    (lambda p: p["actual_dp_scheduler_proof"].update(added_collectives=1), "added collectives"),
    (lambda p: p["tpot_s"].update(mean=0.01), "aggregate tpot_s"),
    (lambda p: p["requests"][0].update(first_stream_chunk_tokens=3), "exactly token 1"),
])
def test_fail_closed_protocol_gates_preserve_measurements(pair, mutation, expected):
    mutate(pair[1] / "ep8/p4096_decode_results.json", mutation)
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert any(expected in e for e in result["errors"])
    assert result["arms"]["ep8"]["summaries"]["timed"]["throughput_tok_s"] > 0


def test_cross_output_mismatch_is_diagnostic_not_discarded_or_explained(pair):
    for name in ("p4096_decode_results.json", "p4096_decode_repeat_results.json"):
        mutate(pair[1] / "ep8" / name, lambda p: p["requests"][0]["output_ids"].__setitem__(7, 12345))
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert result["arms"]["ep8"]["repeat_parity"]["passed"]
    parity = result["cross_topology_output_parity"]["timed"]
    assert parity["requests"][0]["first_different_output_position_zero_based"] == 7
    assert parity["exact_logits_compared"] is False
    assert result["arms"]["ep8"]["summaries"]["timed"]["wall_s"] > 0


def test_within_ep8_repeat_output_mismatch_fails(pair):
    mutate(pair[1] / "ep8/p4096_decode_repeat_results.json", lambda p: p["requests"][0].update(output_text="different"))
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert any("within-process" in e for e in result["errors"])


def test_guard_post_and_same_complete_weight_bank_are_required(pair):
    mutate(pair[1] / "dashboard_guard/checks/123-post.json",
           lambda p: p["model_cache"]["nodes"][0].update(group_key="different-bank"))
    result = compare.compare(*pair)
    assert not result["qualified"]
    assert any("complete-weight bank identities mismatch" in e for e in result["errors"])


def test_missing_run_no_report_created(tmp_path, capsys):
    out = tmp_path / "out"
    assert compare.main(["--pp-run", str(tmp_path / "absent"), "--out", str(out)]) == 2
    assert "NOT_READY_OR_SCHEMA_INVALID" in capsys.readouterr().out
    assert not out.exists()
