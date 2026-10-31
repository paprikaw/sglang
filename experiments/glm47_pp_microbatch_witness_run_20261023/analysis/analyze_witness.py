"""Offline, CPU-only audit of saved PP/EAGLE first-divergence witnesses.

Never imports the serving engine, launches a process, or changes run artifacts.
Missing/invalid receipts are not replaced by unreceipted tensor files. Matching
uses observed request histories, never timestamps or equal verify ordinals.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT))
import witness
import witness_gate
import humming_receipt
from offline_schedule_audit import audit_point

COHORTS = ("timed", "repeat", "witness", "witness-repeat")
WITNESS_COHORTS = ("witness", "witness-repeat")
RANKS = (4, 5, 6, 7)


def stable_json(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise ValueError("JSON changed while being read")
    return json.loads(data), dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def json_artifact(path):
    try:
        value, source = stable_json(path)
        return dict(status="valid", source=source, value=value)
    except FileNotFoundError:
        return dict(status="missing", path=str(path))
    except Exception as error:
        return dict(status="invalid", path=str(path), error=f"{type(error).__name__}: {error}")


def result_path(run, p, cohort):
    suffix = "" if cohort == "timed" else "_" + cohort
    return Path(run) / "pp2ep4" / f"p{p}_decode{suffix}_results.json"


@contextmanager
def cached_references():
    # Runtime comparison helpers call the same full-logit CPU oracle several
    # times per request. Cache only within this offline comparison and retain
    # snapshot references to prevent Python id reuse from aliasing the cache.
    original = witness.chain_reference
    cache = {}
    def reference(snapshot):
        key = id(snapshot)
        if key not in cache:
            cache[key] = (snapshot, original(snapshot))
        return cache[key][1]
    witness.chain_reference = reference
    try:
        yield
    finally:
        witness.chain_reference = original


class Audit:
    def __init__(self, prefix):
        self.prefix = prefix

    @lru_cache(maxsize=2)
    def payload(self, run, cohort, rank):
        directory = Path(run) / "pp2ep4" / "witness"
        expected = f"{cohort}-decode-{self.prefix}"
        candidates, unreadable = [], []
        for path in sorted(directory.glob(f"rank{rank}_flush*.receipt.json")):
            item = json_artifact(path)
            if item["status"] != "valid":
                unreadable.append({k: v for k, v in item.items() if k != "value"})
            elif item["value"].get("cohort") == expected:
                candidates.append(item)
        if not candidates:
            return dict(status="missing", reason="No matching valid JSON receipt", rank=rank,
                        cohort=expected, directory=str(directory), unreadable_receipts=unreadable), None
        if len(candidates) != 1:
            return dict(status="invalid", reason="Ambiguous duplicate rank/cohort receipts", rank=rank,
                        cohort=expected, receipts=[c["source"] for c in candidates]), None
        item = candidates[0]
        receipt = item["value"]
        path = Path(item["source"]["path"])
        try:
            if (receipt.get("passed") is not True or receipt.get("rank") != rank
                    or receipt.get("cohort") != expected
                    or receipt.get("published_after_fsync_and_parse") is not True):
                raise ValueError("Receipt rank/cohort/publication mismatch")
            payload_path = Path(receipt["path"])
            if (payload_path.parent.resolve() != path.parent.resolve()
                    or payload_path.name != path.name.replace(".receipt.json", ".pt")):
                raise ValueError("Receipt payload is outside its owned directory")
            proof, value = witness_gate.read_stable_payload(payload_path,
                expected_sha256=receipt["sha256"], expected_size=receipt["size_bytes"])
            if proof["rank"] != rank or proof["cohort"] != expected:
                raise ValueError("Parsed payload rank/cohort differs from receipt")
            return dict(status="valid", receipt_source=item["source"], payload_proof=proof,
                        validation="Stable receipt + owned path + expected SHA/size + weights-only parse + gate schema"), value
        except Exception as error:
            return dict(status="invalid", receipt_source=item["source"], error=f"{type(error).__name__}: {error}"), None

    def oracle(self, run, cohort, rank):
        proof, payload = self.payload(str(run), cohort, rank)
        if payload is None:
            return proof
        rows, counts, snapshots = payload["rows"], Counter(), []
        with cached_references():
            for row in rows:
                try:
                    check = witness.chain_reference(row)
                except Exception as error:
                    check = dict(passed=False, reference_applicable=False, reason=f"oracle_error:{type(error).__name__}:{error}")
                kind = "pass" if check.get("passed") else "fail" if check.get("reference_applicable") else "inapplicable"
                counts[kind] += 1
                snapshots.append(dict(slot=row["slot"], microstep=row.get("microstep"),
                    verify_ordinal=row["verify_ordinal"], request_indices=[r["request_index"] for r in row["requests"]],
                    oracle_status=kind, oracle=check))
        return dict(**proof, snapshot_count=len(rows), counts={k: counts[k] for k in ("pass", "fail", "inapplicable")},
            all_observed_snapshots_match_cpu_reference=counts["pass"] == len(rows), snapshots=snapshots,
            bounded_to_first_16_verifies_per_slot=True, independent_target_model_correctness_proven=False)

    def comparison(self, left_run, left_cohort, left_rank, right_run, right_cohort, right_rank):
        a, left = self.payload(str(left_run), left_cohort, left_rank)
        b, right = self.payload(str(right_run), right_cohort, right_rank)
        identity = dict(left=dict(run=str(left_run), cohort=left_cohort, rank=left_rank),
                        right=dict(run=str(right_run), cohort=right_cohort, rank=right_rank))
        if left is None or right is None:
            return dict(status="missing" if "missing" in (a["status"], b["status"]) else "invalid",
                        **identity, left_proof=a, right_proof=b)
        try:
            with cached_references():
                batch = witness.compare_cohorts(left["rows"], right["rows"])
                request = witness.compare_requests(left["rows"], right["rows"])
            batch_counts = Counter(row["result"]["classification"] for row in batch["matched"])
            request_counts = Counter(row["classification"] for row in request["matched_requests"])
            first_per_request = {}
            for row in request["matched_requests"]:
                if row["classification"] != "identical_request_logit_and_acceptance_witness":
                    first_per_request.setdefault(str(row["request_index"]), row)
            return dict(status="valid", **identity, left_proof=a, right_proof=b,
                matching_key="Observed ordered RID/history for whole batch; RID/history for per-request; not time/ordinal",
                batch_summary=dict(matched=len(batch["matched"]), unmatched_left=len(batch["unmatched_left"]),
                    unmatched_right=len(batch["unmatched_right"]), classifications=dict(batch_counts)),
                request_summary=dict(matched=len(request["matched_requests"]), unmatched_left=len(request["unmatched_left"]),
                    unmatched_right=len(request["unmatched_right"]), classifications=dict(request_counts),
                    first_nonidentical_observation_per_request=first_per_request),
                batch_comparison=batch, request_comparison=request,
                exact_gpu_kv_equality_proven=False, benign_numerical_branching_proven=False)
        except Exception as error:
            return dict(status="invalid", **identity, error=f"{type(error).__name__}: {error}")


def schedule_audit(point, p, cohort):
    # The existing independent auditor admits a single timed RID prefix.
    # Normalize the cohort label only; preserve exact logical microsteps,
    # rank, slot, ordered RID membership, mode and retirement observations.
    expected = f"{cohort}-decode-{p}-"
    receipts = []
    for receipt in point["all_rank_after"]:
        batches = []
        for row in receipt["batches"]:
            ids = row.get("request_ids", [])
            if ids and all(rid.startswith(expected) for rid in ids):
                batches.append(dict(row, request_ids=[f"timed-decode-{p}-" + rid[len(expected):] for rid in ids]))
        receipts.append(dict(receipt, batches=batches))
    return audit_point(dict(prefix_tokens_per_request=p, all_rank_after=receipts))


def outputs(point, expected_tokens):
    rows = point.get("requests", [])
    indices = [row.get("request_index") for row in rows]
    if sorted(indices) != list(range(8)):
        raise ValueError("Output request indices must be exactly 0..7 once each")
    for row in rows:
        ids = row.get("output_ids")
        if (not isinstance(ids, list) or any(type(x) is not int for x in ids)
                or len(ids) != expected_tokens or not isinstance(row.get("output_text"), str)):
            raise ValueError("Incomplete/invalid actual token IDs or text")
    return {row["request_index"]: row for row in rows}


def output_pair(left, right):
    if left.get("status") != "valid" or right.get("status") != "valid":
        return dict(status="missing" if "missing" in (left.get("status"), right.get("status")) else "invalid",
                    left_status=left.get("status"), right_status=right.get("status"))
    try:
        aa, bb = outputs(left["value"], left["expected_tokens"]), outputs(right["value"], right["expected_tokens"])
        rows = []
        for index in range(8):
            a, b = aa[index], bb[index]
            x, y = a["output_ids"], b["output_ids"]
            prefix = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            rows.append(dict(request_index=index, token_ids_equal=x == y, text_equal=a["output_text"] == b["output_text"],
                lengths=[len(x), len(y)], common_token_prefix=prefix,
                first_differing_token_ids=[x[prefix] if prefix < len(x) else None, y[prefix] if prefix < len(y) else None]
                    if x != y else None))
        return dict(status="valid", exact_token_id_requests=sum(r["token_ids_equal"] for r in rows),
            exact_text_requests=sum(r["text_equal"] for r in rows), request_count=8,
            exact_ids_and_text=all(r["token_ids_equal"] and r["text_equal"] for r in rows), requests=rows,
            boundary="Pairwise actual saved output equality, not independent non-speculative target-model correctness")
    except Exception as error:
        return dict(status="invalid", error=f"{type(error).__name__}: {error}")


def humming_proof(receipts):
    if len(receipts) != 8 or sorted(row["rank"] for row in receipts) != list(range(8)):
        raise ValueError("Humming proof requires all eight unique initialized ranks")
    rows = []
    for receipt in sorted(receipts, key=lambda row: row["rank"]):
        proof = humming_receipt.validate_humming_kernel_receipt(receipt.get("humming", {}))
        kinds = Counter(kernel["gemm_type"] for kernel in receipt["humming"]["kernels"])
        rows.append(dict(rank=receipt["rank"], **proof, registered_kernel_counts_by_gemm_type=dict(kinds)))
    return dict(passed=True, ranks=rows, proves_actual_kernel_launch=False,
                boundary="Loaded dense/grouped_masked SM80 metadata: batch invariant enabled, StreamK disabled; not a launch trace")


def audit_run(audit, run, p):
    cfg = json_artifact(Path(run) / "config.json")
    expected_tokens = cfg.get("value", {}).get("production_output_tokens")
    points, public = {}, {}
    for cohort in COHORTS:
        item = json_artifact(result_path(run, p, cohort))
        item["expected_tokens"] = expected_tokens
        points[cohort] = item
        row = {k: v for k, v in item.items() if k != "value"}
        if item["status"] == "valid":
            value = item["value"]
            try:
                if type(expected_tokens) is not int:
                    raise ValueError("Frozen production_output_tokens missing")
                outputs(value, expected_tokens)
                row["actual_outputs_complete"] = True
            except Exception as error:
                row["actual_outputs_complete"] = False
                row["output_error"] = str(error)
            try:
                row["schedule_audit"] = schedule_audit(value, p, cohort)
            except Exception as error:
                row["schedule_audit"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
            try:
                row["humming_before"] = humming_proof(value["native_cpu_ready_gpu_empty"])
                row["humming_after"] = humming_proof(value["all_rank_after"])
            except Exception as error:
                row["humming_proof"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
            if cohort in ("timed", "repeat"):
                try:
                    row["zero_witness_before"] = witness_gate.validate_uninstrumented(value["native_cpu_ready_gpu_empty"])
                    row["zero_witness_after"] = witness_gate.validate_uninstrumented(value["all_rank_after"])
                except Exception as error:
                    row["zero_witness_proof"] = dict(passed=False, error=f"{type(error).__name__}: {error}")
                row["recorded_metrics_only_not_validated_speedup"] = {k: value.get(k) for k in
                    ("wall_s", "throughput_tok_s", "ttft_s", "tpot_s", "end_to_end_s", "spec_accept_rate")}
        public[cohort] = row
    def task_summary(path):
        item = json_artifact(path)
        if item["status"] != "valid":
            return item
        value = item["value"]
        return dict(status="valid", source=item["source"], fields={key: value.get(key) for key in
            ("passed", "valid_points", "expected", "mode", "gpu_idle", "model_processes_exited", "gpu_resources_released")},
            arms=[{key: arm.get(key) for key in ("layout", "passed", "failure", "cleanup_verified")} for arm in value.get("arms", [])])
    summary = dict(run=str(Path(run).resolve()), config={k: v for k, v in cfg.items() if k != "value"},
        configuration={k: cfg.get("value", {}).get(k) for k in ("id", "mode", "engine_root", "engine_commit", "prefix_lengths", "production_output_tokens")},
        task_results=task_summary(Path(run) / "task" / "results.json"),
        cleanup=task_summary(Path(run) / "task" / "cleanup.json"), result_artifacts=public,
        oracle_by_cohort_rank={}, receipt_and_oracle_coverage={}, within_repeat_by_rank={}, cross_tp_by_cohort={}, output_exactness={})
    for cohort in WITNESS_COHORTS:
        summary["oracle_by_cohort_rank"][cohort] = {str(rank): audit.oracle(run, cohort, rank) for rank in RANKS}
        records = summary["oracle_by_cohort_rank"][cohort]
        valid = [rank for rank in RANKS if records[str(rank)]["status"] == "valid"]
        counts = {kind: sum(records[str(rank)]["counts"][kind] for rank in valid) for kind in ("pass", "fail", "inapplicable")}
        summary["receipt_and_oracle_coverage"][cohort] = dict(required_tail_ranks=list(RANKS), validated_tail_ranks=valid,
            missing_tail_ranks=[rank for rank in RANKS if records[str(rank)]["status"] == "missing"],
            invalid_tail_ranks=[rank for rank in RANKS if records[str(rank)]["status"] == "invalid"],
            all_four_tail_receipts_valid=len(valid) == 4, observed_snapshot_counts=counts,
            all_required_rank_observed_snapshots_match_cpu_reference=(counts["fail"] == counts["inapplicable"] == 0) if len(valid) == 4 else None)
        summary["cross_tp_by_cohort"][cohort] = {f"4_vs_{rank}": audit.comparison(run, cohort, 4, run, cohort, rank) for rank in RANKS[1:]}
    summary["within_repeat_by_rank"] = {str(rank): audit.comparison(run, "witness", rank, run, "witness-repeat", rank) for rank in RANKS}
    for a, b in (("timed", "repeat"), ("witness", "witness-repeat"), ("timed", "witness"), ("repeat", "witness-repeat")):
        summary["output_exactness"][a + "_vs_" + b] = output_pair(points[a], points[b])
    return summary, points


def analyze(baseline, candidate=None, prefix=4096, cpu_threads=1):
    import torch
    torch.set_num_threads(cpu_threads)
    audit = Audit(prefix)
    baseline = str(Path(baseline).resolve())
    base, left = audit_run(audit, baseline, prefix)
    sources = [Path(__file__), Path(witness.__file__), Path(witness_gate.__file__), Path(humming_receipt.__file__), ROOT / "offline_schedule_audit.py"]
    report = dict(schema="pp-eagle-offline-first-divergence-analysis-v1", prefix_tokens_per_request=prefix,
        sources=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
        baseline=base, candidate=dict(status="missing", reason="No candidate run requested"),
        cross_arm=dict(status="missing", reason="No candidate run requested"),
        cpu_only=True, cpu_threads=cpu_threads, runtime_or_artifacts_modified=False, performance_comparison_claim_valid=False,
        limitations=["Only first 16 Verifies per slot/cohort are retained; later divergence may be missing",
                     "Same observed token history/metadata does not establish actual GPU KV-value equality",
                     "CPU oracle validates recorded greedy acceptance, not independent target-model logits",
                     "Unmatched histories are not interpolated; changed trees/masks are not assumed equivalent",
                     "Instrumentation may perturb scheduling in diagnostic cohorts; production outputs are audited separately"])
    if candidate is not None:
        candidate = str(Path(candidate).resolve())
        right_summary, right = audit_run(audit, candidate, prefix)
        report["candidate"] = right_summary
        report["cross_arm"] = dict(status="evaluated", witness_by_cohort_rank={}, output_exactness={})
        for cohort in WITNESS_COHORTS:
            report["cross_arm"]["witness_by_cohort_rank"][cohort] = {
                str(rank): audit.comparison(baseline, cohort, rank, candidate, cohort, rank) for rank in RANKS}
        for cohort in COHORTS:
            report["cross_arm"]["output_exactness"][cohort] = output_pair(left[cohort], right[cohort])
    audit.payload.cache_clear()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    for run in (args.baseline, args.candidate):
        if run is not None and root.is_relative_to(run.resolve()):
            parser.error("Analysis output must be outside run artifacts")
    if root.is_relative_to((ROOT / "runtime").resolve()):
        parser.error("Analysis output must be outside frozen runtime")
    report = analyze(args.baseline, args.candidate, args.prefix_tokens, args.cpu_threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(report=str(args.out.resolve()), cpu_only=True,
        performance_comparison_claim_valid=False, baseline=str(args.baseline),
        candidate=str(args.candidate) if args.candidate else None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
