"""Audit saved PP slot identities; never run inference or modify frozen inputs.

Scheduler hooks record host submission, not GPU execution or overlap. This
auditor matches ranks by logical pp_microstep, never wall-clock timestamps.
"""

import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_point(point):
    receipts = point["all_rank_after"]
    require(sorted(row["rank"] for row in receipts) == list(range(8)), "Expected eight distinct ranks")
    expected = {f"timed-decode-{point['prefix_tokens_per_request']}-{i}" for i in range(8)}
    projections = {}
    summaries = []
    for receipt in sorted(receipts, key=lambda row: row["rank"]):
        rank = receipt["rank"]
        rows = receipt["batches"]
        require(bool(rows), f"Rank {rank}: no timed batches")
        by_step = {}
        memberships = {0: set(), 1: set()}
        rid_slot = {}
        previous_decode = {}
        retired = {0: set(), 1: set()}
        full_decode = {0: 0, 1: 0}
        for row in rows:
            step, slot = row["pp_microstep"], row["pp_slot_id"]
            require(type(step) is int and step >= 0 and step not in by_step, f"Rank {rank}: invalid/duplicate microstep")
            require(slot in (0, 1) and slot == step % 2, f"Rank {rank}: slot/microstep disagreement")
            rids = row["request_ids"]
            require(len(rids) == row["batch_size"] and 1 <= len(rids) <= 4 and len(set(rids)) == len(rids), f"Rank {rank}: invalid batch composition")
            require(set(rids) <= expected, f"Rank {rank}: foreign timed RID")
            for rid in rids:
                require(rid_slot.setdefault(rid, slot) == slot, f"Rank {rank}: RID migrated between slots: {rid}")
            memberships[slot].update(rids)
            # ForwardMode.DECODE == 2 in this pinned engine. The Scheduler
            # invokes MTP target verification inside this run_batch call.
            # Admission/EXTEND chunks may temporarily omit existing slot RIDs;
            # only successive Decode batches establish retirement.
            if row["mode"] == "2":
                now = set(rids)
                require(not now & retired[slot], f"Rank {rank}: retired RID reappeared in Decode")
                if slot in previous_decode:
                    retired[slot].update(previous_decode[slot] - now)
                previous_decode[slot] = now
                full_decode[slot] += len(rids) == 4
            by_step[step] = dict(slot_id=slot, mode=row["mode"], request_ids=rids,
                                 batch_size=row["batch_size"], extend_num_tokens=row.get("extend_num_tokens"))
        require(list(by_step) == sorted(by_step), f"Rank {rank}: nonmonotonic submission microsteps")
        require(all(memberships.values()), f"Rank {rank}: both slots not observed")
        require(not memberships[0] & memberships[1] and memberships[0] | memberships[1] == expected,
                f"Rank {rank}: non-disjoint/incomplete two-slot assignment")
        require(all(len(memberships[slot]) == 4 for slot in (0, 1)), f"Rank {rank}: not two disjoint groups of four")
        require(all(full_decode.values()), f"Rank {rank}: full B4 Decode missing in a slot")
        projections[rank] = by_step
        summaries.append(dict(rank=rank, timed_submissions=len(rows), first_microstep=min(by_step),
                              last_microstep=max(by_step), slot_requests={str(k): sorted(v) for k, v in memberships.items()},
                              full_b4_decode_submissions=full_decode,
                              retired_request_counts={str(k): len(v) for k, v in retired.items()}))
    reference = projections[0]
    for rank in range(1, 8):
        require(projections[rank].keys() == reference.keys(), f"Rank {rank}: different logical submission microsteps")
        for step in reference:
            require(projections[rank][step] == reference[step], f"Rank {rank}: composition mismatch at microstep {step}")
    return dict(passed=True, ranks=summaries, cross_rank_matching_key="pp_microstep",
                observed_slots=[0, 1], request_slot_migration=False,
                identical_ordered_composition_at_each_microstep=True,
                evidence_boundary="Host submission identities only; not GPU completion, numerical correctness, or cross-host overlap")


def output_parity(left, right):
    require(left["prefix_tokens_per_request"] == right["prefix_tokens_per_request"], "Different prefix length")
    aa = {row["request_index"]: row for row in left["requests"]}
    bb = {row["request_index"]: row for row in right["requests"]}
    require(set(aa) == set(bb) == set(range(8)), "Incomplete output request indices")
    rows = []
    for index in range(8):
        a, b = aa[index], bb[index]
        x, y = a["output_ids"], b["output_ids"]
        require(len(x) == len(y) == left["output_tokens_per_request"] == right["output_tokens_per_request"], "Incomplete output IDs")
        prefix = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
        rows.append(dict(request_index=index, token_ids_equal=x == y,
                         text_equal=a["output_text"] == b["output_text"], common_token_prefix=prefix,
                         divergent_positions=sum(u != v for u, v in zip(x, y)),
                         first_differing_token_ids=None if prefix == len(x) else [x[prefix], y[prefix]]))
    return dict(passed=all(row["token_ids_equal"] and row["text_equal"] for row in rows), requests=rows,
                boundary="Pairwise greedy MTP equality; no independent non-speculative reference")


def load_point(run):
    cfg = json.loads((run / "config.json").read_text())
    require(len(cfg["prefix_lengths"]) == 1, "Expected one frozen workload")
    path = run / "pp2ep4" / f"p{cfg['prefix_lengths'][0]}_decode_results.json"
    return json.loads(path.read_text()), dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paired-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    point, source = load_point(args.run)
    result = dict(source=source, slot_audit=audit_point(point))
    if args.paired_run:
        paired, paired_source = load_point(args.paired_run)
        result.update(paired_source=paired_source, paired_slot_audit=audit_point(paired),
                      output_parity=output_parity(point, paired))
    result["passed"] = result["slot_audit"]["passed"] and result.get("output_parity", {}).get("passed", True)
    # Report lives outside frozen runtime and run directories.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
