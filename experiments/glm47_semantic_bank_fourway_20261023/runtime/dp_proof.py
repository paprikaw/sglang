"""CPU receipt gates for actual EP8 Attention TP4 x DP2 placement/admission."""
from deterministic_proof import validate as deterministic_validate
from humming_receipt import validate_humming_kernel_receipt


def validate(receipts, cohort_prefix="timed-"):
    if len(receipts) != 8 or sorted(r["rank"] for r in receipts) != list(range(8)):
        raise RuntimeError("EP8 proof requires all eight ranks")
    deterministic = deterministic_validate(receipts)
    humming, proofs, signatures = [], [], {}
    for receipt in receipts:
        rank = receipt["rank"]
        dp = rank // 4
        expected = list(range(dp*4, dp*4+4))
        mode = receipt["microbatch"]
        if mode.get("enabled") is not False or mode.get("loop_size") != 1:
            raise RuntimeError("EP8 must have PP1 and no PP microbatch feedback")
        if receipt.get("admission_pending") != 0:
            raise RuntimeError("EP8 has unadmitted local requests")
        admissions = [x for x in receipt["atomic_admissions"] if x["cohort"].startswith(cohort_prefix)]
        if len(admissions) != 1:
            raise RuntimeError("Expected one atomic local B4 release")
        admission = admissions[0]
        if (admission.get("request_indices") != expected or admission.get("request_count") != 4
                or admission.get("dp_rank") != dp or admission.get("atomic_local_release") is not True
                or admission.get("observed_at_scheduler_not_inferred_from_http") is not True
                or any(admission.get(k) != 0 for k in
                       ("added_collectives", "device_reads", "gpu_synchronizations"))):
            raise RuntimeError("Actual atomic DP release does not match frozen groups")
        batches = [x for x in receipt["batches"] if x.get("request_ids")
                   and all(rid.startswith(cohort_prefix) for rid in x["request_ids"])]
        if not batches:
            raise RuntimeError("No actual local model forwards")
        expected_ids = admission["request_ids"]
        first = batches[0]
        if first.get("batch_size") != 4 or first.get("request_ids") != expected_ids:
            raise RuntimeError("First model forward was not the full ordered local B4")
        if first.get("extend_num_tokens") != 4 * 65:
            raise RuntimeError("First model forward did not recompute exactly 65 tokens per request")
        observed = set()
        retired = set()
        previous = None
        if [batch.get('local_submission_ordinal') for batch in batches] != list(range(len(batches))):
            raise RuntimeError('Local submission ordinals are not contiguous and monotonic')
        for batch in batches:
            ids = batch["request_ids"]
            if (batch.get("attn_dp_rank") != dp or not 0 < batch["batch_size"] <= 4
                    or len(ids) != batch["batch_size"] or len(set(ids)) != len(ids)
                    or any(rid not in expected_ids for rid in ids)):
                raise RuntimeError("Model batch exceeded/migrated from its local DP group")
            if retired.intersection(ids):
                raise RuntimeError('Request reappeared after retiring from its local Decode batch')
            if previous is not None:
                retired.update(previous.difference(ids))
            previous = set(ids)
            observed.update(ids)
        if observed != set(expected_ids) or len(batches) < 2:
            raise RuntimeError("Incomplete local Decode coverage")
        signature = [(b["mode"], b["batch_size"], tuple(b["request_ids"]),
                      b.get("extend_num_tokens")) for b in batches]
        if dp in signatures and signatures[dp] != signature:
            raise RuntimeError("TP peers disagree about their logical DP batch schedule")
        signatures[dp] = signature
        value = validate_humming_kernel_receipt(receipt.get("humming", {}))
        humming.append(dict(rank=rank, **value))
        proofs.append(dict(rank=rank, dp_rank=dp, request_indices=expected,
            atomic_admission=admission, first_forward=first,
            local_submission_count=len(batches), maximum_local_batch_size=max(b["batch_size"] for b in batches),
            tp_peers_have_identical_ordered_submissions=True, request_dp_migration=False))
    return dict(passed=True, attention_tp=4, attention_dp=2, expert_ep=8, pipeline_stages=1,
        ranks=proofs, deterministic_runtime=deterministic,
        humming_loaded_kernel_invariants=dict(passed=True,ranks=humming,proves_actual_launch=False),
        admission_control="Frozen cohort CPU routing and local ordered B4 release",
        stock_colocated_input_blocker_enabled=False, added_collectives=0,
        device_reads=0, gpu_synchronizations=0,
        matches_pp_logical_groups=[[0,1,2,3],[4,5,6,7]],
        numerical_correctness_claim=False)
