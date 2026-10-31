"""Actual lookahead validation with fixed matched EP/PP request groups."""
import hashlib
import json
import os
from pathlib import Path
import time

from previous_workloads import prompts


def byte(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        if os.pwrite(fd, bytes([value]), 0) != 1:
            raise RuntimeError("Private control write failed")
    finally:
        os.close(fd)


def install(base):
    environment_before = base.environment
    cohort_before = base.cohort
    epoch = [0]

    def environment(cfg, arm, inherited):
        result = environment_before(cfg, arm, inherited)
        result.update(PP_GAP_POLICY_PATH=cfg["gap_policy_path"],
                      PP_FIXED_COHORT_CONTROL=cfg["fixed_cohort_control"],
                      PP_FIXED_COHORT_SIZE=cfg["fixed_cohort_size"],
                      PP_TRANSFER_AUDIT_CONTROL=cfg["transfer_audit_control"],
                      PP_GAP_SEPARATE_CHANNELS="1")
        return result

    def states(run, cfg, policy):
        values = [json.loads((run / "pp2/gap_receipts" / f"policy{policy}.pp{rank}.json").read_text())
                  for rank in (0, 1)]
        if any(s["pending"] or s["ahead_pending"] or s["ahead_produced"] != s["ahead_consumed"]
               or s["prepared_next"] != s["prepared_consumed"] for s in values):
            raise RuntimeError("PP early-work ownership did not fully retire")
        return values

    def cohort(cfg, key, payloads):
        pp = not any("data_parallel_rank" in p for p in payloads)
        full = pp and len(payloads) == cfg["requests"]
        if full:
            epoch[0] = epoch[0] % 254 + 1
            byte(cfg["fixed_cohort_control"], epoch[0])
        else:
            byte(cfg["fixed_cohort_control"], 0)
        result = cohort_before(cfg, key, payloads)
        # Outside measured latency: drain retirement, publish bounded receipts,
        # and only then allow changing a transport policy or batch shape.
        for _ in range(3):
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
        if full:
            receipts = [s.get("experiment_fixed_group_receipt") for s in info["internal_states"]
                        if s.get("experiment_fixed_group_receipt")]
            expected = list(range(5000, 5000 + cfg["requests"], 2)) + list(range(5001, 5000 + cfg["requests"], 2))
            if len(receipts) != 1 or receipts[0]["epoch"] != epoch[0] or receipts[0]["ordered_markers"] != expected:
                raise RuntimeError("Actual PP cohort admission differs from EP even/odd groups")
            result["fixed_group_receipt"] = receipts[0]
        return result

    base.environment = environment
    base.cohort = cohort

    def workloads(run, cfg, arm, key, health):
        from benchmark import policy as expert_policy
        from sampling import set_diagnostic
        from pp_transfer_audit import compare_transfers
        byte(cfg["gap_policy_path"], ord("0"))
        byte(cfg["transfer_audit_control"], ord("0"))
        expert_policy(cfg, b"0")
        set_diagnostic(cfg, False)
        for case_index, case in enumerate(cfg["workloads"]):
            current = dict(cfg, requests=case["requests"])
            root = run / arm / "workloads" / case["id"]
            root.mkdir(parents=True, exist_ok=True)
            Path(cfg["expert_workload_path"]).write_text(case["id"] + "\n")
            byte(cfg["fixed_cohort_size"], case["requests"])
            byte(cfg["gap_policy_path"], ord("0"))
            local = case["requests"] // 2
            health()
            base.progress(run, cfg, "workload_" + arm + "_" + case["id"])
            if arm == "pp2":
                response = base.h.http_json(cfg["port"], "/set_internal_state", key,
                    dict(server_args=dict(pp_max_micro_batch_size=local)), timeout=30)
                if isinstance(response, dict) and response.get("success") is False:
                    raise RuntimeError("Native PP microbatch update rejected")
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
            base.h.save(root / "server_info.json", base.h.safe_server_info(info))
            tokens = prompts(current, case)
            pref = [base.packet(current, arm, t, 1, i) for i, t in enumerate(tokens)]
            dec = [base.packet(current, arm, t + [198], cfg["output_tokens"], i) for i, t in enumerate(tokens)]
            base.h.save(root / "protocol.json", dict(**case, cached_context_per_request=cfg["prefix_tokens"],
                new_input_tokens_per_request=1, output_tokens_per_request=cfg["output_tokens"],
                target_local_decode_tokens=local, pp_slots=2, profiler_off_formal=True,
                prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                same_inputs_as_previous_workload_study=True, no_cpu_kv_restore=True,
                native_shared_expert_overlap=True, two_batch_overlap=False))

            def check(group, cached=True):
                expected_output = cfg["output_tokens"] if cached else 1
                expected_cache = cfg["prefix_tokens"] if cached else 0
                if len(group["requests"]) != case["requests"]:
                    raise RuntimeError("Formal request cohort size changed")
                for row in group["requests"]:
                    if (row.get("cached_tokens") or 0) != expected_cache or row["output_tokens"] != expected_output:
                        raise RuntimeError("Exact cache or output count violated")
                    if cached and len(row.get("output_ids") or []) != expected_output:
                        raise RuntimeError("Full output IDs unavailable")

            base.flush_cache(cfg, key)
            check(base.cohort(current, key, pref), False)
            cold = []
            for repeat in range(cfg["repeats"]):
                health(); base.flush_cache(cfg, key)
                group = base.cohort(current, key, pref); check(group, False)
                cold.append(group)
            base.h.save(root / "metrics.prefill.json", base.summarize(cold, "cold4096->1; lookahead OFF"))
            base.flush_cache(cfg, key)
            check(base.cohort(current, key, pref), False)
            for count in sorted(set([2, 4, 6, 8, local, case["requests"]])) + [case["requests"]] * 2:
                health(); base.cohort(current, key, dec[:min(count, case["requests"])])
            policies = cfg["gap_policies"] if arm == "pp2" else [0]
            values = {p: [] for p in policies}
            order = [p for repeat in range(cfg["repeats"])
                     for p in (policies if repeat % 2 == 0 else list(reversed(policies)))]
            for index, p in enumerate(order):
                health()
                byte(cfg["gap_policy_path"], ord(str(p)))
                check(base.cohort(current, key, dec))
                before = states(run, cfg, p) if arm == "pp2" else None
                base.progress(run, cfg, f"timing_{arm}_{case['id']}_p{p}", repeat=index + 1)
                group = base.cohort(current, key, dec); check(group)
                if arm == "pp2":
                    after = states(run, cfg, p)
                    delta = [dict(ahead_produced=b["ahead_produced"] - a["ahead_produced"],
                                  ahead_consumed=b["ahead_consumed"] - a["ahead_consumed"],
                                  prepared=b["prepared_next"] - a["prepared_next"])
                             for a, b in zip(before, after)]
                    if p == 9 and not all(d["ahead_produced"] > 0 and d["ahead_produced"] == d["ahead_consumed"] for d in delta):
                        raise RuntimeError("Lookahead was selected but did not actually execute in formal cohort")
                    group["scheduling_receipt"] = dict(policy=p, deltas=delta, states=after)
                values[p].append(group)
                base.h.save(root / f"cohort.{index}.p{p}.json", dict(policy=p, performance_sample=True,
                    profiler_enabled=False, result=group))
                base.h.save(root / f"metrics.decode.p{p}.json", base.summarize(values[p], cfg["protocol"]))
            reference = [r["output_ids"] for r in values[0][0]["requests"]]
            agreement = {str(p): all([r["output_ids"] for r in g["requests"]] == reference for g in groups)
                         for p, groups in values.items()}
            base.h.save(root / "output_agreement.json", dict(all_modes_equal=all(agreement.values()),
                by_policy=agreement, output_tokens_checked=sum(case["requests"] * cfg["output_tokens"] * len(g) for g in values.values())))
            if arm == "pp2" and not all(agreement.values()):
                raise RuntimeError("Lookahead changed generated token sequence")
            if arm == "pp2":
                for audit_index, p in enumerate(policies):
                    byte(cfg["gap_policy_path"], ord(str(p)))
                    check(base.cohort(current, key, dec))
                    audit_epoch = case_index * 2 + audit_index + 1
                    byte(cfg["transfer_audit_control"], ord(str(audit_epoch)))
                    check(base.cohort(current, key, dec))
                    byte(cfg["transfer_audit_control"], ord("0"))
                    paths = [run / "pp2/transfer_audit" / f"epoch{audit_epoch}.pp{r}.json" for r in (0, 1)]
                    deadline = time.monotonic() + 45
                    while not all(path.exists() for path in paths):
                        health(); base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Wire receipts did not flush")
                        time.sleep(.05)
                    wire = [json.loads(path.read_text()) for path in paths]
                    result = compare_transfers(wire)
                    expected = {tuple(range(5000, 5000 + case["requests"], 2)),
                                tuple(range(5001, 5000 + case["requests"], 2))}
                    for receipt in wire:
                        groups = {tuple(r["marker"] for r in batch["requests"]) for batch in receipt["batches"]
                                  if len(batch["requests"]) == local and all(8 <= r["generated"] <= 112 for r in batch["requests"])}
                        if groups != expected:
                            raise RuntimeError("Actual steady PP slots differ from declared groups")
                    actual = states(run, cfg, p)
                    if not result["passed"]:
                        raise RuntimeError("Original PP payloads differ between sender and receiver")
                    if p == 9 and not all(s["ahead_produced"] and s["typed_channels"] == 3 for s in actual):
                        raise RuntimeError("Lookahead path was not used")
                    base.h.save(root / f"qualification.p{p}.json", dict(passed=True, policy=p,
                        wire=result, states=actual, fixed_groups=True, target_local_tokens=local))
            for p in policies:
                byte(cfg["gap_policy_path"], ord(str(p)))
                check(base.cohort(current, key, dec)); health()
                directory = root / "traces" / f"p{p}"
                directory.mkdir(parents=True, exist_ok=True)
                base.profile_action(cfg, key, "/start_profile", dict(output_dir=str(directory),
                    num_steps=cfg["profile_steps"], activities=["CPU", "GPU"], record_shapes=True,
                    with_stack=False, profile_id=f"{arm}-{case['id']}-p{p}",
                    merge_profiles=False, detailed_annotations=True))
                group = base.cohort(current, key, dec); check(group)
                base.h.save(directory / "cohort.json", dict(performance_sample=False, result=group))
                exports = base.await_exports(directory, health, timeout=240)
                if len(exports) != 2:
                    raise RuntimeError("Both ranks must export independent profiles")
                base.h.save(directory / "manifest.json", dict(files=exports, policy=p,
                    performance_sample=False, target_local_tokens=local,
                    selected_forwards=cfg["selected_forwards"], drop_initial_graphs=cfg["drop_initial_graphs"]))
            byte(cfg["gap_policy_path"], ord("0"))
            byte(cfg["fixed_cohort_control"], 0)
            base.h.save(root / "completed.json", dict(passed=True, policies=policies, formal_cohorts=len(order)))
        base.h.save(run / arm / "completed.json", dict(passed=True, workloads=len(cfg["workloads"])))

    base.run_workloads = workloads
