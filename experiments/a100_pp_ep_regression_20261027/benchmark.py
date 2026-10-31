"""Profiler-off A100 cohorts and separately measured components.

Reuse the qualified native BF16/Triton A100 bridge for BOTH layouts. PP policy9
is a scheduling experiment; it does not replace or tune Expert kernels.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import threading
import time

from artifact_visibility import wait_for_stable_json
from previous_workloads import prompts


def byte(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        if os.pwrite(fd, bytes([value]), 0) != 1:
            raise RuntimeError("Private control write failed")
    finally:
        os.close(fd)


def install(base):
    original_environment, original_args = base.environment, base.engine_args
    epoch = [0]

    def environment(cfg, arm, inherited):
        env = original_environment(cfg, arm, inherited)
        env.update(EP_STEADY_HARDWARE="A100",
                   PP_GAP_POLICY_PATH=cfg["gap_policy_path"],
                   PP_FIXED_COHORT_CONTROL=cfg["fixed_cohort_control"],
                   PP_FIXED_COHORT_SIZE=cfg["fixed_cohort_size"],
                   PP_TRANSFER_AUDIT_CONTROL=cfg["transfer_audit_control"],
                   PP_GAP_SEPARATE_CHANNELS="1")
        return env

    def engine_args(cfg, arm):
        args = original_args(cfg, arm)
        # The same capacity accommodates the largest declared B32 cohort;
        # never shrink a workload in response to allocator pressure.
        for flag, value in (("--max-total-tokens", "196608"),
                            ("--max-running-requests", "32"),
                            ("--max-mamba-cache-size", "168" if arm == "ep2" else "164")):
            args[args.index(flag) + 1] = value
        # Reuse the validated H100 study's native state-retention budget:
        # EP divides168 into84 states/rank; PP allocates164 half-model states.
        # Keep the final cached prefix state rather than obsolete chunk states.
        args += ["--mamba-max-states-per-path", "1"]
        return args

    def progress(run, cfg, stage, **extra):
        base.h.save(run / "task/progress.json", dict(experiment_id=cfg["id"], task_id="comparison",
            stage=stage, timestamp=time.time(), controller_pid=os.getpid(),
            valid_points=len(list(run.glob("*/workloads/*/metrics.*.p*.json"))),
            expected=cfg["expected_points"], **extra))

    def cohort(cfg, key, packets):
        epoch[0] = epoch[0] % 254 + 1
        byte(cfg["fixed_cohort_control"], epoch[0])
        gate = threading.Barrier(len(packets))
        with ThreadPoolExecutor(max_workers=len(packets)) as pool:
            futures = [pool.submit(base.stream_one, cfg, key, p, gate) for p in packets]
            rows = [future.result() for future in futures]
        duration = max(r["finished"] for r in rows) - min(r["started"] for r in rows)
        result = dict(requests=rows, makespan_s=duration,
            output_tokens_per_s=sum(r["output_tokens"] for r in rows)/duration,
            input_tokens_per_s=sum(len(p["input_ids"]) for p in packets)/duration)
        for _ in range(3):
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
        pp = not any("data_parallel_rank" in p for p in packets)
        field = "experiment_fixed_group_receipt" if pp else "experiment_prefill_ep_receipt"
        receipts = [r[field] for r in info["internal_states"] if field in r]
        if pp:
            expected = list(range(5000, 5000+cfg["requests"], 2)) + list(range(5001, 5000+cfg["requests"], 2))
            if len(receipts) != 1 or receipts[0]["epoch"] != epoch[0] or receipts[0]["ordered_markers"] != expected:
                raise RuntimeError("Actual PP request grouping differs from the declared groups")
        else:
            if len(receipts) != 2 or {r["dp_rank"] for r in receipts} != {0, 1}:
                raise RuntimeError("Both Attention DP ranks must acknowledge admission")
            for r in receipts:
                if r["epoch"] != epoch[0] or r["ordered_markers"] != list(range(5000+r["dp_rank"], 5000+cfg["requests"], 2)):
                    raise RuntimeError("Actual Attention DP admission differs from the declared groups")
        result["admission_receipts"] = receipts
        return result

    def states(run, cfg, p):
        values = [wait_for_stable_json(run / "pp2/gap_receipts" / f"policy{p}.pp{r}.json",
                  timeout_s=30, min_stable_s=.02)[0] for r in (0, 1)]
        for s in values:
            if s["pending"] or s["ahead_pending"] or s["ahead_produced"] != s["ahead_consumed"] or s["prepared_next"] != s["prepared_consumed"]:
                raise RuntimeError("PP speculative ownership did not fully drain")
        return values

    def workloads(run, cfg, arm, key, health):
        from pp_transfer_audit import compare_transfers
        byte(cfg["gap_policy_path"], ord("0"))
        byte(cfg["transfer_audit_control"], ord("0"))
        for case_index, case in enumerate(cfg["workloads"]):
            current = dict(cfg, requests=case["requests"], prefix_tokens=case.get("new_input", 4096))
            root = run / arm / "workloads" / case["id"]
            root.mkdir(parents=True, exist_ok=True)
            byte(cfg["fixed_cohort_size"], case["requests"])
            byte(cfg["gap_policy_path"], ord("0"))
            if arm == "pp2":
                response = base.h.http_json(cfg["port"], "/set_internal_state", key,
                    dict(server_args=dict(pp_max_micro_batch_size=case["requests"]//2)), timeout=30)
                if isinstance(response, dict) and response.get("success") is False:
                    raise RuntimeError("Native PP microbatch update rejected")
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
            base.h.save(root / "server_info.json", base.h.safe_server_info(info))
            tokens = prompts(current, case)
            decode = case["phase"] == "decode"
            packets = [base.packet(current, arm, t+[198] if decode else t, 128 if decode else 1, i)
                       for i, t in enumerate(tokens)]
            pref = [base.packet(current, arm, t, 1, i) for i, t in enumerate(tokens)]
            base.h.save(root / "protocol.json", dict(**case,
                cached_context_per_request=4096 if decode else 0,
                new_input_tokens_per_request=1 if decode else case["new_input"],
                output_tokens_per_request=128 if decode else 1,
                prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                gpu_pair=[0, 1], nvlink_required=True, no_cpu_kv_restore=True,
                graph=False, expert_backend="same native Triton in EP and PP",
                pp_slots=2, target_local_decode_tokens=case["requests"]//2,
                actual_prefill_budget_per_attention=4096, formal_profiler=False))

            def check(group, prepared=False):
                expected_cache = 4096 if decode and not prepared else 0
                expected_output = 1 if prepared else 128 if decode else 1
                expected_prompt = 4097 if decode and not prepared else current["prefix_tokens"]
                if len(group["requests"]) != case["requests"]:
                    raise RuntimeError("Cohort size changed")
                for row in group["requests"]:
                    if (row.get("cached_tokens") or 0) != expected_cache or row["output_tokens"] != expected_output or row["prompt_tokens"] != expected_prompt:
                        raise RuntimeError("Exact input/cache/output workload contract violated")
                    if len(row.get("output_ids") or []) != expected_output:
                        raise RuntimeError("Full output token IDs unavailable")

            health(); base.flush_cache(cfg, key)
            if decode:
                prime = base.cohort(current, key, pref); check(prime, prepared=True)
                base.h.save(root / "prefix_preparation.json", dict(measured=False,
                    cpu_kv_restore=False, real_model_prefix=True, result=prime))
            policies = [0, 9] if decode and arm == "pp2" else [0]
            measured = {p: [] for p in policies}
            for p in policies:
                byte(cfg["gap_policy_path"], ord(str(p)))
                for i in range(cfg["warmups"]):
                    health(); progress(run, cfg, f"warmup_{arm}_{case['id']}_p{p}", repeat=i+1)
                    if not decode: base.flush_cache(cfg, key)
                    check(base.cohort(current, key, packets))
            order = [p for i in range(cfg["repeats"])
                     for p in (policies if i % 2 == 0 else list(reversed(policies)))]
            for index, p in enumerate(order):
                health(); byte(cfg["gap_policy_path"], ord(str(p)))
                if not decode: base.flush_cache(cfg, key)
                before = states(run, cfg, p) if arm == "pp2" else None
                progress(run, cfg, f"timing_{arm}_{case['id']}_p{p}", repeat=index+1)
                group = base.cohort(current, key, packets); check(group)
                if arm == "pp2":
                    after = states(run, cfg, p)
                    delta = [b["ahead_produced"]-a["ahead_produced"] for a,b in zip(before, after)]
                    if p == 9 and not all(x > 0 for x in delta):
                        raise RuntimeError("PP lookahead did not actually execute")
                    group["scheduling_receipt"] = dict(policy=p, ahead_deltas=delta, states=after)
                measured[p].append(group)
                base.h.save(root / f"cohort.{index}.p{p}.json", dict(policy=p,
                    performance_sample=True, profiler_enabled=False, result=group))
                base.h.save(root / f"metrics.{case['phase']}.p{p}.json", base.summarize(measured[p], cfg["protocol"]))
            reference = [r["output_ids"] for r in measured[0][0]["requests"]]
            agreement = {str(p): all([r["output_ids"] for r in g["requests"]] == reference for g in groups)
                         for p,groups in measured.items()}
            base.h.save(root / "output_agreement.json", dict(passed=all(agreement.values()), by_policy=agreement))
            if arm == "pp2" and not all(agreement.values()):
                raise RuntimeError("Scheduling changed the generated output")

            if decode and arm == "pp2":
                for p in policies:
                    health(); byte(cfg["gap_policy_path"], ord(str(p)))
                    audit_epoch = case_index*2 + policies.index(p) + 1
                    if audit_epoch > 9: raise RuntimeError("Audit epoch out of bounds")
                    byte(cfg["transfer_audit_control"], ord(str(audit_epoch)))
                    check(base.cohort(current, key, packets))
                    byte(cfg["transfer_audit_control"], ord("0"))
                    # Disabling capture permits a drained control request to
                    # synchronize and publish the bounded payload hashes.
                    for _ in range(3):
                        base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                    visible = [wait_for_stable_json(run / "pp2/transfer_audit" / f"epoch{audit_epoch}.pp{r}.json",
                               timeout_s=45) for r in (0, 1)]
                    wire = [v[0] for v in visible]
                    result = compare_transfers(wire)
                    if not result["passed"]: raise RuntimeError("PP wire payload hashes differ")
                    expected = {tuple(range(5000,5000+case["requests"],2)), tuple(range(5001,5000+case["requests"],2))}
                    for receipt in wire:
                        groups = {tuple(r["marker"] for r in batch["requests"]) for batch in receipt["batches"]
                            if len(batch["requests"]) == case["requests"]//2 and all(8 <= r["generated"] <= 112 for r in batch["requests"])}
                        if groups != expected: raise RuntimeError("Actual PP steady microbatch composition changed")
                    base.h.save(root / f"qualification.p{p}.json", dict(passed=True, wire=result,
                        states=states(run,cfg,p), wire_visibility=[v[1] for v in visible]))
            for p in policies:
                health(); byte(cfg["gap_policy_path"], ord(str(p)))
                if not decode: base.flush_cache(cfg,key)
                directory = root / "traces" / f"p{p}"
                directory.mkdir(parents=True, exist_ok=True)
                progress(run, cfg, f"profile_{arm}_{case['id']}_p{p}")
                base.profile_action(cfg,key,"/start_profile",dict(output_dir=str(directory),
                    num_steps=56 if decode else 1000000, activities=["CPU","GPU"], record_shapes=True,
                    with_stack=False, profile_id=f"{arm}-{case['id']}-p{p}", merge_profiles=False, detailed_annotations=True))
                group = base.cohort(current,key,packets); check(group)
                if not decode: base.profile_action(cfg,key,"/stop_profile")
                exports = base.await_exports(directory, health, timeout=240)
                if len(exports) != 2: raise RuntimeError("Both actual ranks must export component profiles")
                base.h.save(directory / "cohort.json",dict(performance_sample=False,result=group))
                base.h.save(directory / "manifest.json",dict(files=exports, policy=p, phase=case["phase"],
                    performance_sample=False, selected_forwards=16, drop_initial_graphs=20,
                    target_local_tokens=case["requests"]//2))
            byte(cfg["gap_policy_path"],ord("0")); byte(cfg["fixed_cohort_control"],0)
            base.h.save(root / "completed.json",dict(passed=True,policies=policies,formal_cohorts=len(order)))
        base.h.save(run / arm / "completed.json",dict(passed=True,workloads=len(cfg["workloads"])))

    base.environment, base.engine_args = environment, engine_args
    base.progress, base.cohort, base.run_workloads = progress, cohort, workloads
