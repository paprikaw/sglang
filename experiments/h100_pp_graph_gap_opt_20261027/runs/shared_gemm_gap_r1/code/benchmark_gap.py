"""Controlled shared-GEMM EP reference and cumulative PP scheduling candidates."""
import hashlib
import json
import os
from pathlib import Path
import time

NAMES = ("baseline", "early", "early_local", "early_local_schema")


def byte(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        if os.pwrite(fd, bytes([value]), 0) != 1:
            raise RuntimeError("Policy write failed")
    finally:
        os.close(fd)


def install(base):
    original_environment = base.environment
    original_cohort = base.cohort
    epochs = [0]

    def environment(cfg, arm, inherited):
        env = original_environment(cfg, arm, inherited)
        env.update(PP_GAP_POLICY_PATH=cfg["gap_policy_path"],
                   PP_FIXED_COHORT_CONTROL=cfg["fixed_cohort_control"],
                   PP_TRANSFER_AUDIT_CONTROL=cfg["transfer_audit_control"])
        return env

    def cohort(cfg, key, payloads):
        is_pp = not any("data_parallel_rank" in packet for packet in payloads)
        if is_pp and len(payloads) == 8:
            epochs[0] += 1
            if epochs[0] >= 255:
                raise RuntimeError("Fixed cohort epoch exhausted")
            byte(cfg["fixed_cohort_control"], epochs[0])
        else:
            byte(cfg["fixed_cohort_control"], 0)
        result = original_cohort(cfg, key, payloads)
        # Outside client timing. Let both stages retire the full cohort before
        # changing transport protocol, publishing receipts or taking a profile.
        for _ in range(3):
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
        if is_pp and len(payloads) == 8:
            receipts = [s.get("experiment_fixed_group_receipt") for s in info["internal_states"]
                        if s.get("experiment_fixed_group_receipt")]
            if len(receipts) != 1 or receipts[0]["epoch"] != epochs[0]:
                raise RuntimeError("Fixed PP cohort not acknowledged")
            if receipts[0]["ordered_markers"] != [5000,5002,5004,5006,5001,5003,5005,5007]:
                raise RuntimeError("Unexpected PP request groups")
            result["fixed_group_receipt"] = receipts[0]
        return result

    base.environment = environment
    base.cohort = cohort

    def workloads(run, cfg, arm, key, health):
        from sampling import make_tokens, set_diagnostic
        from pp_transfer_audit import compare_transfers
        prompts = make_tokens(cfg)
        prefix = [base.packet(cfg, arm, t, 1, i) for i, t in enumerate(prompts)]
        decode = [base.packet(cfg, arm, t + [198], 128, i) for i, t in enumerate(prompts)]
        base.h.save(run / arm / "prompt_fingerprints.json", dict(
            prefix_tokens=4096, sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in prompts]))
        byte(cfg["fragment_policy_path"], ord("0"))
        byte(cfg["gap_policy_path"], ord("0"))
        byte(cfg["transfer_audit_control"], ord("0"))
        set_diagnostic(cfg, False)
        base.flush_cache(cfg, key)
        base.h.save(run / arm / "prefix_preparation.json", dict(
            measured=False, real_model_prefix=True, result=base.cohort(cfg, key, prefix)))
        prefills = []
        for _ in range(3):
            health(); base.flush_cache(cfg, key)
            group = base.cohort(cfg, key, prefix)
            if any((r.get("cached_tokens") or 0) != 0 for r in group["requests"]):
                raise RuntimeError("Cold Prefill reused cache")
            prefills.append(group)
        base.h.save(run / arm / "metrics.prefill.json", base.summarize(prefills, "cold4096->1; B8"))
        base.flush_cache(cfg, key)
        base.cohort(cfg, key, prefix)
        for count in (2, 4, 6, 8, 8):
            health(); base.cohort(cfg, key, decode[:count])

        def check(group):
            for row in group["requests"]:
                if row.get("cached_tokens") != 4096 or row["output_tokens"] != 128 or len(row.get("output_ids") or []) != 128:
                    raise RuntimeError("Cache/output mismatch")

        policies = cfg["gap_policies"] if arm == "pp2" else [0]
        # Counterbalance both direction and drift. One untimed warmup after each
        # change, then all formal samples retained. Three samples per policy.
        order = policies + list(reversed(policies)) + policies
        values = {p: [] for p in policies}
        for index, policy in enumerate(order):
            health(); byte(cfg["gap_policy_path"], ord(str(policy)))
            check(base.cohort(cfg, key, decode))
            base.progress(run, cfg, f"{arm}_timing_{index}_p{policy}")
            group = base.cohort(cfg, key, decode); check(group)
            values[policy].append(group)
            base.h.save(run / arm / f"cohort.{index}.p{policy}.json", dict(
                policy=policy, name=NAMES[policy], performance_sample=True,
                profiler_enabled=False, result=group))
            base.h.save(run / arm / f"metrics.decode.p{policy}.json", base.summarize(
                values[policy], "GPUcached4096/new1/output128/B8; shared Triton; fixed request groups"))
        reference = [r["output_ids"] for r in values[0][0]["requests"]]
        agreement = {str(p): all([r["output_ids"] for r in group["requests"]] == reference for group in groups)
                     for p, groups in values.items()}
        base.h.save(run / arm / "output_agreement.json", dict(
            all_modes_equal=all(agreement.values()), by_policy=agreement,
            reference=reference, output_tokens_checked=sum(len(g["requests"]) * 128 for groups in values.values() for g in groups)))
        if arm == "pp2" and not all(agreement.values()):
            raise RuntimeError("PP scheduling changed fixed-group generated tokens")

        # Cloning and wire hashes are diagnostic only. Local tail ownership is
        # independently guarded by slot and request-ancestry checks.
        if arm == "pp2":
            for policy in policies:
                byte(cfg["gap_policy_path"], ord(str(policy)))
                check(base.cohort(cfg, key, decode))
                epoch = policy + 1
                byte(cfg["transfer_audit_control"], ord(str(epoch)))
                check(base.cohort(cfg, key, decode))
                byte(cfg["transfer_audit_control"], ord("0"))
                paths = [run / arm / "transfer_audit" / f"epoch{epoch}.pp{rank}.json" for rank in (0,1)]
                deadline = time.monotonic() + 45
                while not all(path.exists() for path in paths):
                    health(); base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                    if time.monotonic() > deadline:
                        raise TimeoutError("Transfer capture missing")
                    time.sleep(.05)
                receipts = [json.loads(p.read_text()) for p in paths]
                audit = compare_transfers(receipts)
                if not audit["passed"]:
                    raise RuntimeError("PP wire payload hash mismatch")
                expected = {(5000,5002,5004,5006), (5001,5003,5005,5007)}
                for receipt in receipts:
                    groups = {tuple(r["marker"] for r in batch["requests"]) for batch in receipt["batches"]
                              if len(batch["requests"]) == 4 and all(8 <= r["generated"] <= 112 for r in batch["requests"])}
                    if groups != expected:
                        raise RuntimeError("Actual PP groups changed")
                state = [json.loads((run / arm / "gap_receipts" / f"policy{policy}.pp{rank}.json").read_text()) for rank in (0,1)]
                if any(r["pending"] or r["produced"] != r["retired"] for r in state):
                    raise RuntimeError("Local feedback did not drain")
                if policy >= 2 and not state[1]["produced"]:
                    raise RuntimeError("Local tail feedback not used")
                if policy == 3 and not all(r["fast_sent"] and r["fast_received"] for r in state):
                    raise RuntimeError("Cached schema transport not used")
                base.h.save(run / arm / f"qualification.p{policy}.json", dict(
                    passed=True, wire=audit, states=state, fixed_groups=True, numerical_math_unchanged=True))

        for policy in policies:
            byte(cfg["gap_policy_path"], ord(str(policy)))
            check(base.cohort(cfg, key, decode)); health()
            directory = run / arm / "traces" / f"p{policy}"
            directory.mkdir(parents=True)
            base.profile_action(cfg, key, "/start_profile", dict(output_dir=str(directory), num_steps=40,
                activities=["CPU", "GPU"], record_shapes=True, with_stack=False,
                profile_id=f"{arm}-p{policy}", merge_profiles=False, detailed_annotations=True))
            group = base.cohort(cfg, key, decode); check(group)
            base.h.save(directory / "cohort.json", dict(performance_sample=False, result=group))
            exports = base.await_exports(directory, health, timeout=240)
            base.h.save(directory / "manifest.json", dict(files=exports, policy=policy, performance_sample=False))
        byte(cfg["gap_policy_path"], ord("0"))
        byte(cfg["fixed_cohort_control"], 0)
        base.h.save(run / arm / "completed.json", dict(passed=True, policies=policies, formal_cohorts=len(order)))

    base.run_workloads = workloads
