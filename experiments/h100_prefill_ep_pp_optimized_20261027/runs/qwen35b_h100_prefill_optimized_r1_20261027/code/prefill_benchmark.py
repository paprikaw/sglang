"""Profiler-off Prefill cohorts and independent component profiles."""
import hashlib
import os
from pathlib import Path
import time

from artifact_visibility import wait_for_stable_json
from previous_workloads import prompts
from lookahead_workloads import byte


def install(base):
    def progress(run, cfg, stage, **extra):
        base.h.save(run / "task/progress.json", dict(experiment_id=cfg["id"], task_id="comparison",
            stage=stage, timestamp=time.time(), controller_pid=os.getpid(),
            valid_points=len(list(run.glob("*/workloads/*/metrics.prefill.p*.json"))),
            expected=cfg["expected_points"], **extra))

    def states(run, cfg, p):
        base.h.http_json(cfg["port"], "/get_server_info", base.h.read_key(cfg), timeout=30)
        result = [wait_for_stable_json(run / "pp2/gap_receipts" / f"policy{p}.pp{r}.json",
                  timeout_s=25, poll_s=.01, min_stable_s=.02)[0] for r in (0, 1)]
        for row in result:
            if row["pending"] or row["ahead_pending"] or row["early_proxy_pending"]:
                raise RuntimeError("PP work did not drain at the cohort boundary")
            if row["prepared_next"] or row["ahead_produced"] or row["fast_sent"]:
                raise RuntimeError("Decode-only optimization unexpectedly used in Prefill")
        return result

    def workloads(run, cfg, arm, key, health):
        from benchmark import policy as expert_policy
        from sampling import set_diagnostic
        expert_policy(cfg, b"0")
        set_diagnostic(cfg, False)
        byte(cfg["transfer_audit_control"], ord("0"))
        byte(cfg["gap_policy_path"], ord("0"))
        for case in cfg["workloads"]:
            current = dict(cfg, requests=case["requests"], prefix_tokens=case["new_input"])
            root = run / arm / "workloads" / case["id"]
            root.mkdir(parents=True, exist_ok=True)
            Path(cfg["expert_workload_path"]).write_text(case["id"] + "\n")
            byte(cfg["fixed_cohort_size"], case["requests"])
            byte(cfg["gap_policy_path"], ord("0"))
            if arm == "pp2":
                response = base.h.http_json(cfg["port"], "/set_internal_state", key,
                    dict(server_args=dict(pp_max_micro_batch_size=case["requests"] // 2)), timeout=30)
                if isinstance(response, dict) and response.get("success") is False:
                    raise RuntimeError("PP microbatch update rejected")
            info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
            base.h.save(root / "server_info.json", base.h.safe_server_info(info))
            tokens = prompts(current, case)
            packets = [base.packet(current, arm, t, 1, i) for i, t in enumerate(tokens)]
            base.h.save(root / "protocol.json", dict(**case, cached_context_per_request=0,
                new_input_tokens_per_request=case["new_input"], output_tokens_per_request=1,
                unique_total_new_tokens=case["requests"] * case["new_input"],
                input_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                profiler_off_formal=True, prefill_cuda_graph=False,
                shared_expert_optimization_available_to_both=True,
                small_kernel_source_token_limit=16, actual_prefill_token_budget_per_attention=4096,
                native_ep_shared_overlap=True, two_batch_overlap=False,
                pp_policies=[0, 1], decode_lookahead_applicable=False))

            def check(group):
                if len(group["requests"]) != case["requests"]:
                    raise RuntimeError("Prefill cohort size changed")
                for row in group["requests"]:
                    if ((row.get("cached_tokens") or 0) != 0 or row["prompt_tokens"] != case["new_input"]
                            or row["output_tokens"] != 1 or len(row.get("output_ids") or []) != 1):
                        raise RuntimeError("Prefill cache/input/output contract violated")

            policies = [0, 1] if arm == "pp2" else [0]
            measured = {p: [] for p in policies}
            # Two cold untimed cohorts cover both policies before formal timing.
            for p in policies:
                byte(cfg["gap_policy_path"], ord(str(p)))
                health(); base.flush_cache(cfg, key)
                check(base.cohort(current, key, packets))
            order = [p for i in range(cfg["repeats"])
                     for p in (policies if i % 2 == 0 else list(reversed(policies)))]
            for index, p in enumerate(order):
                health(); byte(cfg["gap_policy_path"], ord(str(p)))
                base.flush_cache(cfg, key)
                before = states(run, cfg, p) if arm == "pp2" else None
                progress(run, cfg, f"timing_{arm}_{case['id']}_p{p}", repeat=index + 1)
                group = base.cohort(current, key, packets)
                check(group)
                if arm == "pp2":
                    after = states(run, cfg, p)
                    deltas = [b["early_proxy_submissions"] - a["early_proxy_submissions"]
                              for a, b in zip(before, after)]
                    if (p == 1 and not (deltas[0] > 0 and deltas[1] == 0)) or (p == 0 and any(deltas)):
                        raise RuntimeError("Early activation submission did not match selected policy")
                    group["scheduling_receipt"] = dict(policy=p, early_proxy_deltas=deltas, states=after)
                measured[p].append(group)
                base.h.save(root / f"cohort.{index}.p{p}.json", dict(policy=p,
                    performance_sample=True, profiler_enabled=False, result=group))
                description = f"Cold Prefill B{case['requests']} new{case['new_input']} -> output1; policy{p}"
                base.h.save(root / f"metrics.prefill.p{p}.json", base.summarize(measured[p], description))
            reference = [r["output_ids"] for r in measured[0][0]["requests"]]
            agreement = {str(p): all([r["output_ids"] for r in group["requests"]] == reference
                                    for group in groups) for p, groups in measured.items()}
            base.h.save(root / "output_agreement.json", dict(passed=all(agreement.values()), by_policy=agreement,
                scope="identical one-token outputs across formal repetitions/policies within this layout"))
            if not all(agreement.values()):
                raise RuntimeError("Prefill PP policy changed output tokens")
            for p in policies:
                health(); byte(cfg["gap_policy_path"], ord(str(p)))
                base.flush_cache(cfg, key)
                directory = root / "traces" / f"p{p}"
                directory.mkdir(parents=True, exist_ok=True)
                progress(run, cfg, f"profile_{arm}_{case['id']}_p{p}")
                base.profile_action(cfg, key, "/start_profile", dict(output_dir=str(directory),
                    num_steps=1000000, activities=["CPU", "GPU"], record_shapes=True,
                    with_stack=False, profile_id=f"{arm}-{case['id']}-p{p}-prefill",
                    merge_profiles=False, detailed_annotations=True))
                group = base.cohort(current, key, packets)
                check(group)
                base.profile_action(cfg, key, "/stop_profile")
                exports = base.await_exports(directory, health, timeout=240)
                if len(exports) != 2:
                    raise RuntimeError("Both ranks must export independent Prefill profiles")
                base.h.save(directory / "cohort.json", dict(performance_sample=False, result=group))
                base.h.save(directory / "manifest.json", dict(files=exports, policy=p, phase="cold_prefill",
                    performance_sample=False, cached_context=0, new_input=case["new_input"],
                    global_requests=case["requests"], timestamp=time.time()))
            byte(cfg["gap_policy_path"], ord("0"))
            byte(cfg["fixed_cohort_control"], 0)
            base.h.save(root / "completed.json", dict(passed=True, policies=policies,
                formal_cohorts=len(order), requests_per_cohort=case["requests"]))
        base.h.save(run / arm / "completed.json", dict(passed=True, workloads=len(cfg["workloads"])))

    base.progress = progress
    base.run_workloads = workloads
