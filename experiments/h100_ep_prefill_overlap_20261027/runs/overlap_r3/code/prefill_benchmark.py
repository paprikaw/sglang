"""Profiler-off Prefill cohorts and independent component profiles."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import time

from artifact_visibility import wait_for_stable_json
from previous_workloads import prompts
from lookahead_workloads import byte


def install(base):
    epoch = [0]

    def cohort(cfg, key, packets):
        epoch[0] = epoch[0] % 254 + 1
        byte(cfg["fixed_cohort_control"], epoch[0])
        gate = threading.Barrier(len(packets))
        with ThreadPoolExecutor(max_workers=len(packets)) as pool:
            futures = [pool.submit(base.stream_one, cfg, key, payload, gate) for payload in packets]
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
            expected = list(range(5000,5000+cfg["requests"],2)) + list(range(5001,5000+cfg["requests"],2))
            assert len(receipts) == 1 and receipts[0]["epoch"] == epoch[0] and receipts[0]["ordered_markers"] == expected
        else:
            assert len(receipts) == 2 and {r["dp_rank"] for r in receipts} == {0,1}
            for r in receipts:
                assert r["epoch"] == epoch[0] and r["ordered_markers"] == list(range(5000+r["dp_rank"],5000+cfg["requests"],2))
        result["admission_receipts"] = receipts
        return result

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
                pp_policies=[0, 1], ep_moe_overlap_policies=cfg["ep_overlap_policies"], decode_lookahead_applicable=False))

            def check(group):
                if len(group["requests"]) != case["requests"]:
                    raise RuntimeError("Prefill cohort size changed")
                for row in group["requests"]:
                    if ((row.get("cached_tokens") or 0) != 0 or row["prompt_tokens"] != case["new_input"]
                            or row["output_tokens"] != 1 or len(row.get("output_ids") or []) != 1):
                        raise RuntimeError("Prefill cache/input/output contract violated")

            policies = [0, 1] if arm == "pp2" else cfg["ep_overlap_policies"]
            if arm == "ep2":
                byte(cfg["ep_overlap_policy_path"], ord("8"))
                health(); base.flush_cache(cfg, key)
                qualification_group = base.cohort(current, key, packets)
                check(qualification_group)
                info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                receipts = [r["experiment_ep_overlap"] for r in info["internal_states"]]
                assert len(receipts) == 2 and all(r["qualification_layers"] == 40 and r["qualification_passed"] and r["drained"] for r in receipts)
                base.h.save(root / "numerical_admission.json", dict(performance_sample=False, receipts=receipts, result=qualification_group))
            measured = {p: [] for p in policies}
            # Warm the same fixed native packing geometry before formal timing.
            for p in policies:
                byte(cfg["gap_policy_path"] if arm == "pp2" else cfg["ep_overlap_policy_path"], ord(str(p)))
                for _ in range(cfg["prefill_warmups"]):
                    health(); base.flush_cache(cfg, key)
                    check(base.cohort(current, key, packets))
            order = [p for i in range(cfg["repeats"])
                     for p in (policies if i % 2 == 0 else list(reversed(policies)))]
            for index, p in enumerate(order):
                health(); byte(cfg["gap_policy_path"] if arm == "pp2" else cfg["ep_overlap_policy_path"], ord(str(p)))
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
                if arm == "ep2":
                    info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=30)
                    receipts = [r["experiment_ep_overlap"] for r in info["internal_states"]]
                    assert len(receipts) == 2 and all(r["drained"] and r["policy"] == p and r["counts"].get(str(p), 0) > 0 for r in receipts)
                    group["overlap_receipts"] = receipts
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
            if arm == "pp2" and not all(agreement.values()):
                raise RuntimeError("Prefill PP policy changed output tokens")
            if cfg.get("collect_prefill_logprobs"):
                values = {}
                for p in policies:
                    byte(cfg["gap_policy_path"] if arm == "pp2" else cfg["ep_overlap_policy_path"], ord(str(p)))
                    probes = [dict(packet, return_logprob=True, logprob_start_len=case["new_input"],
                                   top_logprobs_num=10) for packet in packets]
                    rounds = []
                    for _ in range(2):
                        health(); base.flush_cache(cfg, key)
                        group = base.cohort(current, key, probes); check(group)
                        if not all(r["validation_logits"]["output_top_logprobs"] for r in group["requests"]):
                            raise RuntimeError("First-token numerical diagnostic missing top logprobs")
                        rounds.append(group)
                    values[str(p)] = rounds
                base.h.save(root / "validation.prefill_logprobs.json", dict(performance_sample=False,
                    cached_context=0, new_input=case["new_input"], output_tokens=1, by_policy=values))
            for p in policies:
                health(); byte(cfg["gap_policy_path"] if arm == "pp2" else cfg["ep_overlap_policy_path"], ord(str(p)))
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
            byte(cfg["ep_overlap_policy_path"], ord("0"))
            byte(cfg["fixed_cohort_control"], 0)
            base.h.save(root / "completed.json", dict(passed=True, policies=policies,
                formal_cohorts=len(order), requests_per_cohort=case["requests"]))
        base.h.save(run / arm / "completed.json", dict(passed=True, workloads=len(cfg["workloads"])))

    base.progress = progress
    base.cohort = cohort
    base.run_workloads = workloads
