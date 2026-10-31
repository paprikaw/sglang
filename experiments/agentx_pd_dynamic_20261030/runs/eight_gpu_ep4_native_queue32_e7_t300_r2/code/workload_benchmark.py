"""Broader workloads for the shared Expert GEMM policy; native PP scheduling."""
import hashlib
import os
from pathlib import Path
import time


TOPICS = [
    "Write Python code to merge two sorted lists and discuss its complexity.",
    "Explain how a compiler converts an expression into machine instructions.",
    "Derive the quadratic formula and give a worked numerical example.",
    "Describe the water cycle, clouds, rainfall, and evaporation.",
    "Compare relational database indexing with sequential table scans.",
    "Explain how TCP handles packet loss and acknowledges delivery.",
    "Describe photosynthesis and the roles of chlorophyll and sunlight.",
    "Write a short story about a traveller who discovers an old library.",
    "Discuss harmony, rhythm, melody, and the structure of a string quartet.",
    "Give a recipe for vegetable soup, with numbered preparation steps.",
    "Explain conservation of energy using a pendulum as an example.",
    "Compare breadth-first and depth-first search on a directed graph.",
    "Describe the main features of ancient Roman architecture.",
    "Explain numerical integration using the trapezoidal rule.",
    "Discuss methods for testing software and diagnosing a race condition.",
    "Describe the sequence of events during a solar eclipse.",
    "Explain how an operating system schedules runnable processes.",
    "Discuss the functions of roots, stems, leaves, and seeds in plants.",
    "Describe the composition, orbit, and rings of Saturn.",
    "Write a SQL query to group sales by month and compute their total.",
    "Explain the difference between a vector space and a basis.",
    "Describe how a bridge transfers loads through beams and supports.",
    "Compare renewable energy generation from wind and sunlight.",
    "Explain how a hash table stores keys and resolves collisions.",
    "Describe the use of perspective and light in landscape painting.",
    "Explain the Fourier transform using a combination of sine waves.",
    "Discuss how dictionaries record pronunciation, usage, and meaning.",
    "Describe how a telescope collects and focuses light from stars.",
    "Explain matrix multiplication with a small worked example.",
    "Describe the stages involved in designing and building a small robot.",
    "Write pseudocode for binary search with an explanation of its invariant.",
    "Explain how tides depend on the gravitational attraction of the Moon.",
]


def prompts(cfg, case):
    if case["family"] == "repeated":
        from sampling import make_tokens
        return make_tokens(dict(cfg, requests=case["requests"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    result = []
    for i in range(case["requests"]):
        topic = TOPICS[i % len(TOPICS)]
        head = [5000 + i] + tokenizer.encode(f"Document {i}. {topic}\n", add_special_tokens=False)
        body = tokenizer.encode(topic + " Provide clear reasoning and concrete details.\n", add_special_tokens=False)
        tail = tokenizer.encode(f"\nQuestion {i}: {topic}\nAnswer:", add_special_tokens=False)
        limit = cfg["prefix_tokens"] - len(tail)
        row = (head + body * (limit // len(body) + 2))[:limit] + tail
        if len(row) != cfg["prefix_tokens"]:
            raise RuntimeError("Diverse prompt length changed")
        result.append(row)
    return result


def workloads(base, run, cfg, arm, key, health):
    from benchmark import policy
    from sampling import set_diagnostic
    from fair_validation import atomic_validation
    for case in cfg["workloads"]:
        cfg_case = dict(cfg, requests=case["requests"])
        root = run / arm / "workloads" / case["id"]
        root.mkdir(parents=True, exist_ok=True)
        Path(cfg["expert_workload_path"]).write_text(case["id"] + "\n")
        local = case["requests"] // 2
        base.progress(run, cfg, "workload_" + arm + "_" + case["id"])
        health()
        if arm == "pp2":
            response = base.h.http_json(cfg["port"], "/set_internal_state", key,
                dict(server_args=dict(pp_max_micro_batch_size=local)), timeout=30)
            if isinstance(response, dict) and response.get("success") is False:
                raise RuntimeError("Native idle PP microbatch update was rejected")
        else:
            response = None
        # Safe readback also records actual graph, DP, allocator and PP state.
        info = base.h.http_json(cfg["port"], "/get_server_info", key, timeout=15)
        base.h.save(root / "server_info.json", base.h.safe_server_info(info))
        tokens = prompts(cfg_case, case)
        base.h.save(root / "protocol.json", dict(**case, cached_context_per_request=cfg["prefix_tokens"],
            new_input_tokens_per_request=1, output_tokens_per_request=cfg["output_tokens"],
            target_local_decode_tokens=local, pp_slots=2, profiler_off_formal=True,
            no_cpu_kv_restore=True, pp_microbatch_update_accepted=True,
            prompt_sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens]))
        pref = [base.packet(cfg_case, arm, t, 1, i) for i,t in enumerate(tokens)]
        dec = [base.packet(cfg_case, arm, t+[198], cfg["output_tokens"], i) for i,t in enumerate(tokens)]

        def check(group, cached):
            if len(group["requests"]) != case["requests"]:
                raise RuntimeError("Lost requests in formal cohort")
            expected_output = cfg["output_tokens"] if cached else 1
            expected_cache = cfg["prefix_tokens"] if cached else 0
            for row in group["requests"]:
                if row["output_tokens"] != expected_output or (row.get("cached_tokens") or 0) != expected_cache:
                    raise RuntimeError("Formal cohort violated exact output/cache protocol")

        policy(cfg, b"0")
        set_diagnostic(cfg, False)
        base.flush_cache(cfg, key)
        base.cohort(cfg_case, key, pref)
        prefill = []
        for repeat in range(cfg["repeats"]):
            health()
            base.flush_cache(cfg, key)
            group = base.cohort(cfg_case, key, pref)
            check(group, False)
            prefill.append(group)
            base.h.save(root / "prefill.partial.json", dict(rounds=prefill))
        description = f"B{case['requests']}, {case['family']}; cache0 + new{cfg['prefix_tokens']} -> output1"
        base.h.save(root / "metrics.prefill.json", base.summarize(prefill, description))
        base.flush_cache(cfg, key)
        prime = base.cohort(cfg_case, key, pref)
        check(prime, False)
        base.h.save(root / "prefix_preparation.json", dict(performance_sample=False,
            real_model_prefix=True, cpu_kv_restore=False,
            exact_prefix_tokens_per_request=cfg["prefix_tokens"]))
        # Cover every local capture shape up to the target outside timing.
        for count in sorted(set([2, 4, 6, 8, case["requests"]//2, case["requests"]])) + [case["requests"]]*2:
            health()
            base.cohort(cfg_case, key, dec[:min(count, case["requests"])])
        measured = []
        for repeat in range(cfg["repeats"]):
            health()
            base.progress(run, cfg, "timing_"+arm+"_"+case["id"], repeat=repeat+1)
            group = base.cohort(cfg_case, key, dec)
            check(group, True)
            measured.append(group)
            base.h.save(root / "decode.partial.json", dict(rounds=measured))
        description = f"B{case['requests']}, {case['family']}; cache{cfg['prefix_tokens']} + new1 -> output{cfg['output_tokens']}"
        base.h.save(root / "metrics.decode.json", base.summarize(measured, description))
        # Eager fallback is used only for inspection of real routed activations.
        # Clear instrumentation's per-workload keys in the frozen adapters.
        set_diagnostic(cfg, True)
        policy(cfg, b"D")
        diagnostic = base.cohort(cfg_case, key, dec)
        check(diagnostic, True)
        policy(cfg, b"0")
        set_diagnostic(cfg, False)
        for phase, packets in [("prefill", pref), ("decode", dec)]:
            health()
            base.flush_cache(cfg, key)
            if phase == "decode":
                base.cohort(cfg_case, key, pref)
                base.cohort(cfg_case, key, dec)
            trace_root = root / "traces" / phase
            trace_root.mkdir(parents=True, exist_ok=True)
            base.profile_action(cfg, key, "/start_profile", dict(output_dir=str(trace_root),
                num_steps=1000000 if phase == "prefill" else cfg["profile_steps"],
                activities=["CPU", "GPU"], record_shapes=True,
                profile_id=arm+"-"+case["id"]+"-"+phase,
                with_stack=False, detailed_annotations=True, merge_profiles=False))
            base.cohort(cfg_case, key, packets)
            if phase == "prefill":
                base.profile_action(cfg, key, "/stop_profile")
            traces = base.await_exports(trace_root, health, timeout=240)
            if len(traces) != 2:
                raise RuntimeError("Both ranks must export the independent profile")
            base.h.save(trace_root / "manifest.json", dict(phase=phase, performance_sample=False,
                execution="native", target_local_tokens=local,
                selected_forwards=cfg["selected_forwards"], files=traces, timestamp=time.time()))
        base.flush_cache(cfg, key)
        base.cohort(cfg_case, key, pref)
        policy(cfg, b"V")
        validation = [atomic_validation(base, cfg_case, arm, key, tokens) for _ in range(2)]
        base.h.save(root / "validation.probabilities.json", dict(performance_sample=False,
             rounds=validation, output_tokens=8, requests=case["requests"],
             target_local_decode_tokens=local))
        policy(cfg, b"0")
        base.h.save(root / "completed.json", dict(passed=True, formal_rounds=cfg["repeats"]))
    base.h.save(run / arm / "completed.json", dict(passed=True, workloads=len(cfg["workloads"])))


def install(base):
    previous_environment = base.environment
    def environment(cfg, arm, inherited):
        result = previous_environment(cfg, arm, inherited)
        result["EXPERT_WORKLOAD_PATH"] = cfg["expert_workload_path"]
        return result
    base.environment = environment
    base.run_workloads = lambda run,cfg,arm,key,health: workloads(base,run,cfg,arm,key,health)
    def progress(run,cfg,stage,**extra):
        base.h.save(run/"task/progress.json", dict(experiment_id=cfg["id"],task_id="comparison",
            stage=stage,timestamp=time.time(),controller_pid=os.getpid(),
            valid_points=len(list(run.glob("*/workloads/*/metrics.*.json"))),
            expected=cfg["expected_points"], **extra))
    base.progress = progress
