"""Explicit state-preservation qualification, separate from cold/warm equality."""

import hashlib
import json
from pathlib import Path

PRODUCTION_SOURCES = (
    "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
    "python/sglang/srt/layers/attention/linear/gdn_backend.py",
    "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py",
    "python/sglang/kernels/ops/attention/fla/chunk_delta_h.py",
    "python/sglang/srt/mem_cache/memory_pool.py",
    "python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py",
)


def qualify_cached_reference_evidence(receipt, cfg):
    """Validate a frozen independent state review; never grant generation equality.

    Every JSON and original event file is hash-bound. State checks are required
    for each actual global GDN layer, rather than trusting aggregate booleans.
    The original diagnostic must remain failed, with its offline classification
    and derived analysis preserved as separate files.
    """
    if (
        receipt.get("acceptance_class") != "cached_execution_continuity"
        or receipt.get("cold_vs_warm_generation_qualified") is not False
        or receipt.get("cache_boundary_tokens") != 512
    ):
        raise AssertionError("cached reference requires an explicit bounded class")
    row = cfg["layouts"][0]
    if (
        cfg["initial_counts"] != [20, 0, 20]
        or cfg["grown_counts"] != [13, 13, 14]
        or any(
            row[key] != value
            for key, value in {
                "tp": 1,
                "ep": 1,
                "dp": 1,
                "pp": 3,
                "backend": "none",
            }.items()
        )
        or set(receipt["production_sources"]) != set(PRODUCTION_SOURCES)
    ):
        raise AssertionError(
            "cached state review is confined to the original configuration"
        )
    for relative, expected_hash in receipt["production_sources"].items():
        if (
            hashlib.sha256(
                (Path(cfg["engine_root"]) / relative).read_bytes()
            ).hexdigest()
            != expected_hash
        ):
            raise AssertionError("reviewed production state path changed")
    if (
        hashlib.sha256(
            (Path(cfg["model_path"]) / "config.json").read_bytes()
        ).hexdigest()
        != receipt["model_config_sha256"]
    ):
        raise AssertionError("reviewed model configuration changed")
    artifacts = receipt["evidence"]
    required = {
        "raw_results",
        "raw_events_0",
        "raw_events_2",
        "classification",
        "analysis",
        "independent_review",
        "request_contract",
    }
    if set(artifacts) != required:
        raise AssertionError("cached reference evidence set is incomplete")
    contents, hashes = {}, {}
    for label, descriptor in artifacts.items():
        data = Path(descriptor["path"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != descriptor["sha256"]:
            raise AssertionError(f"cached reference evidence hash changed: {label}")
        hashes[label] = digest
        if not label.startswith("raw_events_"):
            contents[label] = json.loads(data)

    raw = contents["raw_results"]["baseline_probe"]
    if (
        any(
            raw.get(key) is not False
            for key in (
                "passed",
                "diagnostic_completed",
                "generation_qualified",
                "dynamic_qualified",
            )
        )
        or raw.get("cleanup_verified") is not True
    ):
        raise AssertionError("original diagnostic failed status must be preserved")
    contract = contents["request_contract"]
    if contract.get("capture") != "actual pre-serialization stream payload":
        raise AssertionError("reference requests require actual captured payloads")
    cold, warm = (contract["requests"][key] for key in ("cold_active", "active"))
    expected_prompt = [2000 + index % 160 for index in range(512)] + [2160]
    for payload, count in ((cold, 256), (warm, 8)):
        if (
            payload.get("input_ids") != expected_prompt
            or payload.get("stream") is not True
            or payload.get("sampling_params")
            != {"temperature": 0, "max_new_tokens": count, "ignore_eos": True}
        ):
            raise AssertionError("cached reference prompt or sampling differs")
    references = raw["references"]
    if (
        len(references["cold_active"]["output_ids"]) != 256
        or len(references["active"]["output_ids"]) != 8
        or references["active"]["meta_info"]["cached_tokens"] != 512
        or references["active"]["meta_info"]["num_retractions"] != 0
    ):
        raise AssertionError("diagnostic reference boundary is incomplete")

    classification = contents["classification"]
    analysis = contents["analysis"]
    review = contents["independent_review"]
    if (
        classification.get("raw_results_sha256") != hashes["raw_results"]
        or analysis.get("postprocessing") != classification
        or review.get("raw_results_sha256") != hashes["raw_results"]
        or review.get("new_gpu_requests") != 0
        or review.get("continuous_fp32_A512_measured") is not False
        or analysis.get("continuous_fp32_state_at_512_observed") is not False
        or review.get("raw_status", {}).get("diagnostic_completed") is not False
        or review.get("raw_status", {}).get("passed") is not False
    ):
        raise AssertionError("offline classification must bind the raw failed run")
    if set(review["ranks"]) != {"0", "2"} or set(analysis["ranks"]) != {"0", "2"}:
        raise AssertionError("independent state review requires both active ranks")

    def require_bits(check):
        if (
            check.get("same_shape_dtype") is not True
            or check.get("bitwise_equal") is not True
            or check.get("all_finite") is not True
            or check.get("max_abs") != 0.0
        ):
            raise AssertionError("independent state chain is not bitwise preserved")

    checked = []
    for rank, start in (("0", 0), ("2", 20)):
        local = review["ranks"][rank]
        expected = [layer for layer in range(start, start + 20) if layer % 4 != 3]
        if (
            local["layer_ids"] != expected
            or set(local["layers"]) != {str(layer) for layer in expected}
            or analysis["ranks"][rank]["layer_ids"] != expected
            or local["events_sha256"] != hashes[f"raw_events_{rank}"]
            or classification["derived_metadata"][rank]["original_sha256"]
            != hashes[f"raw_events_{rank}"]
            or local["selected_CoW_sequences"]
            != classification["dropped_events"][rank][
                "kept_actual_extend_cow_sequences"
            ]
            or not max(local["selected_CoW_sequences"])
            < local["first_warm_GDN_sequence"]
        ):
            raise AssertionError("state review global layers or event identity differ")
        for stage in (
            "gdn_extend:cold_active",
            "gdn_extend:warm_active",
            "checkpoint_birth:cold_active",
        ):
            if local["raw_stage_counts"].get(stage) != len(expected):
                raise AssertionError("state review must cover every GDN layer once")
        if set(local["checkpoint_chain"]) != {"temporal", "conv_0"}:
            raise AssertionError("state review must cover temporal and conv state")
        # These comparisons cover the complete rank tensor in layer_ids order;
        # equality therefore establishes the middle chain for all 15 layers.
        for checks in local["checkpoint_chain"].values():
            if set(checks) != {"B_prime_to_source", "source_to_C"}:
                raise AssertionError("state review omitted a checkpoint chain link")
            for check in checks.values():
                require_bits(check)
        for layer in expected:
            state = local["layers"][str(layer)]
            if set(state["B_to_B_prime"]) != {"temporal", "conv_0"}:
                raise AssertionError("state review omitted a birth tensor")
            for check in state["B_to_B_prime"].values():
                require_bits(check)
            require_bits(state["h_cast_to_B"])
            require_bits(state["C_to_actual_warm_kernel_initial"])
            if (
                state["warm_kernel_uses_C_address"] is not True
                or state["warm_kernel_physical_slots"] != state["warm_C_physical_slots"]
                or len(state["warm_C_physical_slots"]) != 1
                or state["birth_physical_slots"]
                != state["cold_checkpoint_physical_slots"]
                or state["cold_initial_temporal_max_abs"] != 0.0
                or state["query_start_loc"] != {"cold": [0, 513], "warm": [0, 1]}
                or state["actual_dtypes"]
                != {
                    "cold_k": "torch.bfloat16",
                    "cold_h": "torch.bfloat16",
                    "warm_k": "torch.bfloat16",
                    "warm_h": "torch.bfloat16",
                    "temporal": "torch.float32",
                }
            ):
                raise AssertionError(
                    "actual warm kernel state address or semantics differ"
                )
            checked.append(layer)
    if len(checked) != 30 or len(set(checked)) != 30:
        raise AssertionError("state review must cover 30 unique global GDN layers")
    return {
        "acceptance_class": "cached_execution_continuity",
        "cold_vs_warm_generation_qualified": False,
        "cache_boundary_tokens": 512,
        "independent_global_gdn_layers": checked,
        "evidence_sha256": hashes,
        "production_sources_sha256": receipt["production_sources"],
        "model_config_sha256": receipt["model_config_sha256"],
        "scope": "exact IDs at the same warmed boundary; cold/warm equality unqualified",
    }
