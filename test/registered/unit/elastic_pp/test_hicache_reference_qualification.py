"""Reject incomplete or altered evidence for the explicit warmed-cache class."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def qualification(tmp_path):
    path = (
        Path(__file__).resolve().parents[4]
        / "research/pp_elastic/hicache_reference_qualification.py"
    )
    spec = importlib.util.spec_from_file_location("reference_qualification_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bits = {
        "same_shape_dtype": True,
        "bitwise_equal": True,
        "all_finite": True,
        "max_abs": 0.0,
    }
    source_hashes = {}
    for relative in module.PRODUCTION_SOURCES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative)
        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "config.json").write_text("model configuration")
    cfg = {
        "engine_root": str(tmp_path),
        "model_path": str(tmp_path),
        "initial_counts": [20, 0, 20],
        "grown_counts": [13, 13, 14],
        "layouts": [{"tp": 1, "ep": 1, "dp": 1, "pp": 3, "backend": "none"}],
    }
    raw = {
        "baseline_probe": {
            "passed": False,
            "diagnostic_completed": False,
            "generation_qualified": False,
            "dynamic_qualified": False,
            "cleanup_verified": True,
            "references": {
                "cold_active": {"output_ids": [1] * 256},
                "active": {
                    "output_ids": [2] * 8,
                    "meta_info": {"cached_tokens": 512, "num_retractions": 0},
                },
            },
        }
    }
    contract = {"capture": "actual pre-serialization stream payload", "requests": {}}
    for label, count in (("cold_active", 256), ("active", 8)):
        contract["requests"][label] = {
            "input_ids": [2000 + index % 160 for index in range(512)] + [2160],
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        }
    classification = {"dropped_events": {}, "derived_metadata": {}}
    analysis = {"continuous_fp32_state_at_512_observed": False, "ranks": {}}
    review = {
        "new_gpu_requests": 0,
        "continuous_fp32_A512_measured": False,
        "raw_status": {"passed": False, "diagnostic_completed": False},
        "ranks": {},
        "aggregate_checks": {"everything": True},
    }
    files = {
        "raw_results": raw,
        "request_contract": contract,
        "classification": classification,
        "analysis": analysis,
        "independent_review": review,
    }
    for rank, start in (("0", 0), ("2", 20)):
        files[f"raw_events_{rank}"] = f"original events for rank {rank}".encode()
        layers = [layer for layer in range(start, start + 20) if layer % 4 != 3]
        analysis["ranks"][rank] = {"layer_ids": layers}
        classification["dropped_events"][rank] = {
            "kept_actual_extend_cow_sequences": [1, 2]
        }
        classification["derived_metadata"][rank] = {}
        review["ranks"][rank] = {
            "layer_ids": layers,
            "selected_CoW_sequences": [1, 2],
            "first_warm_GDN_sequence": 3,
            "raw_stage_counts": {
                stage: 15
                for stage in (
                    "gdn_extend:cold_active",
                    "gdn_extend:warm_active",
                    "checkpoint_birth:cold_active",
                )
            },
            "checkpoint_chain": {
                name: {
                    link: copy.deepcopy(bits)
                    for link in ("B_prime_to_source", "source_to_C")
                }
                for name in ("temporal", "conv_0")
            },
            "layers": {
                str(layer): {
                    "B_to_B_prime": {
                        name: copy.deepcopy(bits) for name in ("temporal", "conv_0")
                    },
                    "h_cast_to_B": copy.deepcopy(bits),
                    "C_to_actual_warm_kernel_initial": copy.deepcopy(bits),
                    "warm_kernel_uses_C_address": True,
                    "warm_kernel_physical_slots": [11],
                    "warm_C_physical_slots": [11],
                    "birth_physical_slots": [9],
                    "cold_checkpoint_physical_slots": [9],
                    "cold_initial_temporal_max_abs": 0.0,
                    "query_start_loc": {"cold": [0, 513], "warm": [0, 1]},
                    "actual_dtypes": {
                        "cold_k": "torch.bfloat16",
                        "cold_h": "torch.bfloat16",
                        "warm_k": "torch.bfloat16",
                        "warm_h": "torch.bfloat16",
                        "temporal": "torch.float32",
                    },
                }
                for layer in layers
            },
        }

    def write():
        descriptors = {}

        def save(label):
            target = tmp_path / (label + ".json")
            data = (
                files[label]
                if isinstance(files[label], bytes)
                else json.dumps(files[label], sort_keys=True).encode()
            )
            target.write_bytes(data)
            descriptors[label] = {
                "path": str(target),
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        for label in (
            "raw_results",
            "raw_events_0",
            "raw_events_2",
            "request_contract",
        ):
            save(label)
        classification["raw_results_sha256"] = review["raw_results_sha256"] = (
            descriptors["raw_results"]["sha256"]
        )
        for rank in ("0", "2"):
            digest = descriptors[f"raw_events_{rank}"]["sha256"]
            review["ranks"][rank]["events_sha256"] = digest
            classification["derived_metadata"][rank]["original_sha256"] = digest
        analysis["postprocessing"] = copy.deepcopy(classification)
        for label in ("classification", "analysis", "independent_review"):
            save(label)
        return {
            "acceptance_class": "cached_execution_continuity",
            "cold_vs_warm_generation_qualified": False,
            "cache_boundary_tokens": 512,
            "production_sources": source_hashes,
            "model_config_sha256": hashlib.sha256(
                (tmp_path / "config.json").read_bytes()
            ).hexdigest(),
            "evidence": descriptors,
        }

    return module, cfg, files, write


def test_complete_state_review_grants_only_the_cached_execution_class(qualification):
    module, cfg, _, write = qualification
    result = module.qualify_cached_reference_evidence(write(), cfg)
    assert result["acceptance_class"] == "cached_execution_continuity"
    assert result["cold_vs_warm_generation_qualified"] is False
    assert len(set(result["independent_global_gdn_layers"])) == 30
    assert len(result["evidence_sha256"]) == 7


@pytest.mark.parametrize(
    "fault",
    [
        "missing_layer",
        "wrong_global_layer",
        "duplicate_layer",
        "false_bits",
        "nan_state",
        "different_dtype",
        "wrong_slot",
        "missing_conv",
        "middle_chain",
        "kernel_initial",
        "extra_kernel",
        "modified_raw_status",
        "reconstructed_payload",
        "changed_prompt",
        "wrong_boundary",
    ],
)
def test_aggregate_true_cannot_mask_individual_state_or_scope_failures(
    qualification, fault
):
    module, cfg, files, write = qualification
    local = files["independent_review"]["ranks"]["0"]
    state = local["layers"]["0"]
    if fault == "missing_layer":
        del local["layers"]["0"]
    elif fault == "wrong_global_layer":
        local["layer_ids"][0] = 3
    elif fault == "duplicate_layer":
        local["layer_ids"][0] = 1
    elif fault == "false_bits":
        state["B_to_B_prime"]["temporal"]["bitwise_equal"] = False
    elif fault == "nan_state":
        state["B_to_B_prime"]["conv_0"]["all_finite"] = False
    elif fault == "different_dtype":
        state["h_cast_to_B"]["same_shape_dtype"] = False
    elif fault == "wrong_slot":
        state["warm_kernel_physical_slots"] = [12]
    elif fault == "missing_conv":
        del state["B_to_B_prime"]["conv_0"]
    elif fault == "middle_chain":
        local["checkpoint_chain"]["conv_0"]["source_to_C"]["max_abs"] = 0.01
    elif fault == "kernel_initial":
        state["C_to_actual_warm_kernel_initial"]["bitwise_equal"] = False
    elif fault == "extra_kernel":
        local["raw_stage_counts"]["gdn_extend:warm_active"] = 16
    elif fault == "modified_raw_status":
        files["raw_results"]["baseline_probe"]["diagnostic_completed"] = True
    elif fault == "reconstructed_payload":
        files["request_contract"]["capture"] = "historical reconstruction"
    elif fault == "changed_prompt":
        files["request_contract"]["requests"]["active"]["input_ids"][-1] = 0
    elif fault == "wrong_boundary":
        files["raw_results"]["baseline_probe"]["references"]["active"]["meta_info"][
            "cached_tokens"
        ] = 448
    with pytest.raises(AssertionError):
        module.qualify_cached_reference_evidence(write(), cfg)


@pytest.mark.parametrize(
    "fault",
    [
        "changed_hash",
        "changed_source",
        "different_topology",
        "generation_true",
        "missing_artifact",
    ],
)
def test_receipt_integrity_and_explicit_class_are_required(qualification, fault):
    module, cfg, _, write = qualification
    receipt = write()
    if fault == "changed_hash":
        Path(receipt["evidence"]["raw_events_0"]["path"]).write_text("changed")
    elif fault == "changed_source":
        (Path(cfg["engine_root"]) / module.PRODUCTION_SOURCES[0]).write_text("changed")
    elif fault == "different_topology":
        cfg["initial_counts"] = [13, 13, 14]
    elif fault == "generation_true":
        receipt["cold_vs_warm_generation_qualified"] = True
    elif fault == "missing_artifact":
        del receipt["evidence"]["classification"]
    with pytest.raises(AssertionError):
        module.qualify_cached_reference_evidence(receipt, cfg)
