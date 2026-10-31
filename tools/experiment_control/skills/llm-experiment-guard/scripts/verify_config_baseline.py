#!/usr/bin/env python3
"""Verify a vLLM experiment config against the current guard baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1] / "current_experiment_config_baseline.yaml"
)
DEFAULT_PROFILING_CONTROLS = (
    Path(__file__).resolve().parents[1] / "profiling_controls.yaml"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def compare_required(
    actual: Any,
    expected: Any,
    path: str,
    errors: list[str],
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected mapping, got {type(actual).__name__}")
            return
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                errors.append(f"{child_path}: missing")
                continue
            compare_required(actual[key], expected_value, child_path, errors)
        return

    if actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")


def iter_leaf_paths(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(data, dict):
        return [(prefix, data)]

    leaves: list[tuple[str, Any]] = []
    for key, value in data.items():
        child_path = f"{prefix}.{key}" if prefix else str(key)
        leaves.extend(iter_leaf_paths(value, child_path))
    return leaves


def get_dotted_path(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def compare_defaults(
    config: dict[str, Any],
    baseline: dict[str, Any],
    errors: list[str],
) -> list[str]:
    defaults = baseline.get("defaults", {})
    if defaults is None:
        return []
    if not isinstance(defaults, dict):
        errors.append("baseline.defaults: missing or not a mapping")
        return []

    notes: list[str] = []
    for path, default_value in iter_leaf_paths(defaults):
        exists, actual_value = get_dotted_path(config, path)
        if not exists:
            errors.append(f"{path}: missing default-overridable field")
            continue
        if actual_value != default_value:
            notes.append(f"{path}: default {default_value!r}, got {actual_value!r}")
    return notes


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_profiling_controls(
    config: dict[str, Any],
    controls: dict[str, Any],
    *,
    allow_profiling: bool,
    errors: list[str],
) -> list[str]:
    envs = config.get("envs", {})
    if not isinstance(envs, dict):
        errors.append("envs: expected mapping for profiling validation")
        return []

    enable_envs = controls.get("diagnostic_enable_envs", [])
    if not isinstance(enable_envs, list) or not all(
            isinstance(name, str) for name in enable_envs):
        errors.append(
            "profiling_controls.diagnostic_enable_envs: expected string list")
        return []

    enabled = [name for name in enable_envs if is_truthy(envs.get(name))]
    if enabled and not allow_profiling:
        errors.append(
            "profiling is default-off; enabled diagnostic controls "
            f"{enabled!r} require an intentional --allow-profiling check")
        return []

    notes: list[str] = []
    if enabled:
        notes.append(f"intentional profiling enabled by {enabled!r}")

    cuda_controls = controls.get("cuda_event_profiler", {})
    if not isinstance(cuda_controls, dict):
        errors.append(
            "profiling_controls.cuda_event_profiler: expected mapping")
        return notes

    enable_env = cuda_controls.get("enable_env")
    if not isinstance(enable_env, str):
        errors.append(
            "profiling_controls.cuda_event_profiler.enable_env: missing")
        return notes
    if not is_truthy(envs.get(enable_env)):
        runtime_envs = cuda_controls.get("runtime_envs", {})
        if isinstance(runtime_envs, dict):
            inert_companions = sorted(
                name for name in runtime_envs
                if name != enable_env and name in envs)
            if inert_companions:
                notes.append(
                    "CUDA profiler remains disabled; companion controls are "
                    f"inert: {inert_companions!r}")
        return notes

    required = cuda_controls.get("required_when_enabled", [])
    if not isinstance(required, list):
        errors.append(
            "profiling_controls.cuda_event_profiler.required_when_enabled: "
            "expected list")
        return notes
    for name in required:
        value = envs.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(
                f"envs.{name}: required and non-empty when {enable_env} is enabled"
            )

    positive_integer_envs = cuda_controls.get("positive_integer_envs", [])
    if not isinstance(positive_integer_envs, list):
        errors.append(
            "profiling_controls.cuda_event_profiler.positive_integer_envs: "
            "expected list")
        return notes
    for name in positive_integer_envs:
        if name not in envs:
            continue
        try:
            value = int(envs[name])
        except (TypeError, ValueError):
            errors.append(f"envs.{name}: expected a positive integer")
            continue
        if value < 1:
            errors.append(f"envs.{name}: expected a positive integer")

    kinds_env = "VLLM_DYNAMIC_PP_CUDA_PROFILE_KINDS"
    if kinds_env in envs:
        raw_kinds = envs[kinds_env]
        if isinstance(raw_kinds, str):
            kinds = {
                kind.strip()
                for kind in raw_kinds.split(",")
                if kind.strip()
            }
        elif isinstance(raw_kinds, list):
            kinds = {str(kind).strip() for kind in raw_kinds if str(kind).strip()}
        else:
            kinds = set()
        allowed_kinds = set(cuda_controls.get("allowed_kinds", []))
        if not kinds:
            errors.append(f"envs.{kinds_env}: expected at least one kind")
        unknown = kinds - allowed_kinds
        if unknown:
            errors.append(
                f"envs.{kinds_env}: unknown kinds {sorted(unknown)!r}; "
                f"allowed {sorted(allowed_kinds)!r}")

    return notes


def verify(
    config: dict[str, Any],
    baseline: dict[str, Any],
    *,
    profiling_controls: dict[str, Any] | None = None,
    allow_profiling: bool = False,
) -> tuple[list[str], list[str]]:
    required = baseline.get("required", {})
    if not isinstance(required, dict):
        return ["baseline.required: missing or not a mapping"], []

    errors: list[str] = []
    default_notes = compare_defaults(config, baseline, errors)
    if profiling_controls is not None:
        default_notes.extend(
            validate_profiling_controls(
                config,
                profiling_controls,
                allow_profiling=allow_profiling,
                errors=errors,
            ))
    compare_required(config, required.get("top_level", {}), "<root>", errors)
    compare_required(config.get("envs", {}), required.get("envs", {}), "envs", errors)
    compare_required(
        config.get("static_config", {}),
        required.get("static_config", {}),
        "static_config",
        errors,
    )

    sweep_required = required.get("sweep_configs", {})
    if not isinstance(sweep_required, dict):
        errors.append("baseline.required.sweep_configs: not a mapping")
        return errors, default_notes

    required_vllm = sweep_required.get("vllm", {})
    sweep_configs = config.get("sweep_configs")
    if not isinstance(sweep_configs, list) or not sweep_configs:
        errors.append("sweep_configs: missing or empty list")
        return errors, default_notes

    for index, sweep_config in enumerate(sweep_configs):
        if not isinstance(sweep_config, dict):
            errors.append(f"sweep_configs[{index}]: expected mapping")
            continue
        compare_required(
            sweep_config.get("vllm", {}),
            required_vllm,
            f"sweep_configs[{index}].vllm",
            errors,
        )

    return errors, default_notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a vLLM experiment YAML against guard baseline values."
    )
    parser.add_argument("config", type=Path, help="Experiment YAML config to check")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline YAML path (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--profiling-controls",
        type=Path,
        default=DEFAULT_PROFILING_CONTROLS,
        help=(
            "Profiling control policy applied to every baseline "
            f"(default: {DEFAULT_PROFILING_CONTROLS})"
        ),
    )
    parser.add_argument(
        "--allow-profiling",
        action="store_true",
        help=(
            "Allow an intentional diagnostic run to enable profiling or "
            "high-volume traces; normal performance runs must omit this flag"
        ),
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        baseline = load_yaml(args.baseline)
        profiling_controls = load_yaml(args.profiling_controls)
    except Exception as exc:
        print(f"config baseline check failed to load YAML: {exc}", file=sys.stderr)
        return 2

    errors, default_notes = verify(
        config,
        baseline,
        profiling_controls=profiling_controls,
        allow_profiling=args.allow_profiling,
    )
    if errors:
        print("Config baseline check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Config baseline check passed: {args.config}")
    print(f"Baseline: {args.baseline}")
    print(f"Profiling controls: {args.profiling_controls}")
    for note in default_notes:
        print(f"Default-overridable value differs: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
