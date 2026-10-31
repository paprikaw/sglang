"""Prepare reproducible length-only BurstGPT replay cohorts without a model.

Official traces contain token counts and timestamps, not original prompt text.
Length limits filter records; no record is silently clipped.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

REQUIRED_COLUMNS = ("Timestamp", "Model", "Request tokens", "Response tokens")
DEFAULT_SEED = 20261019


def _distribution(values):
    ordered = sorted(values)
    def quantile(fraction):
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "p50": quantile(.5), "p95": quantile(.95)}


def _positive_limit(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_private(path, payload):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def prepare_workload(
    dataset_path: str | os.PathLike[str], out_path: str | os.PathLike[str] | None,
    *, max_requests: int = 64, max_prompt_tokens: int | None = 8192,
    max_output_tokens: int | None = 256, max_total_tokens: int | None = 32768,
    model_filter: str = "GPT-4", seed: int = DEFAULT_SEED,
    min_prompt_tokens: int = 1, min_output_tokens: int = 2,
) -> dict[str, Any]:
    """Select a uniform contiguous window from chronological eligible rows.

    Sliding-window reservoir sampling gives every possible eligible window
    equal probability, with O(max_requests) memory. Original gaps/ties survive.
    out_path=None audits without writing. None removes an upper length bound.
    """
    for name, value in (("max_requests", max_requests),
                        ("min_prompt_tokens", min_prompt_tokens),
                        ("min_output_tokens", min_output_tokens)):
        _positive_limit(value, name)
        if value is None:
            raise ValueError(f"{name} cannot be None")
    for name, value in (("max_prompt_tokens", max_prompt_tokens),
                        ("max_output_tokens", max_output_tokens),
                        ("max_total_tokens", max_total_tokens)):
        _positive_limit(value, name)
    if max_prompt_tokens is not None and min_prompt_tokens > max_prompt_tokens:
        raise ValueError("min_prompt_tokens exceeds max_prompt_tokens")
    if max_output_tokens is not None and min_output_tokens > max_output_tokens:
        raise ValueError("min_output_tokens exceeds max_output_tokens")
    if not model_filter:
        raise ValueError("model_filter must name a model or ALL")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source = Path(dataset_path).expanduser().resolve(strict=True)
    before = source.stat()
    sha256 = _file_sha256(source)
    rng = random.Random(seed)
    window = collections.deque(maxlen=max_requests)
    chosen = None
    chosen_start = 0
    total_rows = eligible_rows = window_count = 0
    global_zero_response = global_negative_response = 0
    exclusions = collections.Counter()
    eligible_models = collections.Counter()
    previous_timestamp = None
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = set(REQUIRED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"BurstGPT CSV is missing required columns: {sorted(missing)}")
        for source_index, row in enumerate(reader):
            total_rows += 1
            try:
                timestamp = float(row["Timestamp"])
                prompt_tokens = int(row["Request tokens"])
                output_tokens = int(row["Response tokens"])
                if not math.isfinite(timestamp):
                    raise ValueError("non-finite timestamp")
            except (ValueError, TypeError, KeyError, OverflowError):
                exclusions["malformed_numeric"] += 1
                continue
            global_zero_response += int(output_tokens == 0)
            global_negative_response += int(output_tokens < 0)
            reason = None
            if output_tokens <= 0:
                reason = "failed_or_nonpositive_output"
            elif prompt_tokens <= 0:
                reason = "nonpositive_prompt"
            elif model_filter != "ALL" and row["Model"] != model_filter:
                reason = "model_filter"
            elif prompt_tokens < min_prompt_tokens:
                reason = "below_min_prompt_tokens"
            elif output_tokens < min_output_tokens:
                reason = "below_min_output_tokens"
            elif max_prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
                reason = "above_max_prompt_tokens"
            elif max_output_tokens is not None and output_tokens > max_output_tokens:
                reason = "above_max_output_tokens"
            elif max_total_tokens is not None and prompt_tokens + output_tokens > max_total_tokens:
                reason = "above_max_total_tokens"
            if reason is not None:
                exclusions[reason] += 1
                continue
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    f"Eligible timestamps are not chronological at source_index={source_index}; "
                    "sort the trace explicitly and preserve its provenance before replay")
            previous_timestamp = timestamp
            eligible_models[row["Model"]] += 1
            window.append({"source_index": source_index, "source_timestamp": timestamp,
                           "source_model": row["Model"], "prompt_tokens": prompt_tokens,
                           "output_tokens": output_tokens})
            eligible_rows += 1
            if len(window) == max_requests:
                window_count += 1
                if rng.randrange(window_count) == 0:
                    chosen = list(window)
                    chosen_start = eligible_rows - max_requests
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("BurstGPT source changed during preparation; no workload was saved")
    if not eligible_rows:
        raise ValueError("No eligible BurstGPT rows remain after the explicit filters")
    if chosen is None:
        chosen = list(window)
    first_timestamp = chosen[0]["source_timestamp"]
    requests = [{"sample_id": index, **row,
                 "offset_s": row["source_timestamp"] - first_timestamp}
                for index, row in enumerate(chosen)]
    prompt_lengths = [row["prompt_tokens"] for row in requests]
    output_lengths = [row["output_tokens"] for row in requests]
    offsets = [row["offset_s"] for row in requests]
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    manifest = {
        "schema_version": 1, "kind": "burstgpt_token_length_replay",
        "contains_original_prompt_text": False,
        "synthetic_prompt_policy": (
            "Generate ephemeral safe-vocabulary token IDs at the exact recorded input length; "
            "force recorded output length with ignore_eos; never persist token IDs or raw payloads"),
        "dataset": {
            "path": str(source), "sha256": sha256, "bytes": before.st_size,
            "columns": columns, "rows": total_rows, "eligible_rows": eligible_rows,
            "eligible_models": dict(eligible_models), "zero_response_rows": global_zero_response,
            "negative_response_rows": global_negative_response,
            "exclusion_counts": dict(sorted(exclusions.items())),
            "exclusion_accounting": "First matching exclusion only; sum(exclusions) + eligible_rows = rows"},
        "selection": {
            "method": "uniform_reservoir_of_contiguous_chronological_eligible_windows",
            "seed": seed, "requested_max_requests": max_requests,
            "candidate_windows": window_count, "selected_eligible_start": chosen_start,
            "model_filter": model_filter, "min_prompt_tokens": min_prompt_tokens,
            "min_output_tokens": min_output_tokens, "max_prompt_tokens": max_prompt_tokens,
            "max_output_tokens": max_output_tokens, "max_total_tokens": max_total_tokens,
            "length_policy": "filter_without_clipping",
            "timestamp_policy": "original chronological gaps and ties; subtract first timestamp only",
            "source_timestamp_start": first_timestamp,
            "source_timestamp_end": chosen[-1]["source_timestamp"],
            "source_index_start": chosen[0]["source_index"],
            "source_index_end": chosen[-1]["source_index"]},
        "summary": {
            "requests": len(requests),
            "models": dict(collections.Counter(row["source_model"] for row in requests)),
            "prompt_tokens": _distribution(prompt_lengths),
            "output_tokens": _distribution(output_lengths),
            "total_prompt_tokens": sum(prompt_lengths), "total_output_tokens": sum(output_lengths),
            "trace_span_s": offsets[-1], "interarrival_s": _distribution(gaps),
            "equal_timestamp_adjacent_pairs": sum(gap == 0 for gap in gaps)},
        "requests": requests}
    if out_path is not None:
        _save_private(Path(out_path).expanduser().resolve(), manifest)
    return manifest


def _optional_positive(value):
    if value.lower() in {"none", "unlimited"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or none")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=_optional_positive, default=8192)
    parser.add_argument("--max-output-tokens", type=_optional_positive, default=256)
    parser.add_argument("--max-total-tokens", type=_optional_positive, default=32768)
    parser.add_argument("--min-prompt-tokens", type=int, default=1)
    parser.add_argument("--min-output-tokens", type=int, default=2)
    parser.add_argument("--model-filter", default="GPT-4")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = prepare_workload(args.dataset, args.output, max_requests=args.max_requests,
                              max_prompt_tokens=args.max_prompt_tokens,
                              max_output_tokens=args.max_output_tokens,
                              max_total_tokens=args.max_total_tokens,
                              min_prompt_tokens=args.min_prompt_tokens,
                              min_output_tokens=args.min_output_tokens,
                              model_filter=args.model_filter, seed=args.seed)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
