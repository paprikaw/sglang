"""Durable chronological BurstGPT length replay, advancing only on commit.

Only source identity, cursor, counters, and token-count metadata are persisted.
No original trace copy, synthetic tokens, prompts, or response text is written.
"""
from __future__ import annotations
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


class TraceStream:
    """Read one pending batch and retry it until the caller commits success.

    A seek cookie makes process restart cheap even near the end of the trace.
    Cursor movement, exclusions, and cycle completion become durable together
    only in commit(). peek_batch() is repeatable and returns an independent
    metadata copy, so caller mutations cannot change the pending cursor.
    """
    def __init__(self, dataset_path, cursor_path, batch_size=256,
                 max_total_tokens=32768, min_output_tokens=2,
                 expected_source_sha256=None, expected_sha256=None):
        for name, value in (("batch_size", batch_size),
                            ("min_output_tokens", min_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_total_tokens is not None and (
            isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int)
            or max_total_tokens <= 0
        ):
            raise ValueError("max_total_tokens must be a positive integer or None")
        if expected_source_sha256 and expected_sha256 and expected_source_sha256 != expected_sha256:
            raise ValueError("Conflicting expected source SHA256 values")
        self.source = Path(dataset_path).expanduser().resolve(strict=True)
        self.cursor_path = Path(cursor_path).expanduser().resolve()
        self.batch_size = batch_size
        self.filters = {"min_output_tokens": min_output_tokens,
                        "max_total_tokens": max_total_tokens}
        initial_stat = self.source.stat()
        digest = hashlib.sha256()
        with self.source.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.sha256 = digest.hexdigest()
        expected = expected_source_sha256 or expected_sha256
        if expected is not None and self.sha256 != expected:
            raise ValueError("BurstGPT source SHA256 differs from the expected source")
        self._source_stat = (initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns)
        self._verify_source()
        self._file = None
        self._reader = None
        self._pending = None
        self._pending_state = None
        with self.source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(iter(stream.readline, ""))
            try:
                self.columns = next(reader)
            except StopIteration:
                raise ValueError("BurstGPT trace has no CSV header") from None
            missing = {"Timestamp", "Model", "Request tokens", "Response tokens"}.difference(self.columns)
            if missing:
                raise ValueError(f"BurstGPT trace is missing required columns: {sorted(missing)}")
            self._header_offset = stream.tell()
        if self.cursor_path.exists():
            with self.cursor_path.open(encoding="utf-8") as stream:
                self.state = json.load(stream)
            self._validate_cursor()
        else:
            self.state = {
                "schema_version": 1, "dataset_path": str(self.source),
                "dataset_sha256": self.sha256, "filters": self.filters,
                "cycle": 0, "last_source_index": -1,
                "source_offset": self._header_offset, "last_source_timestamp": None,
                "committed_requests": 0, "committed_batches": 0,
                "committed_source_rows": 0, "completed_cycles": 0,
                "exclusion_counts": {},
            }
        self._reset_reader(self.state)

    def _verify_source(self):
        stat = self.source.stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != self._source_stat:
            raise RuntimeError("BurstGPT source changed after SHA256 validation")

    def _validate_cursor(self):
        if not isinstance(self.state, dict) or self.state.get("schema_version") != 1:
            raise ValueError("Unsupported or malformed trace cursor")
        if self.state.get("dataset_sha256") != self.sha256:
            raise ValueError("Trace cursor belongs to a different source SHA256")
        if self.state.get("filters") != self.filters:
            raise ValueError("Trace cursor filters differ; use a new cursor for changed filters")
        for key in ("cycle", "source_offset", "committed_requests", "committed_batches",
                    "committed_source_rows", "completed_cycles"):
            value = self.state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Malformed cursor field {key}")
        index = self.state.get("last_source_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < -1:
            raise ValueError("Malformed cursor last_source_index")
        timestamp = self.state.get("last_source_timestamp")
        if timestamp is not None and (not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp)):
            raise ValueError("Malformed cursor last_source_timestamp")
        exclusions = self.state.get("exclusion_counts")
        if not isinstance(exclusions, dict) or any(
            not isinstance(key, str) or isinstance(count, bool)
            or not isinstance(count, int) or count < 0 for key, count in exclusions.items()
        ):
            raise ValueError("Malformed cursor exclusion_counts")

    def _reset_reader(self, state):
        self.close()
        self._file = self.source.open("r", encoding="utf-8-sig", newline="")
        self._file.seek(state["source_offset"])
        self._reader = csv.DictReader(iter(self._file.readline, ""), fieldnames=self.columns)
        self._read_index = state["last_source_index"]

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def peek_batch(self):
        self._verify_source()
        if self._pending is not None:
            return copy.deepcopy(self._pending)
        if self._reader is None:
            self._reset_reader(self.state)
        next_state = copy.deepcopy(self.state)
        requests = []
        excluded = collections.Counter()
        rows_read = 0
        wraps = 0
        end_of_trace = False
        first_cycle = next_state["cycle"]
        try:
            while len(requests) < self.batch_size:
                try:
                    row = next(self._reader)
                except StopIteration:
                    next_state["cycle"] += 1
                    next_state["completed_cycles"] += 1
                    next_state["last_source_index"] = -1
                    next_state["source_offset"] = self._header_offset
                    next_state["last_source_timestamp"] = None
                    wraps += 1
                    end_of_trace = True
                    if requests:
                        break
                    if wraps > 1:
                        raise ValueError(f"No eligible rows in a complete BurstGPT trace; exclusions={dict(excluded)}")
                    self._reset_reader(next_state)
                    first_cycle = next_state["cycle"]
                    continue
                self._read_index += 1
                index = self._read_index
                rows_read += 1
                next_state["last_source_index"] = index
                next_state["source_offset"] = self._file.tell()
                try:
                    timestamp = float(row["Timestamp"])
                    prompt_tokens = int(row["Request tokens"])
                    output_tokens = int(row["Response tokens"])
                    if not math.isfinite(timestamp):
                        raise ValueError("non-finite timestamp")
                except (KeyError, TypeError, ValueError, OverflowError):
                    excluded["malformed_numeric"] += 1
                    continue
                previous = next_state["last_source_timestamp"]
                if previous is not None and timestamp < previous:
                    raise ValueError(f"BurstGPT timestamps moved backward at source_index={index}")
                next_state["last_source_timestamp"] = timestamp
                reason = None
                if output_tokens <= 0:
                    reason = "failed_or_nonpositive_output"
                elif prompt_tokens <= 0:
                    reason = "nonpositive_prompt"
                elif output_tokens < self.filters["min_output_tokens"]:
                    reason = "below_min_output_tokens"
                elif self.filters["max_total_tokens"] is not None and prompt_tokens + output_tokens > self.filters["max_total_tokens"]:
                    reason = "above_max_total_tokens"
                if reason:
                    excluded[reason] += 1
                    continue
                if not requests:
                    first_cycle = next_state["cycle"]
                requests.append({
                    "sample_id": f"source-{index}", "source_index": index,
                    "source_timestamp": timestamp, "source_model": row["Model"],
                    "source_cycle": next_state["cycle"],
                    "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                    "offset_s": timestamp - (requests[0]["source_timestamp"] if requests else timestamp),
                })
        except BaseException:
            self._reset_reader(self.state)
            raise
        next_state["committed_requests"] += len(requests)
        next_state["committed_batches"] += 1
        next_state["committed_source_rows"] += rows_read
        cumulative_exclusions = collections.Counter(next_state["exclusion_counts"])
        cumulative_exclusions.update(excluded)
        next_state["exclusion_counts"] = dict(cumulative_exclusions)
        self._pending_state = next_state
        self._pending = {
            "schema_version": 1, "kind": "burstgpt_full_trace_token_length_stream",
            "contains_original_prompt_text": False,
            "dataset": {"path": str(self.source), "sha256": self.sha256,
                        "columns": self.columns},
            "selection": {"method": "entire_original_trace_chronological_committed_cursor",
                          "model_filter": "ALL", "min_output_tokens": self.filters["min_output_tokens"],
                          "max_total_tokens": self.filters["max_total_tokens"],
                          "max_prompt_tokens": None, "max_output_tokens": None,
                          "length_policy": "filter_without_clipping",
                          "timestamp_policy": "original chronological gaps and ties; subtract first batch timestamp"},
            "summary": {"requests": len(requests),
                        "models": dict(collections.Counter(row["source_model"] for row in requests)),
                        "total_prompt_tokens": sum(row["prompt_tokens"] for row in requests),
                        "total_output_tokens": sum(row["output_tokens"] for row in requests),
                        "trace_span_s": requests[-1]["offset_s"]},
            "stream": {"cycle_start": first_cycle,
                       "cycle_after_commit": next_state["cycle"],
                       "end_of_trace": end_of_trace,
                       "wraps_observed": wraps,
                       "source_rows_read": rows_read,
                       "excluded_rows": dict(excluded),
                       "last_source_index_after_commit": next_state["last_source_index"],
                       "committed_requests_after_commit": next_state["committed_requests"]},
            "requests": requests,
        }
        return copy.deepcopy(self._pending)

    def commit(self):
        """Durably advance the pending batch after the caller confirms success."""
        if self._pending is None:
            raise RuntimeError("There is no pending trace batch to commit")
        self._verify_source()
        self.cursor_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.cursor_path.name}.", dir=self.cursor_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._pending_state, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.cursor_path)
            dir_fd = os.open(self.cursor_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.state = self._pending_state
        self._pending = None
        self._pending_state = None
        # EOF resets are intentionally deferred until after the cursor is durable.
        if self.state["last_source_index"] == -1:
            self._reset_reader(self.state)
        return copy.deepcopy(self.state)
