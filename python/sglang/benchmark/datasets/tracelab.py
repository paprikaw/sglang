# SPDX-License-Identifier: Apache-2.0
"""TraceLab shapes and deterministic session-local prompts.

Adapted from the validated vLLM TraceLab replay adapter; no vLLM dependency.
The source trace contains sanitized shapes, not semantic prompts or tools.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, TextIO


ToolDelayMode = Literal["wall-span", "max-wall", "max-effective", "none"]


@dataclass(frozen=True)
class AgenticTraceTurn:
    session_id: str
    provider: str
    round_index: int
    round_id: str
    input_tokens: int
    prefix_tokens: int
    append_tokens: int
    output_tokens: int
    tool_delay_s: float
    first_event_at: Optional[datetime] = None
    tools: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AgenticTraceSession:
    session_id: str
    provider: str
    turns: tuple[AgenticTraceTurn, ...]

    @property
    def first_event_at(self) -> Optional[datetime]:
        return next(
            (turn.first_event_at for turn in self.turns if turn.first_event_at),
            None,
        )


@dataclass(frozen=True)
class PromptBuild:
    prompt_token_ids: tuple[int, ...]
    requested_prefix_tokens: int
    replayable_prefix_tokens: int
    history_tokens_available: int


def _open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _nonnegative_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def compute_tool_delay_s(
    tools: Sequence[dict[str, Any]],
    mode: ToolDelayMode = "wall-span",
) -> float:
    """Compute the wait before the next LLM turn from one round's tools.

    ``wall-span`` preserves parallel tool execution by measuring from the first
    emission to the last result. It falls back to the longest observed tool
    latency when timestamps are unavailable.
    """
    if mode == "none" or not tools:
        return 0.0

    wall_latencies_ms = [
        value
        for tool in tools
        if (value := _nonnegative_number(tool.get("tool_wall_latency_ms"))) is not None
    ]
    effective_latencies_ms = []
    intervals: list[tuple[datetime, datetime]] = []
    for tool in tools:
        internal = _nonnegative_number(tool.get("tool_internal_latency_ms"))
        wall = _nonnegative_number(tool.get("tool_wall_latency_ms"))
        if internal is not None:
            effective_latencies_ms.append(internal)
        elif wall is not None:
            effective_latencies_ms.append(wall)

        emitted_at = _parse_timestamp(tool.get("emitted_at"))
        result_at = _parse_timestamp(tool.get("result_at"))
        if emitted_at is not None and result_at is not None and result_at >= emitted_at:
            intervals.append((emitted_at, result_at))

    if mode == "wall-span" and intervals:
        start = min(interval[0] for interval in intervals)
        end = max(interval[1] for interval in intervals)
        return (end - start).total_seconds()
    if mode in {"wall-span", "max-wall"}:
        return max(wall_latencies_ms, default=0.0) / 1000.0
    if mode == "max-effective":
        return max(effective_latencies_ms, default=0.0) / 1000.0
    raise ValueError(f"Unknown tool delay mode: {mode}")


def _first_event_at(row: dict[str, Any]) -> Optional[datetime]:
    for event in row.get("timing_events") or []:
        timestamp = _parse_timestamp(event.get("timestamp"))
        if timestamp is not None:
            return timestamp
    return None


def _row_to_turn(
    row: dict[str, Any],
    *,
    line_number: int,
    tool_delay_mode: ToolDelayMode,
) -> AgenticTraceTurn:
    required = (
        "session_id",
        "provider",
        "round_index",
        "input_tokens_total",
        "prefix_tokens",
        "newly_append_tokens",
        "output_tokens",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Trace line {line_number} is missing fields: {missing}")

    input_tokens = int(row["input_tokens_total"])
    prefix_tokens = int(row["prefix_tokens"])
    append_tokens = int(row["newly_append_tokens"])
    output_tokens = int(row["output_tokens"])
    if min(input_tokens, prefix_tokens, append_tokens, output_tokens) < 0:
        raise ValueError(f"Trace line {line_number} has a negative token count")
    if input_tokens != prefix_tokens + append_tokens:
        raise ValueError(
            f"Trace line {line_number} violates input=prefix+append: "
            f"{input_tokens}!={prefix_tokens}+{append_tokens}"
        )
    if input_tokens == 0 or output_tokens == 0:
        raise ValueError(f"Trace line {line_number} has a zero-length request")

    return AgenticTraceTurn(
        session_id=str(row["session_id"]),
        provider=str(row["provider"]),
        round_index=int(row["round_index"]),
        round_id=str(row.get("round_id") or f"round-{line_number}"),
        input_tokens=input_tokens,
        prefix_tokens=prefix_tokens,
        append_tokens=append_tokens,
        output_tokens=output_tokens,
        tool_delay_s=compute_tool_delay_s(row.get("tools") or [], mode=tool_delay_mode),
        first_event_at=_first_event_at(row),
        tools=tuple(row.get("tools") or []),
    )


def load_tracelab_sessions(
    dataset_path: str | Path,
    *,
    session_ids: Optional[Sequence[str]] = None,
    provider: Optional[str] = None,
    max_sessions: Optional[int] = None,
    min_rounds: int = 2,
    max_rounds_per_session: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    require_initial_prefix_zero: bool = False,
    tool_delay_mode: ToolDelayMode = "wall-span",
) -> list[AgenticTraceSession]:
    """Load a deterministic, session-complete subset of TraceLab JSONL."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if min_rounds < 1:
        raise ValueError("min_rounds must be positive")
    if max_sessions is not None and max_sessions < 1:
        raise ValueError("max_sessions must be positive")

    requested = list(dict.fromkeys(session_ids or []))
    requested_set = set(requested)
    grouped: dict[str, list[AgenticTraceTurn]] = defaultdict(list)
    with _open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on trace line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Trace line {line_number} is not a JSON object")
            if requested_set and str(row.get("session_id")) not in requested_set:
                continue
            if provider is not None and row.get("provider") != provider:
                continue
            turn = _row_to_turn(
                row,
                line_number=line_number,
                tool_delay_mode=tool_delay_mode,
            )
            grouped[turn.session_id].append(turn)

    sessions_by_id: dict[str, AgenticTraceSession] = {}
    rejection_reasons: dict[str, str] = {}
    for session_id, turns in grouped.items():
        turns.sort(key=lambda turn: turn.round_index)
        round_indices = [turn.round_index for turn in turns]
        if len(set(round_indices)) != len(round_indices):
            raise ValueError(f"Session {session_id} has duplicate round indices")
        selected_turns = (
            turns[:max_rounds_per_session]
            if max_rounds_per_session is not None
            else turns
        )
        reason = None
        if len(selected_turns) < min_rounds:
            reason = f"has only {len(selected_turns)} selected rounds"
        elif any(
            b.round_index != a.round_index + 1
            for a, b in zip(selected_turns, selected_turns[1:])
        ):
            reason = "selected rounds are not consecutive"
        elif max_input_tokens is not None and any(
            turn.input_tokens + turn.output_tokens > max_input_tokens
            for turn in selected_turns
        ):
            reason = f"exceeds max input+output tokens {max_input_tokens}"
        elif max_output_tokens is not None and any(
            turn.output_tokens > max_output_tokens for turn in selected_turns
        ):
            reason = f"exceeds max output tokens {max_output_tokens}"
        elif require_initial_prefix_zero and selected_turns[0].prefix_tokens != 0:
            reason = "does not start with a cold prefix"
        if reason:
            rejection_reasons[session_id] = reason
            continue
        sessions_by_id[session_id] = AgenticTraceSession(
            session_id=session_id,
            provider=selected_turns[0].provider,
            turns=tuple(selected_turns),
        )

    if requested:
        missing = [session_id for session_id in requested if session_id not in grouped]
        rejected = {
            session_id: rejection_reasons[session_id]
            for session_id in requested
            if session_id in rejection_reasons
        }
        if missing or rejected:
            raise ValueError(
                f"Requested sessions unavailable; missing={missing}, rejected={rejected}"
            )
        ordered = [sessions_by_id[session_id] for session_id in requested]
    else:
        ordered = sorted(
            sessions_by_id.values(),
            key=lambda session: (
                session.first_event_at or datetime.max.astimezone(),
                session.session_id,
            ),
        )

    if max_sessions is not None:
        ordered = ordered[:max_sessions]
    if not ordered:
        raise ValueError("No TraceLab sessions matched the selection filters")
    return ordered


class SyntheticPromptBuilder:
    """Build exact-length prompts while preserving each recorded prefix."""

    def __init__(
        self,
        session_id: str,
        *,
        token_id_base: int = 1000,
        token_id_span: int = 1000,
    ) -> None:
        if token_id_base < 0:
            raise ValueError("token_id_base must be nonnegative")
        if token_id_span < 2:
            raise ValueError("token_id_span must be at least 2")
        self.session_id = session_id
        self.token_id_base = token_id_base
        self.token_id_span = token_id_span
        self._history: list[int] = []

    @property
    def history(self) -> tuple[int, ...]:
        return tuple(self._history)

    def _tokens(self, segment: str, length: int) -> list[int]:
        seed_bytes = hashlib.sha256(
            f"{self.session_id}:{segment}".encode("utf-8")
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big")
        # Do not reduce a single seed modulo token_id_span: that creates only
        # token_id_span possible entire prefixes and false cross-session hits.
        rng = random.Random(seed)
        return [
            self.token_id_base + rng.randrange(self.token_id_span)
            for _ in range(length)
        ]

    def build(self, turn: AgenticTraceTurn) -> PromptBuild:
        requested_prefix = min(turn.prefix_tokens, turn.input_tokens)
        history_available = len(self._history)
        replayable_prefix = min(requested_prefix, history_available)

        prompt = list(self._history[:replayable_prefix])
        unavailable_prefix = requested_prefix - replayable_prefix
        if unavailable_prefix:
            prompt.extend(
                self._tokens(
                    f"round-{turn.round_index}-external-prefix",
                    unavailable_prefix,
                )
            )

        suffix_len = turn.input_tokens - requested_prefix
        suffix = self._tokens(f"round-{turn.round_index}-append", suffix_len)
        if suffix and requested_prefix < history_available:
            old_next_token = self._history[requested_prefix]
            if suffix[0] == old_next_token:
                offset = suffix[0] - self.token_id_base
                suffix[0] = self.token_id_base + ((offset + 1) % self.token_id_span)
        prompt.extend(suffix)

        if len(prompt) != turn.input_tokens:
            raise AssertionError(
                f"Built {len(prompt)} tokens, expected {turn.input_tokens}"
            )
        if prompt[:replayable_prefix] != self._history[:replayable_prefix]:
            raise AssertionError("Synthetic prompt failed to preserve its prefix")
        return PromptBuild(
            prompt_token_ids=tuple(prompt),
            requested_prefix_tokens=requested_prefix,
            replayable_prefix_tokens=replayable_prefix,
            history_tokens_available=history_available,
        )

    def observe_completion(
        self,
        prompt_token_ids: Iterable[int],
        output_token_ids: Iterable[int],
    ) -> None:
        self._history = [*prompt_token_ids, *output_token_ids]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
