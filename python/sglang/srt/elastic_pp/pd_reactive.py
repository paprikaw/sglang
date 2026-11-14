"""A mechanism-independent reactive P/D controller; no future trace inputs.

Observations are per-instance (not summed over pipeline ranks). The caller
must classify compute, capacity and transfer queues before constructing them.
Unknown target-layout feasibility deliberately prevents scale-down.
"""
from dataclasses import dataclass
import math
from typing import Literal

Role = Literal["prefill", "decode"]


@dataclass(frozen=True)
class Observation:
    role: Role
    observed_at: float
    expanded: bool = False
    compute_wait_s: float = 0.0
    compute_queue: int = 0
    capacity_wait_s: float = 0.0
    capacity_blocked: bool = False
    blocked_by_decode: bool = False
    overloaded: bool = False
    next_layout_improves: bool = False
    target_fits_state: bool | None = None
    target_fits_load: bool | None = None
    in_flight_handoff: bool = False
    metrics_valid: bool = True


@dataclass(frozen=True)
class Decision:
    role: Role
    direction: Literal["up", "down"]
    gpu_delta: int
    reason: str


@dataclass(frozen=True)
class Parameters:
    # Initial policy knobs, not model-independent performance thresholds.
    high_wait_s: float = 2.0
    up_hold_s: float = 2.0
    down_hold_s: float = 30.0
    residence_s: float = 15.0
    failure_backoff_s: float = 10.0
    max_sample_age_s: float = 3.0
    max_sample_gap_s: float = 3.0
    max_gpus: int = 8

    def __post_init__(self):
        values = (self.high_wait_s, self.up_hold_s, self.down_hold_s,
                  self.residence_s, self.failure_backoff_s,
                  self.max_sample_age_s, self.max_sample_gap_s)
        if any(not math.isfinite(x) or x < 0 for x in values) or self.high_wait_s == 0:
            raise ValueError("Policy times must be finite and nonnegative; high wait > 0")
        if self.max_gpus < 1:
            raise ValueError("Positive GPU budget required")


class ReactivePDPolicy:
    def __init__(self, mechanism: Literal["inplace", "replica"],
                 parameters=Parameters(), stage_width=1, replica_gpus=2):
        if mechanism not in ("inplace", "replica"):
            raise ValueError("Unknown mechanism")
        if stage_width < 1 or replica_gpus < 1:
            raise ValueError("Positive action sizes required")
        self.mechanism, self.p = mechanism, parameters
        self.step = stage_width if mechanism == "inplace" else replica_gpus
        self.pending: Decision | None = None
        self._since = {}
        self._last_sample = {}
        self._last_commit = {}
        self._retry_at = 0.0

    def observe(self, samples, *, now: float, occupied_gpus: int,
                free_by_role: dict[str, int]) -> Decision | None:
        if not math.isfinite(now) or occupied_gpus < 0:
            raise ValueError("Invalid observation time/GPU count")
        roles = [s.role for s in samples]
        if len(set(roles)) != len(roles) or any(r not in ("prefill", "decode") for r in roles):
            raise ValueError("Exactly one observation per distinct P/D role")
        candidates = []
        blocked_by_d = any(s.role == "prefill" and s.blocked_by_decode for s in samples)
        for s in samples:
            old = self._last_sample.get(s.role)
            fresh = (s.metrics_valid and math.isfinite(s.observed_at)
                     and 0 <= now - s.observed_at <= self.p.max_sample_age_s)
            if not fresh:
                self._since.pop(s.role, None)
                continue
            if old is not None and s.observed_at <= old:
                continue  # Repeated scrapes cannot advance persistence.
            if old is not None and s.observed_at - old > self.p.max_sample_gap_s:
                self._since.pop(s.role, None)
            self._last_sample[s.role] = s.observed_at
            values = (s.compute_wait_s, s.capacity_wait_s)
            if any(not math.isfinite(v) or v < 0 for v in values):
                self._since.pop(s.role, None)
                continue
            high = False
            wait = s.compute_wait_s
            reason = "compute_backlog"
            if s.role == "prefill":
                high = (not s.blocked_by_decode and s.compute_queue > 0
                        and wait >= self.p.high_wait_s)
            else:
                wait = s.capacity_wait_s
                high = s.capacity_blocked and wait >= self.p.high_wait_s
                reason = "decode_capacity"
                if not high and s.overloaded and s.next_layout_improves:
                    high, wait, reason = True, self.p.high_wait_s, "decode_service_load"
            direction = None
            if high and not s.expanded:
                direction = "up"
            elif (not high and s.expanded and s.compute_queue == 0
                  and not s.capacity_blocked and not s.blocked_by_decode
                  and not s.overloaded and not s.in_flight_handoff
                  and s.target_fits_state is True and s.target_fits_load is True):
                direction, reason = "down", "smaller_layout_safe"
            if direction is None:
                self._since.pop(s.role, None)
                continue
            previous = self._since.get(s.role)
            if previous is None or previous[0] != direction:
                previous = (direction, s.observed_at)
                self._since[s.role] = previous
            hold = self.p.up_hold_s if direction == "up" else self.p.down_hold_s
            if s.observed_at - previous[1] < hold:
                continue
            if now - self._last_commit.get(s.role, -math.inf) < self.p.residence_s:
                continue
            delta = self.step if direction == "up" else -self.step
            if direction == "up" and (occupied_gpus + delta > self.p.max_gpus
                                        or free_by_role.get(s.role, 0) < delta):
                continue
            priority = (direction == "up", s.role == "decode" and blocked_by_d,
                        wait / self.p.high_wait_s, s.role == "decode")
            candidates.append((priority, Decision(s.role, direction, delta, reason)))
        if self.pending is not None or now < self._retry_at or not candidates:
            return None
        self.pending = max(candidates, key=lambda x: x[0])[1]
        return self.pending

    def complete(self, *, now: float, success: bool):
        if self.pending is None:
            raise RuntimeError("No owned pending operation")
        if success:
            self._last_commit[self.pending.role] = now
        else:
            self._retry_at = now + self.p.failure_backoff_s
        self._since.clear()
        self.pending = None
