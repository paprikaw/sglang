"""Load-driven policy primitives for elastic pipeline parallelism.

The policy intentionally stays independent of HTTP, scheduler, and CUDA state.
The tokenizer-side controller supplies one consistent scheduler observation and
uses the returned partition to invoke the existing transactional reconfiguration
path.  Keeping policy and mechanism separate makes the decision logic cheap to
unit test and prevents an automatic transition from bypassing the generation
fence used by manual transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

LayerPartition = Tuple[int, ...]


def parse_elastic_pp_partitions(raw: str) -> Tuple[LayerPartition, ...]:
    """Parse ``"40,0,0,40;20,20,20,20"`` into ordered partitions."""

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("elastic PP autoscale partitions must not be empty")

    partitions = []
    for partition_text in raw.split(";"):
        partition_text = partition_text.strip()
        if not partition_text:
            raise ValueError(
                "elastic PP autoscale partitions must not contain empty entries"
            )
        try:
            partition = tuple(int(item.strip()) for item in partition_text.split(","))
        except ValueError as exc:
            raise ValueError(
                "elastic PP autoscale partitions must contain integers: "
                f"{partition_text!r}"
            ) from exc
        if not partition or any(count < 0 for count in partition):
            raise ValueError(
                "elastic PP autoscale layer counts must be non-negative: "
                f"{partition}"
            )
        partitions.append(partition)

    if len(partitions) < 2:
        raise ValueError(
            "elastic PP autoscaling requires at least two ordered partitions"
        )
    if len(set(partitions)) != len(partitions):
        raise ValueError("elastic PP autoscale partitions must be unique")
    return tuple(partitions)


def serialize_elastic_pp_partitions(
    partitions: Sequence[Sequence[int]],
) -> str:
    return ";".join(
        ",".join(str(int(count)) for count in partition) for partition in partitions
    )


def validate_elastic_pp_autoscale_partitions(
    partitions: Sequence[Sequence[int]],
    *,
    pp_size: int,
    num_layers: int,
) -> Tuple[LayerPartition, ...]:
    """Validate an increasing logical-PP resource frontier.

    Rank 0 remains active because it owns the input embedding.  The logical
    last rank may move because elastic Llama materializes norm/lm-head endpoint
    state on every resident candidate rank.  This supports vLLM-compatible
    PP2 <-> PP4 placement in a resident four-rank group (for example
    ``40,40,0,0`` <-> ``20,20,20,20``).
    """

    normalized = tuple(tuple(int(count) for count in part) for part in partitions)
    for partition in normalized:
        if len(partition) != pp_size:
            raise ValueError(
                "each elastic PP autoscale partition must contain one count per "
                f"resident rank: got={len(partition)}, expected={pp_size}"
            )
        if any(count < 0 for count in partition):
            raise ValueError(
                "elastic PP autoscale layer counts must be non-negative: "
                f"{partition}"
            )
        if sum(partition) != num_layers:
            raise ValueError(
                "each elastic PP autoscale partition must cover every model "
                f"layer: got={sum(partition)}, expected={num_layers}"
            )
        if pp_size > 1 and partition[0] == 0:
            raise ValueError(
                "elastic PP autoscale partitions require non-empty rank 0 "
                "because it owns the input embedding"
            )

    active_counts = tuple(
        sum(count > 0 for count in partition) for partition in normalized
    )
    if any(left >= right for left, right in zip(active_counts, active_counts[1:])):
        raise ValueError(
            "elastic PP autoscale partitions must be ordered by strictly "
            f"increasing active-rank count: {active_counts}"
        )
    return normalized


@dataclass
class ElasticPPBacklogPolicy:
    """Move one step on a calibrated PP frontier using sustained backlog.

    Scale-up and scale-down thresholds form hysteresis.  Stable checks suppress
    one-poll spikes, and the operation interval starts when a migration commits,
    rather than when the decision is made.  This mirrors the validated vLLM
    controller semantics while leaving all migration work to SGLang's existing
    scheduler-fenced coordinator.
    """

    partitions: Sequence[Sequence[int]]
    scale_up_total_requests: int
    scale_down_total_requests: int
    scale_up_stable_checks: int = 2
    scale_down_stable_checks: int = 4
    min_operation_interval_s: float = 10.0
    failure_backoff_s: float = 5.0
    max_operations: Optional[int] = None
    _up_checks: int = field(default=0, init=False)
    _down_checks: int = field(default=0, init=False)
    _last_operation_s: Optional[float] = field(default=None, init=False)
    _retry_not_before_s: float = field(default=0.0, init=False)
    _completed_operations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        partitions = tuple(
            tuple(int(count) for count in partition) for partition in self.partitions
        )
        if len(partitions) < 2:
            raise ValueError("partitions must contain at least two configurations")
        if len(set(partitions)) != len(partitions):
            raise ValueError("partitions must be unique")
        if self.scale_up_total_requests <= self.scale_down_total_requests:
            raise ValueError(
                "scale_up_total_requests must exceed scale_down_total_requests"
            )
        if self.scale_down_total_requests < 0:
            raise ValueError("request thresholds must be non-negative")
        if self.scale_up_stable_checks < 1 or self.scale_down_stable_checks < 1:
            raise ValueError("stable-check counts must be positive")
        if self.min_operation_interval_s < 0 or self.failure_backoff_s < 0:
            raise ValueError("operation intervals must be non-negative")
        if self.max_operations is not None and self.max_operations < 1:
            raise ValueError("max_operations must be positive or None")
        self.partitions = partitions

    def reset(self) -> None:
        self._up_checks = 0
        self._down_checks = 0
        self._last_operation_s = None
        self._retry_not_before_s = 0.0
        self._completed_operations = 0

    @property
    def completed_operations(self) -> int:
        return self._completed_operations

    def mark_operation_completed(
        self,
        now_s: float,
        *,
        count_toward_limit: bool = True,
    ) -> None:
        self._last_operation_s = float(now_s)
        self._retry_not_before_s = 0.0
        self._up_checks = 0
        self._down_checks = 0
        if count_toward_limit:
            self._completed_operations += 1

    def mark_operation_failed(self, now_s: float) -> None:
        self._retry_not_before_s = float(now_s) + self.failure_backoff_s
        self._up_checks = 0
        self._down_checks = 0

    def observe(
        self,
        *,
        current_partition: Sequence[int],
        waiting_requests: int,
        running_requests: int,
        now_s: float,
    ) -> Optional[LayerPartition]:
        current = tuple(int(count) for count in current_partition)
        try:
            position = self.partitions.index(current)
        except ValueError as exc:
            raise ValueError(
                f"current partition {current} is not on the autoscale frontier "
                f"{self.partitions}"
            ) from exc

        waiting = max(0, int(waiting_requests))
        running = max(0, int(running_requests))
        total = waiting + running
        wants_up = (
            position < len(self.partitions) - 1
            and total >= self.scale_up_total_requests
        )
        wants_down = position > 0 and total <= self.scale_down_total_requests
        self._up_checks = self._up_checks + 1 if wants_up else 0
        self._down_checks = self._down_checks + 1 if wants_down else 0

        can_operate = (
            now_s >= self._retry_not_before_s
            and (
                self.max_operations is None
                or self._completed_operations < self.max_operations
            )
            and (
                self._last_operation_s is None
                or now_s - self._last_operation_s >= self.min_operation_interval_s
            )
        )
        if not can_operate:
            return None

        if self._up_checks >= self.scale_up_stable_checks:
            target = self.partitions[position + 1]
        elif self._down_checks >= self.scale_down_stable_checks:
            target = self.partitions[position - 1]
        else:
            return None

        self._up_checks = 0
        self._down_checks = 0
        return target
