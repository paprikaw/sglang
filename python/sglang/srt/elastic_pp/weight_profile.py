"""Exact first-load weight accounting for elastic pipeline parallelism.

The initial checkpoint iterator contains weights for every Transformer layer,
including layers filtered out by the startup PP placement.  Elastic PP records
that iterator once, then calibrates the layers that are resident after loader
post-processing against the model's actual parameters.  This provides a stable
per-layer capacity model for every future topology without re-reading weights
or assuming that all layers have the same byte size.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import torch
from torch import nn

from sglang.srt.layers.utils import get_layer_id

logger = logging.getLogger(__name__)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass(frozen=True)
class WeightTensorMetadata:
    """Immutable, value-free metadata for one checkpoint or resident tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    layer_id: int | None
    kind: str

    @classmethod
    def from_tensor(
        cls, name: str, tensor: torch.Tensor, *, kind: str
    ) -> WeightTensorMetadata:
        return cls(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=_tensor_nbytes(tensor),
            layer_id=get_layer_id(name),
            kind=kind,
        )


def _sum_bytes_by_layer(
    weight_bytes_by_name: dict[str, int],
) -> tuple[dict[int, int], int]:
    layer_sizes: dict[int, int] = {}
    non_layer_size = 0
    for name, weight_bytes in weight_bytes_by_name.items():
        layer_id = get_layer_id(name)
        if layer_id is None:
            non_layer_size += weight_bytes
        else:
            layer_sizes[layer_id] = layer_sizes.get(layer_id, 0) + weight_bytes
    return layer_sizes, non_layer_size


@dataclass(frozen=True)
class ElasticPPWeightProfile:
    """Complete first-load weight accounting for one model worker.

    ``layer_weight_sizes`` has one entry for every Transformer layer.  Layers
    initially resident on this rank use their physical post-processing size;
    all other layers retain their checkpoint size.  Non-layer and total sizes
    describe tensors physically resident on this worker.
    """

    layer_weight_sizes: tuple[int, ...]
    non_layer_weight_size: int
    total_weight_size: int
    checkpoint_total_weight_size: int
    checkpoint_non_layer_weight_size: int
    checkpoint_tensor_count: int
    resident_tensor_count: int

    def transformer_weight_size(self, start_layer: int, end_layer: int) -> int:
        """Return exact Transformer bytes for half-open ``[start, end)``."""

        if start_layer < 0 or end_layer < start_layer:
            raise ValueError(
                "layer range must be non-negative and half-open: "
                f"[{start_layer}, {end_layer})"
            )
        if end_layer > len(self.layer_weight_sizes):
            raise ValueError(
                "layer range exceeds first-load weight profile: "
                f"end={end_layer}, layers={len(self.layer_weight_sizes)}"
            )
        return sum(self.layer_weight_sizes[start_layer:end_layer])

    def target_weight_size(self, start_layer: int, end_layer: int) -> int:
        """Return exact target resident bytes, including rank-local endpoints."""

        return self.non_layer_weight_size + self.transformer_weight_size(
            start_layer, end_layer
        )


class FirstLoadWeightProfiler:
    """Record exactly one complete checkpoint stream and calibrate residency."""

    def __init__(self, num_hidden_layers: int) -> None:
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        self.num_hidden_layers = int(num_hidden_layers)
        self._checkpoint_weight_bytes_by_name: dict[str, int] = {}
        self._recording_started = False
        self._recording_complete = False
        self._profile: ElasticPPWeightProfile | None = None

    @property
    def recording_complete(self) -> bool:
        return self._recording_complete

    def record_initial_stream(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Record every tensor before PP filtering on the first load only."""

        if self._recording_started:
            yield from weights
            return

        self._recording_started = True
        for name, weight in weights:
            weight_bytes = _tensor_nbytes(weight)
            previous = self._checkpoint_weight_bytes_by_name.setdefault(
                name, weight_bytes
            )
            if previous != weight_bytes:
                raise ValueError(
                    "weight size changed in the initial checkpoint stream: "
                    f"{name} was {previous} bytes, now {weight_bytes} bytes"
                )
            yield name, weight
        self._recording_complete = True

    @staticmethod
    def _resident_parameter_bytes(module: nn.Module) -> dict[str, int]:
        named_parameters = list(module.named_parameters())
        accelerator_types = {
            parameter.device.type
            for _, parameter in named_parameters
            if parameter.device.type not in {"cpu", "meta"}
        }
        return {
            name: _tensor_nbytes(parameter)
            for name, parameter in named_parameters
            if parameter.device.type != "meta"
            and (not accelerator_types or parameter.device.type in accelerator_types)
        }

    def finalize(self, module: nn.Module) -> ElasticPPWeightProfile:
        """Calibrate checkpoint sizes against post-processed resident tensors."""

        if self._profile is not None:
            return self._profile
        if not self._recording_started or not self._recording_complete:
            raise ValueError(
                "initial checkpoint weight stream was not completely consumed"
            )

        checkpoint_layer_sizes, checkpoint_non_layer_size = _sum_bytes_by_layer(
            self._checkpoint_weight_bytes_by_name
        )
        expected_layers = set(range(self.num_hidden_layers))
        missing_layers = expected_layers - checkpoint_layer_sizes.keys()
        unexpected_layers = checkpoint_layer_sizes.keys() - expected_layers
        if missing_layers or unexpected_layers:
            raise ValueError(
                "initial weight stream does not match the model layer set: "
                f"missing={sorted(missing_layers)}, "
                f"unexpected={sorted(unexpected_layers)}"
            )

        resident_weight_bytes_by_name = self._resident_parameter_bytes(module)
        resident_layer_sizes, resident_non_layer_size = _sum_bytes_by_layer(
            resident_weight_bytes_by_name
        )

        # Packed or quantized resident tensors are authoritative for local
        # layers.  Checkpoint sizes remain authoritative for nonresident layers.
        effective_layer_sizes = dict(checkpoint_layer_sizes)
        effective_layer_sizes.update(resident_layer_sizes)
        ordered_layer_sizes = tuple(
            effective_layer_sizes[layer_id]
            for layer_id in range(self.num_hidden_layers)
        )
        profile = ElasticPPWeightProfile(
            layer_weight_sizes=ordered_layer_sizes,
            non_layer_weight_size=resident_non_layer_size,
            total_weight_size=sum(resident_weight_bytes_by_name.values()),
            checkpoint_total_weight_size=sum(
                self._checkpoint_weight_bytes_by_name.values()
            ),
            checkpoint_non_layer_weight_size=checkpoint_non_layer_size,
            checkpoint_tensor_count=len(self._checkpoint_weight_bytes_by_name),
            resident_tensor_count=len(resident_weight_bytes_by_name),
        )
        self._profile = profile
        logger.info(
            "Elastic PP first-load weight profile: checkpoint_tensors=%d "
            "checkpoint_total=%.3f GiB checkpoint_non_layer=%.3f GiB "
            "resident_tensors=%d resident_total=%.3f GiB "
            "resident_non_layer=%.3f GiB layer_total=%.3f GiB "
            "layer_min=%.3f GiB layer_max=%.3f GiB",
            profile.checkpoint_tensor_count,
            profile.checkpoint_total_weight_size / 1024**3,
            profile.checkpoint_non_layer_weight_size / 1024**3,
            profile.resident_tensor_count,
            profile.total_weight_size / 1024**3,
            profile.non_layer_weight_size / 1024**3,
            sum(profile.layer_weight_sizes) / 1024**3,
            min(profile.layer_weight_sizes) / 1024**3,
            max(profile.layer_weight_sizes) / 1024**3,
        )
        logger.debug(
            "Elastic PP checkpoint tensor bytes: %s",
            self._checkpoint_weight_bytes_by_name,
        )
        logger.debug(
            "Elastic PP resident tensor bytes: %s", resident_weight_bytes_by_name
        )
        return profile

    def get_profile(self) -> ElasticPPWeightProfile:
        if self._profile is None:
            raise ValueError("elastic PP weight profile has not been finalized")
        return self._profile
