"""Opt-in, drained-fence byte audit; never a migration performance sample.

Only row indices, tensor geometry and digests cross the CPU control group.
Tensor contents stay local. A mismatch after commit must fail closed rather
than resume a generation whose state could not be verified.
"""

from __future__ import annotations

import hashlib
import time

import torch


def _rows(values):
    rows = torch.as_tensor(values, dtype=torch.int64).detach().cpu()
    if rows.ndim != 1 or torch.any(rows < 0) or rows.unique().numel() != rows.numel():
        raise ValueError("state audit requires unique nonnegative row indices")
    return rows.sort().values


def _tensor_record(layer, kind, tensor, rows, chunk_rows=16):
    """Hash bytes, including NaN payloads and signed zeros, in bounded chunks."""
    if rows.numel() and rows[-1].item() >= tensor.shape[0]:
        raise ValueError("state audit row exceeds the committed tensor capacity")
    digest = hashlib.sha256()
    for chunk in rows.split(chunk_rows):
        selected = tensor.index_select(0, chunk.to(tensor.device)).detach().cpu()
        digest.update(selected.contiguous().view(torch.uint8).numpy().tobytes())
    shape = (rows.numel(), *tensor.shape[1:])
    return dict(
        layer=layer,
        kind=kind,
        shape=shape,
        dtype=str(tensor.dtype),
        digest=digest.hexdigest(),
        nbytes=(
            rows.numel() * tensor[0].numel() * tensor.element_size()
            if tensor.shape[0]
            else 0
        ),
    )


class FencedStateContentAudit:
    def __init__(self, pool, rank):
        self.pool, self.rank = pool, rank
        self.stats = dict(enabled=True, passed=False)

    def _tensors(self, interval):
        selector = getattr(self.pool, "elastic_pp_kv_layer_ids", None)
        kv_layers = (
            selector(interval) if selector else range(interval.start, interval.end)
        )
        for layer in kv_layers:
            yield layer, "k", self.pool.get_key_buffer(layer)
            yield layer, "v", self.pool.get_value_buffer(layer)
        recurrent = getattr(self.pool, "recurrent_layer_ids", None)
        if recurrent is not None:
            for layer in recurrent(interval):
                index = self.pool.req_to_token_pool.mamba_map[layer]
                state = self.pool.mamba_pool.mamba_cache.at_layer_idx(index)
                for i, tensor in enumerate(state.conv):
                    yield layer, f"conv_{i}", tensor
                yield layer, "temporal", state.temporal

    def capture(self, transition, kv_slots):
        started = time.monotonic()
        if getattr(self.pool, "_elastic_pp_cpu_spill", None) is not None:
            raise NotImplementedError(
                "state content audit requires live source backing"
            )
        kv_rows = _rows(kv_slots)
        recurrent_rows = torch.empty(0, dtype=torch.int64)
        if hasattr(self.pool, "recurrent_slot_count"):
            free = set(
                self.pool.req_to_token_pool.mamba_allocator.free_slots.cpu().tolist()
            )
            recurrent_rows = _rows(
                [i for i in range(self.pool.recurrent_slot_count) if i not in free]
            )
        records = [
            _tensor_record(
                layer, kind, tensor, kv_rows if kind in ("k", "v") else recurrent_rows
            )
            for layer, kind, tensor in self._tensors(
                transition.source.layer_range(self.rank)
            )
        ]
        self.stats = dict(
            enabled=True,
            passed=False,
            capture_seconds=time.monotonic() - started,
            source_tensors=len(records),
            source_bytes=sum(r["nbytes"] for r in records),
        )
        return dict(
            rank=self.rank,
            generation=transition.source.generation,
            kv_rows=kv_rows.tolist(),
            recurrent_rows=recurrent_rows.tolist(),
            records=records,
        )

    def verify(self, transition, snapshots):
        started = time.monotonic()
        source = {}
        if len(snapshots) != transition.source.max_pp_size:
            raise ValueError("state audit requires every source rank exactly once")
        ranks = set()
        for snapshot in snapshots:
            rank = snapshot["rank"]
            if rank in ranks or rank not in range(transition.source.max_pp_size):
                raise ValueError("state audit source rank set is invalid")
            ranks.add(rank)
            if snapshot["generation"] != transition.source.generation:
                raise ValueError("state audit source generation differs")
            interval = transition.source.layer_range(rank)
            for record in snapshot["records"]:
                key = record["layer"], record["kind"]
                if key in source or not interval.start <= key[0] < interval.end:
                    raise ValueError(
                        "state audit record has duplicate or wrong layer owner"
                    )
                rows = (
                    snapshot["kv_rows"]
                    if key[1] in ("k", "v")
                    else snapshot["recurrent_rows"]
                )
                source[key] = (record, _rows(rows))
        target = list(self._tensors(transition.target.layer_range(self.rank)))
        interval = transition.target.layer_range(self.rank)
        expected_keys = {
            key for key in source if interval.start <= key[0] < interval.end
        }
        if {(l, k) for l, k, _ in target} != expected_keys:
            raise RuntimeError("state audit target tensor coverage differs from source")
        nbytes = 0
        for layer, kind, tensor in target:
            expected, rows = source[layer, kind]
            actual = _tensor_record(layer, kind, tensor, rows)
            if actual != expected:
                self.stats.update(
                    passed=False,
                    failed_layer=layer,
                    failed_component=kind,
                    verify_seconds=time.monotonic() - started,
                )
                raise RuntimeError(
                    "elastic PP committed state content differs: "
                    f"generation={transition.target.generation} "
                    f"layer={layer} component={kind}"
                )
            nbytes += actual["nbytes"]
        self.stats.update(
            passed=True,
            generation=transition.target.generation,
            verified_tensors=len(target),
            verified_bytes=nbytes,
            verified_kv_tensors=sum(k in ("k", "v") for _, k, _ in target),
            verified_recurrent_tensors=sum(k not in ("k", "v") for _, k, _ in target),
            verify_seconds=time.monotonic() - started,
        )
