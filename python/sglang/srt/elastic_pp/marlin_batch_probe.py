"""Opt-in startup diagnostic: one MoE anchor with different companion routes.

No attention, cache or migration participates. Keep the anchor's input,
weights, routes and gate weights fixed, and test only batch composition.
Only scalar error/count metrics are retained; never save inputs or weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_finished = False


def probe(kwargs, forward):
    hidden = kwargs["hidden_states"]
    experts = kwargs["w1"].shape[0]
    if kwargs.get("expert_map") is not None or experts < 2:
        raise ValueError("batch probe requires the unsharded Marlin expert set")
    single = dict(kwargs, inplace=False)
    for name in ("hidden_states", "gating_output", "topk_weights", "topk_ids"):
        single[name] = kwargs[name][:1].detach().clone().contiguous()
    if kwargs.get("workspace") is not None:
        single["workspace"] = torch.zeros_like(kwargs["workspace"])
    reference = forward(**single).clone()
    repeat = forward(**single).clone()
    result = dict(
        experts=experts,
        hidden_size=hidden.shape[1],
        intermediate_size=kwargs["w2"].shape[1] * 16,
        top_k=single["topk_ids"].shape[1],
        activation_dtype=str(hidden.dtype),
        weight_dtype=str(kwargs["w1"].dtype),
        deterministic=kwargs.get("deterministic", False),
        input_rms=(hidden[:1].float().square().sum() / hidden[:1].numel())
        .sqrt()
        .item(),
        reference_rms=(reference.float().square().sum() / reference.numel())
        .sqrt()
        .item(),
        reference_nonzero_values=reference.count_nonzero().item(),
        canonical_k=os.environ.get("SGLANG_MARLIN_MOE_CANONICAL_K") == "1",
        single_repeat_equal=torch.equal(reference, repeat),
        cases={},
    )
    for name in ("same_routes", "shifted_routes"):
        batch = dict(single)
        for key in ("hidden_states", "gating_output", "topk_weights", "topk_ids"):
            tensor = single[key]
            batch[key] = tensor.repeat(2, *([1] * (tensor.ndim - 1)))
        if name == "shifted_routes":
            batch["topk_ids"][1] = (batch["topk_ids"][1] + experts // 2) % experts
        output = forward(**batch).clone()
        again = forward(**batch).clone()
        delta = output[:1].float() - reference.float()
        result["cases"][name] = dict(
            anchor_equal=torch.equal(output[:1], reference),
            batch_repeat_equal=torch.equal(output, again),
            differing_values=(output[:1] != reference).sum().item(),
            max_abs_delta=delta.abs().max().item(),
            relative_l2=(
                delta.norm() / reference.float().norm().clamp_min(1e-20)
            ).item(),
            active_experts=batch["topk_ids"].unique().numel(),
        )
    return result


def maybe_probe(kwargs, forward, directory):
    global _finished
    if _finished or torch.cuda.is_current_stream_capturing():
        return
    _finished = True
    result = probe(kwargs, forward)
    from sglang.srt.elastic_pp.batch_numerics_probe import synthetic_rows

    result["synthetic_nonzero"] = {}
    for amplitude in (0.25, 1.0, 4.0):
        synthetic = dict(kwargs)
        synthetic["hidden_states"] = synthetic_rows(
            (1, kwargs["hidden_states"].shape[1]),
            device=kwargs["hidden_states"].device,
            dtype=kwargs["hidden_states"].dtype,
            amplitude=amplitude,
        )
        result["synthetic_nonzero"][str(amplitude)] = probe(synthetic, forward)
    result.update(
        scope="startup MoE operator only; no cache, migration or performance claim",
        device=torch.cuda.current_device(),
        probe_completed=True,
    )
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"marlin_batch_gpu{result['device']}.json"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
