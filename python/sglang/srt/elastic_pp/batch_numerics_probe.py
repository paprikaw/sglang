"""Opt-in, startup-only batch arithmetic probes with synthetic nonzero inputs.

Use the loaded dense/SSM parameters, but private inputs, workspaces and states.
Only scalar diagnostics are written. These checks do not qualify migration or
performance and are never installed by an ordinary serving launch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

_installed_directory = None


def synthetic_rows(shape, *, device, dtype, amplitude=0.25):
    count = 1
    for dim in shape[1:]:
        count *= dim
    columns = torch.arange(count, device=device, dtype=torch.float32)
    rows = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
    values = ((columns[None, :] + 13 * rows) % 97 - 48) * (amplitude / 48)
    return values.reshape(shape).to(dtype)


def _difference(reference, actual):
    delta = actual.float() - reference.float()
    finite = bool(torch.isfinite(delta).all())
    return dict(
        equal=torch.equal(reference, actual),
        differing_values=int((reference != actual).sum()),
        finite=finite,
        max_abs_delta=float(delta.abs().max()) if finite else None,
    )


def compare_batches(run):
    reference = run(1)
    repeated = run(1)
    result = dict(
        reference={
            name: dict(
                finite=bool(torch.isfinite(value).all()),
                nonzero_values=int(torch.count_nonzero(value)),
                rms=float(value.float().square().mean().sqrt()),
            )
            for name, value in reference.items()
        },
        single_repeat={
            name: _difference(value, repeated[name])
            for name, value in reference.items()
        },
        cases={},
    )
    for batch_size in (2, 4):
        actual = run(batch_size)
        repeat = run(batch_size)
        result["cases"][str(batch_size)] = {
            name: dict(
                **_difference(value, actual[name]),
                batch_repeat_equal=torch.equal(actual[name], repeat[name]),
            )
            for name, value in reference.items()
        }
    return result


def probe_dense(method, layer, bias, forward):
    inputs = synthetic_rows(
        (4, layer.input_size_per_partition),
        device=layer.weight.device,
        dtype=layer.weight_scale.dtype,
    )
    private = SimpleNamespace(
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=torch.zeros_like(layer.workspace),
        output_size_per_partition=layer.output_size_per_partition,
        input_size_per_partition=layer.input_size_per_partition,
    )

    def run(size):
        output = forward(method, private, inputs[:size].clone(), bias)
        return {"output": output[0].detach().clone()}

    result = compare_batches(run)
    result.update(
        input_size=layer.input_size_per_partition,
        output_size=layer.output_size_per_partition,
        input_rms=float(inputs[0].float().square().mean().sqrt()),
    )
    return result


def probe_gdn(mixed_qkv, a, b, kwargs, forward):
    if any(
        value is not None
        for name, value in kwargs.items()
        if name.startswith("replayssm_")
    ):
        return {"skipped": "ReplaySSM state is outside this probe"}
    inputs = {
        name: synthetic_rows(
            (4, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype
        )
        for name, tensor in {"mixed_qkv": mixed_qkv, "a": a, "b": b}.items()
    }
    original_state = kwargs["ssm_states"]
    initial = synthetic_rows(
        (4, *original_state.shape[1:]),
        device=original_state.device,
        dtype=original_state.dtype,
        amplitude=0.01,
    )

    def run(size):
        state = initial[:size].clone()
        call = dict(kwargs, ssm_states=state)
        call["cache_indices"] = torch.arange(
            size, device=state.device, dtype=kwargs["cache_indices"].dtype
        )
        output = forward(
            **{name: value[:size].clone() for name, value in inputs.items()}, **call
        )
        return {
            "output": output.reshape(size, -1)[0].detach().clone(),
            "state": state[0].detach().clone(),
        }

    result = compare_batches(run)
    result.update(qkv_width=mixed_qkv.shape[-1], state_shape=list(initial.shape[1:]))
    return result


def _save(directory, operator, geometry, result):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    device = torch.cuda.current_device()
    target = path / f"{operator}_gpu{device}_{geometry}.json"
    result.update(
        operator=operator,
        device=device,
        probe_completed=True,
        scope="synthetic nonzero operator inputs; no migration or performance qualification",
    )
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)


def install(directory):
    global _installed_directory
    if _installed_directory is not None:
        if _installed_directory != directory:
            raise RuntimeError("batch numerical probe directory changed")
        return
    from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    seen_dense, seen_gdn = set(), set()
    original_dense = Fp8LinearMethod.apply
    original_gdn = GDNKernelDispatcher.packed_decode

    def dense(method, layer, x, bias=None):
        output = original_dense(method, layer, x, bias)
        geometry = (layer.input_size_per_partition, layer.output_size_per_partition)
        if (
            method.use_marlin
            and geometry not in seen_dense
            and not torch.cuda.is_current_stream_capturing()
        ):
            seen_dense.add(geometry)
            _save(
                directory,
                "dense_marlin",
                "x".join(map(str, geometry)),
                probe_dense(method, layer, bias, original_dense),
            )
        return output

    def gdn(dispatcher, mixed_qkv, a, b, **kwargs):
        output = original_gdn(dispatcher, mixed_qkv, a, b, **kwargs)
        geometry = (mixed_qkv.shape[-1], *kwargs["ssm_states"].shape[1:])
        if (
            dispatcher.supports_packed_decode
            and geometry not in seen_gdn
            and not torch.cuda.is_current_stream_capturing()
        ):
            seen_gdn.add(geometry)
            _save(
                directory,
                "gdn_packed",
                "x".join(map(str, geometry)),
                probe_gdn(
                    mixed_qkv,
                    a,
                    b,
                    kwargs,
                    lambda **call: original_gdn(dispatcher, **call),
                ),
            )
        return output

    Fp8LinearMethod.apply = dense
    GDNKernelDispatcher.packed_decode = gdn
    _installed_directory = directory
