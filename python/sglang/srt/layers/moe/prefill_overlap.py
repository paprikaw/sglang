"""Opt-in, eager BF16 DeepEP Prefill overlap without splitting Attention/GDN.

Experimental admission is intentionally explicit: two balanced DP ranks, standard
top-k, no quantization, and a control file selected only at drained cohort boundaries.
The default policy delegates to the unmodified implementation.
"""

from collections import Counter
from contextlib import nullcontext
from functools import wraps
import json
import mmap
import os
from pathlib import Path


def execute_two(dispatchers, inputs, core, *, overlap):
    """Keep handles independent and capture both inputs before the first GEMM.

    combine_a captures the producer event; moving it past the next core would
    introduce a false dependency and prevent combine/compute concurrency.
    """
    a, b = dispatchers
    if not overlap:
        outputs = []
        for dispatcher, (hidden, topk) in zip(dispatchers, inputs):
            dispatcher.dispatch_a(hidden, topk)
            combined = core(dispatcher.dispatch_b())
            dispatcher.combine_a(combined)
            outputs.append(dispatcher.combine_b())
        return outputs
    a.dispatch_a(*inputs[0])
    b.dispatch_a(*inputs[1])
    combined_a = core(a.dispatch_b())
    a.combine_a(combined_a)
    combined_b = core(b.dispatch_b())
    b.combine_a(combined_b)
    output_a = a.combine_b()
    output_b = b.combine_b()
    return [output_a, output_b]


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get("EP2_PP2_ARM") != "ep2":
        return
    import torch
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPConfig, DeepEPDispatcher, DeepEPMode,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import configure_deep_gemm_num_sms
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch, get_dp_global_num_tokens
    from sglang.srt.managers.scheduler import Scheduler

    run = Path(os.environ["EP2_PP2_RUN"])
    control = (run / "ep_overlap_policy.bin").open("r+b")
    selector = mmap.mmap(control.fileno(), 1, access=mmap.ACCESS_READ)
    counts = Counter()
    shapes = Counter()
    modules = {}
    qualification = {}

    def mode():
        return int(chr(selector[0]))

    def scope(name):
        return torch.profiler.record_function(name) if torch.autograd._profiler_enabled() else nullcontext()

    class ScopedDispatcher:
        def __init__(self, inner, layer):
            self.inner, self.layer = inner, layer

        def dispatch_a(self, *args):
            with scope(f"component.communication.dispatch.{self.layer}"):
                return self.inner.dispatch_a(*args)

        def dispatch_b(self):
            with scope(f"component.communication.dispatch.{self.layer}"):
                return self.inner.dispatch_b()

        def combine_a(self, *args):
            with scope(f"component.communication.combine.{self.layer}"):
                return self.inner.combine_a(*args)

        def combine_b(self):
            with scope(f"component.communication.combine.{self.layer}"):
                return self.inner.combine_b()

    def children(layer):
        if not hasattr(layer, "_prefill_overlap_dispatchers"):
            native = layer.dispatcher
            if hasattr(native, "_inners"):
                native = native._inners[0]
            impl = native._normal_dispatcher
            kwargs = {key: getattr(impl, key) for key in (
                "group", "router_topk", "permute_fusion", "num_experts",
                "num_local_experts", "hidden_size", "params_dtype",
            )}
            pair = [DeepEPDispatcher(**kwargs, deepep_mode=DeepEPMode.NORMAL,
                                     async_finish=True) for _ in range(2)]
            if impl.quant_config is not None:
                for child in pair:
                    child.set_quant_config(impl.quant_config)
            layer._prefill_overlap_dispatchers = pair
            modules[layer.layer_id] = layer
        return [ScopedDispatcher(child, layer.layer_id) for child in layer._prefill_overlap_dispatchers]

    def split_forward(layer, hidden, topk, policy):
        assert not layer._dwdp_bound and hidden.dtype == torch.bfloat16
        assert isinstance(topk, StandardTopKOutput)
        origin_dim = hidden.shape[-1]
        source_rows = hidden.shape[0]
        cut = source_rows // 2
        inputs = []
        for low, high in ((0, cut), (cut, hidden.shape[0])):
            logits = None if topk.router_logits is None else topk.router_logits[low:high]
            inputs.append((hidden[low:high], StandardTopKOutput(
                topk.topk_weights[low:high], topk.topk_ids[low:high], logits)))
        sms = None
        if policy == 3:
            sms = torch.cuda.get_device_properties(hidden.device).multi_processor_count - DeepEPConfig.get_instance().num_sms
            assert sms > 0

        def core(dispatched):
            with configure_deep_gemm_num_sms(sms):
                return layer.run_moe_core(dispatch_output=dispatched)

        outputs = execute_two(children(layer), inputs, core, overlap=policy != 1)
        result = torch.cat(outputs, dim=0)[..., :origin_dim].contiguous()
        if layer.reduce_results and (layer.moe_tp_size > 1 or layer.moe_ep_size > 1):
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            result = tensor_model_parallel_all_reduce(result)
        counts[policy] += 1
        shapes[f"p{policy}:T{source_rows}:chunks{cut},{source_rows-cut}:gemm_sms{sms}"] += 1
        return result

    def error(reference, value):
        if reference.numel() == 0:
            return dict(exact=torch.equal(reference, value), relative_l2=0., max_abs=0., finite=True)
        diff = value.float() - reference.float()
        return dict(exact=torch.equal(reference, value),
                    relative_l2=(torch.linalg.vector_norm(diff) /
                                 torch.linalg.vector_norm(reference.float()).clamp_min(1e-20)).item(),
                    max_abs=diff.abs().max().item(), finite=bool(torch.isfinite(value).all()))

    original = FusedMoE.forward_impl

    @wraps(original)
    def forward(self, hidden_states, topk_output, pre_quant_input=None):
        policy = mode()
        if policy == 0 or not get_is_extend_in_batch():
            counts[0] += 1
            return original(self, hidden_states, topk_output, pre_quant_input=pre_quant_input)
        global_rows = get_dp_global_num_tokens()
        # Every rank makes the same decision from the synchronized CPU list.
        # A tiny health request must not create an all-empty first sub-dispatch.
        if max(global_rows) < 2 or (policy == 8 and min(global_rows) == 0):
            counts[f"bypass_p{policy}"] += 1
            return original(self, hidden_states, topk_output, pre_quant_input=pre_quant_input)
        assert pre_quant_input is None, "The Prefill overlap admission only supports unquantized BF16"
        if policy == 8:
            saved_hidden = hidden_states.clone() if self.layer_id not in qualification else None
            source_rows = hidden_states.shape[0]
            reference = original(self, hidden_states, topk_output)
            if self.layer_id not in qualification:
                serial = split_forward(self, saved_hidden.clone(), topk_output, 1)
                overlap = split_forward(self, saved_hidden.clone(), topk_output, 2)
                reserved = split_forward(self, saved_hidden.clone(), topk_output, 3)
                row = dict(native_vs_split=error(reference, serial),
                           serial_vs_overlap=error(serial, overlap),
                           serial_vs_reserved=error(serial, reserved), source_rows=source_rows, global_rows=list(global_rows))
                row["passed"] = (row["serial_vs_overlap"]["exact"] and
                                 all(row[k]["finite"] and row[k]["relative_l2"] <= .015
                                     for k in ("native_vs_split", "serial_vs_reserved")))
                qualification[self.layer_id] = row
                directory = run / "ep2/overlap_qualification"
                directory.mkdir(exist_ok=True)
                rank = torch.distributed.get_rank()
                (directory / f"rank{rank}.json").write_text(json.dumps(qualification, indent=2) + "\n")
                if not row["passed"]:
                    raise RuntimeError(f"Prefill overlap numerical admission failed: layer{self.layer_id}: {row}")
            return reference
        assert policy in (1, 2, 3), policy
        return split_forward(self, hidden_states, topk_output, policy)

    FusedMoE.forward_impl = forward
    original_state = Scheduler.get_internal_state

    @wraps(original_state)
    def state(self, *args, **kwargs):
        result = original_state(self, *args, **kwargs)
        drained = all(child._stage.name == "INITIAL" and child._normal_dispatcher.handle is None
                      for layer in modules.values() for child in layer._prefill_overlap_dispatchers)
        result.internal_state["experiment_ep_overlap"] = dict(
            rank=torch.distributed.get_rank(), policy=mode(), counts=dict(counts),
            shapes=dict(shapes), layers=len(modules), drained=drained,
            qualification_layers=len(qualification),
            qualification_passed=bool(qualification) and all(row["passed"] for row in qualification.values()),
            comm_sms=DeepEPConfig.get_instance().num_sms,
            total_sms=torch.cuda.get_device_properties(0).multi_processor_count)
        return result

    Scheduler.get_internal_state = state
    _INSTALLED = True
