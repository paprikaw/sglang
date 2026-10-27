"""Opt-in two-chunk BF16 DeepEP overlap in fixed-policy Decode CUDA graphs.

Attention/GDN remain whole. Each process captures one immutable MoE policy;
changing a control byte after capture cannot change an already captured graph.
"""
from collections import Counter
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import json
import mmap
import os
from pathlib import Path

_INSTALLED = False
_GRAPH_BUILD = ContextVar("decode_overlap_graph_build", default=False)


def eligible(source_rows, global_rows, graph_build=False):
    if source_rows < 2:
        return False
    # Graph capture uses equal padded source shapes on both DP ranks.
    if graph_build:
        return True
    return bool(global_rows) and min(global_rows) >= 2


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get("EP2_PP2_ARM") != "ep2":
        return
    import torch
    from sglang.srt.layers.moe.prefill_overlap import execute_two
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher, DeepEPMode
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch, get_dp_global_num_tokens
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.managers.scheduler import Scheduler

    policy = int(os.environ["EP_DECODE_OVERLAP_POLICY"])
    assert policy in (0, 1, 2)
    run = Path(os.environ["EP2_PP2_RUN"])
    control_file = (run / "decode_overlap_verify.bin").open("r+b")
    control = mmap.mmap(control_file.fileno(), 1, access=mmap.ACCESS_READ)
    counts, replay_counts = Counter(), Counter()
    modules, qualifications = {}, {}

    def scope(name):
        return torch.profiler.record_function(name) if torch.autograd._profiler_enabled() else nullcontext()

    class Scoped:
        def __init__(self, inner, layer): self.inner, self.layer = inner, layer
        def dispatch_a(self, *args):
            with scope(f"component.communication.dispatch.{self.layer}"): return self.inner.dispatch_a(*args)
        def dispatch_b(self):
            with scope(f"component.communication.dispatch.{self.layer}"): return self.inner.dispatch_b()
        def combine_a(self, *args):
            with scope(f"component.communication.combine.{self.layer}"): return self.inner.combine_a(*args)
        def combine_b(self):
            with scope(f"component.communication.combine.{self.layer}"): return self.inner.combine_b()

    def children(layer):
        if not hasattr(layer, "_decode_overlap_pair"):
            native = layer.dispatcher
            if hasattr(native, "_inners"): native = native._inners[0]
            impl = native._low_latency_dispatcher
            kwargs = {key: getattr(impl, key) for key in (
                "group", "router_topk", "permute_fusion", "num_experts",
                "num_local_experts", "hidden_size", "params_dtype")}
            layer._decode_overlap_pair = [DeepEPDispatcher(**kwargs,
                deepep_mode=DeepEPMode.LOW_LATENCY, async_finish=True,
                return_recv_hook=False) for _ in range(2)]
            for child in layer._decode_overlap_pair:
                child.set_quant_config(impl.quant_config)
            modules[layer.layer_id] = layer
        return [Scoped(child, layer.layer_id) for child in layer._decode_overlap_pair]

    def split(layer, hidden, topk, overlap):
        assert isinstance(topk, StandardTopKOutput) and hidden.dtype == torch.bfloat16
        cut, width = hidden.shape[0] // 2, hidden.shape[-1]
        inputs = [(hidden[lo:hi], StandardTopKOutput(
            topk.topk_weights[lo:hi], topk.topk_ids[lo:hi],
            None if topk.router_logits is None else topk.router_logits[lo:hi]))
            for lo, hi in ((0, cut), (cut, hidden.shape[0]))]
        outputs = execute_two(children(layer), inputs,
            lambda value: layer.run_moe_core(dispatch_output=value), overlap=overlap)
        result = torch.cat(outputs, dim=0)[..., :width].contiguous()
        if layer.reduce_results and (layer.moe_tp_size > 1 or layer.moe_ep_size > 1):
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            result = tensor_model_parallel_all_reduce(result)
        return result

    def error(ref, value):
        delta = value.float() - ref.float()
        return dict(exact=torch.equal(ref, value), finite=bool(torch.isfinite(value).all()),
                    relative_l2=float(delta.norm() / ref.float().norm().clamp_min(1e-20)),
                    max_abs=float(delta.abs().max()) if delta.numel() else 0.)

    original = FusedMoE.forward_impl
    @wraps(original)
    def forward(layer, hidden_states, topk_output, pre_quant_input=None):
        build = _GRAPH_BUILD.get()
        if (get_is_extend_in_batch() and not build) or not eligible(
                hidden_states.shape[0], get_dp_global_num_tokens(), build):
            counts["native_bypass"] += 1
            return original(layer, hidden_states, topk_output, pre_quant_input=pre_quant_input)
        verify = not build and control[0] == ord("1")
        if verify:
            workload = (run / "active_workload.txt").read_text().strip()
            key = (workload, layer.layer_id)
            saved = hidden_states.clone()
            ref = original(layer, hidden_states, topk_output, pre_quant_input=pre_quant_input).clone()
            if key not in qualifications:
                serial = split(layer, saved.clone(), topk_output, False)
                overlapped = split(layer, saved.clone(), topk_output, True)
                row = dict(native_vs_serial=error(ref, serial), serial_vs_overlap=error(serial, overlapped),
                           source_rows=saved.shape[0], global_rows=list(get_dp_global_num_tokens()))
                row["passed"] = row["serial_vs_overlap"]["exact"] and row["native_vs_serial"]["finite"] and row["native_vs_serial"]["relative_l2"] <= .015
                qualifications[key] = row
                path = run / "ep2/workloads" / workload / "overlap_qualification"
                path.mkdir(parents=True, exist_ok=True)
                rank = torch.distributed.get_rank()
                (path / f"rank{rank}.json").write_text(json.dumps({str(k[1]):v for k,v in qualifications.items() if k[0] == workload},indent=2)+"\n")
                if not row["passed"]: raise RuntimeError(f"Decode overlap qualification failed: {key}: {row}")
            return ref
        if policy == 0:
            return original(layer, hidden_states, topk_output, pre_quant_input=pre_quant_input)
        assert pre_quant_input is None
        counts[f"{'capture' if build else 'eager'}:p{policy}:T{hidden_states.shape[0]}"] += 1
        return split(layer, hidden_states, topk_output, policy == 2)
    FusedMoE.forward_impl = forward

    previous_capture = DecodeCudaGraphRunner.capture_one_shape
    @wraps(previous_capture)
    def capture(self, *args, **kwargs):
        token = _GRAPH_BUILD.set(True)
        try: return previous_capture(self, *args, **kwargs)
        finally: _GRAPH_BUILD.reset(token)
    DecodeCudaGraphRunner.capture_one_shape = capture

    previous_replay = DecodeCudaGraphRunner.execute
    @wraps(previous_replay)
    def replay(self, forward_batch, *args, **kwargs):
        replay_counts[str(forward_batch.batch_size)] += 1
        return previous_replay(self, forward_batch, *args, **kwargs)
    DecodeCudaGraphRunner.execute = replay

    previous_state = Scheduler.get_internal_state
    @wraps(previous_state)
    def state(self, *args, **kwargs):
        result = previous_state(self, *args, **kwargs)
        result.internal_state["experiment_decode_overlap"] = dict(
            rank=torch.distributed.get_rank(), policy=policy, counts=dict(counts),
            graph_replays=dict(replay_counts),
            qualification_records=len(qualifications),
            qualification_passed=all(row["passed"] for row in qualifications.values()),
            layers=len(modules), drained=all(child._stage.name == "INITIAL" and child._low_latency_dispatcher.handle is None
                for layer in modules.values() for child in layer._decode_overlap_pair))
        return result
    Scheduler.get_internal_state = state
    _INSTALLED = True
