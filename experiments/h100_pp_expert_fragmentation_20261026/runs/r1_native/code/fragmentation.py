"""Private shape-adaptive Expert execution and untimed actual-input diagnostics.

Default mode is the unchanged DeepGEMM implementation. All numerical comparisons
and timing below run only during explicitly marked eager diagnostic requests.
"""
from dataclasses import replace
from functools import wraps
import json
import mmap
import os
from pathlib import Path
import statistics

_DONE = False
_SEEN = set()
_BENCHED = set()


def install():
    global _DONE
    if _DONE or os.environ.get('EP2_PP2_ARM') != 'pp2':
        return
    import torch
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    fd = os.open(os.environ['FRAGMENT_POLICY_PATH'], os.O_RDONLY)
    control = mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    os.close(fd)
    mode = os.environ.get('FRAGMENT_EXPERT_MODE', 'native')
    original = UnquantizedFusedMoEMethod.apply

    def alternate(method, layer, dispatch):
        if not hasattr(method, '_fragment_triton'):
            method._fragment_triton = MoeRunner(MoeRunnerBackend.TRITON, method.moe_runner_config)
        quant = TritonMoeQuantInfo(w13_weight=layer.w13_weight, w2_weight=layer.w2_weight,
                                 fuse_swiglu_interleaved=False)
        return method._fragment_triton.run(dispatch, quant)

    def diagnostic(method, layer, dispatch):
        from sglang.srt.runtime_context import get_parallel
        ps = get_parallel()
        x = dispatch.hidden_states
        ids = dispatch.topk_output.topk_ids
        phase = chr(control[0])
        key = (id(layer), phase, x.shape[0])
        if key in _SEEN:
            return
        _SEEN.add(key)
        counts = torch.bincount(ids.flatten().to(torch.int64).clamp_min(0), minlength=layer.num_local_experts).cpu().tolist()
        root = Path(os.environ['EP2_PP2_RUN']) / 'pp2' / 'routing'
        root.mkdir(exist_ok=True)
        layer_id = getattr(layer, 'layer_id', -1)
        record = dict(phase=phase, tokens=x.shape[0], hidden=x.shape[1],
            num_local_experts=layer.num_local_experts, topk=ids.shape[-1], layer_id=layer_id,
            pp_rank=ps.pp_rank, active_experts=sum(n > 0 for n in counts),
            assignments=sum(counts), counts=counts,
            padded_rows={str(b): sum(((n + b - 1) // b) * b for n in counts) for b in (8, 16, 32, 64, 128)})
        path = root / f'pp{ps.pp_rank}.{phase}.layer{layer_id}.t{x.shape[0]}.{len(_SEEN)}.json'
        path.write_text(json.dumps(record, indent=2))
        if phase != 'D' or x.shape[0] > 32:
            return
        # One real layer per stage and shape; benchmark both kernels using the
        # same non-inplace runner contract and cloned real activations.
        bench_key = (ps.pp_rank, phase, x.shape[0])
        if bench_key in _BENCHED:
            return
        _BENCHED.add(bench_key)
        conf = replace(method.moe_runner_config, inplace=False)
        native_runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, conf)
        triton_runner = MoeRunner(MoeRunnerBackend.TRITON, conf)
        native_quant = DeepGemmMoeQuantInfo(w13_weight=layer.w13_weight, w2_weight=layer.w2_weight, use_fp8=False)
        triton_quant = TritonMoeQuantInfo(w13_weight=layer.w13_weight, w2_weight=layer.w2_weight, fuse_swiglu_interleaved=False)
        fixed = dispatch._replace(hidden_states=x.clone())
        funcs = dict(native=lambda: native_runner.run(fixed, native_quant).hidden_states,
                     triton=lambda: triton_runner.run(fixed, triton_quant).hidden_states)
        ref = funcs['native']().clone()
        result = dict(routing=record, modes={}, numerical={}, scope='Actual layer weights and routed activations; Expert execution only; not E2E')
        for name, fn in funcs.items():
            for _ in range(3):
                out = fn()
            torch.cuda.synchronize()
            err = (out.float() - ref.float())
            relative_l2 = (err.norm() / ref.float().norm().clamp_min(1e-8)).item()
            passed = bool(torch.isfinite(out).all().item() and relative_l2 < .015)
            result['numerical'][name] = dict(relative_l2=relative_l2, max_abs=err.abs().max().item(), passed=passed)
            timings = []
            for _ in range(3):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for __ in range(20):
                    out = fn()
                end.record(); end.synchronize()
                timings.append(begin.elapsed_time(end) / 20)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = fn()
            graph_times = []
            for _ in range(3):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for __ in range(50):
                    graph.replay()
                end.record(); end.synchronize()
                graph_times.append(begin.elapsed_time(end) / 50)
            result['modes'][name] = dict(eager_ms=timings, graph_ms=graph_times,
                graph_median_ms=statistics.median(graph_times))
        (root / f'microbench.pp{ps.pp_rank}.{phase}.t{x.shape[0]}.json').write_text(json.dumps(result, indent=2))

    @wraps(original)
    def apply(method, layer, dispatch_output):
        if control[0] in (ord('D'), ord('P')) and not torch.cuda.is_current_stream_capturing():
            diagnostic(method, layer, dispatch_output)
        if mode == 'triton_small' and dispatch_output.hidden_states.shape[0] <= 32:
            return alternate(method, layer, dispatch_output)
        return original(method, layer, dispatch_output)
    UnquantizedFusedMoEMethod.apply = apply
    _DONE = True
