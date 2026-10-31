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
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_module
    import h100_runtime_adapter
    fd = os.open(os.environ['FRAGMENT_POLICY_PATH'], os.O_RDONLY)
    control = mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    os.close(fd)
    mode = os.environ.get('FRAGMENT_EXPERT_MODE', 'native')
    original = UnquantizedFusedMoEMethod.apply
    # Nested CUDA tensors otherwise travel inside a Python pickle in this
    # archived engine's PP probability feedback. Unpickling creates a second
    # GPU context on the sender's device in the receiving process. Convert
    # only nested metadata leaves during the untimed validation protocol;
    # top-level tensors retain the existing typed tensor transport.
    import sglang.srt.managers.scheduler_pp_mixin as pp_mixin
    previous_logprob = pp_mixin.get_logprob_dict_from_result
    def cpu_metadata(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k:cpu_metadata(v) for k,v in value.items()}
        if isinstance(value, list):
            return [cpu_metadata(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_metadata(v) for v in value)
        return value
    def safe_logprob(result):
        value = previous_logprob(result)
        if control[0] == ord('V'):
            value = {k:v if isinstance(v, torch.Tensor) else cpu_metadata(v) for k,v in value.items()}
        return value
    pp_mixin.get_logprob_dict_from_result = safe_logprob
    tile_options = {
        'm16n64k64': dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                          GROUP_SIZE_M=1, num_warps=4, num_stages=3),
        'm16n128k64': dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                           GROUP_SIZE_M=1, num_warps=4, num_stages=3),
        'm16n64k128': dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128,
                           GROUP_SIZE_M=1, num_warps=4, num_stages=3),
    }

    def alternate(method, layer, dispatch):
        if not hasattr(method, '_fragment_triton'):
            method._fragment_triton = MoeRunner(MoeRunnerBackend.TRITON, method.moe_runner_config)
        quant = TritonMoeQuantInfo(w13_weight=layer.w13_weight, w2_weight=layer.w2_weight,
                                 fuse_swiglu_interleaved=False)
        config = tile_options.get(mode.removeprefix('triton_'))
        with override_config(config):
            return method._fragment_triton.run(dispatch, quant)

    def diagnostic(method, layer, dispatch):
        from sglang.srt.runtime_context import get_parallel
        ps = get_parallel()
        x = dispatch.hidden_states
        ids = dispatch.topk_output.topk_ids
        phase = chr(control[0])
        workload = Path(os.environ['EXPERT_WORKLOAD_PATH']).read_text().strip()
        key = (workload, id(layer), phase, x.shape[0])
        if key in _SEEN:
            return
        _SEEN.add(key)
        flat = ids.flatten().to(torch.int64)
        counts = torch.bincount(flat[flat >= 0], minlength=layer.num_local_experts).cpu().tolist()
        root = Path(os.environ['EP2_PP2_RUN']) / 'pp2' / 'workloads' / workload / 'routing'
        root.mkdir(exist_ok=True)
        layer_id = getattr(layer, 'layer_id', -1)
        record = dict(phase=phase, tokens=x.shape[0], hidden=x.shape[1],
            num_local_experts=layer.num_local_experts, topk=ids.shape[-1], layer_id=layer_id,
            pp_rank=ps.pp_rank, active_experts=sum(n > 0 for n in counts),
            inplace=method.moe_runner_config.inplace,
            router_weight_on_input=method.moe_runner_config.apply_router_weight_on_input,
            routed_scaling_factor=method.moe_runner_config.routed_scaling_factor,
            assignments=sum(counts), counts=counts,
            padded_rows={str(b): sum(((n + b - 1) // b) * b for n in counts) for b in (8, 16, 32, 64, 128)})
        path = root / f'pp{ps.pp_rank}.{phase}.layer{layer_id}.t{x.shape[0]}.{len(_SEEN)}.json'
        path.write_text(json.dumps(record, indent=2))
        if phase != 'D' or x.shape[0] > 32:
            return
        # One real layer per stage and shape; benchmark both kernels using the
        # same non-inplace runner contract and cloned real activations.
        bench_key = (workload, ps.pp_rank, phase, x.shape[0])
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
        if mode.startswith('triton_') and x.shape[0] == 4:
            def tiled(config):
                with override_config(config):
                    return triton_runner.run(fixed, triton_quant).hidden_states
            for name, config in tile_options.items():
                funcs[name] = lambda config=config: tiled(config)
            def compact():
                selector = deep_module._should_use_masked_standard_layout
                deep_module._should_use_masked_standard_layout = lambda *args: False
                try:
                    return native_runner.run(fixed, native_quant).hidden_states
                finally:
                    deep_module._should_use_masked_standard_layout = selector
            funcs['deepgemm_compact'] = compact
            if not conf.apply_router_weight_on_input and conf.routed_scaling_factor in (None, 1., 1):
                import route_gemv
                funcs['route_gemv8'] = lambda: route_gemv.run(fixed, layer.w13_weight, layer.w2_weight, 8).hidden_states
                funcs['route_gemv16'] = lambda: route_gemv.run(fixed, layer.w13_weight, layer.w2_weight, 16).hidden_states
        if mode == 'native_mechanism' and x.shape[0] == 4:
            # Equal active experts, input, routing weights and effective FLOPs.
            # Place the same selected weights first in one 128-expert bank;
            # vary only the resident expert dimension passed to DeepGEMM.
            selected = sorted(set(ids.flatten().cpu().tolist()))
            assert min(selected) >= 0 and len(selected) <= 32
            order = selected + [i for i in range(layer.num_local_experts) if i not in selected]
            index = torch.tensor(order[:128], device=x.device, dtype=torch.int64)
            bank13 = layer.w13_weight.index_select(0, index)
            bank2 = layer.w2_weight.index_select(0, index)
            mapping = torch.empty(layer.num_local_experts, dtype=torch.int64, device=x.device)
            mapping[index] = torch.arange(128, device=x.device)
            remapped = fixed._replace(topk_output=fixed.topk_output._replace(topk_ids=mapping[ids.long()].to(ids.dtype)))
            for resident in (32, 64, 128):
                resident_runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, replace(conf, num_local_experts=resident))
                resident_quant = DeepGemmMoeQuantInfo(w13_weight=bank13[:resident], w2_weight=bank2[:resident], use_fp8=False)
                funcs['native_resident' + str(resident)] = lambda rr=resident_runner, rq=resident_quant: rr.run(remapped, rq).hidden_states
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
        if mode.startswith('triton_') and dispatch_output.hidden_states.shape[0] <= 16:
            if torch.autograd._profiler_enabled():
                with torch.profiler.record_function('component.expert.routed.' + str(h100_runtime_adapter._LAYER.get())):
                    return alternate(method, layer, dispatch_output)
            return alternate(method, layer, dispatch_output)
        return original(method, layer, dispatch_output)
    UnquantizedFusedMoEMethod.apply = apply
    _DONE = True
