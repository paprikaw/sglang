"""Private EP counterpart of PP's small-batch Triton expert policy."""
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
    if _DONE or os.environ.get('EP2_PP2_ARM') != 'ep2':
        return
    import torch
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput, DeepEPLLCombineInput
    from sglang.srt.model_executor.model_runner import ModelRunner
    import ep_triton
    import h100_runtime_adapter

    fd = os.open(os.environ['FRAGMENT_POLICY_PATH'], os.O_RDONLY)
    control = mmap.mmap(fd, 1, access=mmap.ACCESS_READ)
    os.close(fd)
    mode = os.environ['FRAGMENT_EXPERT_MODE']
    original = UnquantizedFusedMoEMethod.apply
    root = Path(os.environ['EP2_PP2_RUN']) / 'ep2'

    def candidate(layer, dispatch):
        return DeepEPLLCombineInput(ep_triton.run(dispatch, layer.w13_weight, layer.w2_weight),
                                    dispatch.topk_ids, dispatch.topk_weights)

    def compare(method, layer, dispatch, *, metadata=False):
        # The native runner is permitted to dispose its input storage. Each
        # control therefore gets its own clone of the identical routed input.
        fixed = dispatch._replace(hidden_states=dispatch.hidden_states.clone())
        reference = original(method, layer, fixed).hidden_states
        fixed = dispatch._replace(hidden_states=dispatch.hidden_states.clone())
        value = ep_triton.run(fixed, layer.w13_weight, layer.w2_weight,
                              return_metadata=metadata)
        output, meta = value if metadata else (value, None)
        mask = torch.arange(output.shape[1], device=output.device)[None, :] < dispatch.masked_m[:, None]
        a, b = output[mask].float(), reference[mask].float()
        error = dict(relative_l2=float((a-b).norm() / b.norm().clamp_min(1e-20)),
                     max_abs=float((a-b).abs().max()) if a.numel() else 0.0,
                     finite=bool(torch.isfinite(a).all()),
                     valid_rows=int(mask.sum()))
        if not error['finite'] or error['relative_l2'] > .015:
            raise RuntimeError('EP small-batch Triton failed the same 1.5% layer L2 gate as PP: ' + str(error))
        if metadata:
            count = dispatch.masked_m.cpu().tolist()
            padded = int(meta['padded'].cpu())
            if sum(count) > meta['assignments_bound'] or padded > meta['padded_bound']:
                raise RuntimeError('EP metadata capacity was exceeded')
            ids = meta['sorted_ids'][:padded].cpu().tolist()
            actual = sorted(i for i in ids if i < output.shape[0] * output.shape[1])
            expected = [e*output.shape[1]+r for e,c in enumerate(count) for r in range(c)]
            if actual != expected:
                raise RuntimeError('EP index adapter dropped or duplicated an assignment')
            expert_ids = meta['expert_ids'][:padded//meta['block_m']].cpu().tolist()
            for block, expert in enumerate(expert_ids):
                if any(i // output.shape[1] != expert for i in ids[block*meta['block_m']:(block+1)*meta['block_m']]
                       if i < output.shape[0]*output.shape[1]):
                    raise RuntimeError('EP index adapter selected the wrong expert weight')
            error.update(metadata_coverage=True, **{k:v for k,v in meta.items() if not isinstance(v,torch.Tensor)})
        return error

    def diagnostic(method, layer, dispatch):
        from sglang.srt.runtime_context import get_parallel
        rank = get_parallel().tp_rank
        t = dispatch.topk_ids.shape[0]
        layer_id = getattr(layer, 'layer_id', -1)
        workload = Path(os.environ['EXPERT_WORKLOAD_PATH']).read_text().strip()
        key = (workload, rank, layer_id, t)
        if key in _SEEN:
            return
        _SEEN.add(key)
        counts = dispatch.masked_m.cpu().tolist()
        record = dict(rank=rank, layer_id=layer_id, local_input_tokens=t,
            expert_buffer_shape=list(dispatch.hidden_states.shape), counts=counts,
            assignments=sum(counts), active_experts=sum(c>0 for c in counts),
            padded_rows={str(b):sum((c+b-1)//b*b for c in counts) for b in (8,16,32,64,128)},
            numerical=compare(method,layer,dispatch,metadata=True))
        routing = root/'workloads'/workload/'routing';routing.mkdir(parents=True,exist_ok=True)
        if (workload,rank,t) not in _BENCHED:
            _BENCHED.add((workload,rank,t))
            timing={}
            for name,fn in [('native',lambda d:original(method,layer,d).hidden_states),
                            ('triton',lambda d:candidate(layer,d).hidden_states)]:
                storage=dispatch.hidden_states.clone()
                fixed=dispatch._replace(hidden_states=storage.view_as(storage))
                # Native dispose_tensor may detach the view during this
                # diagnostic capture. Retain its backing storage explicitly.
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result=fn(fixed)
                samples=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/50)
                timing[name]=dict(graph_ms=samples,graph_median_ms=statistics.median(samples))
                del graph,result,storage
            record['microbenchmark']=timing
        (routing/f'ep{rank}.layer{layer_id}.t{t}.json').write_text(json.dumps(record,indent=2))

    @wraps(original)
    def apply(method, layer, dispatch_output):
        dispatch=dispatch_output
        small = isinstance(dispatch,DeepEPLLDispatchOutput) and dispatch.topk_ids.shape[0] <= 16
        if small and control[0] == ord('D') and not torch.cuda.is_current_stream_capturing():
            diagnostic(method,layer,dispatch)
        if small and mode.startswith('triton_'):
            if torch.autograd._profiler_enabled():
                with torch.profiler.record_function('component.expert.routed.'+str(h100_runtime_adapter._LAYER.get())):
                    return candidate(layer,dispatch)
            return candidate(layer,dispatch)
        return original(method,layer,dispatch)
    UnquantizedFusedMoEMethod.apply=apply

    # Compile and test before the first communication-bearing model step.
    # These probes use the real first-layer weight bank, never timed requests.
    previous_load=ModelRunner.load_model
    @wraps(previous_load)
    def load(self,*args,**kwargs):
        result=previous_load(self,*args,**kwargs)
        layer=next(m for m in self.model.modules() if hasattr(m,'w13_weight') and hasattr(m,'dispatcher'))
        e,n,k=layer.w13_weight.shape
        cap=int(os.environ['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'])*self.ps.moe_ep_size
        generator=torch.Generator(device=layer.w13_weight.device).manual_seed(20261026+self.ps.tp_rank)
        x=torch.randn((e,cap,k),device=layer.w13_weight.device,dtype=torch.bfloat16,generator=generator)*.1
        counts=torch.zeros(e,device=x.device,dtype=torch.int32)
        counts[:7]=torch.tensor([0,1,3,15,16,17,cap],device=x.device,dtype=torch.int32)
        dispatch=DeepEPLLDispatchOutput(x,None,
            torch.zeros((4,8),device=x.device,dtype=torch.int64),
            torch.ones((4,8),device=x.device),counts,1)
        tests=[]
        for logical_tokens in (4,8,16):
            probe=dispatch._replace(topk_ids=torch.zeros((logical_tokens,8),device=x.device,dtype=torch.int64),
                                    topk_weights=torch.ones((logical_tokens,8),device=x.device))
            tests.append(dict(logical_tokens=logical_tokens,**compare(layer.quant_method,layer,probe,metadata=True)))
        empty=dispatch._replace(masked_m=torch.zeros_like(counts))
        tests.append(compare(layer.quant_method,layer,empty,metadata=True))
        # All experts active: validates the maximum sparse-index block count.
        dense=dispatch._replace(masked_m=torch.ones_like(counts))
        tests.append(compare(layer.quant_method,layer,dense,metadata=True))
        torch.cuda.synchronize()
        (root/'ep_kernel_qualification').mkdir(exist_ok=True)
        (root/'ep_kernel_qualification'/f'tp{self.ps.tp_rank}.json').write_text(json.dumps(dict(
            passed=True,performance_sample=False,real_weights=True,tests=tests,
            shared_gemm='sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel',
            no_dispatch_or_combine_changes=True),indent=2))
        return result
    ModelRunner.load_model=load
    _DONE=True
