"""Startup-only warmup before the first DeepEP dispatch; original kernels unchanged."""
import time,torch
from common import save,case

def preload(model,arm,rank):
    started=time.monotonic();seen=set();records=[]
    with torch.inference_mode():
        for name,layer in model.named_modules():
            if not name.endswith('.mlp.experts'):continue
            core=layer.quant_method.runner.runner_core
            metas=tuple((k,v.to_str()) for k,v in sorted(layer.humming_metas.items()))
            if metas in seen:continue
            seen.add(metas);old=core.layer;core.layer=layer
            n=layer.num_local_experts;k=layer.hidden_size;device=next(layer.parameters()).device
            capacity=case(arm)['ep_capacity']*case(arm)['ep']
            hidden=torch.zeros((n,capacity,k),device=device,dtype=torch.bfloat16)
            ids=torch.zeros((1,8),device=device,dtype=torch.int64)
            weights=torch.ones((1,8),device=device,dtype=torch.float32)/8
            counts=torch.zeros((n,),device=device,dtype=torch.int32)
            output=core._run_grouped_masked_gemm(hidden,ids,weights,counts,1)
            torch.cuda.synchronize();records.append(dict(layer=name,experts=n,hidden=k,output_shape=list(output.shape)))
            core.layer=old
            del hidden,ids,weights,counts,output
    save(f'{arm}/preload_rank{rank}.json',dict(passed=True,rank=rank,startup_only=True,no_communication=True,elapsed_s=time.monotonic()-started,unique_expert_shapes=records))


def warm_communication(model,arm,rank):
    """Initialize actual DeepEP kernels while no model collective is outstanding."""
    import torch.distributed as dist
    from sglang.srt.distributed.parallel_state import get_world_group
    layer=next(l for name,l in model.named_modules() if name.endswith('.mlp.experts'))
    impl=resolve_low_latency_dispatcher(layer.dispatcher)
    torch.cuda.synchronize();dist.barrier(group=get_world_group().cpu_group)
    buffer=impl._get_buffer()
    torch.cuda.synchronize();dist.barrier(group=get_world_group().cpu_group)
    device=next(layer.parameters()).device
    x=torch.ones((1,layer.hidden_size),dtype=torch.bfloat16,device=device)
    ids=(torch.arange(8,device=device,dtype=torch.int64)*20).view(1,8)
    weights=torch.ones((1,8),device=device,dtype=torch.float32)/8
    save(f'{arm}/commwarm_start_rank{rank}.json',dict(rank=rank,phase='before_dispatch',startup_only=True))
    begin=time.monotonic()
    recv,counts,handle,event,hook=buffer.low_latency_dispatch(x,ids,case(arm)['ep_capacity'],160,use_fp8=False,async_finish=False,return_recv_hook=False)
    save(f'{arm}/commwarm_dispatch_rank{rank}.json',dict(rank=rank,phase='dispatch_returned',startup_only=True))
    out,event,hook=buffer.low_latency_combine(recv,ids,weights,handle,async_finish=False,return_recv_hook=False)
    torch.cuda.synchronize()
    assert torch.allclose(out,x,atol=.02,rtol=.02)
    save(f'{arm}/commwarm_rank{rank}.json',dict(passed=True,rank=rank,startup_only=True,no_expert_compute=True,identity_checked=True,elapsed_s=time.monotonic()-begin))


def resolve_low_latency_dispatcher(dispatcher):
    if hasattr(dispatcher,'_inners'):
        assert len(dispatcher._inners)==1,'This protocol disables TBO'
        dispatcher=dispatcher._inners[0]
    return dispatcher._low_latency_dispatcher
