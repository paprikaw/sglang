"""Guard-only real HY3 layer1 surface; imports CUDA only after authentication."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

from common import ROOT,MODEL,save,sha
from geometry import HIDDEN,INTERMEDIATE,TOKENS,EXPERTS,BYTES_PER_EXPERT,counts,summary
from guard_authorization import validate_guard


def main(run):
    cfg=json.loads((run/'config.json').read_text())
    receipt=validate_guard(run,cfg,os.environ,run/'visibility/worker')
    spec=json.loads((run/'dashboard_spec.json').read_text())
    binding=json.loads(Path(os.environ['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS']).read_text())
    assert binding['policy']=='expert_layer_only' and binding['lease_id']==receipt['lease_id']
    assert spec['model_cache']['benchmark']=='hy3_bf16_local_experts'
    assert spec['model_cache']['expert_ids']==list(range(48))
    authorization=dict(experiment_id=cfg['id'],lease_id=receipt['lease_id'],node=cfg['nodes'][0])
    selection=json.loads((run/'selection.json').read_text())
    assert sha(run/'selection.json')==spec['model_cache']['selection']['sha256']
    logging.basicConfig(level=logging.INFO,format='[TP0] %(message)s')
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerCore,TritonRunnerInput,TritonMoeQuantInfo,
        fused_experts_deepep_ll_to_triton_bf16,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import _prepare_fused_moe_run
    from sglang.srt.layers.moe.moe_runner.bf16_deepep import prepare_bf16_deepep_ll
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    torch.cuda.set_device(0);torch.set_grad_enabled(False)
    set_global_server_args_for_scheduler(ServerArgs(
        model_path=str(MODEL),dtype='bfloat16',moe_runner_backend='triton',
        attention_backend='triton',disable_radix_cache=True,disable_custom_all_reduce=True,
        disable_shared_experts_fusion=True,
    ))
    torch.backends.cuda.matmul.allow_tf32=False
    # World-size-one control groups satisfy production allocation interfaces.
    # No collectives or dispatch/combine are called in any measured span.
    import socket
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='gloo',
                                distributed_init_method=f'tcp://127.0.0.1:{port}')
    initialize_model_parallel(tensor_model_parallel_size=1,backend='gloo')
    w13=torch.empty((48,2*INTERMEDIATE,HIDDEN),device='cuda',dtype=torch.bfloat16)
    w2=torch.empty((48,HIDDEN,INTERMEDIATE),device='cuda',dtype=torch.bfloat16)
    byfile={}
    for name,item in selection['tensors'].items():byfile.setdefault(item['file'],[]).append(name)
    checked=0
    for filename,names in sorted(byfile.items()):
        with safe_open(Path(MODEL)/filename,framework='pt',device='cpu') as f:
            for name in names:
                value=f.get_tensor(name).clone()
                assert hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()==selection['tensors'][name]['sha256']
                e=int(name.split('.')[5]);projection=name.split('.')[6]
                if projection=='gate_proj':w13[e,:INTERMEDIATE].copy_(value)
                elif projection=='up_proj':w13[e,INTERMEDIATE:].copy_(value)
                else:w2[e].copy_(value)
                checked+=1
    assert checked==144
    torch.cuda.synchronize()
    torch.manual_seed(20261022)
    all_x=torch.randn((max(TOKENS),HIDDEN),device='cuda',dtype=torch.bfloat16)*.125
    numerical=[];results=[]

    def runner_config(resident):
        return MoeRunnerConfig(num_experts=192,num_local_experts=resident,hidden_size=HIDDEN,
                               intermediate_size_per_partition=INTERMEDIATE,top_k=1,
                               activation='silu',inplace=False,no_combine=False)

    def compact(total,active,resident=48,cohort='surface'):
        cs=counts(total,active,resident)
        ids=torch.tensor([e for e,n in enumerate(cs) for _ in range(n)],device='cuda',dtype=torch.int32).view(-1,1)
        x=all_x[:total]
        weights=torch.ones((total,1),device='cuda',dtype=torch.float32)
        q=TritonMoeQuantInfo(w13[:resident],w2[:resident])
        preparation=_prepare_fused_moe_run(x,q.w13_weight,q.w2_weight,ids,
                     use_fp8_w8a8=False,use_int8_w8a8=False,use_int8_w8a16=False,
                     use_int4_w4a16=False,per_channel_quant=False,block_shape=None)
        config,down,down_tma,up_tma,sorted_ids,expert_ids,padded=preparation
        state=dict(config=config,down_config=down,down_moe_use_tma=down_tma,up_moe_use_tma=up_tma)
        core=TritonRunnerCore(runner_config(resident))
        inp=TritonRunnerInput(x,weights,ids,sorted_ids,expert_ids,padded)
        def invoke():return core.run(inp,q,state).hidden_states
        output=invoke();torch.cuda.synchronize()
        bm=config['BLOCK_SIZE_M'];predicted=sum(math.ceil(n/bm)*bm for n in cs if n)
        assert padded.item()==predicted,(padded.item(),predicted)
        for expert in (next(e for e,n in enumerate(cs) if n),max(e for e,n in enumerate(cs) if n)):
            offset=sum(cs[:expert]);n=min(2,cs[expert]);xx=x[offset:offset+n]
            gateup=F.linear(xx,w13[expert])
            act=(F.silu(gateup[:,:INTERMEDIATE].float())*gateup[:,INTERMEDIATE:].float()).to(torch.bfloat16)
            reference=F.linear(act,w2[expert])
            actual=output[offset:offset+n]
            delta=actual.float()-reference.float()
            nrms=(delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-8)).item()
            error=delta.abs().max().item();peak=reference.float().abs().max().item()
            valid=nrms<=.015 and error<=.03*peak+1e-4
            numerical.append(dict(total=total,active=active,resident=resident,cohort=cohort,
                                  expert=expert,nrms=nrms,max_abs=error,passed=valid))
            assert valid,numerical[-1]
        meta=dict(cohort=cohort,component='expert_core',T=total,E=active,resident=resident,
                  counts=cs,weight_bytes=active*BYTES_PER_EXPERT,gemm_flops=6*HIDDEN*INTERMEDIATE*total,
                  config=config,down_config=down,actual_padded_rows=predicted,
                  m_tiles=predicted//bm,padding_ratio=predicted/total,
                  aligned_prepare_in_timing=False,up_tma=up_tma,down_tma=down_tma)
        return meta,invoke

    def measure(cases,modes):
        for meta,fn in cases:
            for _ in range(5):fn()
        torch.cuda.synchronize()
        for mode in modes:
            retained=[];fns=[]
            for _,fn in cases:
                if mode=='eager':fns.append(fn)
                else:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=fn()
                    retained.append((graph,out));fns.append(graph.replay)
            raw=[[] for _ in cases];blocks=[[] for _ in cases]
            size=len(cases)
            for block in range(3):
                order=list(range(size))
                if block==1:order.reverse()
                elif block==2:order=order[size//2:]+order[:size//2]
                for j in order:
                    for _ in range(3):fns[j]()
                    starts=[torch.cuda.Event(enable_timing=True) for _ in range(10)]
                    ends=[torch.cuda.Event(enable_timing=True) for _ in range(10)]
                    for a,b in zip(starts,ends):a.record();fns[j]();b.record()
                    ends[-1].synchronize()
                    times=[a.elapsed_time(b) for a,b in zip(starts,ends)]
                    raw[j].extend(times);blocks[j].append(summary(times)['median_ms'])
            for (meta,fn),times,bs in zip(cases,raw,blocks):
                results.append(dict(**meta,mode=mode,timing=summary(times),
                                    raw_ms=times,block_median_ms=bs))
            torch.cuda.synchronize()
            del retained,fns

    for total in TOKENS:
        cases=[compact(total,e) for e in EXPERTS if e<=total]
        measure(cases,('eager','cuda_graph'))
        del cases
        save(run/'profile/partial.json',dict(passed=None,rows=results))
        print('HY3_SURFACE_DONE',total,flush=True)

    for total in TOKENS:
        cases=[compact(total,min(total,resident),resident,'ep_pp_matched') for resident in (24,48)]
        measure(cases,('eager','cuda_graph'));del cases
        bridge_cases=[];prep_cases=[]
        # A controlled common per-expert slot stride, not a claim about the
        # full engine's configured receive capacity or its transport latency.
        capacity=math.ceil(total/24)
        for resident in (24,48):
            cs=counts(total,min(total,resident),resident)
            recv=torch.zeros((resident,capacity,HIDDEN),device='cuda',dtype=torch.bfloat16)
            offset=0
            for e,n in enumerate(cs):
                recv[e,:n].copy_(all_x[offset:offset+n]);offset+=n
            masked=torch.tensor(cs,device='cuda',dtype=torch.int32)
            topids=torch.zeros((1,8),device='cuda',dtype=torch.int64)
            topweights=torch.ones((1,8),device='cuda',dtype=torch.float32)/8
            dispatch=DeepEPLLDispatchOutput(recv,None,topids,topweights,masked,0)
            q=TritonMoeQuantInfo(w13[:resident],w2[:resident]);config=runner_config(resident)
            def bridge(dispatch=dispatch,q=q,config=config):
                return fused_experts_deepep_ll_to_triton_bf16(dispatch,q,config).hidden_states
            def prep(recv=recv,masked=masked):return prepare_bf16_deepep_ll(recv,masked)
            output=bridge();torch.cuda.synchronize()
            valid=torch.arange(capacity,device='cuda').view(1,-1)<masked.view(-1,1)
            assert torch.isfinite(output[valid]).all() and (output[~valid]==0).all()
            for expert in (0,max(e for e,n in enumerate(cs) if n)):
                n=min(2,cs[expert]);xx=recv[expert,:n]
                gu=F.linear(xx,w13[expert]);aa=(F.silu(gu[:,:INTERMEDIATE].float())*gu[:,INTERMEDIATE:].float()).to(torch.bfloat16)
                expected=F.linear(aa,w2[expert]);actual=output[expert,:n]
                nrms=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-8)).item()
                numerical.append(dict(total=total,resident=resident,cohort='bridge',expert=expert,nrms=nrms,passed=nrms<=.015))
                assert nrms<=.015
            meta=dict(cohort='ep_pp_matched',T=total,E=min(total,resident),resident=resident,
                      capacity=capacity,allocated_slots=resident*capacity,counts=cs,
                      capacity_scope='controlled common stride; not production transport capacity')
            bridge_cases.append(({**meta,'component':'bridge_full_compute'},bridge))
            prep_cases.append(({**meta,'component':'layout_prepare_only'},prep))
        measure(bridge_cases,('cuda_graph',));measure(prep_cases,('cuda_graph',))
        del bridge_cases,prep_cases
        save(run/'profile/partial.json',dict(passed=None,rows=results))
        print('HY3_MATCHED_DONE',total,flush=True)

    assert len(results)==112 and all(r['passed'] for r in numerical)
    save(run/'profile/result.json',dict(passed=True,rows=results,numerical=numerical,
         authorization=authorization,selection_sha256=sha(run/'selection.json'),
         backend='HY3 BF16 production Triton expert core and LL-layout bridge',
         hardware_counters_measured=False,actual_world_size=1,timed_communication=False,
         gpu=dict(name=torch.cuda.get_device_name(0),total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())))
    print('HY3_EXPERT_SURFACE_COMPLETE',flush=True)
    while True:time.sleep(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
