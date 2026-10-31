"""Actual DeepEP communication; synthetic vectors, identity experts, no GEMM."""
import argparse
import ctypes
import datetime
import gc
import json
import os
from pathlib import Path
import socket
import time
from common import save,sha,native_for
from geometry import layout
from native_backend import initialize_with_log,verify_backend_text
from ce_support import geometry as ce_geometry,setup as ce_setup,snapshot,validate_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True)
    args=p.parse_args();cfg=json.loads((args.run/'config.json').read_text())
    phase=next(x for x in cfg['phases'] if x['id']==args.phase)
    out=args.run/phase['id'];world=phase['world'];cap=phase['capacity']
    hidden=phase['hidden'];experts=phase['experts'];ce=phase['ce_enabled']
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==world
    import torch
    import torch.distributed as dist
    import deep_ep,deep_ep_cpp
    torch.set_grad_enabled(False);torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=240))
    hosts=[None]*world;gpus=[None]*world;host=socket.gethostname().split('.')[0]
    physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    dist.all_gather_object(hosts,host);dist.all_gather_object(gpus,physical[local])
    assert hosts==[r['node'] for r in phase['placement']]
    assert gpus==[r['gpu'] for r in phase['placement']]
    assert 'A100' in torch.cuda.get_device_name()
    expected=native_for(args.run,phase['model']);native=Path(deep_ep_cpp.__file__).resolve()
    assert str(native)==expected['path'] and sha(native)==expected['sha256']
    rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(cap,hidden,world,experts)
    extra=ce_geometry(world,cap,experts,hidden)['extra_bytes'] if ce else 0
    initlog=out/f'native_init_rank{rank}.log'
    buffer=initialize_with_log(initlog,lambda:deep_ep.Buffer(dist.group.WORLD,0,rdma_bytes+extra,
                low_latency_mode=True,num_qps_per_rank=experts//world,
                allow_nvlink_for_low_latency_mode=True,explicitly_destroy=True))
    backend=verify_backend_text(initlog.read_text())
    assert buffer.runtime.get_low_latency_nvlink_pair_only() is True
    assert buffer.runtime.get_low_latency_pcie_p2p_max_tokens()==phase['pcie_threshold']
    assert int(os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'])==experts//world
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if 'libnvshmem_host.so' in line and '/' in line.split()[-1]}
    assert len(paths)==1;nv=ctypes.CDLL(next(iter(paths)))
    nv.nvshmem_ptr.argtypes=[ctypes.c_void_p,ctypes.c_int];nv.nvshmem_ptr.restype=ctypes.c_void_p
    view=buffer.get_local_buffer_tensor(torch.uint8,use_rdma_buffer=True)
    ptrs=[int(nv.nvshmem_ptr(view.data_ptr(),peer) or 0) for peer in range(world)]
    assert [bool(v) for v in ptrs]==[h==host for h in hosts]
    api,tensors=(ce_setup(torch,native,view,ptrs,rdma_bytes,rank,world,cap,
                         True,True,phase['skip_local_drain'],True,experts,hidden) if ce else (None,None))
    route=dict(passed=True,rank=rank,world=world,native=expected,capacity=cap,
               model_shape=phase['model'],hidden=hidden,experts=experts,topk=8,
               ce_enabled=ce,pcie_threshold=phase['pcie_threshold'],cuda_graph=False,
               skip_local_drain=phase['skip_local_drain'],peer_mapped=[bool(v) for v in ptrs],
               actual_nic_backend=backend,rdma_buffer_bytes=rdma_bytes,ce_receive_scratch_bytes=extra,
               expected_payload_routes=[dict(peer=q,transport='local' if q==rank else 'nvlink' if q==rank^1
                  else 'ce_pcie' if ce and hosts[q]==host else 'rdma') for q in range(world)],
               gpu_name=torch.cuda.get_device_name(),placement=phase['placement'])
    save(out/f'route_rank{rank}.json',route)
    channel=torch.arange(hidden,device='cuda')[None,:]
    def barrier():torch.cuda.synchronize();dist.barrier()
    def data(ids,iteration):
        ids=ids[:,None];values=(ids*7+channel*3+iteration*13)%61-30
        values[:,0]=ids[:,0]%128+iteration*128
        values[:,1]=ids[:,0]//128+iteration*128
        return (values/16).to(torch.bfloat16).contiguous()
    points=[]
    for work in phase['workloads']:
        shape=layout(world,work['tokens'],experts,work['offset'],cap)
        ids=torch.tensor(shape['ids'][rank],device='cuda',dtype=torch.int64)
        indices=torch.tensor(shape['routes'][rank],device='cuda',dtype=deep_ep.topk_idx_t)
        weights=torch.full((shape['local_tokens'],8),.125,device='cuda',dtype=torch.float32)
        x=data(ids,0)
        def dispatch():
            recv,counts,handle,_,hook=buffer.low_latency_dispatch(x,indices,cap,experts,
                           use_fp8=False,async_finish=False,return_recv_hook=True)
            hook();return recv,counts,handle
        def combine(recv,handle):
            y,_,hook=buffer.low_latency_combine(recv,indices,weights,handle,
                          use_logfmt=False,zero_copy=False,async_finish=False,return_recv_hook=True)
            hook();return y
        def pair():
            recv,counts,handle=dispatch();return recv,counts,handle,combine(recv,handle)
        def validate(result,iteration):
            recv,counts,handle,y=result;torch.cuda.synchronize()
            got=counts.cpu().tolist();ranges=handle[1].cpu().tolist();source_ids=handle[0].cpu().tolist()
            check_e=[];check_slot=[];check_ids=[]
            for e in range(experts//world):
                expert=rank*(experts//world)+e
                assert got[e]==shape['counts'][expert]
                for source in range(world):
                    packed=ranges[e][source];begin,count=packed>>32,packed&((1<<32)-1)
                    expect=[i for i,choices in enumerate(shape['routes'][source]) if expert in choices]
                    actual=source_ids[e][begin:begin+count]
                    assert count==len(expect) and sorted(actual)==expect
                    check_e.extend([e]*count);check_slot.extend(range(begin,begin+count))
                    check_ids.extend(shape['ids'][source][i] for i in actual)
            if check_ids:
                ei=torch.tensor(check_e,device='cuda');si=torch.tensor(check_slot,device='cuda')
                gi=torch.tensor(check_ids,device='cuda')
                torch.testing.assert_close(recv[ei,si],data(gi,iteration),atol=0,rtol=0)
            torch.testing.assert_close(y,data(ids,iteration),atol=0,rtol=0)
            return dict(passed=True,counts_exact=True,source_ids_exact=True,payload_exact=True,
                        weighted_identity_exact=True,received_assignments=len(check_ids))
        barrier();before=validate(pair(),0);barrier()
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for event in events:event.record()
        events[-1].synchronize()
        for _ in range(cfg['warmup']):barrier();pair();torch.cuda.synchronize()
        samples=[];was_gc=gc.isenabled();gc.disable()
        try:
            for iteration in range(cfg['iterations']):
                barrier();counter0=snapshot(api) if ce else None
                started=time.perf_counter();events[0].record()
                recv,counts,handle=dispatch();events[1].record()
                y=combine(recv,handle);events[2].record();events[2].synchronize()
                wall=(time.perf_counter()-started)*1000
                delta=[b-a for a,b in zip(counter0,snapshot(api))] if ce else None
                if ce:validate_metrics(delta)
                samples.append(dict(iteration=iteration,pair_cuda_ms=events[0].elapsed_time(events[2]),
                         dispatch_cuda_ms=events[0].elapsed_time(events[1]),combine_cuda_ms=events[1].elapsed_time(events[2]),
                         pair_cpu_wall_ms=wall,ce_counter_delta=delta))
                torch.testing.assert_close(y,data(ids,0),atol=0,rtol=0)
        finally:
            if was_gc:gc.enable()
        x.copy_(data(ids,1));barrier();after=validate(pair(),1);barrier()
        distinct=None;stress=[]
        if work['global_batch'] in (8,4096):
            # Untimed: combine must preserve distinct expert contributions and unequal weights.
            recv,counts,handle=dispatch();torch.cuda.synchronize()
            w=torch.arange(1,9,device='cuda',dtype=torch.float32)/64
            weights.copy_(w[None,:]);ranges=handle[1].cpu().tolist()
            for e in range(experts//world):
                expert=rank*(experts//world)+e
                for source in range(world):
                    packed=ranges[e][source];begin,count=packed>>32,packed&((1<<32)-1)
                    if count:recv[e,begin:begin+count].fill_((expert%16+source%4)/32)
            barrier();y=combine(recv,handle);torch.cuda.synchronize()
            reference=(((indices.long()%16+rank%4).float()/32)*w[None,:]).sum(-1).to(torch.bfloat16)
            torch.testing.assert_close(y,reference[:,None].expand_as(y),atol=0,rtol=0)
            distinct=dict(passed=True,unequal_weights=True,distinct_expert_outputs=True)
            weights.fill_(.125)
            for changed in range(2,6):
                x.copy_(data(ids,changed));barrier()
                if rank==(changed%world):time.sleep(.001)
                stress.append(validate(pair(),changed))
                barrier()
        public_shape={k:v for k,v in shape.items() if k not in ('ids','routes','counts')}
        point=dict(passed=True,id=work['id'],cohort=work['cohort'],global_batch=work['global_batch'],
                   call_unique_tokens=work['tokens'],source_tokens_per_rank=shape['local_tokens'],offset=work['offset'],
                   shape=public_shape,samples=samples,correctness_before=before,
                   correctness_after_changed_payload=after,distinct_expert_validation=distinct,
                   skewed_changed_payload_checks=stress,cuda_graph=False,native_sha256=expected['sha256'])
        save(out/f'point_{work["id"]}_rank{rank}.json',point);points.append(point)
        del recv,counts,handle,y,x
        torch.cuda.synchronize()
    barrier();ce_final=snapshot(api) if ce else None;buffer.destroy();barrier()
    save(out/f'result_rank{rank}.json',dict(passed=True,rank=rank,phase=phase['id'],
              model_loaded=False,expert_compute=False,attention_compute=False,points=points,
              native_sha256=expected['sha256'],route=route,ce_final_counters=ce_final,
              buffer_explicitly_destroyed=True))
    dist.destroy_process_group()


if __name__=='__main__':main()
