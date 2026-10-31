"""Real DeepEP D/C, synthetic identity expert, fixed global chunk boundaries."""
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--phase',required=True)
    args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text())
    phase=next(p for p in cfg['phases'] if p['id']==args.phase)
    out=args.run/phase['id']
    world,cap,hidden,experts=phase['world'],phase['capacity'],phase['hidden'],phase['experts']
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==world
    import torch
    import torch.distributed as dist
    import deep_ep,deep_ep_cpp
    torch.set_grad_enabled(False)
    torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=300))
    host=socket.gethostname().split('.')[0]
    cpu_affinity=sorted(os.sched_getaffinity(0))
    assert len(cpu_affinity)==cfg['cpus_per_node_step'], 'CPU NIC proxies require the declared full booked CPU set'
    physical=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    hosts,gpus=[None]*world,[None]*world
    dist.all_gather_object(hosts,host)
    dist.all_gather_object(gpus,physical[local])
    assert hosts==[p['node'] for p in phase['placement']]
    assert gpus==[p['gpu'] for p in phase['placement']]
    assert 'H100' in torch.cuda.get_device_name()
    expected=native_for(args.run,phase['model'])
    native=Path(deep_ep_cpp.__file__).resolve()
    assert str(native)==expected['path'] and sha(native)==expected['sha256']
    rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(cap,hidden,world,experts)
    initlog=out/f'native_init_rank{rank}.log'
    buffer=initialize_with_log(initlog,lambda:deep_ep.Buffer(dist.group.WORLD,0,rdma_bytes,
        low_latency_mode=True,num_qps_per_rank=experts//world,
        allow_nvlink_for_low_latency_mode=True,explicitly_destroy=True))
    backend=verify_backend_text(initlog.read_text())
    assert buffer.runtime.get_low_latency_nvlink_pair_only() is False
    assert buffer.runtime.get_low_latency_pcie_p2p_max_tokens()==0
    assert int(os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'])==experts//world
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if 'libnvshmem_host.so' in line and '/' in line.split()[-1]}
    assert len(paths)==1
    nv=ctypes.CDLL(next(iter(paths)))
    nv.nvshmem_ptr.argtypes=[ctypes.c_void_p,ctypes.c_int]
    nv.nvshmem_ptr.restype=ctypes.c_void_p
    view=buffer.get_local_buffer_tensor(torch.uint8,use_rdma_buffer=True)
    ptrs=[int(nv.nvshmem_ptr(view.data_ptr(),peer) or 0) for peer in range(world)]
    assert [bool(v) for v in ptrs]==[h==host for h in hosts]
    route=dict(passed=True,rank=rank,node=host,physical_gpu=physical[local],world=world,
        native=expected,wrapper_path=str(Path(deep_ep.__file__).resolve()),capacity=cap,
        shape=phase['model'],hidden=hidden,experts=experts,topk=8,dtype='bfloat16',
        use_fp8=False,use_logfmt=False,ce_enabled=False,pcie_threshold=0,pair_only=False,
        transport_policy=phase['transport'],peer_mapped=[bool(v) for v in ptrs],
        actual_nic_backend=backend,rdma_buffer_bytes=rdma_bytes,nvshmem_qp_depth=int(os.environ['NVSHMEM_QP_DEPTH']),
        expected_payload_routes=[dict(peer=q,transport='local' if q==rank else 'nvlink' if hosts[q]==host else 'rdma') for q in range(world)],
        gpu_name=torch.cuda.get_device_name(),gpu_total_memory=torch.cuda.get_device_properties(local).total_memory,
        placement=phase['placement'],slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_step_id=os.environ.get('SLURM_STEP_ID'),cpu_affinity=cpu_affinity,
        cpus_per_node_step=cfg['cpus_per_node_step'])
    save(out/f'route_rank{rank}.json',route)
    channel=torch.arange(hidden,device='cuda')[None,:]

    def barrier():
        torch.cuda.synchronize()
        dist.barrier()

    def data(ids,iteration):
        ids=ids[:,None]
        values=(ids*7+channel*3+iteration*13)%61-30
        values[:,0]=ids[:,0]%128+iteration*128
        values[:,1]=ids[:,0]//128+iteration*128
        return (values/16).to(torch.bfloat16).contiguous()

    def run_point(work):
        shapes=[layout(world,p['tokens'],experts,p['offset'],cap) for p in work['chunks']]
        inputs=[]
        for shape in shapes:
            ids=torch.tensor(shape['ids'][rank],device='cuda',dtype=torch.int64)
            indices=torch.tensor(shape['routes'][rank],device='cuda',dtype=deep_ep.topk_idx_t)
            weights=torch.full((shape['local_tokens'],8),.125,device='cuda',dtype=torch.float32)
            inputs.append(dict(ids=ids,indices=indices,weights=weights,x=data(ids,0)))

        def dispatch(inp):
            recv,counts,handle,_,hook=buffer.low_latency_dispatch(inp['x'],inp['indices'],cap,experts,
                use_fp8=False,async_finish=False,return_recv_hook=True)
            hook()
            return recv,counts,handle

        def combine(inp,recv,handle):
            y,_,hook=buffer.low_latency_combine(recv,inp['indices'],inp['weights'],handle,
                use_logfmt=False,zero_copy=False,async_finish=False,return_recv_hook=True)
            hook()
            return y

        def execute(events=None):
            results=[]
            if events is not None:
                events[0].record()
            for index,inp in enumerate(inputs):
                recv,counts,handle=dispatch(inp)
                if events is not None:
                    events[2*index+1].record()
                y=combine(inp,recv,handle)
                if events is not None:
                    events[2*index+2].record()
                results.append((recv,counts,handle,y))
            return results

        def validate(results,iteration):
            torch.cuda.synchronize()
            received=0
            for shape,inp,result in zip(shapes,inputs,results):
                recv,counts,handle,y=result
                got=counts.cpu().tolist()
                ranges=handle[1].cpu().tolist()
                source_ids=handle[0].cpu().tolist()
                check_e,check_slot,check_ids=[],[],[]
                for e in range(experts//world):
                    expert=rank*(experts//world)+e
                    assert got[e]==shape['counts'][expert]
                    for source in range(world):
                        packed=ranges[e][source]
                        begin,count=packed>>32,packed&((1<<32)-1)
                        expect=shape['source_rows'][expert][source]
                        actual=source_ids[e][begin:begin+count]
                        assert count==len(expect) and sorted(actual)==expect
                        check_e.extend([e]*count)
                        check_slot.extend(range(begin,begin+count))
                        check_ids.extend(shape['ids'][source][i] for i in actual)
                # Bound correctness temporaries independently of graph-held buffers.
                for start in range(0,len(check_ids),1024):
                    ei=torch.tensor(check_e[start:start+1024],device='cuda')
                    si=torch.tensor(check_slot[start:start+1024],device='cuda')
                    gi=torch.tensor(check_ids[start:start+1024],device='cuda')
                    torch.testing.assert_close(recv[ei,si],data(gi,iteration),atol=0,rtol=0)
                torch.testing.assert_close(y,data(inp['ids'],iteration),atol=0,rtol=0)
                received+=len(check_ids)
            assert received==work['tokens']*8//world
            return dict(passed=True,counts_exact=True,source_ids_exact=True,payload_exact=True,
                weighted_identity_exact=True,received_assignments=received)

        barrier()
        before=validate(execute(),0)
        barrier()
        for _ in range(cfg['warmup']):
            barrier()
            result=execute()
            torch.cuda.synchronize()
            del result
        events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(2*len(inputs)+1)]
        for event in events:
            event.record()
        events[-1].synchronize()
        graph=None
        graph_result=None
        graph_checks=None
        if work['cuda_graph']:
            # Release eager warmup blocks before allocating a graph's private pool.
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_result=execute(events)
            barrier()
            graph.replay()
            torch.cuda.synchronize()
            graph_checks=dict(original_payload=validate(graph_result,0))
            for _ in range(cfg['warmup']):
                barrier()
                graph.replay()
                torch.cuda.synchronize()
        samples=[]
        was_gc=gc.isenabled()
        gc.disable()
        try:
            for iteration in range(cfg['iterations']):
                barrier()
                started=time.perf_counter()
                if graph is not None:
                    graph.replay()
                    result=graph_result
                else:
                    result=execute(events)
                events[-1].synchronize()
                wall=(time.perf_counter()-started)*1000
                per_chunk=[dict(dispatch_ms=events[2*i].elapsed_time(events[2*i+1]),
                    combine_ms=events[2*i+1].elapsed_time(events[2*i+2])) for i in range(len(inputs))]
                d=sum(c['dispatch_ms'] for c in per_chunk)
                c=sum(c['combine_ms'] for c in per_chunk)
                total=events[0].elapsed_time(events[-1])
                assert abs(d+c-total)<1e-3
                samples.append(dict(iteration=iteration,event_1based=iteration+1,pair_cuda_ms=total,
                    dispatch_cuda_ms=d,combine_cuda_ms=c,pair_cpu_wall_ms=wall,chunks=per_chunk))
                # Outside the measured event span, check every returned source tensor.
                for inp,(_,_,_,y) in zip(inputs,result):
                    torch.testing.assert_close(y,data(inp['ids'],0),atol=0,rtol=0)
                del result
        finally:
            if was_gc:
                gc.enable()
        for inp in inputs:
            inp['x'].copy_(data(inp['ids'],1))
        barrier()
        if graph is not None:
            graph.replay()
            torch.cuda.synchronize()
            after=validate(graph_result,1)
            graph_checks['changed_payload']=after
        else:
            after=validate(execute(),1)
        barrier()
        distinct=None
        stress=[]
        if work['tokens'] in (8,32768):
            # Untimed first-chunk test of distinct expert outputs and unequal weights.
            inp=inputs[0]
            recv,counts,handle=dispatch(inp)
            torch.cuda.synchronize()
            w=torch.arange(1,9,device='cuda',dtype=torch.float32)/64
            inp['weights'].copy_(w[None,:])
            ranges=handle[1].cpu().tolist()
            for e in range(experts//world):
                expert=rank*(experts//world)+e
                for source in range(world):
                    packed=ranges[e][source]
                    begin,count=packed>>32,packed&((1<<32)-1)
                    if count:
                        recv[e,begin:begin+count].fill_((expert%16+source%4)/32)
            barrier()
            y=combine(inp,recv,handle)
            torch.cuda.synchronize()
            ref=(((inp['indices'].long()%16+rank%4).float()/32)*w[None,:]).sum(-1).to(torch.bfloat16)
            torch.testing.assert_close(y,ref[:,None].expand_as(y),atol=0,rtol=0)
            distinct=dict(passed=True,unequal_weights=True,distinct_expert_outputs=True,chunk_index=0)
            inp['weights'].fill_(.125)
            del recv,counts,handle,y
            for changed in (2,3):
                for inp in inputs:
                    inp['x'].copy_(data(inp['ids'],changed))
                barrier()
                if rank==changed%world:
                    time.sleep(.001)
                if graph is not None:
                    graph.replay()
                    torch.cuda.synchronize()
                    stress.append(validate(graph_result,changed))
                else:
                    stress.append(validate(execute(),changed))
                barrier()
        public_shapes=[{k:v for k,v in shape.items() if k not in ('ids','routes','counts','source_rows')} for shape in shapes]
        return dict(passed=True,id=work['id'],cohort=work['cohort'],global_batch=work['tokens'],
            global_original_tokens=work['tokens'],source_tokens_per_rank_total=work['tokens']//world,
            received_assignments_per_rank=work['tokens']*8//world,global_remote_assignments=work['tokens']*8*(world-1)//world,
            chunks=len(inputs),global_chunk_limit=cfg['global_chunk_limit'],chunk_shapes=public_shapes,
            samples=samples,correctness_before=before,correctness_after_changed_payload=after,
            distinct_expert_validation=distinct,skewed_changed_payload_checks=stress,
            cuda_graph=work['cuda_graph'],graph_correctness=graph_checks,native_sha256=expected['sha256'])

    points=[]
    for work in phase['workloads']:
        point=run_point(work)
        save(out/f'point_{work["id"]}_rank{rank}.json',point)
        points.append(point)
        gc.collect()
        torch.cuda.empty_cache()
        barrier()
    barrier()
    buffer.destroy()
    barrier()
    save(out/f'result_rank{rank}.json',dict(passed=True,rank=rank,phase=phase['id'],
        model_loaded=False,expert_compute=False,attention_compute=False,points=points,
        native_sha256=expected['sha256'],route=route,buffer_explicitly_destroyed=True))
    dist.destroy_process_group()


if __name__=='__main__':
    main()
