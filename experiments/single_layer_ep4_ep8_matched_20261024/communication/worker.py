"""Pure identity-expert communication, no model, NCCL collectives or GEMM."""
import argparse,ctypes,datetime,json,os,socket,time
from pathlib import Path
from common import save,sha,NATIVE_SO,NATIVE_SHA,CAPACITY,HIDDEN,EXPERTS,TOPK
from helpers import layout,expected_source_ids,validate_placement
from native_backend import initialize_with_log,verify_backend_text

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--phase',required=True)
    args=parser.parse_args();cfg=json.loads((args.run/'config.json').read_text())
    phase=next(p for p in cfg['phases'] if p['id']==args.phase);rows=phase['placement'];validate_placement(rows)
    global HIDDEN,EXPERTS
    HIDDEN,EXPERTS=phase['hidden'],phase['experts']
    import helpers
    helpers.EXPERTS=EXPERTS
    outdir=args.run/phase['id'];world=phase['world'];capacity=phase['capacity'];rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==world
    import torch,torch.distributed as dist,deep_ep,deep_ep_cpp
    torch.set_grad_enabled(False);torch.cuda.set_device(local)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=240))
    visible=list(map(int,os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    host=socket.gethostname().split('.')[0];hosts=[None]*world;gpus=[None]*world
    dist.all_gather_object(hosts,host);dist.all_gather_object(gpus,visible[local])
    assert hosts==[r['node'] for r in rows] and gpus==[r['gpu'] for r in rows]
    assert 'A100' in torch.cuda.get_device_name()
    native=Path(deep_ep_cpp.__file__).resolve();assert native==NATIVE_SO and sha(native)==NATIVE_SHA
    rdma_bytes=deep_ep.Buffer.get_low_latency_rdma_size_hint(capacity,HIDDEN,world,EXPERTS)
    log=outdir/f'native_init_rank{rank}.log'
    buffer=initialize_with_log(log,lambda:deep_ep.Buffer(dist.group.WORLD,0,rdma_bytes,low_latency_mode=True,
        num_qps_per_rank=EXPERTS//world,allow_nvlink_for_low_latency_mode=True,explicitly_destroy=True))
    backend=verify_backend_text(log.read_text())
    assert buffer.runtime.get_low_latency_nvlink_pair_only() is True
    assert buffer.runtime.get_low_latency_pcie_p2p_max_tokens()==0
    assert int(os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'])==EXPERTS//world
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnvshmem_host.so' in line and '/' in line.split()[-1]}
    assert len(paths)==1;nv=ctypes.CDLL(next(iter(paths)))
    nv.nvshmem_ptr.argtypes=[ctypes.c_void_p,ctypes.c_int];nv.nvshmem_ptr.restype=ctypes.c_void_p
    view=buffer.get_local_buffer_tensor(torch.uint8,use_rdma_buffer=True)
    mapped=[bool(nv.nvshmem_ptr(view.data_ptr(),peer)) for peer in range(world)];del view
    assert mapped==[h==host for h in hosts]
    route=dict(passed=True,rank=rank,world=world,placement=rows,peer_mapped=mapped,
        native_path=str(native),native_sha256=NATIVE_SHA,pcie_p2p_max_tokens=0,nvlink_pair_only_enabled=True,
        actual_nic_backend=backend,num_qps_per_rank=EXPERTS//world,capacity=capacity,gpu_name=torch.cuda.get_device_name(),
        expected_payload_routes=[dict(peer=p,payload_transport='local' if p==rank else 'nvlink' if p==(rank^1) else 'rdma',
            scope='cross_node' if hosts[p]!=host else 'same_node') for p in range(world)],
        scope='Startup native policy and mapped peers; not formal transport tracing')
    save(outdir/f'route_rank{rank}.json',route)
    channel=torch.arange(HIDDEN,device='cuda')[None,:]
    def barrier():torch.cuda.synchronize();dist.barrier()
    def data(ids,mask,iteration):
        ids=ids[:,None];values=(ids*7+channel*3+iteration*13)%61-30
        values[:,0]=ids[:,0]%128+iteration*128;values[:,1]=ids[:,0]//128+iteration*128
        values[~mask]=0
        return (values/16).to(torch.bfloat16).contiguous()
    points=[]
    for workload in cfg['workloads']:
        total=workload['source_tokens_per_rank']*world;shape=layout(world,total,local_ranks=phase['local_ranks'])
        if phase['id'].endswith('local167'):
            offset=total
            shape['ids_by_rank']=[[t+offset for t in rr] for rr in shape['ids_by_rank']]
            shape['routes_by_rank']=[[helpers.expert_choices(t) for t in rr] for rr in shape['ids_by_rank']]
            shape['global_expert_counts']=[0]*EXPERTS
            for rr in shape['routes_by_rank']:
                for choices in rr:
                    for expert in choices:shape['global_expert_counts'][expert]+=1
        ids=torch.tensor(shape['ids_by_rank'][rank],device='cuda',dtype=torch.int64)
        mask=torch.tensor(shape['real_masks_by_rank'][rank],device='cuda',dtype=torch.bool)
        indices=torch.tensor(shape['routes_by_rank'][rank],device='cuda',dtype=deep_ep.topk_idx_t)
        weights=torch.full((shape['local_tokens'],TOPK),.125,device='cuda',dtype=torch.float32)
        x=data(ids,mask,0)
        def dispatch():
            recv,counts,handle,_,hook=buffer.low_latency_dispatch(x,indices,capacity,EXPERTS,use_fp8=False,async_finish=False,return_recv_hook=True)
            hook();return recv,counts,handle
        def combine(recv,handle):
            y,_,hook=buffer.low_latency_combine(recv,indices,weights,handle,use_logfmt=False,zero_copy=False,async_finish=False,return_recv_hook=True)
            hook();return y
        def pair():
            recv,counts,handle=dispatch();return recv,counts,handle,combine(recv,handle)
        def validate(result,iteration):
            recv,counts,handle,y=result;torch.cuda.synchronize()
            got=counts.cpu().tolist();ranges=handle[1].cpu().tolist();checked=0
            for e in range(EXPERTS//world):
                expert=rank*(EXPERTS//world)+e;assert got[e]==shape['global_expert_counts'][expert]
                for source in range(world):
                    packed=ranges[e][source];begin,count=packed>>32,packed&((1<<32)-1)
                    expected=expected_source_ids(shape,source,expert)
                    actual=handle[0][e,begin:begin+count].long().cpu().tolist()
                    assert count==len(expected) and sorted(actual)==expected
                    if count:
                        sid=torch.tensor([shape['ids_by_rank'][source][i] for i in actual],device='cuda')
                        smask=torch.tensor([shape['real_masks_by_rank'][source][i] for i in actual],device='cuda',dtype=torch.bool)
                        torch.testing.assert_close(recv[e,begin:begin+count],data(sid,smask,iteration),atol=0,rtol=0)
                    checked+=count
            torch.testing.assert_close(y,data(ids,mask,iteration),atol=0,rtol=0)
            return dict(passed=True,source_ids_exact=True,counts_exact=True,payload_exact=True,
                weighted_identity_exact=True,padding_not_routed=True,received_assignments=checked)
        barrier();before=validate(pair(),0);barrier()
        graph=None;captured=None
        if workload['cuda_graph']:
            for _ in range(cfg['warmup']):barrier();pair();torch.cuda.synchronize()
            barrier();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=pair()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            x.copy_(data(ids,mask,1));barrier();graph.replay();graph_check=validate(captured,1)
            x.copy_(data(ids,mask,0));barrier()
        else:graph_check=None
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for event in events:event.record()
        events[-1].synchronize()
        for _ in range(cfg['warmup']):
            barrier();graph.replay() if graph else pair();torch.cuda.synchronize()
        samples=[]
        for iteration in range(cfg['iterations']):
            barrier();started=time.perf_counter();events[0].record()
            if graph:
                graph.replay();events[2].record();result=captured
                dispatch_ms=combine_ms=None
            else:
                recv,counts,handle=dispatch();events[1].record();y=combine(recv,handle);events[2].record()
                result=(recv,counts,handle,y)
            events[2].synchronize();wall=(time.perf_counter()-started)*1000
            if not graph:dispatch_ms=events[0].elapsed_time(events[1]);combine_ms=events[1].elapsed_time(events[2])
            samples.append(dict(iteration=iteration,pair_cuda_ms=events[0].elapsed_time(events[2]),pair_cpu_wall_ms=wall,
                dispatch_cuda_ms=dispatch_ms,combine_cuda_ms=combine_ms))
            torch.testing.assert_close(result[3],data(ids,mask,0),atol=0,rtol=0)
        x.copy_(data(ids,mask,1));barrier()
        if graph:graph.replay();after=validate(captured,1)
        else:after=validate(pair(),1)
        barrier()
        geometry={k:v for k,v in shape.items() if k not in ('ids_by_rank','real_masks_by_rank','routes_by_rank')}
        geometry.update(active_experts=sum(c>0 for c in shape['global_expert_counts']),
            capacity=capacity,
            per_owner_assignment_counts=[sum(shape['global_expert_counts'][r*(EXPERTS//world):(r+1)*(EXPERTS//world)]) for r in range(world)],
            logical_payload_bytes_per_leg=total*TOPK*HIDDEN*2)
        point=dict(passed=True,id=workload['id'],source_tokens_per_rank=workload['source_tokens_per_rank'],global_tokens=total,cuda_graph=workload['cuda_graph'],shape=geometry,
            samples=samples,correctness_before=before,correctness_after_changed_payload=after,graph_correctness=graph_check,
            graph_replays_per_sample=1 if graph else 0,graph_capture_eager_warmups=cfg['warmup'] if graph else 0,
            native_sha256=NATIVE_SHA,route=route)
        save(outdir/f'point_t{total}_rank{rank}.json',point);points.append(point)
        torch.cuda.synchronize()
        if graph is not None:graph.reset()
        captured=None;result=None;graph=None
        torch.cuda.synchronize()
    barrier();buffer.destroy();barrier()
    save(outdir/f'result_rank{rank}.json',dict(passed=True,rank=rank,phase=phase['id'],model_loaded=False,
        expert_compute=False,points=points,native_sha256=NATIVE_SHA,buffer_explicitly_destroyed=True))
    dist.destroy_process_group()

if __name__=='__main__':main()
