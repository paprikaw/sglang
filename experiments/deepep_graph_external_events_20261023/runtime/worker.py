"""Actual external CUDA timing events inside a real EP8 dispatch/combine Graph.

No model, Attention or Expert compute. BF16 identity dispatch/combine only.
Every replay has a durable pre-checkpoint; a native fault remains a failed mode.
"""
import argparse
import ctypes
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import time

from common import (MODES, FLAGS, NATIVE, NATIVE_SHA, DEEP, TRANSPORT,
                    expert_counts, routes, save, sha, EVENT_NAMES, validate_event_sample, validate_freshness)


def mapped_library_paths(text):
    names = ('libcupti','libcudart','libcuda.so','libtorch_cuda','libkineto','libnvshmem','deep_ep_cpp')
    return sorted({line.split(maxsplit=5)[-1].removesuffix(' (deleted)')
        for line in text.splitlines() if len(line.split(maxsplit=5)) == 6
        and line.split(maxsplit=5)[-1].startswith('/')
        and any(name in line.split(maxsplit=5)[-1] for name in names)})


def loaded_libraries():
    """Read already-mapped paths only; no CUPTI API or profiling collector."""
    return [dict(path=p,realpath=os.path.realpath(p))
            for p in mapped_library_paths(Path('/proc/self/maps').read_text())]


def cleanup_graph_and_buffer(graph, buffer, cuda, dist):
    cuda.synchronize()
    if graph is not None:
        graph.reset()
    dist.barrier()
    cuda.synchronize()
    if buffer is not None:
        buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    output = args.run/args.mode
    faulthandler.enable(all_threads=True)
    for key,value in TRANSPORT.items():
        if os.environ.get(key) != value:
            raise RuntimeError('Frozen transport mismatch: '+key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise RuntimeError('Pinned NVLink physical pairs require GPU order 0,1,2,3')
    import torch
    import torch.distributed as dist
    import deep_ep
    import deep_ep_cpp

    rank,local_rank,world = int(os.environ['RANK']),int(os.environ['LOCAL_RANK']),int(os.environ['WORLD_SIZE'])
    if world != 8 or local_rank != rank%4:
        raise RuntimeError('Exactly two four-GPU nodes required')
    torch.set_grad_enabled(False)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('gloo',timeout=timedelta(seconds=cfg['worker_timeout_s']))
    native_path = Path(deep_ep_cpp.__file__).resolve()
    if native_path != NATIVE.resolve() or sha(native_path) != NATIVE_SHA:
        raise RuntimeError('Loaded native does not match real v7 experiment')
    if Path(deep_ep.__file__).resolve().parent != (DEEP/'deep_ep').resolve():
        raise RuntimeError('Wrong DeepEP wrapper')
    hosts = [None]*world
    dist.all_gather_object(hosts,os.uname().nodename.split('.')[0])
    if hosts != [n for n in cfg['nodes'] for _ in range(4)]:
        raise RuntimeError('Rank to node mapping differs from frozen configuration')
    if 'A100' not in torch.cuda.get_device_name(local_rank):
        raise RuntimeError('Only A100 diagnosis admitted')
    graph,buffer,libraries = None,None,[]
    started = time.monotonic()

    def checkpoint(phase,**extra):
        value = dict(phase=phase,rank=rank,local_rank=local_rank,world=world,mode=args.mode,
            timestamp=time.time(),torch_version=torch.__version__,torch_git_version=torch.version.git_version,
            cuda_version=torch.version.cuda,flags={k:os.environ.get(k) for k in FLAGS},**extra)
        save(output/f'progress_rank{rank}.json',value)
        print(json.dumps(value,sort_keys=True),flush=True)

    def sample_libraries(phase):
        libraries.append(dict(phase=phase,libraries=loaded_libraries()))
        save(output/f'loaded_libraries_rank{rank}.json',dict(rank=rank,samples=libraries))

    try:
        sample_libraries('torch_imported_maps_only')
        rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(256,5120,8,160)
        buffer = deep_ep.Buffer(dist.group.WORLD,0,rdma_bytes,low_latency_mode=True,
            num_qps_per_rank=20,allow_nvlink_for_low_latency_mode=True,explicitly_destroy=True)
        if not buffer.runtime.get_low_latency_nvlink_pair_only() or buffer.runtime.get_low_latency_pcie_p2p_max_tokens() != 0:
            raise RuntimeError('Native transport getters differ from real experiment')
        if int(os.environ.get('NVSHMEM_IBGDA_NUM_RC_PER_PE','0')) != 20:
            raise RuntimeError('Actual QP count differs from local experts=20')
        host_libraries = [p for p in mapped_library_paths(Path('/proc/self/maps').read_text()) if 'libnvshmem_host.so' in p]
        if len(host_libraries) != 1:
            raise RuntimeError('Expected exactly one already-loaded NVSHMEM host library')
        nv = ctypes.CDLL(host_libraries[0],mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
        nv.nvshmem_ptr.argtypes=[ctypes.c_void_p,ctypes.c_int]
        nv.nvshmem_ptr.restype=ctypes.c_void_p
        view = buffer.get_local_buffer_tensor(torch.uint8,use_rdma_buffer=True)
        mapped = [bool(nv.nvshmem_ptr(view.data_ptr(),peer)) for peer in range(world)]
        if mapped != [host == hosts[rank] for host in hosts]:
            raise RuntimeError('Unexpected same-node NVSHMEM peer mappings')
        del view
        save(output/f'route_rank{rank}.json',dict(rank=rank,host=hosts[rank],hosts=hosts,
            physical_gpu=local_rank,peer_mapped=mapped,rdma_bytes=rdma_bytes,policy=TRANSPORT,
            num_qps_per_rank=20,runtime_pair_only=True,runtime_pcie_threshold=0,
            mapping_scope='Addressability only: physical pairs NVLink; other peers RDMA; not hardware path tracing',
            native_path=str(native_path),native_sha256=NATIVE_SHA,wrapper=str(deep_ep.__file__)))
        channels = torch.arange(5120,device='cuda').view(1,-1)
        global_ids = torch.arange(rank*12,(rank+1)*12,device='cuda').view(-1,1)
        def data(phase):
            values = ((global_ids*7+channels*3+phase*11)%31-15)/16
            values[:,0] = global_ids[:,0]+phase*128
            return values.to(torch.bfloat16).contiguous()
        x = data(0)
        indices = torch.tensor(routes(rank),device='cuda',dtype=deep_ep.topk_idx_t).contiguous()
        weights = torch.full((12,8),.125,device='cuda',dtype=torch.float32)
        def exchange():
            recv,counts,handle,event,hook = buffer.low_latency_dispatch(x,indices,256,160,
                use_fp8=False,async_finish=False,return_recv_hook=True)
            hook()
            out,event2,hook2 = buffer.low_latency_combine(recv,indices,weights,handle,
                use_logfmt=False,zero_copy=False,async_finish=False,return_recv_hook=True)
            hook2()
            return dict(recv=recv,counts=counts,handle=handle,out=out,event=event,event2=event2)
        counts_expected = expert_counts()[rank*20:(rank+1)*20]
        def validate(row,phase,already_synchronized=False):
            if not already_synchronized:torch.cuda.synchronize()
            if row['counts'].cpu().tolist() != counts_expected:
                raise RuntimeError('Balanced expert receive counts mismatch')
            layout = row['handle'][1].cpu().tolist()
            source_ids = row['handle'][0]
            checked = 0
            for local_expert in range(20):
                expert = rank*20+local_expert
                for source in range(8):
                    packed = layout[local_expert][source]
                    begin,count = packed>>32,packed&((1<<32)-1)
                    selected = [i for i,r in enumerate(routes(source)) if expert in r]
                    actual = source_ids[local_expert,begin:begin+count].long()
                    if sorted(actual.cpu().tolist()) != selected or count != len(selected):
                        raise RuntimeError('Dispatch source IDs mismatch')
                    if count:
                        ids = actual.view(-1,1)+source*12
                        expected = ((ids*7+channels*3+phase*11)%31-15)/16
                        expected[:,0] = ids[:,0]+phase*128
                        if not torch.equal(row['recv'][local_expert,begin:begin+count],expected.to(torch.bfloat16)):
                            raise RuntimeError('Dispatch BF16 payload mismatch')
                    checked += count
            if not torch.equal(row['out'],x):
                raise RuntimeError('Weighted identity combine mismatch')
            return dict(passed=True,phase=phase,received_assignments=checked,
                source_ids_exact=True,source_payload_exact=True,combine_identity_exact=True,
                local_expert_counts=counts_expected)
        eager = exchange()
        validate(eager,0)
        for _ in range(2):
            eager = exchange()
            torch.cuda.synchronize()
        del eager
        dist.barrier()
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()
        events = {name:torch.cuda.Event(enable_timing=True,external=True) for name in EVENT_NAMES}
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=stream):
            events['graph_start'].record()
            events['dispatch_start'].record()
            recv,counts,handle,event,hook = buffer.low_latency_dispatch(x,indices,256,160,
                use_fp8=False,async_finish=False,return_recv_hook=True)
            hook()
            events['dispatch_end'].record()
            events['combine_start'].record()
            out,event2,hook2 = buffer.low_latency_combine(recv,indices,weights,handle,
                use_logfmt=False,zero_copy=False,async_finish=False,return_recv_hook=True)
            hook2()
            events['combine_end'].record()
            events['graph_end'].record()
            captured = dict(recv=recv,counts=counts,handle=handle,out=out,event=event,event2=event2)
        # Graph, origin, all timing events, input and captured metadata stay live.
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
        before = validate(captured,0)
        checkpoint('unprofiled_graph_verified',collector_lifecycles=0,external_event_count=len(events))
        sample_libraries('after_capture_no_profiler')
        samples=[]
        for index in range(3):
            phase=index+1
            x.copy_(data(phase))
            torch.cuda.synchronize()
            dist.barrier()
            if index:time.sleep(.02)  # CPU-only gap, before Graph timing start.
            checkpoint('before_external_event_graph_replay',sample=index,payload_phase=phase)
            graph.replay()
            torch.cuda.synchronize()
            # Read timestamps only after all captured work is complete.
            stamps={name:origin.elapsed_time(events[name]) for name in EVENT_NAMES}
            sample=dict(sample=index,payload_phase=phase,timestamps_from_origin_ms=stamps,
                dispatch_ms=events['dispatch_start'].elapsed_time(events['dispatch_end']),
                combine_ms=events['combine_start'].elapsed_time(events['combine_end']),
                graph_ms=events['graph_start'].elapsed_time(events['graph_end']))
            validate_event_sample(sample)
            payload=validate(captured,phase,already_synchronized=True)
            sample.update(payload_valid=payload['passed'],payload=payload)
            samples.append(sample)
            save(output/f'samples_rank{rank}.json',dict(rank=rank,samples=samples))
            checkpoint('after_external_event_graph_replay',sample=index,spans=sample)
        freshness=validate_freshness(samples)
        sample_libraries('complete_no_profiler')
        save(output/f'result_rank{rank}.json',dict(passed=True,event_valid=True,rank=rank,world=world,mode=args.mode,
            host=hosts[rank],collector_lifecycles=0,cupti_api_calls=0,prior_calibration_sessions=0,
            graph_count=1,measured_graph_replays=3,captured_external_event_count=len(events),
            event_flags=dict(enable_timing=True,external=True),samples=samples,freshness=freshness,
            before=before,native_path=str(native_path),native_sha256=NATIVE_SHA,wrapper_path=str(deep_ep.__file__),
            torch_version=torch.__version__,torch_git_version=torch.version.git_version,cuda_version=torch.version.cuda,
            elapsed_s=time.monotonic()-started,model_loaded=False,attention_compute=False,expert_compute=False,
            workload=dict(local_tokens=12,global_tokens=96,hidden=5120,experts=160,topk=8,
                capacity=256,dtype='BF16',identity_weight=.125,num_qps_per_rank=20),
            note='Actual per-rank GPU event spans including communication waits and event-node overhead; not wire-only latency, not full-model time. No CUPTI collector. CPU gaps and correctness readbacks lie outside event bounds.'))
        checkpoint('complete',event_valid=True)
    finally:
        cleanup_graph_and_buffer(graph,buffer,torch.cuda,dist)


if __name__ == '__main__':
    main()
