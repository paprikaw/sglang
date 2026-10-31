"""Minimal real EP8 LL Graph: first CPU+CUDA / CUDA / kernel-only collector.

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

from common import (MODES, FLAGS, NATIVE, NATIVE_SHA, DEEP, TRANSPORT, activity_config,
                    expert_counts, routes, save, sha, trace_counts, profile_valid)


def mapped_library_paths(text):
    names = ('libcupti','libcudart','libcuda.so','libtorch_cuda','libkineto','libnvshmem','deep_ep_cpp')
    return sorted({line.split(maxsplit=5)[-1].removesuffix(' (deleted)')
        for line in text.splitlines() if len(line.split(maxsplit=5)) == 6
        and line.split(maxsplit=5)[-1].startswith('/')
        and any(name in line.split(maxsplit=5)[-1] for name in names)})


def loaded_libraries(query_versions=False):
    """Read maps before capture; query CUPTI only AFTER profiler.start()."""
    rows = []
    for path in mapped_library_paths(Path('/proc/self/maps').read_text()):
        row = dict(path=path, realpath=os.path.realpath(path))
        if query_versions and 'libcupti' in Path(path).name:
            try:
                library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
                value = ctypes.c_uint32()
                function = library.cuptiGetVersion
                function.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
                function.restype = ctypes.c_int
                row.update(cupti_get_version_rc=function(ctypes.byref(value)),cupti_version=value.value)
            except Exception as error:
                row['version_query_error'] = type(error).__name__ + ': ' + str(error)
        rows.append(row)
    return rows


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
    from torch.profiler import ProfilerActivity, profile, record_function

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

    def sample_libraries(phase,query_versions=False):
        libraries.append(dict(phase=phase,query_versions=query_versions,libraries=loaded_libraries(query_versions)))
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
        def validate(row,phase):
            torch.cuda.synchronize()
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
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph,stream=stream):
            captured = exchange()
        # Retain all captured pointers, metadata and graph inputs for full lifetime.
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
        before = validate(captured,0)
        checkpoint('unprofiled_graph_verified',prior_profiler_sessions=0)
        sample_libraries('after_capture_before_first_profiler_maps_only')
        dist.barrier()
        torch.cuda.synchronize()
        activities = activity_config(args.mode,ProfilerActivity)
        collector = profile(activities=activities,record_shapes=False,profile_memory=False,
                            with_stack=False,with_modules=False)
        collector.start()
        time.sleep(.25)
        sample_libraries('first_profiler_started',query_versions=True)
        prime = torch.empty(1,device='cuda',dtype=torch.int32)
        prime.fill_(0)
        torch.cuda.synchronize()
        for index in range(3):
            with record_function(f'stage-marker-proof:{index}'):
                torch.cuda._sleep(100000000)
            torch.cuda.synchronize()
        checkpoint('idle_marker_proofs_submitted')
        with record_function('diagnostic:eager_deepep_before_graph'):
            eager = exchange()
        torch.cuda.synchronize()
        del eager
        sample_libraries('after_eager_before_first_profiled_graph',query_versions=True)
        for index in range(3):
            checkpoint('before_profiled_graph_replay',graph_replay_index=index)
            with record_function('batch:diagnostic'):
                with record_function(f'target:VERIFY:{index}'):
                    torch.cuda._sleep(1)
                    graph.replay()
                    torch.cuda._sleep(1)
            torch.cuda.synchronize()
            checkpoint('after_profiled_graph_replay',graph_replay_index=index)
        collector.stop()
        path = output/f'activity_rank{rank}.json'
        collector.export_chrome_trace(str(path))
        counts = trace_counts(json.loads(path.read_text()))
        after = validate(captured,0)
        sample_libraries('collector_stopped',query_versions=True)
        x.copy_(data(1))
        graph.replay()
        changed = validate(captured,1)
        valid = profile_valid(args.mode,counts)
        save(output/f'result_rank{rank}.json',dict(passed=True,rank=rank,world=world,mode=args.mode,
            host=hosts[rank],profile_valid=valid,collector_lifecycles=1,prior_calibration_sessions=0,
            graph_count=1,profiled_graph_replays=3,raw_gpu_proof=counts,trace_path=str(path),
            cpu_marker_signature_available=args.mode==MODES[0],
            runtime_driver_callbacks_expected=args.mode!=MODES[2],
            before=before,after=after,changed_payload_graph=changed,
            native_path=str(native_path),native_sha256=NATIVE_SHA,wrapper_path=str(deep_ep.__file__),
            torch_version=torch.__version__,torch_git_version=torch.version.git_version,cuda_version=torch.version.cuda,
            elapsed_s=time.monotonic()-started,model_loaded=False,attention_compute=False,expert_compute=False,
            workload=dict(local_tokens=12,global_tokens=96,hidden=5120,experts=160,topk=8,
                capacity=256,dtype='BF16',identity_weight=.125,num_qps_per_rank=20),
            note='Diagnostic only. Real payload and raw GPU kernel coverage; filtered mode intentionally has no runtime correlation. No full-model latency or A/E attribution.'))
        checkpoint('complete',profile_valid=valid)
    finally:
        cleanup_graph_and_buffer(graph,buffer,torch.cuda,dist)


if __name__ == '__main__':
    main()
