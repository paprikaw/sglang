"""First CUPTI collector after capture: same real NCCL Graph, two activity modes.

No model, weights, Attention, Expert or earlier calibration/profiler lifecycle.
Crash evidence is saved before every Graph replay; the supervisor owns cleanup.
"""
import argparse
import ctypes
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import time

from common import FLAGS, MODES, save, trace_counts


def cleanup_graphs_and_group(graphs, cuda, dist):
    """Release captured NCCL graph references before group shutdown."""
    cuda.synchronize()
    for graph in graphs:
        graph.reset()
    graphs.clear()
    dist.barrier()
    cuda.synchronize()
    dist.destroy_process_group()


def mapped_library_paths(text):
    names = ('libcupti', 'libcudart', 'libcuda.so', 'libtorch_cuda', 'libkineto')
    return sorted({line.split(maxsplit=5)[-1].removesuffix(' (deleted)')
        for line in text.splitlines() if len(line.split(maxsplit=5)) == 6
        and line.split(maxsplit=5)[-1].startswith('/')
        and any(name in line.split(maxsplit=5)[-1] for name in names)})


def loaded_libraries():
    """Query already mapped library handles only; never choose a pip package."""
    rows = []
    for path in mapped_library_paths(Path('/proc/self/maps').read_text()):
        row = dict(path=path, realpath=os.path.realpath(path))
        if 'libcupti' in Path(path).name:
            try:
                library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
                value = ctypes.c_uint32()
                function = library.cuptiGetVersion
                function.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
                function.restype = ctypes.c_int
                row.update(cupti_get_version_rc=function(ctypes.byref(value)), cupti_version=value.value)
            except Exception as error:
                row['version_query_error'] = type(error).__name__ + ': ' + str(error)
        rows.append(row)
    return rows


def profile_valid(mode, counts):
    """Raw Graph/kernel coverage is mandatory; CUDA-only has no CPU scopes."""
    coverage = (counts['cuda_graph_launches'] >= 3
                and counts['observed_nccl_kernels'] >= 4
                and counts['observed_spin_kernels'] >= 3)
    if mode == 'cpu_cuda_after_capture':
        return coverage and len(counts['scoped_proofs']) == 3 and all(
            p['scope_count'] == 1 and len(p['kernels']) == 1 for p in counts['scoped_proofs'])
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=MODES)
    args = parser.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    initial_flags = {key: os.environ.get(key) for key in FLAGS}
    faulthandler.enable(all_threads=True)
    import torch
    import torch.distributed as dist
    from torch.profiler import ProfilerActivity, profile, record_function

    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    output = args.run / args.mode
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=cfg['worker_timeout_s']))
    if dist.get_world_size() != 2:
        raise RuntimeError('Exactly two local GPUs required')
    graphs, libraries = [], []
    started = time.monotonic()

    def checkpoint(phase, **extra):
        row = dict(phase=phase, rank=rank, mode=args.mode, timestamp=time.time(),
            initial_flags=initial_flags, torch_version=torch.__version__,
            torch_git_version=torch.version.git_version, cuda_version=torch.version.cuda, **extra)
        save(output / f'progress_rank{rank}.json', row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def sample_libraries(phase):
        libraries.append(dict(phase=phase, libraries=loaded_libraries()))
        save(output / f'loaded_libraries_rank{rank}.json', dict(rank=rank, samples=libraries))

    try:
        sample_libraries('torch_imported')
        send = torch.full((cfg['payload_elements'],), rank + 1, device='cuda', dtype=torch.bfloat16)
        recv = torch.empty_like(send)

        def exchange():
            operations = [dist.P2POp(dist.isend, send, 1-rank), dist.P2POp(dist.irecv, recv, 1-rank)]
            for work in dist.batch_isend_irecv(operations):
                work.wait()

        for _ in range(2):
            exchange()
            torch.cuda.synchronize()
        if not bool((recv == (2-rank)).all().item()):
            raise RuntimeError('NCCL eager payload mismatch')
        graph = torch.cuda.CUDAGraph()
        graphs.append(graph)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            exchange()
        # v7 ordering: capture and normal replay before the first collector.
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
        if not bool((recv == (2-rank)).all().item()):
            raise RuntimeError('Unprofiled Graph payload mismatch')
        checkpoint('unprofiled_graph_verified', prior_profiler_sessions=0)
        sample_libraries('after_capture_before_first_profiler')
        dist.barrier()
        torch.cuda.synchronize()
        activities = [ProfilerActivity.CUDA]
        if args.mode == 'cpu_cuda_after_capture':
            activities.insert(0, ProfilerActivity.CPU)
        collector = profile(activities=activities, record_shapes=False,
                            profile_memory=False, with_stack=False, with_modules=False)
        collector.start()
        time.sleep(.25)
        sample_libraries('first_profiler_started')
        prime = torch.empty(1, device='cuda', dtype=torch.int32)
        prime.fill_(0)
        torch.cuda.synchronize()
        for index in range(3):
            with record_function(f'stage-marker-proof:{index}'):
                torch.cuda._sleep(100000000)
            torch.cuda.synchronize()
        checkpoint('idle_marker_proofs_submitted')
        # The actual model did eager Extend before its first target Graph replay.
        with record_function('diagnostic:eager_p2p_before_graph'):
            exchange()
        torch.cuda.synchronize()
        sample_libraries('after_eager_before_first_profiled_graph')
        for index in range(cfg['profiled_graph_replays']):
            checkpoint('before_profiled_graph_replay', graph_replay_index=index)
            with record_function(f'diagnostic:captured_p2p_replay:{index}'):
                graph.replay()
            torch.cuda.synchronize()
            checkpoint('after_profiled_graph_replay', graph_replay_index=index)
        collector.stop()
        trace_path = output / f'activity_rank{rank}.json'
        collector.export_chrome_trace(str(trace_path))
        counts = trace_counts(json.loads(trace_path.read_text()))
        sample_libraries('collector_stopped')
        verified = bool((recv == (2-rank)).all().item())
        if not verified:
            raise RuntimeError('Final captured P2P payload mismatch')
        valid = profile_valid(args.mode, counts)
        save(output / f'result_rank{rank}.json', dict(passed=True, mode=args.mode,
            rank=rank, local_rank=local_rank, node=os.uname().nodename,
            initial_flags=initial_flags, final_flags={key: os.environ.get(key) for key in FLAGS},
            torch_version=torch.__version__, torch_git_version=torch.version.git_version,
            cuda_version=torch.version.cuda, nccl_version=torch.cuda.nccl.version(),
            elapsed_s=time.monotonic()-started, prior_calibration_sessions=0,
            collector_lifecycles=1, graph_count=1,
            profile_valid=valid, cpu_marker_signature_available=args.mode == 'cpu_cuda_after_capture',
            raw_gpu_proof=counts, loaded_library_evidence=str(output / f'loaded_libraries_rank{rank}.json'),
            bidirectional_payload_verified=verified, one_way_payload_bytes=2*cfg['payload_elements'],
            trace_path=str(trace_path), no_model=True, expert_compute=False, attention_compute=False,
            note='passed proves NCCL payload; profile_valid separately requires real Graph launches and kernel activity. No model latency.'))
        checkpoint('complete', profile_valid=valid, **counts)
    finally:
        cleanup_graphs_and_group(graphs, torch.cuda, dist)


if __name__ == '__main__':
    main()
