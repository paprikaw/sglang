"""Actual NCCL P2P plus profiler/Graph lifecycle; no model/A/E computation.

Each mode is a fresh torchrun process. Rank 0 alone profiles six existing
communication warmups, as do the model's calibrated leaders. Rank 1 executes
the same communication/Graph schedule without calibration profilers. Both
then collect CPU+CUDA bootstrap and CUDA-only replay/marker evidence.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from common import FLAGS, MODES, save, trace_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=MODES)
    args = parser.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    # Flags must be supplied to a fresh process before the first Torch import.
    initial_flags = {key: os.environ.get(key) for key in FLAGS}
    import torch
    import torch.distributed as dist
    from torch.profiler import ProfilerActivity, profile, record_function

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    output = args.run / args.mode
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=cfg['worker_timeout_s']))
    if dist.get_world_size() != 2:
        raise RuntimeError('Exactly two local GPUs required')
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    graphs, buffers, traces, validations = [], [], [], []
    started = time.monotonic()

    def capture_trace(label, body, selected_activities=activities):
        torch.cuda.synchronize()
        with profile(activities=selected_activities, record_shapes=False,
                     profile_memory=False, with_stack=False) as collector:
            body()
        path = output / f'{label}_rank{rank}.json'
        collector.export_chrome_trace(str(path))
        counts = trace_counts(json.loads(path.read_text()))
        row = dict(label=label, path=str(path), rank=rank, **counts)
        traces.append(row)
        print(json.dumps(dict(mode=args.mode, rank=rank, phase=label,
            gpu_events=counts['gpu_events'], launches=counts['cuda_launches'],
            graph_launches=counts['cuda_graph_launches'], nccl_kernels=counts['observed_nccl_kernels']), sort_keys=True), flush=True)

    try:
        for capture_index, elements in enumerate(cfg['payload_elements']):
            send = torch.full((elements,), rank + 1, device='cuda', dtype=torch.bfloat16)
            recv = torch.empty_like(send)
            buffers.append((send, recv))

            def exchange():
                operations = [dist.P2POp(dist.isend, send, 1-rank),
                              dist.P2POp(dist.irecv, recv, 1-rank)]
                for work in dist.batch_isend_irecv(operations):
                    work.wait()

            # Real activation P2P also initializes NCCL before any Graph capture.
            exchange()
            torch.cuda.synchronize()
            if not bool((recv == (2-rank)).all().item()):
                raise RuntimeError('NCCL warmup payload mismatch')
            for warm in range(2):
                torch.cuda.synchronize()
                dist.barrier()
                label = f'calibration_graph{capture_index}_warm{warm}'
                if rank == 0:
                    capture_trace(label, exchange)
                else:
                    exchange()
                torch.cuda.synchronize()
                dist.barrier()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                exchange()
            graph.replay()
            torch.cuda.synchronize()
            if not bool((recv == (2-rank)).all().item()):
                raise RuntimeError('NCCL captured replay payload mismatch')
            validations.append(dict(capture_index=capture_index, elements=elements,
                dtype='bfloat16', one_way_payload_bytes=2*elements,
                bidirectional_payload_verified=True))
            graphs.append(graph)
            dist.barrier()

        def markers():
            prime = torch.empty(1, dtype=torch.int32, device='cuda')
            prime.fill_(0)
            torch.cuda.synchronize()
            for index in range(3):
                with record_function(f'stage-marker-proof:{index}'):
                    torch.cuda._sleep(100000)
                torch.cuda.synchronize()

        dist.barrier()
        capture_trace('post_capture_cpu_cuda_markers', markers)
        dist.barrier()

        def cuda_only():
            graphs[-1].replay()
            markers()

        capture_trace('post_capture_cuda_only_replay_markers', cuda_only, [ProfilerActivity.CUDA])
        dist.barrier()
        # Validation is outside profiler and not part of measured communication.
        verified = all(bool((recv == (2-rank)).all().item()) for _, recv in buffers)
        if not verified:
            raise RuntimeError('Final replay payload mismatch')
        save(output / f'result_rank{rank}.json', dict(passed=True, mode=args.mode, rank=rank,
            local_rank=local_rank, node=os.uname().nodename, initial_flags=initial_flags,
            final_flags={key: os.environ.get(key) for key in FLAGS},
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            nccl_version=torch.cuda.nccl.version(), elapsed_s=time.monotonic()-started,
            calibration_leader=(rank == 0), prior_calibration_sessions=6 if rank == 0 else 0,
            graph_count=len(graphs), validations=validations, traces=traces,
            no_model=True, expert_compute=False, attention_compute=False,
            note='passed means real NCCL payload correctness; missing GPU collector events are a diagnostic result, not silently accepted profiling'))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
