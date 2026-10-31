"""Initialize the exact native PP GPU P2P groups before the scheduler loop.

This performs no inference and sends nothing through the CPU metadata channel.
The final world CPU barrier is only a startup completion barrier.
"""
import hashlib
import os
from pathlib import Path
import time


def _start_ticks():
    raw = Path('/proc/self/stat').read_text()
    return int(raw[raw.rfind(')') + 2:].split()[19])


def warm(model_runner, arm, global_rank, *, torch_module=None,
         dist_module=None, world_group=None, receipt_writer=None, phase=None):
    """Warm each [TP rank, TP rank + 4] PP pair with matched native P2P.

    Optional dependencies support CPU-only tests; production uses native torch
    isend/irecv on the same device_group as send_tensor_dict, not PyNCCL.
    Each rank sends once and receives once, then all eight startup ranks join
    a Gloo barrier before any successful receipt is written.
    """
    if arm != 'pp2ep4' or type(global_rank) is not int or global_rank not in range(8):
        raise ValueError('PP P2P warmup requires the exact PP2/TP4 eight-rank layout')
    phase = phase or os.environ['BANK_PHASE']
    if not (phase in ('pp2ep4_build', 'pp2ep4_restore', 'pp2ep4_measure') or (phase.startswith('pp2ep4_b') and phase.endswith('_measure'))):
        raise ValueError('PP P2P warmup requires an owned PP bank phase')
    if int(os.environ['NODE_RANK']) != global_rank // 4:
        raise ValueError('PP P2P warmup node/global rank mismatch')
    if torch_module is None:
        import torch as torch_module
    if dist_module is None:
        dist_module = torch_module.distributed
    if world_group is None:
        from sglang.srt.distributed.parallel_state import get_world_group
        world_group = get_world_group()
    if receipt_writer is None:
        from common import save
        receipt_writer = save

    pp = model_runner.pp_group
    local_rank = global_rank % 4
    ranks = (local_rank, local_rank + 4)
    if (pp.world_size != 2 or tuple(pp.ranks) != ranks or
            pp.rank_in_group != global_rank // 4 or
            dist_module.get_rank() != global_rank or
            dist_module.get_world_size() != 8 or
            dist_module.get_rank(group=pp.device_group) != global_rank // 4 or
            dist_module.get_world_size(group=pp.device_group) != 2 or
            tuple(dist_module.get_process_group_ranks(pp.device_group)) != ranks or
            str(dist_module.get_backend(pp.device_group)) != 'nccl' or
            tuple(world_group.ranks) != tuple(range(8)) or
            dist_module.get_world_size(group=world_group.cpu_group) != 8 or
            str(dist_module.get_backend(world_group.cpu_group)) != 'gloo'):
        raise ValueError('PP P2P warmup native group/rank/backend mismatch')
    device = torch_module.device(model_runner.device)
    if (device.type != 'cuda' or torch_module.cuda.current_device() != local_rank or
            device.index not in (None, local_rank)):
        raise ValueError('PP P2P warmup is not on this rank owned CUDA device')

    started = time.monotonic()
    sent = torch_module.full((1,), float(global_rank + 1),
                             dtype=torch_module.bfloat16, device=device)
    received = torch_module.empty((1,), dtype=torch_module.bfloat16, device=device)
    if (sent.numel() != 1 or received.numel() != 1 or
            sent.dtype != torch_module.bfloat16 or received.dtype != torch_module.bfloat16 or
            sent.device.type != 'cuda' or received.device.type != 'cuda'):
        raise RuntimeError('PP P2P warmup must use nonempty native BF16 CUDA tensors')
    operations = []
    # Both ranks enter the first direction together, before either scheduler
    # may wait for another request. The reverse covers the output direction.
    for src, dst in (ranks, ranks[::-1]):
        sending = global_rank == src
        if sending:
            work = dist_module.isend(sent, dst=dst, group=pp.device_group)
        else:
            work = dist_module.irecv(received, src=src, group=pp.device_group)
        if work.wait() is not True:
            raise RuntimeError('Native PP P2P work did not complete successfully')
        operations.append(dict(src=src, dst=dst,
            local_operation='send' if sending else 'receive',
            tensor_elements=1, tensor_bytes=2, native_work_completed=True))
    torch_module.cuda.synchronize(device=device)
    expected = ranks[1 - pp.rank_in_group] + 1
    actual = received.item()
    if actual != expected:
        raise RuntimeError('Native PP P2P received sentinel mismatch')
    # No send_object/recv_object or request/proxy metadata is injected here.
    dist_module.barrier(group=world_group.cpu_group)
    receipt = dict(passed=True, rank=global_rank, node_rank=global_rank // 4,
        local_rank=local_rank, pp_rank=pp.rank_in_group, bank_phase=phase,
        pid=os.getpid(), process_start_ticks=_start_ticks(), timestamp=time.time(),
        source_path=str(Path(__file__).resolve()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pp_group_ranks=list(ranks), device_group_backend='nccl',
        operation_api='torch.distributed.isend/irecv',
        tensor_dtype='bfloat16', tensor_elements=1,
        operations=operations, received_value=actual, expected_received_value=expected,
        received_sentinel_verified=True, cuda_synchronized=True,
        world_cpu_barrier_completed=True, world_cpu_barrier_backend='gloo',
        cpu_metadata_messages_sent=False, pynccl_used=False,
        before_pp_event_loop=True, inference_performed=False,
        performance_metric=False, elapsed_startup_s=time.monotonic() - started)
    receipt_writer(f'{arm}/pp_p2p_warm_rank{global_rank}.json', receipt)
    return receipt
