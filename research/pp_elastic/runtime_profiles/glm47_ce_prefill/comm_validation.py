import ctypes, os
from pathlib import Path
LOCAL_GPUS, EXPERTS, HIDDEN, TOPK = 4, 160, 5120, 8
WORLD = LOCAL_EXPERTS = EP_CAPACITY = PREFILL_DISPATCH_ROWS = 0

def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)

def validate_topology_rows(rows):
    assert len(rows) == WORLD
    base_node_rank = rows[0]['node_rank']
    assert isinstance(base_node_rank, int) and base_node_rank >= 0
    hosts = [row['host'] for row in rows]
    assert len(set(hosts)) == WORLD // LOCAL_GPUS
    for start in range(0, WORLD, LOCAL_GPUS):
        block = rows[start:start + LOCAL_GPUS]
        assert len({row['host'] for row in block}) == 1
        assert len({row['uuid'] for row in block}) == LOCAL_GPUS
    for rank, row in enumerate(rows):
        assert row['rank'] == rank and row['local_rank'] == rank % LOCAL_GPUS
        assert row['node_rank'] == base_node_rank + rank // LOCAL_GPUS
        assert 'A100' in row['name']
        assert row['nvlink_local_peers'] == [
            peer in (rank % LOCAL_GPUS, (rank % LOCAL_GPUS) ^ 1)
            for peer in range(LOCAL_GPUS)]
    return rows

def pair_topology(group):
    import torch.distributed as dist
    import pynvml
    rank, world = group.rank(), group.size()
    assert world == WORLD
    local_rank = torch.cuda.current_device()
    assert local_rank == rank % LOCAL_GPUS
    visible = [int(value) for value in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(visible) == len(set(visible)) == LOCAL_GPUS
    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(value) for value in visible]
        handle = handles[local_rank]
        links = [other == local_rank or pynvml.nvmlDeviceGetP2PStatus(
            handle, peer, pynvml.NVML_P2P_CAPS_INDEX_NVLINK) == pynvml.NVML_P2P_STATUS_OK
            for other, peer in enumerate(handles)]
        local = dict(rank=rank, local_rank=local_rank,
            node_rank=int(os.environ['NODE_RANK']), host=os.uname().nodename.split('.')[0],
            physical_gpu=visible[local_rank], visible=visible,
            uuid=text(pynvml.nvmlDeviceGetUUID(handle)),
            name=text(pynvml.nvmlDeviceGetName(handle)), nvlink_local_peers=links)
    finally:
        pynvml.nvmlShutdown()
    rows = [None] * world
    dist.all_gather_object(rows, local, group=group)
    return validate_topology_rows(rows)

def routing_ids(source, n):
    return [[((slot + source) % WORLD) * LOCAL_EXPERTS
             + ((token * TOPK + slot + source * 3) % LOCAL_EXPERTS)
             for slot in range(TOPK)] for token in range(n)]

def routing_counts(rank, rows_per_source):
    counts = [0] * LOCAL_EXPERTS
    for source, n in enumerate(rows_per_source):
        for ids in routing_ids(source, n):
            for expert in ids:
                if expert // LOCAL_EXPERTS == rank:
                    counts[expert % LOCAL_EXPERTS] += 1
    return counts

def peer_mapping(buffer):
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'libnvshmem_host.so' in line and '/' in line.split()[-1]}
    assert len(paths) == 1, paths
    nv = ctypes.CDLL(next(iter(paths)))
    nv.nvshmem_my_pe.restype = nv.nvshmem_n_pes.restype = ctypes.c_int
    assert nv.nvshmem_my_pe() == buffer.rank and nv.nvshmem_n_pes() == WORLD
    nv.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    nv.nvshmem_ptr.restype = ctypes.c_void_p
    view = buffer.get_local_buffer_tensor(torch.uint8, use_rdma_buffer=True)
    mapped = [bool(nv.nvshmem_ptr(view.data_ptr(), peer)) for peer in range(WORLD)]
    assert mapped == [peer // LOCAL_GPUS == buffer.rank // LOCAL_GPUS for peer in range(WORLD)]
    return mapped

def check_communication(buffer, group, capture=False):
    """Untimed startup-only checks, all ranks active even with an empty source."""
    import torch.distributed as dist
    assert not torch.cuda.is_current_stream_capturing()
    rank = buffer.rank
    cases = []

    def make_inputs(rows_per_source):
        n = rows_per_source[rank]
        ids = torch.tensor(routing_ids(rank, n), dtype=torch.int64, device='cuda').reshape(n, TOPK)
        weights = torch.full((n, TOPK), 1 / TOPK, device='cuda')
        # Dyadic BF16 values make reference rounding reproducible.
        x = (torch.arange(HIDDEN, device='cuda').remainder(17)[None, :] / 128
             + rank / 32 + torch.arange(n, device='cuda')[:, None] / 128).to(torch.bfloat16)
        return x.contiguous(), ids, weights, routing_counts(rank, rows_per_source)

    def pipeline(x, ids, weights, counts):
        recv, got_counts, handle, _, hook = buffer.low_latency_dispatch(
            x, ids, EP_CAPACITY, EXPERTS, use_fp8=False, return_recv_hook=True)
        hook()
        transformed = torch.empty_like(recv)
        for expert, count in enumerate(counts):
            if count:
                transformed[expert, :count] = recv[expert, :count] * 2 + (rank * LOCAL_EXPERTS + expert) / 16
        combined, _, hook = buffer.low_latency_combine(
            transformed, ids, weights, handle, return_recv_hook=True)
        hook()
        return combined, got_counts

    def verify(out, got_counts, x, ids, weights, counts):
        torch.cuda.synchronize()
        assert got_counts.cpu().tolist() == counts
        expected = (x.float() * 2 + (ids.float() / 16 * weights).sum(-1, keepdim=True)).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, rtol=0, atol=0.0625)
        return float((out.float() - expected.float()).abs().max()) if out.numel() else 0.0

    for name, rows in [('balanced1', [1] * WORLD), ('balanced_prefill_capacity', [PREFILL_DISPATCH_ROWS] * WORLD),
                       ('empty0_decode', [0] + [1] * (WORLD - 1))]:
        dist.barrier(group=group)
        x, ids, weights, counts = make_inputs(rows)
        out, got_counts = pipeline(x, ids, weights, counts)
        error = verify(out, got_counts, x, ids, weights, counts)
        cases.append(dict(case=name, local_rows=x.shape[0], counts=counts,
                          passed=True, max_abs_error=error))
        del out, got_counts, x, ids, weights
    # The CE adapter performs host count work and cannot be captured. Exercise
    # changed payload and unequal weights in eager mode for buffer reuse.
    x, ids, weights, counts = make_inputs([1] * WORLD)
    weights.copy_(torch.arange(1, TOPK + 1, device='cuda', dtype=torch.float32)[None, :] / 36)
    for step in range(3):
        if step:
            x.add_(0.25)
        dist.barrier(group=group)
        out, got_counts = pipeline(x, ids, weights, counts)
        error = verify(out, got_counts, x, ids, weights, counts)
        cases.append(dict(case=f'eager_changed_unequal_weights_{step}', local_rows=1,
                          passed=True, max_abs_error=error))
    del out, got_counts, x, ids, weights
    assert capture is False, 'CE serving protocol requires eager startup correctness'
    buffer.clean_low_latency_buffer(EP_CAPACITY, HIDDEN, EXPERTS)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    return dict(passed=True, rank=rank, startup_only=True, timed=False, cases=cases,
                independent_cpu_routing_reference=True,
                graph_captured=False, eager_changed_input=True, unequal_weights=True, actual_model_compute_checked=False)
