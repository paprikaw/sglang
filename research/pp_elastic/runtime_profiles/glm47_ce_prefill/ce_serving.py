"""Install the experimental eager CE transport on the singleton DeepEP buffer.

This adapter allocates staging before SGLang's untimed communication checks.
Model routing and the masked Humming expert layout remain the native API's own.
"""
import ctypes
import inspect
import json
import os
import time
import threading
from pathlib import Path

_INSTALLED = None


def geometry(world, capacity, experts=160, hidden=5120):
    if world not in (4, 8) or experts != 160 or hidden != 5120 or not 0 < capacity <= 2048:
        raise ValueError("Unsupported CE geometry")
    local = experts // world
    header = 16 + ((local * 4 + 15) // 16) * 16
    frame = header + capacity * local * (hidden * 2 + 32)
    extra = ((world * frame + 2 * world * 128 + 127) // 128) * 128
    return dict(local_experts=local, header_bytes=header, frame_bytes=frame, extra_bytes=extra)


def snapshot(buffer):
    api = getattr(buffer, "_ce_api", None)
    if api is None:
        return None
    counters = (ctypes.c_uint64 * 6)()
    epoch = api.deepep_ce_snapshot(counters)
    if epoch < 0:
        raise RuntimeError("CE is not configured")
    return dict(epoch=epoch, counters=list(counters))


def register_counter_listener(run,arm,rank,buffer,ticks,save_fn,snapshot_fn=snapshot):
    """Host-only snapshots after matching post-formal requests, never live stats."""
    from counter_socket import start_listener
    from common import sha
    def receive(request):
        received_at=time.time()
        marker=json.loads((Path(run)/arm/'CE_SNAPSHOT_REQUEST.json').read_text())
        if (marker.get('arm')!=arm or marker.get('request_id')!=request.get('request_id')
                or not marker.get('after_formal_requests') or not marker.get('no_other_model_requests')
                or not request.get('after_formal_requests')
                or (request.get('rank'),request.get('pid'),request.get('process_start_ticks'))!=(rank,os.getpid(),ticks)):
            raise RuntimeError('CE socket lacks a matching quiescent post-formal request')
        # Native counters are plain host uint64 values, not concurrent-statistics
        # atomics. The client has completed all HTTP requests before the marker;
        # no subsequent model requests run. No Torch/CUDA call occurs here.
        counters=snapshot_fn(buffer)
        if counters!=snapshot_fn(buffer):
            raise RuntimeError('CE counters changed during post-formal snapshot')
        receipt=dict(passed=True,rank=rank,pid=os.getpid(),process_start_ticks=ticks,
            counters=counters,request_id=marker['request_id'],received_at=received_at,completed_at=time.time(),
            source='unix_socket_listener',listener_thread_id=threading.get_ident(),
            after_formal_requests=True,cpu_only=True,timed=False)
        path=Path(run)/arm/f'ce_final_rank{rank}.json';save_fn(path,receipt)
        return dict(passed=True,receipt=receipt,receipt_sha256=sha(path))
    return start_listener(run,arm,rank,ticks,receive)


def install(run, arm):
    global _INSTALLED
    if _INSTALLED == os.getpid():
        return
    from common import case, native_info, save
    selected = case(arm, run)
    if selected.get("transport", "ce") != "ce":
        _INSTALLED = os.getpid()
        return
    import deep_ep
    import torch
    import torch.distributed as dist

    native = native_info(arm, run)
    original = deep_ep.Buffer.__init__
    signature = inspect.signature(original)
    capacity, world = selected["ep_capacity"], selected["ep"]
    g = geometry(world, capacity)

    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["low_latency_mode"] is not True:
            raise ValueError("CE serving adapter requires low_latency")
        group = bound.arguments["group"]
        if group.size() != world:
            raise ValueError("CE group size differs from the serving contract")
        base_bytes = bound.arguments["num_rdma_bytes"]
        if base_bytes % 128 or base_bytes < deep_ep.Buffer.get_low_latency_rdma_size_hint(capacity,5120,world,160):
            raise ValueError("CE scratch must follow the full aligned DeepEP LL layout")
        bound.arguments["num_rdma_bytes"] = base_bytes + g["extra_bytes"]
        original(*bound.args, **bound.kwargs)
        view = self.get_local_buffer_tensor(torch.uint8, use_rdma_buffer=True)
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "libnvshmem_host.so" in line and "/" in line.split()[-1]}
        if len(paths) != 1:
            raise RuntimeError("Expected exactly one active NVSHMEM host library")
        nv = ctypes.CDLL(next(iter(paths)))
        nv.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        nv.nvshmem_ptr.restype = ctypes.c_void_p
        raw_peers = [int(nv.nvshmem_ptr(view.data_ptr(), peer) or 0) for peer in range(world)]
        expected_mapping = [peer // 4 == self.rank // 4 for peer in range(world)]
        if [bool(ptr) for ptr in raw_peers] != expected_mapping:
            raise RuntimeError("NVSHMEM peer mapping differs from the selected placement")
        view[base_bytes:base_bytes + g["extra_bytes"]].zero_()
        tensors = [torch.empty(world * g["frame_bytes"], device="cuda", dtype=torch.uint8),
                   torch.empty(world, device="cuda", dtype=torch.int32),
                   torch.empty(160, device="cuda", dtype=torch.int32),
                   torch.empty(world, dtype=torch.int32, pin_memory=True)]
        api = ctypes.CDLL(native["path"])
        api.deepep_ce_configure_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        api.deepep_ce_configure_v2.restype = ctypes.c_int
        peers = (ctypes.c_uint64 * world)(*[ptr + base_bytes if ptr else 0 for ptr in raw_peers])
        torch.cuda.synchronize()
        rc = api.deepep_ce_configure_v2(tensors[0].data_ptr(), view.data_ptr() + base_bytes,
            peers, tensors[1].data_ptr(), tensors[2].data_ptr(), tensors[3].data_ptr(),
            capacity * g["local_experts"], self.rank, world, capacity)
        if rc:
            raise RuntimeError(f"CE configuration failed: {rc}")
        settings = dict(dispatch_dedup=1, matched_peer_order=1,
                        skip_local_drain=int(world == 4), local_notify=1)
        for name, enabled in settings.items():
            function = getattr(api, "deepep_ce_set_" + name)
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_int
            rc = function(enabled)
            if rc:
                raise RuntimeError(f"CE setting {name}={enabled} failed: {rc}")
        api.deepep_ce_snapshot.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        api.deepep_ce_snapshot.restype = ctypes.c_int
        # Every backing tensor outlives all calls and receive hooks on this buffer.
        self._ce_api, self._ce_tensors, self._ce_view = api, tensors, view
        self._ce_geometry, self._ce_settings = g, settings
        self._ce_calls = 0
        dist.barrier(group=group)
        global_rank = int(os.environ["NODE_RANK"]) * 4 + self.rank if selected["pp"] > 1 else self.rank
        stat_text = Path(f"/proc/{os.getpid()}/stat").read_text()
        ticks = int(stat_text[stat_text.rfind(")") + 2:].split()[19])
        self._ce_counter_listener=register_counter_listener(run,arm,global_rank,self,ticks,save)
        save(Path(run) / arm / f"ce_config_rank{global_rank}.json", dict(
            passed=True, rank=global_rank, local_ep_rank=self.rank, world=world,
            pid=os.getpid(), process_start_ticks=ticks,counter_socket=str(self._ce_counter_listener.path),
            counter_socket_protocol=1,counter_socket_uid=os.getuid(),counter_socket_mode=0o600,
            counter_socket_directory_mode=0o700,
            counter_collection='post-formal Unix socket background listener',
            native=native, geometry=g, capacity=capacity, settings=settings,
            peer_mapping=expected_mapping, cuda_graph=False, startup_only=True,
            extra_symmetric_bytes=g["extra_bytes"], send_staging_bytes=world*g["frame_bytes"],
            cross_node_payload="RDMA" if world == 8 else "none",
            cross_pair_payload="CE PCIe", counters=snapshot(self)))

    deep_ep.Buffer.__init__ = init
    original_dispatch=deep_ep.Buffer.low_latency_dispatch
    dispatch_signature=inspect.signature(original_dispatch)
    def dispatch(self,*args,**kwargs):
        bound=dispatch_signature.bind(self,*args,**kwargs)
        if bound.arguments['x'].shape[0]>capacity or bound.arguments['topk_idx'].shape[0]>capacity:
            raise ValueError("Source rows exceed the rank-consistent CE threshold")
        if bound.arguments['x'].shape[1]!=5120 or bound.arguments['num_experts']!=160:
            raise ValueError("CE accepts only BF16 GLM H5120/E160")
        return original_dispatch(self,*args,**kwargs)
    deep_ep.Buffer.low_latency_dispatch=dispatch
    _INSTALLED = os.getpid()
