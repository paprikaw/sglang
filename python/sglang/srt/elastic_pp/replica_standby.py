"""Keep a fixed PP2 Decode engine ready with all model-weight pages unmapped.

Only the private two-stage startup controller publishes the one-shot activation.
Neither rank enters its serving loop until both have restored their own stage.
"""

import os
from pathlib import Path
import secrets
import time

import torch

from sglang.srt.elastic_pp.weight_backing import LayerWeightBacking, weight_storage_plan
from sglang.srt.weight_cache.replica_preparation import (
    EXPERT_DTYPES,
    packed_expert_views,
    preparation_key,
    publish,
    read_packet,
)


def byte_view(tensor):
    storage = tensor.untyped_storage()
    return torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(
        storage, 0, (storage.nbytes(),)
    )


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def memory_usage():
    free, total = torch.cuda.mem_get_info()
    rss_kib = next(
        int(line.split()[1])
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    return dict(
        free=free,
        total=total,
        device_used=total - free,
        worker_rss_bytes=rss_kib * 1024,
        torch_allocated=torch.cuda.memory_allocated(),
        torch_reserved=torch.cuda.memory_reserved(),
    )


def graph_identity(mr):
    runner = mr.decode_cuda_graph_runner
    graphs = runner.backend._graphs
    return {str(k): id(v) for k, v in graphs.items()}


class ReplicaStandby:
    def __init__(self, mr, directory):
        self.directory = Path(directory)
        self.mr = mr
        self.rank = mr.ps.pp_rank
        self.key = preparation_key(
            self.directory, mr.model._canonical_cpu_weight_cache_key
        )
        if (
            self.directory != Path(os.environ["SGLANG_REPLICA_PREPARED_DIR"])
            or mr.server_args.disaggregation_mode != "decode"
            or not mr.server_args.skip_server_warmup
            or mr.server_args.elastic_pp_initial_partition != "24,24"
            or (mr.ps.pp_size, mr.ps.tp_size) != (2, 1)
        ):
            raise ValueError("Standby requires the prepared fixed PP2 Decode replica")
        self.root = Path(os.environ["SGLANG_SLLMSTORE_STAGE_ROOT"])
        self.packet = read_packet(self.root, "packed.json", self.key)
        self.layers = mr._elastic_pp_weight_backing.layers
        lo, hi = self.key["pp_layers"][self.rank]
        if set(self.layers) != set(range(lo, hi)):
            raise ValueError("Standby VMM does not cover this PP stage")
        layer_keys = set().union(
            *(owner.storage_keys for owner in self.layers.values())
        )
        # Endpoint embedding, output head and final norm also need stable VAs.
        # Discover their aliases across the entire model before capturing Graphs.
        self.endpoints = LayerWeightBacking(
            mr.model, torch.cuda.current_device(), exclude_storage_keys=layer_keys
        )
        self.all_keys = layer_keys | self.endpoints.storage_keys
        self.owners = [*self.layers.values(), self.endpoints]
        self.device_bytes = sum(owner.size for owner in self.owners)
        self.experts = {
            int(m.layer_id): m
            for m in mr.model.modules()
            if getattr(m, "_qwen_a100_fp8_marlin_prepared", False)
        }
        if set(self.experts) != set(self.layers):
            raise ValueError(
                "Standby expert coverage differs from the live layer range"
            )
        self.ticket = secrets.token_hex(16)
        self.cpu_dense = []
        self.state = "preparing"

    @torch.inference_mode()
    def park(self):
        mr = self.mr
        groups, _ = weight_storage_plan(mr.model)
        if set(groups) != self.all_keys:
            raise RuntimeError(
                "Model gained or rebound weight storage after graph capture"
            )
        expert_keys = {
            storage_key(getattr(m, field))
            for m in self.experts.values()
            for field in EXPERT_DTYPES
        }
        dense = [
            byte_view(g["source"]) for k, g in groups.items() if k not in expert_keys
        ]
        dense_bytes = sum(t.numel() for t in dense)
        if dense_bytes > 8 << 30 or not expert_keys <= self.all_keys:
            raise ValueError(
                "Processed dense state exceeds the admitted 8 GiB per rank"
            )
        for tensor in dense:
            host = torch.empty(
                tensor.shape, dtype=torch.uint8, pin_memory=True, device="cpu"
            )
            host.copy_(tensor, non_blocking=True)
            self.cpu_dense.append((tensor, host))
        torch.cuda.synchronize()
        self.graphs = graph_identity(mr)
        self.batches = sorted(mr.decode_cuda_graph_runner.capture_bs)
        if (
            {key.size for key in mr.decode_cuda_graph_runner.backend._graphs}
            != {1, 2, 4}
            or self.batches != [1, 2, 4]
            or mr._elastic_pp_prepared_graphs.building
        ):
            raise RuntimeError("Expected sealed B1/B2/B4 Decode Graphs before standby")
        self.host_bank = mr._replica_attached_bank
        if mr._elastic_pp_shared_host_bank is not self.host_bank:
            raise RuntimeError("Prepared HiCache is not attached")
        before = memory_usage()
        for owner in self.owners:
            owner.evict()
        torch.cuda.empty_cache()
        after = memory_usage()
        if (
            any(o.mapped for o in self.owners)
            or after["free"] - before["free"] < 0.95 * self.device_bytes
        ):
            raise RuntimeError(
                "Physical weight release not observed on the standby GPU"
            )
        self.state = "parked"
        publish(
            self.directory / f"parked-{self.rank}.json",
            dict(
                key=self.key,
                ticket=self.ticket,
                pid=os.getpid(),
                rank=self.rank,
                unix=time.time(),
                device=torch.cuda.current_device(),
                before=before,
                standby=after,
                weight_mapped_bytes=0,
                evicted_bytes=self.device_bytes,
                dense_cpu_bytes=dense_bytes,
                weight_bytes=sum(k[2] for k in groups),
                graph_batches=self.batches,
                graph_objects=self.graphs,
                host_bytes=self.host_bank.bytes,
                hicache_attach_s=mr._replica_attach_seconds,
                startup_graph_s=sum(
                    e.capture_seconds
                    for e in mr._elastic_pp_prepared_graphs.entries.values()
                ),
            ),
        )

    @torch.inference_mode()
    def wait_and_restore(self):
        path = self.directory / "activate.json"
        while not path.exists():
            time.sleep(0.02)
        command = read_packet(self.directory, path.name, self.key)
        if self.state != "parked" or command["tickets"][self.rank] != self.ticket:
            raise RuntimeError("Stale standby activation")
        self.state = "restoring"
        started = time.perf_counter()
        for owner in self.owners:
            owner.restore()
        mapped = time.perf_counter()
        for target, host in self.cpu_dense:
            target.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        dense_done = time.perf_counter()
        expert_bytes = 0
        for idx, layer in sorted(self.experts.items()):
            entry = self.packet["layers"][str(idx)]
            shapes = {k: tuple(getattr(layer, k).shape) for k in EXPERT_DTYPES}
            views = packed_expert_views(entry, self.root, self.rank, shapes)
            for name, src in views.items():
                dst = getattr(layer, name)
                if src.dtype != dst.dtype or src.shape != dst.shape:
                    raise ValueError(
                        "Prepared expert differs from captured destination"
                    )
                dst.copy_(src)
            # The official Store owns the IPC source until this stream completes.
            torch.cuda.current_stream().synchronize()
            del src, views
            expert_bytes += entry["size"]
        restored = time.perf_counter()
        groups, _ = weight_storage_plan(self.mr.model)
        if (
            set(groups) != self.all_keys
            or graph_identity(self.mr) != self.graphs
            or self.mr._replica_attached_bank is not self.host_bank
            or not all(o.mapped for o in self.owners)
        ):
            raise RuntimeError(
                "Restoration changed weights, Graphs or the host cache binding"
            )
        dense_bytes = sum(t.numel() for t, _ in self.cpu_dense)
        if expert_bytes + dense_bytes != sum(key[2] for key in groups):
            raise RuntimeError("Incomplete restored model-weight coverage")
        self.cpu_dense.clear()
        torch.distributed.barrier(group=self.mr.pp_group.cpu_group)
        self.state = "active"
        publish(
            self.directory / f"restored-{self.rank}.json",
            dict(
                key=self.key,
                ticket=self.ticket,
                rank=self.rank,
                pid=os.getpid(),
                unix=time.time(),
                expert_store_bytes=expert_bytes,
                expert_layers=sorted(self.experts),
                dense_h2d_bytes=dense_bytes,
                mapped_weight_bytes=self.device_bytes,
                graph_objects=graph_identity(self.mr),
                graph_batches=self.batches,
                host_bytes=self.host_bank.bytes,
                memory=memory_usage(),
                map_s=mapped - started,
                dense_h2d_s=dense_done - mapped,
                expert_store_s=restored - dense_done,
                rank_restore_s=restored - started,
                through_barrier_s=time.perf_counter() - started,
            ),
        )
