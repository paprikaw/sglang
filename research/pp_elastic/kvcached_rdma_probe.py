"""Bounded startup diagnostic; run only as an owned child of the Guard worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket


def run(output: Path):
    import torch
    from kvcached import vmm_ops
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )
    from sglang.srt.utils.network import get_local_ip_auto

    step = 2 * 1024**2
    engine = MooncakeTransferEngine(get_local_ip_auto(), 0, "mlx5_bond_0").engine
    ordinary = torch.zeros(2 * step, device="cuda", dtype=torch.uint8)
    vmm_ops.init_kvcached("cuda:0", step, True)
    tensor = vmm_ops.create_kv_tensors(
        14 * step, 1, "cuda:0", 1, 2, 0, False, 1, step, 14 * step
    )[0]
    assert vmm_ops.map_to_kv_tensors([i * step for i in range(10)], 0)
    tensor[:10 * step].fill_(31)
    torch.cuda.synchronize()
    assert tensor[10 * step - 1].item() == 31
    records = []

    def publish(value):
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2)+"\n")
        temporary.replace(output)

    def attempt(name, ptr, size):
        status = int(engine.register_memory(ptr, size))
        row = dict(name=name, ptr=ptr, length=size, status=status)
        if status == 0:
            row["unregister_status"] = int(engine.unregister_memory(ptr))
        records.append(row)
        publish(dict(node=socket.gethostname(), records=records))
        return status == 0 and row.get("unregister_status") == 0

    ordinary_ok = attempt("ordinary_cuda", ordinary.data_ptr(), 2 * step)
    span_ok = attempt("vmm_ten_allocations", tensor.data_ptr(), 10 * step)
    pages_ok = [attempt(f"vmm_page_{i}", tensor.data_ptr()+i*step, step)
                for i in range(10)]
    passed = ordinary_ok and all(pages_ok)
    publish(dict(node=socket.gethostname(), records=records,
        passed=passed, contiguous_span_registered=span_ok,
        scope="Registration only; no model, P/D transfer or capacity qualification"))
    if not passed:
        raise RuntimeError("KVCacheD per-allocation RDMA registration is unavailable; stop before model load")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
