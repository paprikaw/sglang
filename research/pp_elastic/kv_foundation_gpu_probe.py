"""Guard-owned GPU checks; isolated primitives, not model Graph qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


def probe():
    import torch

    from kvcached import tp_ipc_util as ipc, vmm_ops as native
    from sglang.srt.elastic_pp.graph_bank import (
        ElasticGraphBank,
        GraphKey,
        TorchGraphBackend,
    )
    from sglang.srt.elastic_pp.kv_patch import (
        ContinuousKVPatch,
        KVPatchBusy,
        TensorKVPatchCopier,
    )
    from sglang.srt.mem_cache.kvcached_compat import probe_layer_mapping_capabilities

    probe_layer_mapping_capabilities().require()
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    # Warm runtime allocations before measuring driver-visible free memory.
    torch.zeros(1, device=device).add_(1)
    torch.cuda.synchronize()
    page, groups, pages = 2 * 1024**2, 4, 2
    map_size = 2 * page
    span = pages * map_size
    total = groups * span
    rows, head = pages * page // 128, 64
    active = {0, 1}
    evidence = dict(
        passed=False,
        scope="isolated GPU VMM / Graph bank / immutable-row patch primitives",
        model_graph_or_hicache_backend_qualified=False,
        gpu=torch.cuda.get_device_name(0),
        physical_bytes_per_layer_group=span,
        timings_ms={},
    )
    native.init_kvcached(str(device), page_size=page, contiguous_layout=True)
    raw = native.create_kv_tensors(
        pages * page * 2,
        2,
        str(device),
        groups,
        num_kv_buffers=2,
        layer_group_granularity=1,
        contiguous_page_size=map_size,
        contiguous_total_size=total,
    )[0].view(dtype=torch.bfloat16)
    kv = [
        tuple(
            raw.as_strided(
                (rows, 1, head), (head * 2, head, 1), layer * span // 2 + kind * head
            )
            for kind in (0, 1)
        )
        for layer in range(groups)
    ]
    addresses = [tuple(tensor.data_ptr() for tensor in pair) for pair in kv]
    ipc.register_worker_offset_filter(
        lambda offsets, group: [
            offset for offset in offsets if offset // span in active
        ],
        map_size=map_size,
        layer_group_span=span,
        num_layer_groups=groups,
    )

    def offsets(layer):
        return [layer * span + i * map_size for i in range(pages)]

    @contextmanager
    def lease(layers):
        held = ipc.pin_worker_offsets(
            [offset for layer in layers for offset in offsets(layer)]
        )
        try:
            yield
        finally:
            ipc.unpin_worker_offsets(held)

    def timed(name, function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        evidence["timings_ms"][name] = (time.perf_counter() - start) * 1000
        return value

    def free_bytes():
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info()[0]

    bank = patch = None
    try:
        before = free_bytes()
        requested = [offset for layer in range(groups) for offset in offsets(layer)]
        assert ipc._apply_worker_mapping_command("map_to_kv_tensors", requested)
        torch.cuda.synchronize()
        assert ipc.snapshot_worker_mapped_offsets() == tuple(offsets(0) + offsets(1))
        evidence["initial_map_driver_bytes"] = before - free_bytes()
        for tensor in kv[0]:
            tensor[:8].fill_(3)
        for tensor in kv[1]:
            tensor[:8].fill_(7)
        assert ipc.map_worker_offsets_for_migration(offsets(0))
        assert all(bool(torch.all(tensor[:8] == 3)) for tensor in kv[0])
        with lease([1]):
            try:
                ipc.unmap_worker_offsets_for_layer_groups([1], span)
            except RuntimeError as error:
                assert "pinned" in str(error)
            else:
                raise AssertionError("pinned source page was reclaimed")
        assert timed(
            "map_target", lambda: ipc.map_worker_offsets_for_migration(offsets(2))
        )
        assert all(bool(torch.all(tensor[:8] == 0)) for tensor in kv[2])
        for tensor in kv[2]:
            tensor[0].fill_(5)

        bank = ElasticGraphBank(backend=TorchGraphBackend(device), max_entries=2)
        source_key, target_key = GraphKey((2, 0, 2), 1), GraphKey((1, 1, 2), 1)
        source_out = torch.zeros((1, head), device=device, dtype=torch.bfloat16)
        target_out = torch.zeros_like(source_out)
        source_bindings = dict(k0=kv[0][0], k1=kv[1][0], output=source_out)
        target_bindings = dict(k1=kv[1][0], k2=kv[2][0], output=target_out)
        source_forward = lambda: torch.add(kv[0][0][0], kv[1][0][0], out=source_out)
        target_forward = lambda: torch.add(kv[1][0][0], kv[2][0][0], out=target_out)
        timed(
            "capture_source",
            lambda: bank.capture(
                source_key,
                bindings=source_bindings,
                forward=source_forward,
                lease=lambda: lease([0, 1]),
            ),
        )
        timed(
            "capture_target",
            lambda: bank.capture(
                target_key,
                bindings=target_bindings,
                forward=target_forward,
                lease=lambda: lease([1, 2]),
            ),
        )
        bank.activate()
        bank.run(
            source_key,
            bindings=source_bindings,
            eager=source_forward,
            lease=lambda: lease([0, 1]),
        )
        torch.cuda.synchronize()
        bank.poll()
        assert bool(torch.all(source_out == 10))

        copier = TensorKVPatchCopier(
            {0: kv[1]}, {0: kv[2]}, page_lease=lambda rows: lease([1, 2])
        )
        patch = ContinuousKVPatch(
            generation=1, capacity=rows, copier=copier, max_rows=2
        )

        def write(slot, value):
            epoch = patch.allocate(slot, generation=1)
            for tensor in kv[1]:
                tensor[slot].fill_(value)
            ready = torch.cuda.Event()
            ready.record()
            patch.publish(slot, epoch, ready=ready, generation=1)
            return epoch

        write(1, 11)
        old_epoch = write(2, 13)
        patch.submit(generation=1)
        try:
            patch.free(2, old_epoch, generation=1)
        except KVPatchBusy:
            pass
        else:
            raise AssertionError("in-flight source row was reused")
        # Source inference appends a distinct row while the first copy is active.
        write(3, 17)
        try:
            ipc.unmap_worker_offsets_for_layer_groups([1], span)
        except RuntimeError as error:
            assert "pinned" in str(error)
        else:
            raise AssertionError("patch copy lost its physical page lease")
        patch.poll(generation=1, wait=True)
        patch.free(2, old_epoch, generation=1)
        new_epoch = write(2, 19)
        assert new_epoch > old_epoch
        evidence["patch"] = timed(
            "patch_final_fence", lambda: patch.finish_at_fence(generation=1)
        )
        assert evidence["patch"]["rounds"] >= 2
        for source, target in zip(kv[1], kv[2], strict=True):
            assert torch.equal(source[[1, 2, 3]], target[[1, 2, 3]])

        before = free_bytes()
        assert timed(
            "unmap_retired",
            lambda: ipc.unmap_worker_offsets_for_layer_groups([0], span),
        )
        reclaimed = free_bytes() - before
        evidence["retired_group_driver_bytes_reclaimed"] = reclaimed
        # Driver free memory is independent evidence, not a Python-view count.
        assert reclaimed >= span - map_size, (reclaimed, span)
        assert addresses == [tuple(tensor.data_ptr() for tensor in pair) for pair in kv]
        assert all(bool(torch.all(tensor[0] == 7)) for tensor in kv[1])
        active.clear()
        active.update({1, 2})
        bank.run(
            target_key,
            bindings=target_bindings,
            eager=target_forward,
            lease=lambda: lease([1, 2]),
        )
        torch.cuda.synchronize()
        bank.poll()
        assert bool(torch.all(target_out == 12))
        # A changed output allocation must execute eager rather than a stale graph.
        fallback_out = torch.zeros_like(target_out)
        fallback_bindings = dict(target_bindings, output=fallback_out)
        bank.run(
            target_key,
            bindings=fallback_bindings,
            eager=lambda: torch.add(kv[1][0][0], kv[2][0][0], out=fallback_out),
            lease=lambda: lease([1, 2]),
        )
        torch.cuda.synchronize()
        bank.poll()
        assert bool(torch.all(fallback_out == 12))
        try:
            bank.run(
                source_key,
                bindings=source_bindings,
                eager=source_forward,
                pages_ready=False,
            )
        except RuntimeError as error:
            assert "not mapped" in str(error)
        else:
            raise AssertionError("stale source graph accessed retired backing")
        evidence["graph"] = dict(bank.counters)
        assert timed(
            "remap_source", lambda: ipc.map_worker_offsets_for_migration(offsets(0))
        )
        assert all(bool(torch.all(tensor[:8] == 0)) for tensor in kv[0])
        assert all(bool(torch.all(tensor[0] == 7)) for tensor in kv[1])
        evidence["retained_addresses_and_values"] = True
        evidence["reuse_epoch_protected"] = True
        evidence["passed"] = True
    finally:
        if patch is not None and patch.phase not in ("complete", "aborted"):
            patch.abort(generation=1)
        if bank is not None:
            bank.close()
        assert ipc.unmap_worker_offsets_for_layer_groups(range(groups), span)
        assert not ipc.snapshot_worker_mapped_offsets()
        torch.cuda.synchronize()
        kv.clear()
        del raw
        native.shutdown_kvcached()
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    # Require an authenticated exact-node lease, even for this small probe.
    from hicache_transition_smoke import h

    cfg = json.loads((args.run / "config.json").read_text())
    h.authenticate(args.run, cfg, os.environ, local=True)
    dependency = cfg["kv_foundation_probe"]
    root = Path(dependency["root"])
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip()
        != dependency["tree"]
    ):
        raise RuntimeError("KVCacheD source tree changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip():
        raise RuntimeError("KVCacheD tracked source changed")
    binary = Path(dependency["native"]["path"])
    if (
        hashlib.sha256(binary.read_bytes()).hexdigest()
        != dependency["native"]["sha256"]
    ):
        raise RuntimeError("KVCacheD binary changed")
    started = time.time()
    try:
        result = probe()
    except BaseException as error:
        result = dict(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(started_at=started, elapsed_s=time.time() - started)
        h.save(args.run / "component_gpu.json", result)


if __name__ == "__main__":
    main()
