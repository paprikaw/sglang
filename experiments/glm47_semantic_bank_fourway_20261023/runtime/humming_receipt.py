"""Read-only receipt of actual loaded Humming kernels, for idle/bootstrap hooks.

No CUDA calls, synchronization, kernel compilation, tensor readback, or file I/O.
Loaded cubin metadata proves resolved available kernel settings, NOT that a
particular kernel launched. Requiring every loaded dense/grouped-masked kernel
to satisfy the invariant covers whichever of these candidates was selected.
Call only after normal model/Graph warmup, and again after the measured cohort.
"""

from __future__ import annotations

import hashlib
import sys


SCHEMA = "humming-loaded-kernel-invariant-v1"


def collect_humming_kernel_receipt():
    module = sys.modules.get("humming.kernel.humming")
    if module is None:
        raise RuntimeError("Humming kernel module has not been initialized")
    registry = module.HummingKernel._id2kernel
    rows = []
    for kernel_id, kernel in sorted(registry.items()):
        gemm_type = getattr(kernel, "gemm_type", None)
        if hasattr(gemm_type, "value"):
            gemm_type = gemm_type.value
        code = getattr(kernel, "code", None)
        rows.append({
            "kernel_id": int(kernel_id),
            "gemm_type": gemm_type,
            "sm_version": getattr(kernel, "sm_version", None),
            "use_batch_invariant": getattr(kernel, "use_batch_invariant", None),
            "use_stream_k": getattr(kernel, "use_stream_k", None),
            "block_shape": list(getattr(kernel, "block_shape", ())),
            "warp_shape": list(getattr(kernel, "warp_shape", ())),
            "kernel_filename": getattr(kernel, "kernel_filename", None),
            "kernel_name": getattr(kernel, "kernel_name", None),
            "source_sha256": hashlib.sha256(code.encode()).hexdigest()
            if isinstance(code, str) else None,
        })
    return {
        "schema": SCHEMA,
        "evidence_kind": "registered_cubin_metadata_not_launch_trace",
        "proves_actual_launch": False,
        "kernels": rows,
    }


def validate_humming_kernel_receipt(
    receipt, required_gemm_types=("dense", "grouped_masked")
):
    """Fail closed if any required type is absent or has unsafe loaded kernels."""
    if receipt.get("schema") != SCHEMA:
        raise ValueError("invalid Humming kernel receipt schema")
    kernels = receipt.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("no registered Humming kernels")
    required = set(required_gemm_types)
    if not required:
        raise ValueError("required Humming GEMM types must not be empty")
    seen = set()
    ids = set()
    for row in kernels:
        kernel_id = row.get("kernel_id")
        if type(kernel_id) is not int or kernel_id in ids:
            raise ValueError("invalid or duplicate Humming kernel id")
        ids.add(kernel_id)
        kind = row.get("gemm_type")
        if kind not in required:
            continue
        seen.add(kind)
        if row.get("use_batch_invariant") is not True:
            raise ValueError(f"Humming {kind} kernel is not batch invariant")
        if row.get("use_stream_k") is not False:
            raise ValueError(f"Humming {kind} kernel has StreamK enabled/unknown")
        if row.get("sm_version") != 80:
            raise ValueError(f"Humming {kind} kernel is not the expected SM80 path")
        for field in ("block_shape", "warp_shape"):
            shape = row.get(field)
            if (not isinstance(shape, list) or len(shape) != 3
                    or any(type(value) is not int or value <= 0 for value in shape)):
                raise ValueError(f"Humming {kind} missing effective {field}")
        if not row.get("kernel_filename") or not row.get("kernel_name"):
            raise ValueError(f"Humming {kind} missing loaded cubin identity")
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Humming {kind} missing generated source digest")
    missing = required - seen
    if missing:
        raise ValueError(f"Humming kernel types absent: {sorted(missing)}")
    return {"passed": True, "required_gemm_types": sorted(required),
            "registered_kernel_count": len(kernels),
            "proves_actual_launch": False}
