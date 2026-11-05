"""CPU regression for DP-padded Triton metadata; no CUDA kernels are launched."""

import ast
import copy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
BACKEND = ROOT / "python/sglang/srt/layers/attention/triton_backend.py"
COMMON = ROOT / "python/sglang/srt/utils/common.py"
FORWARD_BATCH = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"


def compile_nodes(path, nodes, env):
    # Isolate CPU metadata code from backend imports that initialize CUDA.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *copy.deepcopy(nodes),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), env)


def eager_max_batch_size(server_args, max_bs):
    node = next(
        node
        for node in ast.parse(COMMON.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_eager_max_batch_size"
    )
    env = {
        "require_mlp_sync": lambda args: args.require_mlp_sync,
        "get_parallel": lambda: SimpleNamespace(attn_tp_size=server_args.attn_tp_size),
        "ceil_align": lambda value, align: (value + align - 1) // align * align,
    }
    compile_nodes(COMMON, [node], env)
    cp_padding = ModuleType("sglang.srt.layers.cp.padding")
    cp_padding.get_cp_padding_align_size = lambda: server_args.cp_align_size
    with patch.dict(sys.modules, {cp_padding.__name__: cp_padding}):
        return env[node.name](server_args, max_bs)


class CPUKVIndexKernel:
    """Model the KV index gather, preserving each padded row's request index."""

    def __getitem__(self, grid):
        self.grid = grid
        return self.run

    def run(self, req_to_token, req_indices, seq_lens, indptr, _, output, stride):
        self.req_indices = req_indices.clone()
        self.seq_lens = seq_lens.clone()
        for row in range(self.grid[0]):
            start, end = indptr[row : row + 2].tolist()
            output[start:end] = req_to_token[req_indices[row], : seq_lens[row]]


def metadata_backend():
    source = ast.parse(BACKEND.read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "TritonAttnBackend"
    )
    init = next(node for node in cls.body if getattr(node, "name", None) == "__init__")
    # Run the production capacity assignment and scratch allocation statements,
    # leaving out model/kernel initialization unrelated to metadata storage.
    allocations = [
        node
        for node in init.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "max_bs" for t in node.targets)
        )
        or (
            isinstance(node, ast.If)
            and ast.unparse(node.test)
            in {
                "kv_indptr_buf is None",
                "self.sliding_window_size is not None and self.sliding_window_size > 0",
                "not self.skip_prefill",
            }
        )
    ]
    assert len(allocations) == 4
    init = copy.deepcopy(init)
    init.body = (
        ast.parse(
            "self.skip_prefill = skip_prefill\n"
            "self.sliding_window_size = model_runner.sliding_window_size\n"
        ).body
        + allocations
    )
    cls = copy.deepcopy(cls)
    cls.bases = []
    cls.body = [init] + [
        node
        for node in cls.body
        if getattr(node, "name", None)
        in {"_fill_kv_indptr_and_indices", "get_cuda_graph_seq_len_fill_value"}
    ]
    kernel = CPUKVIndexKernel()
    env = {
        "torch": torch,
        "get_eager_max_batch_size": eager_max_batch_size,
        "create_flashinfer_kv_indices_triton": kernel,
    }
    compile_nodes(BACKEND, [cls], env)
    return env[cls.name], kernel


def pad_tensor(tensor, size, *, value=0):
    cls = next(
        node
        for node in ast.parse(FORWARD_BATCH.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
    )
    method = next(
        node
        for node in cls.body
        if getattr(node, "name", None) == "_pad_tensor_to_size"
    )
    env = {"torch": torch}
    compile_nodes(FORWARD_BATCH, [method], env)
    return env[method.name](None, tensor, size, value=value)


class TestTritonDPMetadataCapacity(unittest.TestCase):
    def make_backend(self, *, mlp_sync=True, pool_size=2, sliding_window_size=None):
        cls, kernel = metadata_backend()
        runner = SimpleNamespace(
            server_args=SimpleNamespace(
                require_mlp_sync=mlp_sync, attn_tp_size=4, cp_align_size=1
            ),
            req_to_token_pool=SimpleNamespace(size=pool_size),
            sliding_window_size=sliding_window_size,
            device="cpu",
        )
        backend = cls(runner)
        # Request 0 is the reserved dummy row; real requests retain distinct KV.
        backend.req_to_token = torch.zeros((pool_size + 1, 8), dtype=torch.int32)
        for request in range(1, pool_size + 1):
            backend.req_to_token[request] = 100 * request + torch.arange(8)
        return backend, kernel

    def test_dp_decode_padding_preserves_real_and_dummy_kv_rows(self):
        for real_reqs in (1, 2):
            with self.subTest(real_reqs=real_reqs):
                backend, kernel = self.make_backend()
                requests = pad_tensor(torch.arange(1, real_reqs + 1), 4)
                lengths = pad_tensor(
                    torch.full((real_reqs,), 7, dtype=torch.int32),
                    4,
                    value=backend.get_cuda_graph_seq_len_fill_value(),
                )
                indices = torch.empty(int(lengths.sum()), dtype=torch.int32)
                indptr = backend._fill_kv_indptr_and_indices(
                    4, lengths, requests, indices
                )

                self.assertEqual(
                    indptr.tolist(), [0] + torch.cumsum(lengths, 0).tolist()
                )
                self.assertEqual(kernel.grid, (4,))
                self.assertEqual(kernel.req_indices.tolist(), requests.tolist())
                self.assertEqual(kernel.seq_lens.tolist(), lengths.tolist())
                expected = torch.cat(
                    [
                        backend.req_to_token[request, :length]
                        for request, length in zip(requests, lengths)
                    ]
                )
                torch.testing.assert_close(indices, expected)
                self.assertTrue((indices[7 * real_reqs :] == 0).all())
                self.assertEqual(backend.req_to_token.shape[0], 3)

    def test_non_mlp_sync_keeps_unpadded_capacity(self):
        backend, _ = self.make_backend(mlp_sync=False)
        self.assertEqual(backend.kv_indptr.numel(), 3)
        lengths = torch.tensor([7, 5], dtype=torch.int32)
        indices = torch.empty(12, dtype=torch.int32)
        indptr = backend._fill_kv_indptr_and_indices(
            2, lengths, torch.tensor([1, 2]), indices
        )
        self.assertEqual(indptr.tolist(), [0, 7, 12])
        self.assertEqual(
            indices.tolist(), list(range(100, 107)) + list(range(200, 205))
        )

    def test_all_metadata_scratch_uses_the_same_aligned_capacity(self):
        backend, _ = self.make_backend(sliding_window_size=4)
        for name in ("kv_indptr", "window_kv_indptr", "qo_indptr", "mask_indptr"):
            with self.subTest(buffer=name):
                self.assertEqual(getattr(backend, name).numel(), 5)
        # Non-divisible larger local pools must use the same policy as eager inputs.
        backend, _ = self.make_backend(pool_size=5)
        self.assertEqual(backend.kv_indptr.numel(), 9)


if __name__ == "__main__":
    unittest.main()
