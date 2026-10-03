import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestMoERunner(CustomTestCase):
    BASE_URL = DEFAULT_URL_FOR_TEST
    TIMEOUT = 6000
    DEFAULT_EVAL_KWARGS = {
        "eval_name": "mmlu",
        "num_examples": 5,
        "num_threads": 1,
    }

    CONFIGS = {
        "moe_runner_auto": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_triton": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_triton_kernel": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton_kernel",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_cutlass": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,  # requires model with modelopt_fp4 quantization
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutlass",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_deep_gemm": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "deep_gemm",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_trtllm": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,  # modelopt_fp4 or fp8 quantization is required for Flashinfer trtllm MOE
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_trtllm",
            ],
        },
        "moe_runner_flashinfer_mxfp4": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--quantization",
                "mxfp4",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_flashinfer_cutedsl": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutedsl",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_cutlass": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
        "moe_runner_cutlass_fp8": {
            "model": DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
            "timeout": 3600,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "cutlass",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--disable-cuda-graph",
            ],
        },
        "moe_runner_speculative": {
            "model": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
            "other_args": [
                "--trust-remote-code",
                "--moe-runner-backend",
                "triton",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
                "--speculative-moe-runner-backend",
                "triton",
                "--speculative-num-steps",
                "2",
                "--speculative-num-draft-tokens",
                "4",
                "--attention-backend",
                "torch_native",
                "--sampling-backend",
                "pytorch",
            ],
        },
    }

    def _run_config(self, config: dict) -> None:
        model = config["model"]
        other_args = config.get("other_args", [])
        eval_kwargs = self.DEFAULT_EVAL_KWARGS
        env = dict(os.environ)
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
        env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
        env.update(config.get("env_overrides", {}))
        timeout = config.get("timeout", self.TIMEOUT)

        process = popen_launch_server(
            model,
            self.BASE_URL,
            timeout=timeout,
            other_args=other_args,
            env=env,
        )
        try:
            args = SimpleNamespace(
                base_url=self.BASE_URL,
                model=model,
                **eval_kwargs,
            )
            metrics = run_eval(args)
            print(f"{metrics=}")
            self.assertGreaterEqual(metrics["score"], 0.48)
        finally:
            kill_process_tree(process.pid)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHummingMaskedActivation(CustomTestCase):
    def test_counts_and_padding_with_changing_cuda_graph(self):
        from sgl_kernel import silu_and_mul

        from sglang.kernels.ops.moe.humming_masked_activation import (
            masked_silu_and_mul,
        )

        # Non-power-of-two columns and a capacity beyond one persistent-grid
        # stride exercise both the column mask and the last partial row tile.
        experts, capacity, intermediate = 4, 517, 1536
        torch.manual_seed(42)
        inputs = torch.randn(
            (experts * capacity, 2 * intermediate),
            device="cuda",
            dtype=torch.bfloat16,
        )
        actual = torch.empty(
            (experts * capacity, intermediate), device="cuda", dtype=inputs.dtype
        )
        reference = torch.empty_like(actual)
        counts = torch.ones(experts, device="cuda", dtype=torch.int32)

        def run():
            masked_silu_and_mul(inputs, actual, counts)

        def check(cpu_counts):
            silu_and_mul(inputs, reference)
            view = actual.view(experts, capacity, intermediate)
            ref_view = reference.view(experts, capacity, intermediate)
            for expert, count in enumerate(cpu_counts):
                if count:
                    torch.testing.assert_close(
                        view[expert, :count],
                        ref_view[expert, :count],
                        rtol=0.008,
                        atol=0.0001,
                    )
                if count < capacity:
                    self.assertTrue(torch.isnan(view[expert, count:]).all().item())

        cases = ([0, 0, 0, 0], [0, 1, 64, capacity], [capacity] * experts)
        for cpu_counts in cases:
            with self.subTest(graph=False, counts=cpu_counts):
                counts.copy_(torch.tensor(cpu_counts, device="cuda", dtype=counts.dtype))
                actual.fill_(float("nan"))
                run()
                check(cpu_counts)

        # Capture with every expert empty. The same graph must later execute
        # nonempty experts, loop over multiple tiles, and stop writing padding
        # again when counts become zero. Inputs change as well to reject stale
        # output that accidentally matches a previous replay.
        counts.zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for cpu_counts in (*cases, cases[0]):
            with self.subTest(graph=True, counts=cpu_counts):
                counts.copy_(torch.tensor(cpu_counts, device="cuda", dtype=counts.dtype))
                inputs.add_(0.03125)
                actual.fill_(float("nan"))
                graph.replay()
                check(cpu_counts)


for _name, _cfg in TestMoERunner.CONFIGS.items():
    setattr(
        TestMoERunner,
        f"test_{_name}",
        (lambda self, cfg=_cfg: self._run_config(cfg)),
    )


if __name__ == "__main__":
    unittest.main()
