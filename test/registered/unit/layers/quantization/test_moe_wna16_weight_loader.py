"""Packed AWQ zero-point loading must honor EP ownership and MoE TP ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization import moe_wna16
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMoeWNA16ExpertWeightLoader(unittest.TestCase):
    def setUp(self):
        self.group = patch.object(
            moe_wna16, "get_tp_group", return_value=SimpleNamespace(device="cpu")
        )
        self.parallel = patch.object(
            moe_wna16,
            "get_parallel",
            return_value=SimpleNamespace(tp_rank=5),
            create=True,
        )
        self.group.start()
        self.parallel.start()
        self.addCleanup(self.group.stop)
        self.addCleanup(self.parallel.stop)

    def make_loader(self, *, start=100, count=20, tp_size=1, tp_rank=0):
        mapper = Mock(
            side_effect=lambda expert: (
                expert - start if start <= expert < start + count else -1
            )
        )
        layer = SimpleNamespace(
            quant_config=SimpleNamespace(
                has_zp=True, linear_quant_method="awq", weight_bits=4
            ),
            intermediate_size_per_partition=16,
            group_size_div_factor=1,
            moe_tp_size=tp_size,
            moe_tp_rank=tp_rank,
            _map_global_expert_id_to_local_expert_id=mapper,
        )
        fallback = Mock()
        return (
            moe_wna16.MoeWNA16Method.get_weight_loader(layer, fallback),
            mapper,
            fallback,
        )

    @staticmethod
    def parameter(shape):
        return torch.nn.Parameter(
            torch.full(shape, 0xEE, dtype=torch.uint8), requires_grad=False
        )

    def test_ep8_gate_and_up_zero_points_use_local_expert(self):
        loader, mapper, fallback = self.make_loader()
        param = self.parameter((20, 16, 4))
        for shard, nibble in [("w1", 1), ("w3", 2)]:
            raw = torch.full((4, 2), int(str(nibble) * 8, 16), dtype=torch.int32)
            loader(param, raw, "w13_qzeros", shard, 100)
        self.assertTrue(torch.all(param[0, :8] == 0x11).item())
        self.assertTrue(torch.all(param[0, 8:] == 0x22).item())
        self.assertTrue(torch.all(param[1:] == 0xEE).item())
        self.assertEqual(mapper.call_count, 2)
        fallback.assert_not_called()

    def test_ep8_down_zero_points_use_local_expert(self):
        loader, _, _ = self.make_loader()
        param = self.parameter((20, 8, 4))
        loader(
            param,
            torch.full((4, 2), 0x33333333, dtype=torch.int32),
            "w2_qzeros",
            "w2",
            119,
        )
        self.assertTrue(torch.all(param[19] == 0x33).item())
        self.assertTrue(torch.all(param[:19] == 0xEE).item())

    def test_non_owned_expert_is_skipped_before_device_copy(self):
        loader, _, fallback = self.make_loader()
        param = self.parameter((20, 8, 4))
        unloaded = Mock()
        unloaded.to.side_effect = AssertionError(
            "Non-owned zero points must not be copied"
        )
        loader(param, unloaded, "w2_qzeros", "w2", 99)
        self.assertTrue(torch.all(param == 0xEE).item())
        fallback.assert_not_called()

    def test_mixed_tp_ep_uses_expert_tensor_parallel_rank(self):
        loader, _, _ = self.make_loader(start=80, count=80, tp_size=2, tp_rank=1)
        param = self.parameter((80, 8, 4))
        raw = torch.cat(
            [
                torch.full((4, 2), 0x11111111, dtype=torch.int32),
                torch.full((4, 2), 0x22222222, dtype=torch.int32),
            ]
        )
        loader(param, raw, "w2_qzeros", "w2", 90)
        self.assertTrue(torch.all(param[10] == 0x22).item())
        self.assertTrue(torch.all(param[:10] == 0xEE).item())
        self.assertTrue(torch.all(param[11:] == 0xEE).item())

    def test_ep1_tensor_parallel_zero_points_are_unchanged(self):
        loader, _, _ = self.make_loader(start=0, count=160, tp_size=8, tp_rank=5)
        param = self.parameter((160, 8, 4))
        raw = torch.cat(
            [
                torch.full((4, 2), int(str(i + 1) * 8, 16), dtype=torch.int64).to(
                    torch.int32
                )
                for i in range(8)
            ]
        )
        loader(param, raw, "w2_qzeros", "w2", 100)
        self.assertTrue(torch.all(param[100] == 0x66).item())

    def test_full_expert_parameter_keeps_global_index(self):
        loader, mapper, _ = self.make_loader()
        param = self.parameter((160, 8, 4))
        param._sglang_require_global_experts = True
        loader(
            param,
            torch.full((4, 2), 0x11111111, dtype=torch.int32),
            "w2_qzeros",
            "w2",
            100,
        )
        self.assertTrue(torch.all(param[100] == 0x11).item())
        mapper.assert_not_called()

    def test_generic_weight_loader_still_receives_global_expert_id(self):
        loader, mapper, fallback = self.make_loader()
        param = self.parameter((20, 16, 8))
        loader(
            param,
            torch.full((16, 2), 0x11111111, dtype=torch.int32),
            "w13_qweight",
            "w1",
            100,
        )
        self.assertEqual(fallback.call_args.args[-1], 100)
        mapper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
