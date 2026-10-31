import copy,json,unittest
from awq_tma_fix import zero_point_tma_fallback

class FallbackTest(unittest.TestCase):
    def test_only_zero_point_tma_is_disabled_and_serialized(self):
        configs={"compute_config":{"gemm_type":"grouped_masked"},"compute_config_str":"original",
            "w13_tuning_config":{"use_tma":True,"use_tma_bzp":True,"block_shape":[16,128,128]},
            "w2_tuning_config":{"use_tma":True,"use_tma_bzp":None,"use_warp_spec":True}}
        original=copy.deepcopy(configs)
        fixed=zero_point_tma_fallback(configs)
        self.assertEqual(configs,original)
        self.assertEqual(fixed["compute_config_str"],"original")
        for name in ("w13_tuning_config","w2_tuning_config"):
            expected=dict(configs[name],use_tma_bzp=False)
            self.assertEqual(fixed[name],expected)
            self.assertEqual(json.loads(fixed[name+"_str"]),expected)
