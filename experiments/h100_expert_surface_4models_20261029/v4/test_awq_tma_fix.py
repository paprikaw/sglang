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

    def test_native_shape_selector_parser_receives_fallback(self):
        import ast,os,types
        from unittest.mock import patch
        from pathlib import Path
        source=Path('/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/lib/python3.12/site-packages/humming/kernel/humming.py')
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare_kernels')
        method.decorator_list=[]
        calls=[]
        class FakeKernel:
            def __init__(self,**kwargs):
                calls.append(kwargs);self.kernel_id=len(calls)
            def load_cubin(self):pass
        namespace={'json':json,'os':os,'HummingKernel':FakeKernel,
            'torch':types.SimpleNamespace(tensor=lambda values,**kw:values,int64=int)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        native=[[0,127,{'use_tma':True,'block_shape':[16,128,128]}],
                [128,1<<30,{'use_tma':True,'block_shape':[64,128,128]}]]
        original=copy.deepcopy(native)
        fixed=zero_point_tma_fallback({'w13_tuning_config':native,'w2_tuning_config':native})
        cls=types.SimpleNamespace(_str2kernel_cache={})
        with patch.dict(os.environ,{'HUMMING_DISABLE_PARALLEL_BUILD':'1'}):
            parsed=namespace['prepare_kernels'](cls,{'b_dtype':'int4'}, {},fixed['w13_tuning_config_str'])
        self.assertEqual(native,original)
        self.assertEqual(parsed,[0,127,1,0,128,1<<30,2,0])
        self.assertEqual([x['block_shape'] for x in calls],[[16,128,128],[64,128,128]])
        self.assertTrue(all(x['use_tma'] and x['use_tma_bzp'] is False for x in calls))
