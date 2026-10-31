"""CPU checks for actual admission boundaries and scientific postcheck failures."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h100_surface_policy as policy
from common import PLAN,MODELS

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.workers=[dict(id=f'{m}_{node}',node=node,gpu=i,model=m) for node in ['n0','n1'] for i,m in enumerate(MODELS)]
        self.plan=copy.deepcopy(PLAN)
        self.selections={m:dict(model_path=policy.MODEL_PATHS[m],layer=g[0],resident=g[1],hidden=g[2],intermediate=g[3],
              expert_ids=list(range(g[1])),active_experts=MODELS[m]['experts'],tensors={'a.mlp.experts.b':{}},config_sha256='a'*64,index_sha256='b'*64) for m,g in policy.GEOMETRY.items()}
        self.paths={}
        for name,obj in [('plan',self.plan),('workers',self.workers),('selection',self.selections)]:
            p=self.root/(name+'.json');p.write_text(json.dumps(obj));self.paths[name]=dict(path=str(p),sha256=policy.digest(p))
        for name in ['controller','worker']:
            p=self.root/(name+'.py');p.write_text('pass\n');self.paths[name]=dict(path=str(p),sha256=policy.digest(p))
        c=dict(policy='expert_layer_only',reason='User requested exact four-model benchmark',benchmark=policy.BENCHMARK,
            selection=self.paths['selection'],measurement_plan=self.paths['plan'],workers=self.paths['workers'],entrypoint=self.paths['controller'],worker=self.paths['worker'],
            command=['python3',self.paths['controller']['path'],'--run',str(self.root)],worker_command=['python3',self.paths['worker']['path'],'--run',str(self.root)])
        count=sum(sum(e<=t for t in PLAN['tokens'] for e in m['experts'])*2 for m in MODELS.values())
        self.spec=dict(engine='sglang',max_active_nodes=2,tasks=[dict(nodes=[n],expected=count) for n in ['n0','n1']],
            workload=dict(kind='h100_four_model_expert_layer_compute',model_loaded=False,expert_compute=True,attention_compute=False,communication=False,gpu_count=8),
            frozen_inputs=list(self.paths.values()),model_cache=c)
    def fake_frozen(self,spec,d):return Path(d['path'])
    def test_valid_bounded_plan(self):
        with patch.object(policy,'frozen',side_effect=self.fake_frozen):self.assertEqual(policy.validate(self.spec)['benchmark'],policy.BENCHMARK)
    def test_reject_model_or_gpu_expansion(self):
        for workload in [dict(self.spec['workload'],model_loaded=True),dict(self.spec['workload'],gpu_count=16),dict(self.spec['workload'],communication=True)]:
            with self.subTest(workload=workload),patch.object(policy,'frozen',side_effect=self.fake_frozen),self.assertRaises(ValueError):policy.validate(dict(self.spec,workload=workload))
    def test_changed_shape_or_sample_count(self):
        for key,value in [('tokens',[65536]),('samples_per_mode',1)]:
            p=Path(self.paths['plan']['path']);p.write_text(json.dumps(dict(self.plan,**{key:value})))
            with patch.object(policy,'frozen',side_effect=self.fake_frozen),self.assertRaises(ValueError):policy.validate(self.spec)
    def test_gpu_affinity_authentication(self):
        c=self.spec['model_cache'];self.assertEqual(policy.worker_gpu(c,'n0','glm47_n0','0'),'0')
        for args in [('n0','glm47_n0','1'),('n1','glm47_n0','0'),('n0','unknown','0')]:
            with self.subTest(args=args),self.assertRaises(RuntimeError):policy.worker_gpu(c,*args)
    def test_frozen_input_mutation(self):
        d=self.paths['plan'];Path(d['path']).write_text('{}')
        with self.assertRaises(ValueError):policy.frozen(self.spec,d)
    def test_duplicate_or_missing_cells_rejected(self):
        (self.root/'profile').mkdir()
        (self.root/'profile/result.json').write_text(json.dumps(dict(passed=True,rows=[],numerical=[dict(passed=True)])))
        with self.assertRaises(RuntimeError):policy.verify_result(self.spec['model_cache'],dict(admitted_at=0,lease_id='l'))

if __name__=='__main__':unittest.main()
