import sys,json,time,tempfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('tools/idle_gpu_benchmark').resolve()))
import hy3_manager as m
cfg=json.loads(Path('experiments/idle_gpu_fleet_20261019/hy3_prefill_20261019/profile.json').read_text())
now=time.time();books=m.discover_bookings(cfg,now);ledger=m.read_json(Path(cfg['gpu_registry_dir']).parent/'cpu_weight_daemons/registry.json')
assert m.eligibility(cfg,books,{},ledger,now) is None
assert 'busy' in m.eligibility(cfg,books,{cfg['nodes'][0]:'busy'},ledger,now)
assert 'no current' in m.eligibility(cfg,{}, {},ledger,now)
original=hashlib.sha256((m.BASE/'dashboard_spec.json').read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
 run,source,spec=m.prepare(cfg,Path(tmp),now+1000)
 assert spec['resource_priority']=='background' and spec['preemptible']
 assert spec['model_cache']['reuse_only'] and spec['model_cache']['on_miss']=='fail'
 assert source['trace']['max_context_tokens']==2048
 for node in cfg['nodes']: assert spec['model_cache']['usage_logs'][node][0]['path'].startswith(str(run))
 for item in spec['frozen_inputs']: assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
assert hashlib.sha256((m.BASE/'dashboard_spec.json').read_bytes()).hexdigest()==original
import hy3_controller as c
class Fake:
 def post(self,url,**kwargs):return kwargs['json']
assert c.RankSession(Fake(),1).post('unused',json={'stream':True})=={'stream':True,'data_parallel_rank':1}
print('PASS live cache admission, blocked/missing booking, immutable reuse-only background spec and frozen inputs, DP routing')
