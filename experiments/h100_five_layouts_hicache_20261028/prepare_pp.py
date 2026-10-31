from pathlib import Path
import ast,hashlib,json,shutil,sys
import prepare
ROOT=prepare.ROOT
run=prepare.prepare('pp','r1');cfg=json.loads((run/'config.json').read_text())
# Use the latest generic PP2 scheduler, without importing 122B model adapters.
source=prepare.WS/'experiments/h100_qwen122b_pp_refine_20261027/balance_warm_source/gap_schedule.py'
shutil.copy2(source,run/'code/gap_schedule.py')
p=run/'code/gap_schedule.py';s=p.read_text();s=s.replace("if self.ps.pp_size != 2 or self._pp_topology() is not None:\n                raise RuntimeError(\"Early activation requires static PP2\")","if self.ps.pp_size not in (2,4) or self._pp_topology() is not None:\n                raise RuntimeError(\"Early activation requires static PP2 or PP4\")");p.write_text(s)
shutil.copy2(ROOT/'pp_optim.py',run/'code/pp_optim.py')
p=run/'code/runtime_adapter.py';p.write_text(p.read_text()+"\n_hicache_install=install\ndef install():\n    _hicache_install()\n    import pp_optim\n    pp_optim.install()\n")
p=run/'code/runner.py';s=p.read_text();s=s.replace("HICACHE_BENCHMARK='1')","HICACHE_BENCHMARK='1',HICACHE_PP_SIZE=str(row['pp']),PP_GAP_SEPARATE_CHANNELS='1')")
a=s.index('def replay_workload(');s=s[:a]+(ROOT/'pp_selection.py.inc').read_text()+'\n\n'+s[a:]
s=s.replace("        for phase in ('prefill','decode'):\n            rounds=[];audits=[]", "        for phase in ('prefill','decode'):\n            selected=select_pp(run,cfg,arm,key,tokens,length,cache_mode,phase,health,out) if layout(cfg,arm)['pp']>1 else None\n            rounds=[];audits=[]")
s=s.replace("cache_tiers_by_round=audits))", "cache_tiers_by_round=audits,selected_optimization=selected))")
p.write_text(s)
# Retain complete generated text for optimization equivalence checks.
p=run/'code/base_runner.py';p.write_text(p.read_text().replace("final.get('text', '')[-200:]","final.get('text', '')"))
for p in (run/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
spec=json.loads((run/'dashboard_spec.json').read_text())
for row in spec['frozen_inputs']:row['sha256']=prepare.sha(row['path'])
known={r['path'] for r in spec['frozen_inputs']}
for p in (run/'code').glob('*.py'):
 if str(p) not in known:spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
prepare.save(run/'dashboard_spec.json',spec)
print('PP optimizations frozen',run)
