"""Complete the five-layout comparison on node178 with qualified early-send."""
import ast,copy,json,shutil
import prepare
run=prepare.prepare('ep','pp2_r1');previous=prepare.ROOT/'runs/pp_171_r7'
for p in (previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
adapter='''"""Only the qualified native and early-send PP paths are installed."""
import os
_DONE=False
def install():
 global _DONE
 if _DONE:return
 pp=int(os.environ.get('HICACHE_PP_SIZE','1'))
 if pp in (2,4):
  import gap_schedule
  from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
  from sglang.srt.managers.scheduler import Scheduler
  from pp_schedule_opt import open_policy
  gap_schedule._POLICY=open_policy(os.environ['PP_GAP_POLICY_PATH'])
  gap_schedule.install_early(SchedulerPPMixin,gap_schedule.mode)
  previous=Scheduler.get_internal_state
  def state(self,*args,**kwargs):
   result=previous(self,*args,**kwargs)
   result.internal_state['benchmark_pp_policy']=dict(policy=gap_schedule.mode(),pipeline_size=pp,native_transport=True,microbatch=getattr(self,'pp_max_micro_batch_size',None))
   return result
  Scheduler.get_internal_state=state
 _DONE=True
'''
(run/'code/pp_optim.py').write_text(adapter)
p=run/'code/runner.py';s=p.read_text()
old="candidates=[(0,16),(1,16),(2,16),(1,8),(2,8)] if phase=='prefill' else [(0,16),(1,16),(2,16),(1,8),(2,8),(1,4),(2,4)]"
assert old in s;s=s.replace(old,"candidates=[(policy,microbatch) for microbatch in (16,8,4) for policy in (0,1)]")
p.write_text(s)
cfg=json.loads((run/'config.json').read_text());row=copy.deepcopy(next(r for r in json.loads((previous/'config.json').read_text())['layouts'] if r['id']=='two_pp2'))
row['usage_logs']=[dict(id=cfg['id']+f'.two_pp2.r{r}',path=str(run/'cache'/f'two_pp2.r{r}.log'),expected_ranks=['PP0 TP0','PP1 TP0']) for r in range(2)]
cfg['layouts']=[row];cfg['expected_points']=10;cfg['model_cache']['on_miss']='build';cfg['model_cache']['usage_logs']={cfg['node']:row['usage_logs']}
cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000,pp_mem_fraction_static=.92,pp_decode_graph_max_bs=16)
cfg['protocol']+=' Final five layouts all on node178. PP2 installs only native and event-preserving early-send; selects both policies at microbatch16/8/4. Lookahead experimental transport failed qualification; tail-local2 produced a slot overwrite under long-context HiCache pressure. Both excluded. PP static memory0.92, Graph maxB16, native message transport. HiCache remains enabled throughout.'
prepare.save(run/'config.json',cfg)
prepare.save(run/'unsupported_optimizations.json',dict(lookahead=dict(status='not_qualified',evidence=str(prepare.ROOT/'runs/pp_171_r6')),tail_local2=dict(status='rejected_long_context_slot_overwrite',evidence=str(previous/'cache/two_pp2.r1.log')),final_candidates='Native0 and early-send1 x microbatch16/8/4'))
spec=json.loads((run/'dashboard_spec.json').read_text());spec['model_cache']=cfg['model_cache'];spec['tasks'][0]['expected']=10
spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
for p in [*(run/'code').glob('*.py'),run/'unsupported_optimizations.json']:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
prepare.save(run/'dashboard_spec.json',spec)
print('Frozen final same-node PP2 comparison',run)
