"""Use output-qualified PP2 candidates; exclude unsafe HiCache lookahead."""
import ast,json,shutil
import prepare
run=prepare.prepare('pp','r7');previous=prepare.ROOT/'runs/pp_171_r6'
for p in (previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
p=run/'code/pp_optim.py';s=p.read_text().replace('  import dynamic_pp_schema\n  gap_schedule.install_schema=dynamic_pp_schema.install_schema\n','');p.write_text(s)
p=run/'code/runner.py';s=p.read_text().replace("PP_GAP_SEPARATE_CHANNELS='1'","PP_GAP_SEPARATE_CHANNELS='0'")
old="candidates=[(0,16),(1,16),(1,8)] if phase=='prefill' else [(0,16),(9,16),(10,16),(9,8)]"
new="candidates=[(0,16),(1,16),(2,16),(1,8),(2,8)] if phase=='prefill' else [(0,16),(1,16),(2,16),(1,8),(2,8),(1,4),(2,4)]"
assert old in s;s=s.replace(old,new);p.write_text(s)
cfg=json.loads((run/'config.json').read_text());cfg['layouts']=[r for r in cfg['layouts'] if r['id']!='pp4'];cfg['expected_points']=20
cfg['model_cache']['usage_logs']={cfg['node']:[u for row in cfg['layouts'] for u in row['usage_logs']]}
cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000,pp_mem_fraction_static=.92,pp_decode_graph_max_bs=16,ep_control_mem_fraction_static=.94)
cfg['protocol']+=' PP2 policy9 HiCache pilot failed output equality and later fixed output-length qualification in pp_171_r6. Exclude lookahead9 and unqualified dependent10. Final PP2 candidates use native typed messages: native0, early-send1, tail-local2, microbatch16/8/4, no extra NCCL or Gloo channels. PP static memory0.92, Graph maxB16. Best valid candidate selected separately per cache mode/context/phase.'
prepare.save(run/'config.json',cfg)
prepare.save(run/'unsupported_optimizations.json',dict(policy9=dict(status='rejected_output_validation',pilot=str(previous/'results/two_pp2_p8256/pilot.natural_replay.decode.p9.mb16.r0.json'),later_failure_log=str(previous/'controller.log')),policy10=dict(status='not_qualified',reason='Depends on rejected policy9 lookahead path'),final_candidates='Native0 / early-send1 / tail-local2, native message transport'))
spec=json.loads((run/'dashboard_spec.json').read_text());spec['model_cache']=cfg['model_cache'];spec['tasks'][0]['expected']=20
spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
for p in [*(run/'code').glob('*.py'),run/'unsupported_optimizations.json']:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
prepare.save(run/'dashboard_spec.json',spec)
print('Frozen output-qualified PP2 candidates',run)
