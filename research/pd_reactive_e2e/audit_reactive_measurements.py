"""Verify completed measurements without changing a failed supervisor receipt."""
from pathlib import Path
import hashlib,json,sys,time
import analyze_reactive_pilot as analysis

r=Path(sys.argv[1]).resolve()
s=analysis.analyze(r)
assert s['complete_workload'] and s['expected_tasks']==4 and s['expected_requests']==76
read=lambda p:json.loads((r/p).read_text())
assert read('replay_workload_done.json')['exit_code']==0
start=read('start_smoke.json');post=read('post_smoke.json')
ids=start['responses'][0]['output_ids']
assert start['passed'] and post['passed'] and len(ids)==8 and all(isinstance(x,int) for x in ids)
assert all(x['output_ids']==ids and x['output_tokens']==8 for x in start['responses'])
assert post['response']['output_ids']==ids and post['response']['output_tokens']==8
actions=read('actions.json')
assert [a['decision']['direction'] for a in actions]==['up','down']
assert all(a['success'] and a['answer']['success'] for a in actions)
for action,counts in zip(actions,[[16,16,16],[24,0,24]]):
 states=action['after']['internal_states']
 assert len(states)==3 and all(x['elastic_pp_state']['layer_counts']==counts and x['transition_metrics']['phase']=='done' for x in states)
final=read('state.final.json')['internal_states']
assert all(x['elastic_pp_state']['layer_counts']==[24,0,24] and x['elastic_pp_state']['migration_phase']=='done' for x in final)
checks=sorted((r/'dashboard_guard/checks').glob('*post.json'))
assert len(checks)==1
check=json.loads(checks[0].read_text())
assert check['cleanup_verified'] and check['model_cache']['passed'] and check['passed'] is False
assert read('task/cleanup.json')['passed']
for p in (r/'servers').glob('*/cleanup.json'):
 j=json.loads(p.read_text());assert j['cleanup_verified'] and j['node_gpu_idle_observed']
client=read('client_cleanup.json')
assert client['children_reaped'] and client['process_groups_exited']
assert not client['owned_gpu_processes'] and not client['cleanup_errors']
assert client['cleanup_verified'] is False and client['observed_gpu_processes']
failure=read('failure.json')['error']
assert 'TimeoutExpired' in failure and '--client' in failure
service_end=float((r/'release_router').read_text())
assert actions[-1]['finished']<=service_end
assert service_end<= (r/'failure.json').stat().st_mtime
assert '"gpu_registry_state": "released"' in (r/'guard.launch.log').read_text()
paths=['config.json','start_smoke.json','post_smoke.json','state.final.json','actions.json','release_router','failure.json','client_cleanup.json','task/cleanup.json','analysis_export_manifest.json',str(checks[0].relative_to(r))]
value=dict(audited_at_unix=time.time(),measurements_complete=True,supervisor_passed=False,
  original_supervisor_failure=failure,service_end_unix=service_end,formal_requests=s['formal_requests'],task_count=s['task_count'],
  scaling_round_trip_observed=True,output_ids_compared=8,final_gpu_cleanup_verified=True,
  scope='Completed workload and transitions may be described; original failed supervisor and client cleanup receipts remain unchanged. This is not a fully passing experiment.',
  files={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in paths})
(r/'analysis/evidence_audit.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k!='files'}))
