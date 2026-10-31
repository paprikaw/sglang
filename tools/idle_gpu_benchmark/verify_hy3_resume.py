import json,time,runpy
from pathlib import Path
root=Path.cwd()/'experiments/idle_gpu_fleet_20261019/hy3_prefill_20261019'
nodes=['spartan-gpgpu168','spartan-gpgpu169']
def ready(run):
 try:
  rows=[json.loads((root/'nodes'/(n+'.json')).read_text())for n in nodes]
  return all(r['run']==run and r['status']=='running' and r['cumulative']['successful_requests']>=2 and time.time()-r['timestamp']<10 for r in rows)
 except (FileNotFoundError,KeyError):return False
old=json.loads((root/'state.json').read_text())['run'];deadline=time.time()+900
while not ready(old):
 if time.time()>deadline:raise TimeoutError('initial requests not ready')
 time.sleep(2)
print('initial model serving on both DP replicas',flush=True)
runpy.run_path(str(Path(__file__).resolve().with_name('verify_hy3_handoff.py')),run_name='__main__')
started=time.time();deadline=started+900
while True:
 state=json.loads((root/'state.json').read_text())
 if state.get('run')!=old and state.get('run') and ready(state['run']):break
 if time.time()>deadline:raise TimeoutError('automatic recovery failed')
 time.sleep(2)
result=dict(passed=True,timestamp=time.time(),old_run=old,new_run=state['run'],manager_pid=state['manager_pid'],manager_rounds=state['rounds'],automatic_resume=True,ready_after_foreground_release_s=time.time()-started,nodes={n:json.loads((root/'nodes'/(n+'.json')).read_text())for n in nodes})
(root/'validation/automatic_recovery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='nodes'}),flush=True)
