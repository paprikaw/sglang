"""Read-only aggregate; sequential telemetry, no model traffic or resource mutation."""
import sys,time,json,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'tools/idle_gpu_benchmark'))
from prefill_monitor import save,query_gpu

def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}

def main():
 lock=(ROOT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  now=time.time();rows={};attempts={}
  for n in ['spartan-gpgpu065','spartan-gpgpu069']:
   r=ROOT/n;state=read(r/'state.json');a=state.get('attempts',{}).get(n,{});w=read(r/'prefill/nodes'/f'{n}.json');run=a.get('run');d=read(Path(run)/'deployment.json')if run else {}
   current=w.get('run')==run and now-w.get('timestamp',0)<15
   row=dict(run=run,deployment_id=d.get('deployment_id'),worker_updated_at=w.get('timestamp')if current else None,worker_status=w.get('status')if current else 'loading',model=Path(d.get('model_path','unknown')).name,state_reason=state.get('nodes',{}).get(n,{}).get('reason'),cumulative=w.get('cumulative')if current else None)
   if current and w.get('status')=='running':
    g=query_gpu(n);row.update(gpus=g['gpus'],gpu_stale=bool(g['error']))
   rows[n]=row
   if a:attempts[n]=a
  rows['spartan-gpgpu169']=dict(worker_status='waiting',state_reason='HY3 分层缓存缺少配对节点 168；等待完整模型所需资源')
  save(ROOT/'state.json',dict(state='running',updated_at=time.time(),attempts=attempts))
  save(ROOT/'all_nodes_state.json',dict(timestamp=time.time(),nodes=rows))
  time.sleep(5)
if __name__=='__main__':main()
