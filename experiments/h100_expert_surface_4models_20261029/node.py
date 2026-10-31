"""One authenticated booked-node supervisor, four isolated GPU workers."""
import argparse,hmac,json,os,select,signal,socket,subprocess,sys,time
from pathlib import Path
from common import ROOT,GUARD_SCRIPTS,KEYS,authenticate,save,signature,ticks,worker_environment
from remote_safety import enable_subreaper,track_descendants,stop_owned,gpu_cleanup_query

def main(run):
    packet=json.loads(sys.stdin.readline(65536));payload=packet['payload']
    receipt,spec,binding=authenticate(run,payload['env'])
    assert hmac.compare_digest(packet['signature'],signature(payload,receipt['token']))
    node=socket.gethostname().split('.')[0]
    cfg=json.loads((run/'config.json').read_text());assert node in cfg['nodes'] and payload['node']==node
    workers=[w for w in cfg['workers'] if w['node']==node]
    task=run/'tasks'/node;own={};children=[];cancelled=[];errors=[];passed=False
    owner=os.getpid();owner_ticks=ticks(owner);enable_subreaper()
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(s,lambda n,f:cancelled.append(n))
    poll=select.poll();poll.register(sys.stdin.fileno(),select.POLLIN|select.POLLHUP|select.POLLERR)
    try:
        for worker in workers:
            folder=run/'workers'/worker['id'];folder.mkdir(parents=True,exist_ok=True)
            env=worker_environment(run,worker,payload['env'])
            cmd=['python3',str(GUARD_SCRIPTS/'model_cache_adapter.py'),'exec-local','--bindings',payload['env'][KEYS[3]],'--',*spec['model_cache']['worker_command']]
            with (folder/'worker.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((worker,proc));own[proc.pid]=ticks(proc.pid)
        save(task/'launch.json',dict(timestamp=time.time(),node=node,job_id=cfg['jobs'][node],lease_id=receipt['lease_id'],
             owner_pid=owner,owner_start_ticks=owner_ticks,workers=[dict(**w,pid=p.pid,start_ticks=own[p.pid]) for w,p in children]))
        deadline=time.monotonic()+3600;next_progress=0
        while True:
            track_descendants(owner,owner_ticks,own)
            if cancelled:raise InterruptedError('Guard/controller cancellation')
            if time.monotonic()>deadline:raise TimeoutError('Single-layer workers exceeded one hour')
            if poll.poll(100):
                data=os.read(sys.stdin.fileno(),4096)
                if not data or data.strip()==b'STOP':raise InterruptedError('Controller control link closed')
            if time.monotonic()>next_progress:
                states=[];valid=0
                for w,p in children:
                    folder=run/'workers'/w['id'];partial={}
                    try:partial=json.loads((folder/'progress.json').read_text())
                    except (OSError,ValueError):pass
                    valid+=partial.get('valid_points',0)
                    states.append(dict(id=w['id'],gpu=w['gpu'],returncode=p.poll(),**partial))
                save(task/'progress.json',dict(timestamp=time.time(),stage='running',controller_pid=owner,process_start_ticks=owner_ticks,
                    valid_points=valid,expected_points=cfg['expected_per_node'],workers=states,note='Four independent single-layer H100 workers'))
                next_progress=time.monotonic()+2
            if all(p.poll() is not None for w,p in children):
                passed=all(p.returncode==0 for w,p in children);break
    except BaseException as e:
        errors.append(dict(type=type(e).__name__,error=str(e)))
    finally:
        for w,p in children:stop_owned(p,own,owner,owner_ticks)
        query=gpu_cleanup_query(20)
        results=[]
        for w,p in children:
            try:
                from common import stable
                r=stable(run/'workers'/w['id']/'result.json');assert r['worker']==w['id']
                results.append(r)
            except BaseException as e:errors.append(dict(worker=w['id'],type=type(e).__name__,error=str(e)))
        passed=passed and len(results)==4 and all(r.get('passed') is True for r in results)
        cleaned=query['passed'] and all(p.poll() is not None for w,p in children)
        save(task/'cleanup.json',dict(timestamp=time.time(),experiment_id=cfg['id'],task_id=node,passed=cleaned,
             model_processes_exited=cleaned,gpu_idle=query['passed'],evidence=query,owned_identities=[dict(pid=p,start_ticks=t) for p,t in own.items()]))
        valid=sum(len(r.get('rows',[])) for r in results)
        save(task/'result.json',dict(passed=passed,cleaned=cleaned,workers=results,errors=errors))
        save(task/'progress.json',dict(timestamp=time.time(),stage='complete' if passed and cleaned else 'failed',
             valid_points=valid,expected_points=cfg['expected_per_node'],errors=errors))
    return 0 if passed and cleaned else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);raise SystemExit(main(p.parse_args().run.resolve()))
