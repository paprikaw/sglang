"""One exact-node lease, preloaded full bank, two sequential complete-model arms."""
import argparse,json,os,shlex,signal,subprocess,time,urllib.error,urllib.request,sys,threading
from pathlib import Path
from common import ROOT,config,save,environment,GUARD
from auth import authenticate
from timing import extract_weight_load
from measure import measure_all
BOOTSTRAP="import json,os,sys\np=json.loads(sys.stdin.readline())\nos.environ.update(p['env'])\nos.execv(p['command'][0],p['command'])\n"

def run_arm(run,cfg,guard_env,cancelled):
    children=[];cleanups=[];timings=[];failure=None;result=[];ready_s=None;started=time.monotonic()
    finished=threading.Event()
    def cancellation_watch():
        while not finished.wait(.2):
            if cancelled['signal']:
                for _,process,_ in list(children):
                    if process.poll() is None:
                        try:process.stdin.write('stop\n');process.stdin.flush()
                        except (BrokenPipeError,OSError,ValueError):pass
                return
    watcher=threading.Thread(target=cancellation_watch,daemon=True)
    watcher.start()
    try:
        for rank,node in enumerate(cfg['nodes']):
            command=[cfg['python'],str(GUARD/'model_cache_adapter.py'),'exec-local','--bindings',guard_env['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'],'--',cfg['python'],str(ROOT/'runtime/model_node.py'),'--run',str(run),'--node-rank',str(rank)]
            log=(run/cfg['layout']/(node+'.ssh.log')).open('x')
            p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,shlex.join(['/usr/bin/python3','-c',BOOTSTRAP])],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            children.append((node,p,log));p.stdin.write(json.dumps(dict(env=environment(cfg,rank,guard_env),command=command))+'\n');p.stdin.flush()
        url=f'http://{cfg["nodes"][0]}:{cfg["api_port"]}'
        while True:
            if cancelled['signal'] or any(p.poll() is not None for _,p,_ in children):raise RuntimeError('Cancelled or model worker exited')
            authenticate(run,cfg,frozen=False)
            try:
                with urllib.request.urlopen(url+'/health_generate',timeout=5) as r:r.read()
                ready_s=time.monotonic()-started;break
            except (urllib.error.URLError,TimeoutError):pass
            if time.monotonic()-started>cfg['timeout_s']:raise TimeoutError('Model readiness timed out')
            time.sleep(2)
        deadline=time.monotonic()+120
        while not all((run/cfg['layout']/'capacity'/f'rank{r}.json').exists() for r in range(8)):
            if time.monotonic()>deadline:raise TimeoutError('Missing all-rank natural capacity proof')
            time.sleep(.5)
        save(run/'task/progress.json',dict(stage='measuring',layout=cfg['layout'],timestamp=time.time(),valid_points=0))
        result=measure_all(url,run,cfg)
    except BaseException as e:
        failure=dict(type=type(e).__name__,message=str(e))
    finally:
        finished.set();watcher.join(timeout=1)
        for _,p,_ in children:
            if p.poll() is None:
                try:p.stdin.write('stop\n');p.stdin.flush()
                except (BrokenPipeError,OSError):pass
            p.stdin.close()
        sys.path.insert(0,str(GUARD));from artifact_visibility import wait_for_stable_json
        for node,p,log in children:
            try:p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:failure=failure or dict(message='SSH owner did not exit')
            log.close()
            try:
                row,visibility=wait_for_stable_json(run/cfg['layout']/node/'exit.json',timeout_s=20)
                cleanups.append(row);save(run/cfg['layout']/(node+'.visibility.json'),visibility)
            except Exception as e:cleanups.append(dict(node=node,owned_children_exited=False,error=str(e)))
        for rank,node in enumerate(cfg['nodes']):
            try:timings.append(dict(node=node,**extract_weight_load((run/cfg['layout']/node/'target_weight.log').read_text(),range(4*rank,4*rank+4))))
            except Exception as e:failure=failure or dict(message='Missing complete weight-load timing',node=node,error=str(e))
    clean=len(cleanups)==2 and all(r.get('owned_children_exited') and r.get('gpu_cleanup',{}).get('passed') for r in cleanups)
    row=dict(layout=cfg['layout'],passed=failure is None and len(result)==1 and clean,failure=failure,results=result,
        cleanup_verified=clean,cleanups=cleanups,weights_load=timings,launch_to_ready_s=ready_s)
    save(run/cfg['layout']/'arm_result.json',row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run;cfg=config(run)
    authenticate(run,cfg);cancelled={'signal':0}
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,lambda n,_:cancelled.update(signal=n))
    env={k:os.environ[k] for k in ('EXPERIMENT_GUARD_LEASE_RECEIPT','EXPERIMENT_GUARD_ALLOCATION_FILE','EXPERIMENT_GUARD_ACQUIRED_NODES','EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')}
    arms=[]
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',controller_pid=os.getpid(),valid_points=0))
    for layout in cfg['arm_order']:
        if cancelled['signal']:break
        selected=dict(cfg,layout=layout)
        save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',layout=layout,valid_points=sum(r['passed'] for r in arms),controller_pid=os.getpid()))
        row=run_arm(run,selected,env,cancelled);arms.append(row)
        if not row['passed']:break
    clean=bool(arms) and all(r['cleanup_verified'] for r in arms)
    passed=len(arms)==2 and all(r['passed'] for r in arms)
    save(run/'task/results.json',dict(passed=passed,valid_points=sum(r['passed'] for r in arms),expected=2,arms=arms))
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='complete' if passed else 'failed',valid_points=sum(r['passed'] for r in arms),progress_percent=100 if passed else 0))
    save(run/'task/cleanup.json',dict(timestamp=time.time(),experiment_id=cfg['id'],task_id='decode',passed=clean,
        gpu_idle=clean,model_processes_exited=clean,gpu_resources_released=clean))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
