"""One exact-node lease, reused full bank, one baseline/candidate PP arm."""
import argparse,json,os,shlex,signal,subprocess,time,urllib.error,urllib.request,sys,threading
from pathlib import Path
from common import ROOT,config,save,environment,GUARD,ARMS,select_arm,sha,execution_arms
from auth import authenticate
from timing import extract_weight_load
from measure import measure_all
BOOTSTRAP="import json,os,sys\np=json.loads(sys.stdin.readline())\nos.environ.update(p['env'])\nos.execv(p['command'][0],p['command'])\n"

def may_continue(row):
    if row.get('passed'):return row.get('cleanup_verified') is True
    return (row.get('role')=='consumer' and row.get('performance_valid') is True
        and row.get('profile_valid') is False and bool(row.get('profile_failure'))
        and row.get('failure') is None and row.get('cleanup_verified') is True)

def completed_counts(arms):
    rows=[row for row in arms if row.get('role')=='consumer']
    return dict(valid_points=sum(row.get('passed') is True and row.get('cleanup_verified') is True for row in rows),
        performance_points=sum(row.get('performance_valid') is True and row.get('cleanup_verified') is True for row in rows),
        profile_points=sum(row.get('profile_valid') is True and row.get('cleanup_verified') is True for row in rows))

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
            log=(run/cfg['arm']/(node+'.ssh.log')).open('x')
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
        while not all((run/cfg['arm']/'capacity'/f'rank{r}.json').exists() for r in range(8)):
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
                row,visibility=wait_for_stable_json(run/cfg['arm']/node/'exit.json',timeout_s=20)
                cleanups.append(row);save(run/cfg['arm']/(node+'.visibility.json'),visibility)
            except Exception as e:cleanups.append(dict(node=node,owned_children_exited=False,error=str(e)))
        for rank,node in enumerate(cfg['nodes']):
            try:timings.append(dict(node=node,**extract_weight_load((run/cfg['arm']/node/'target_weight.log').read_text(),range(4*rank,4*rank+4))))
            except Exception as e:failure=failure or dict(message='Missing complete weight-load timing',node=node,error=str(e))
    clean=len(cleanups)==2 and all(r.get('owned_children_exited') and r.get('gpu_cleanup',{}).get('passed') for r in cleanups)
    consumer=cfg['role']=='consumer'
    performance_valid=bool(failure is None and len(result)==1 and result[0].get('performance_valid') is True) if consumer else None
    profile_valid=bool(performance_valid and result[0].get('profile_valid') is True) if consumer else None
    profile_failure=result[0].get('profile_failure') if consumer and len(result)==1 else None
    passed=failure is None and len(result)==1 and clean and (not consumer or (performance_valid and profile_valid))
    row=dict(arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],role=cfg['role'],passed=passed,
        performance_valid=performance_valid,profile_valid=profile_valid,profile_failure=profile_failure,failure=failure,results=result,
        cleanup_verified=clean,cleanups=cleanups,weights_load=timings,launch_to_ready_s=ready_s)
    save(run/cfg['arm']/'arm_result.json',row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run;cfg=config(run)
    authenticate(run,cfg);cancelled={'signal':0}
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,lambda n,_:cancelled.update(signal=n))
    env={k:os.environ[k] for k in ('EXPERIMENT_GUARD_LEASE_RECEIPT','EXPERIMENT_GUARD_ALLOCATION_FILE','EXPERIMENT_GUARD_ACQUIRED_NODES','EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')}
    arms=[]
    if cfg.get('reuse_semantic_bank'):
        save(run/'bank_reuse.json',dict(passed=True,source=cfg['source_semantic_bank_ready'],
            source_bank_modified=False,producer_skipped=True,republication=False))
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',controller_pid=os.getpid(),valid_points=0))
    for arm in cfg['arm_order']:
        if cancelled['signal']:break
        selected=select_arm(cfg,arm)
        save(run/'task/progress.json',dict(timestamp=time.time(),stage='loading',arm=arm,**completed_counts(arms),controller_pid=os.getpid()))
        row=run_arm(run,selected,env,cancelled);arms.append(row)
        if not may_continue(row):break
        if arm=='producer':
            if cfg.get('reuse_semantic_bank'):raise RuntimeError('Reuse mode must never publish or run a producer')
            try:
                from semantic_bank import finalize_bank
                from workload import prompts
                evidence=[dict(path=str(run/arm/node/'exit.json'),sha256=sha(run/arm/node/'exit.json')) for node in cfg['nodes']]
                bank=finalize_bank(cfg['semantic_bank_dir'],prompts(),cfg['bank_model_identity'],
                    dict(producer_exited=True,cleanup_passed=True,evidence=evidence))
                save(run/'bank_publication.json',bank)
            except BaseException as exc:
                row.update(passed=False,failure=dict(type=type(exc).__name__,message=str(exc)))
                save(run/arm/'arm_result.json',row);break
    clean=bool(arms) and all(r['cleanup_verified'] for r in arms)
    counts=completed_counts(arms)
    passed=(len(arms)==len(execution_arms(cfg)) and [r['arm'] for r in arms]==cfg['arm_order']
        and all(r['passed'] for r in arms) and counts['performance_points']==counts['profile_points']==4)
    save(run/'task/results.json',dict(passed=passed,**counts,expected=4,expected_performance=4,expected_profiles=4,mode='fourway',
        reuse_semantic_bank=cfg.get('reuse_semantic_bank',False),source_semantic_bank_ready=cfg.get('source_semantic_bank_ready'),arms=arms))
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='complete' if passed else 'failed',**counts,progress_percent=100 if passed else 0))
    save(run/'task/cleanup.json',dict(timestamp=time.time(),experiment_id=cfg['id'],task_id='decode',passed=clean,
        gpu_idle=clean,model_processes_exited=clean,gpu_resources_released=clean))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
