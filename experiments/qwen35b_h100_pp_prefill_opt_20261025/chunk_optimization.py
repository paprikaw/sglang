"""One warmed PP process: four native Prefill budgets in mirrored order."""
import json
import gzip
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

BUDGETS=(4096,2048,1024,8192)


def set_mode(path,mode):
    # This is a private policy, validated by the same one-byte reader as the
    # producer-ready experiment. Only three frozen admission budgets exist.
    if mode not in range(4):raise ValueError('Unknown chunk policy')
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        import stat
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or st.st_mode&0o077 or st.st_size!=1:
            raise RuntimeError('Chunk policy must be a private one-byte regular file')
        os.pwrite(fd,str(mode).encode(),0)
    finally:os.close(fd)


def worker(run,cfg):
    q=sys.modules['__main__']
    incoming=json.loads(sys.stdin.readline())
    env=dict(os.environ,**incoming['environment'])
    q.h.authenticate(run,cfg,env,local=True)
    def interrupted(signum,frame):raise RuntimeError('Guard/controller requested cleanup')
    signal.signal(signal.SIGTERM,interrupted)
    def watch_parent():
        sys.stdin.read();os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watch_parent,daemon=True).start()
    import pp_ready_send as policy
    for path in (cfg['policy_path'],cfg['coalesce_policy_path'],cfg['chunk_policy_path']):
        policy.create_policy(path)
    children=[];failure=None;points=[];key=q.h.read_key(cfg)
    def health():
        q.h.authenticate(run,cfg,env,local=True)
        if any(x['process'].poll() is not None for x in children):raise RuntimeError('PP server exited')
    try:
        topology=subprocess.run(['nvidia-smi','topo','-m'],text=True,capture_output=True,check=True,timeout=20)
        (run/'topology.txt').write_text(topology.stdout)
        q.make_tokens(cfg,1,16)
        q.progress(run,cfg,'pp_model_loading')
        local=q.arm_cfg(cfg,'pp2')
        child=q.launch(run,local,'pp2',env,key);children.append(child)
        capacity=q.wait_ready(run,local,'pp2',child,health)
        q.h.save(run/'capacities.json',dict(pp2=capacity))
        for row in cfg['optimization_cases']:
            n=row['requests'];cached=row['cached_tokens'];fresh=row['new_tokens'];cold=cached==0
            q.set_microbatch(local,'pp2',key,n)
            tokens=q.make_tokens(cfg,n,fresh if cold else cached)
            def prepare(branch, subset=None):
                set_mode(cfg['chunk_policy_path'],0)
                chosen=tokens if subset is None else tokens[:subset]
                if cold:
                    q.b.flush_cache(local,key)
                    return [q.b.packet(local,'pp2',t,1,i) for i,t in enumerate(chosen)],None
                prep=q.prime(local,'pp2',key,chosen,health)
                return q.payloads(local,'pp2',chosen,fresh,1,branch),prep
            def validate(result,count=n):
                if len(result['requests'])!=count:raise RuntimeError('Incomplete cohort')
                if not cold:q.check_cached(result,cached)
                elif any((x.get('cached_tokens') or 0)!=0 for x in result['requests']):
                    raise RuntimeError('Cold Prefill unexpectedly hit cache')
                if any(x['output_tokens']!=1 for x in result['requests']):raise RuntimeError('Output length changed')
            # Warm every candidate budget outside timing. Eight long-context
            # requests exercise every full per-forward request shape, without
            # constructing all 32 prefixes merely to warm a kernel.
            for mode in range(4):
                q.progress(run,cfg,f'warm_{row["id"]}_{mode}',workload=row)
                req,prep=prepare(1100+mode,subset=min(n,8))
                set_mode(cfg['chunk_policy_path'],mode)
                result=q.b.cohort(local,key,req);validate(result,len(req))
                q.h.save(run/'pp2'/f'warmup.{row["id"]}.{mode}.json',
                         dict(performance_sample=False,budget=BUDGETS[mode],preparation=prep))
            for index,mode in enumerate((0,1,2,3,3,2,1,0)):
                q.progress(run,cfg,f'{row["id"]}_{index}',workload=row,chunk_tokens=BUDGETS[mode])
                req,prep=prepare(1200+index)
                set_mode(cfg['chunk_policy_path'],mode)
                health();result=q.b.cohort(local,key,req);validate(result)
                point=dict(case=row,arm='pp2',trial=index,chunk_tokens=BUDGETS[mode],
                           preparation=prep,result=result,new_token_throughput=n*fresh/result['makespan_s'])
                q.h.save(run/'pp2'/f'metrics.{row["id"]}.{index}.json',point);points.append(point)
            if row['id']=='short_cached32':
                outputs=[]
                for mode in range(4):
                    req,prep=prepare(1299);set_mode(cfg['chunk_policy_path'],mode)
                    result=q.b.cohort(local,key,req);validate(result)
                    outputs.append([x['output_ids'] for x in result['requests']])
                q.h.save(run/'pp2/chunk_output_agreement.json',dict(
                    passed=all(x==outputs[0] for x in outputs),output_ids=outputs,
                    limitation='Different native chunk partitions may change floating-point accumulation; performance is not a quality benchmark'))
            if row['id']=='long_cached32':
                # Profile baseline and the fastest measured candidate, after
                # all formal cohorts; ranking uses only the unprofiled runs.
                import statistics
                speeds={m:statistics.mean(p['new_token_throughput'] for p in points
                        if p['case']['id']==row['id'] and p['chunk_tokens']==BUDGETS[m]) for m in range(4)}
                winner=max((1,2,3),key=lambda m:speeds[m])
                for mode in (0,winner):
                    req,prep=prepare(1300);set_mode(cfg['chunk_policy_path'],mode)
                    d=run/'pp2/traces'/f'chunk{BUDGETS[mode]}';d.mkdir(parents=True)
                    q.b.profile_action(local,key,'/start_profile',dict(output_dir=str(d),
                        activities=['CPU','GPU'],record_shapes=True,with_stack=False,
                        profile_id=f'chunk-{mode}',merge_profiles=False,detailed_annotations=True))
                    try:result=q.b.cohort(local,key,req)
                    finally:q.b.profile_action(local,key,'/stop_profile')
                    validate(result)
                    q.h.save(d/'cohort.json',dict(performance_sample=False,chunk_tokens=BUDGETS[mode],result=result,preparation=prep))
                    deadline=time.monotonic()+120
                    previous=None
                    while True:
                        health()
                        files=sorted(d.glob('*.trace.json.gz'))
                        try:
                            parsed=[]
                            for p in files:
                                raw=p.read_bytes();json.loads(gzip.decompress(raw))
                                parsed.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
                            if len(parsed)==2 and previous==parsed:
                                q.h.save(d/'manifest.json',dict(passed=True,files=parsed,
                                    performance_sample=False,stable_observations=True));break
                            previous=parsed if len(parsed)==2 else None
                        except (OSError,EOFError,ValueError):previous=None
                        if time.monotonic()>deadline:raise TimeoutError('Both PP traces must finish exporting')
                        time.sleep(.2)
        q.h.save(run/'completed.json',dict(passed=True,points=len(points),expected_points=cfg['expected_points']))
    except BaseException:
        failure=traceback.format_exc();(run/'worker_error.txt').write_text(failure);raise
    finally:
        evidence=q.h.cleanup(children) if children else dict(cleanup_verified=False)
        q.h.save(run/'worker_cleanup.json',evidence)
        for path in (cfg['policy_path'],cfg['coalesce_policy_path'],cfg['chunk_policy_path']):Path(path).unlink(missing_ok=True)
        if failure is None and not evidence['cleanup_verified']:raise RuntimeError('Owned cleanup failed')
    return 0
