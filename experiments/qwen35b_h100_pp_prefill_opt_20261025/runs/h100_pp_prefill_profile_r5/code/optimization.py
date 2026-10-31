"""Independent diagnostic only; join and CRC-check BOTH PP trace exports."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import zlib


def await_exports(directory, health, timeout=120):
    deadline=time.monotonic()+timeout
    while True:
        health()
        files=sorted(directory.glob('*.trace.json.gz'))
        exports=[]
        if len(files)==2:
            try:
                for path in files:
                    raw=path.read_bytes();data=json.loads(gzip.decompress(raw))
                    if not data.get('traceEvents'):raise ValueError('Empty trace')
                    exports.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            except (OSError,EOFError,ValueError,zlib.error):exports=[]
        if len(exports)==2:return exports
        if time.monotonic()>deadline:raise TimeoutError('Incomplete two-stage trace export')
        time.sleep(1)


def worker(run,cfg):
    q=sys.modules['__main__']
    incoming=json.loads(sys.stdin.readline());env=dict(os.environ,**incoming['environment'])
    q.h.authenticate(run,cfg,env,local=True)
    def interrupted(signum,frame):raise RuntimeError('Guard requested owned cleanup')
    signal.signal(signal.SIGTERM,interrupted)
    def watch_parent():
        sys.stdin.read();os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watch_parent,daemon=True).start()
    import pp_ready_send as policy
    policy.create_policy(cfg['policy_path']);policy.create_policy(cfg['coalesce_policy_path'])
    children=[];failure=None;profiles=[];key=q.h.read_key(cfg)
    def health():
        q.h.authenticate(run,cfg,env,local=True)
        if any(c['process'].poll() is not None for c in children):raise RuntimeError('Owned model exited')
    def progress(stage):
        q.h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',
            timestamp=time.time(),stage=stage,valid_points=len(profiles),expected=2))
    try:
        (run/'topology.txt').write_text(subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,check=True,timeout=20).stdout)
        progress('loading_PP2');q.make_tokens(cfg,1,16)
        local=q.arm_cfg(cfg,'pp2');child=q.launch(run,local,'pp2',env,key);children.append(child)
        capacities=q.wait_ready(run,local,'pp2',child,health);q.h.save(run/'capacities.json',dict(pp2=capacities))
        q.set_microbatch(local,'pp2',key,32);tokens=q.make_tokens(cfg,32,89984)
        def prepare():
            policy.set_policy(cfg['policy_path'],False)
            prep=q.prime(local,'pp2',key,tokens,health)
            return q.payloads(local,'pp2',tokens,512,1,1500),prep
        def validate(result):
            q.check_cached(result,89984)
            if len(result['requests'])!=32 or any(r['output_tokens']!=1 for r in result['requests']):raise RuntimeError('Invalid diagnostic requests')
        progress('exact_shape_warmup');req,_=prepare();validate(q.b.cohort(local,key,req))
        outputs=[]
        for candidate in (False,True):
            label='ready_send' if candidate else 'baseline';progress('profile_'+label)
            req,prep=prepare();policy.set_policy(cfg['policy_path'],candidate)
            directory=run/'pp2/traces'/label;directory.mkdir(parents=True)
            q.b.profile_action(local,key,'/start_profile',dict(output_dir=str(directory),activities=['CPU','GPU'],
                record_shapes=True,with_stack=False,profile_id='joined-'+label,merge_profiles=False,detailed_annotations=True))
            try:result=q.b.cohort(local,key,req)
            finally:q.b.profile_action(local,key,'/stop_profile')
            validate(result);exports=await_exports(directory,health)
            sample=dict(performance_sample=False,result=result,preparation=prep,ready_send=candidate)
            q.h.save(directory/'cohort.json',sample)
            manifest=dict(passed=True,performance_sample=False,all_prefill_chunks=True,files=exports,ready_send=candidate)
            q.h.save(directory/'manifest.json',manifest);profiles.append(manifest)
            outputs.append([r['output_ids'] for r in result['requests']])
        agrees=outputs[0]==outputs[1] and all(x is not None for x in outputs[0])
        q.h.save(run/'pp2/output_agreement.json',dict(passed=agrees,scope='independent two-stage diagnostic, identical inputs'))
        if not agrees:raise RuntimeError('Profile outputs changed')
        q.h.save(run/'completed.json',dict(passed=True,profile_only=True,points=2,expected_points=2));progress('complete')
    except BaseException:
        failure=traceback.format_exc();(run/'worker_error.txt').write_text(failure);raise
    finally:
        evidence=q.h.cleanup(children) if children else dict(cleanup_verified=False)
        q.h.save(run/'worker_cleanup.json',evidence)
        for name in ('policy_path','coalesce_policy_path'):Path(cfg[name]).unlink(missing_ok=True)
        if failure is None and not evidence['cleanup_verified']:raise RuntimeError('Unverified cleanup')
    return 0
