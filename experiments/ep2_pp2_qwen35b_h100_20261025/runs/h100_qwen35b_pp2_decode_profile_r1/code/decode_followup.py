"""Fill two missing PP Decode profile windows without rerunning formal metrics."""
import gzip
import json
import os
from pathlib import Path
import signal
import threading
import traceback


def worker(q,run,cfg):
    incoming=json.loads(q.sys.stdin.readline())
    env=dict(os.environ,**incoming['environment'])
    q.h.authenticate(run,cfg,env,local=True)
    def interrupted(signum,frame):
        raise RuntimeError('Guard requested owned diagnostic cleanup')
    signal.signal(signal.SIGTERM,interrupted)
    def watch_parent():
        q.sys.stdin.read();os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watch_parent,daemon=True).start()
    import pp_ready_send as ready
    ready.create_policy(cfg['policy_path']);ready.set_policy(cfg['policy_path'],False)
    key=q.h.read_key(cfg);children=[]
    try:
        def health():
            q.h.authenticate(run,cfg,env,local=True)
            if any(c['process'].poll() is not None for c in children):
                raise RuntimeError('Diagnostic PP server exited')
        q.make_tokens(cfg,1,16)
        local=q.arm_cfg(cfg,'pp2')
        q.progress(run,cfg,'pp_loading')
        children.append(q.launch(run,local,'pp2',env,key))
        capacity=q.wait_ready(run,local,'pp2',children[0],health)
        assert capacity == cfg['reference_capacities']['pp2'], (capacity,cfg['reference_capacities'])
        for row in cfg['reference_cases']:
            q.progress(run,cfg,'profile_'+row['id'])
            q.set_microbatch(local,'pp2',key,row['requests'])
            tokens=q.make_tokens(local,row['requests'],row['cached_context_tokens'])
            preparation=q.prime(local,'pp2',key,tokens,health)
            req=q.payloads(local,'pp2',tokens,1,row['output_tokens'],600)
            warm=q.b.cohort(local,key,req);q.check_cached(warm,row['cached_context_tokens'])
            q.profile_case(run,local,'pp2',key,row,req,health)
            directory=run/'pp2/traces'/row['id']
            for path in directory.glob('*.trace.json.gz'):
                events=json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
                assert sum(e.get('name')=='cudaGraphLaunch' for e in events)>=2, path
            q.h.save(directory/'sampling_validation.json',dict(passed=True,
                performance_sample=False,num_steps=cfg['decode_profile_steps'],
                preparation=preparation,warm_cached_tokens=[r['cached_tokens'] for r in warm['requests']],
                same_gpu_pair=[2,3],same_reference_case=row,reference_run=cfg['reference_run']))
        q.h.save(run/'completed.json',dict(passed=True,profile_only=True,
            cases=cfg['reference_cases'],logical_capacities=cfg['reference_capacities']))
    except BaseException:
        (run/'worker_error.txt').write_text(traceback.format_exc());raise
    finally:
        evidence=q.h.cleanup(children)
        q.h.save(run/'worker_cleanup.json',evidence)
        Path(cfg['policy_path']).unlink(missing_ok=True)
        if not evidence['cleanup_verified']:
            raise RuntimeError('Owned diagnostic cleanup failed')
