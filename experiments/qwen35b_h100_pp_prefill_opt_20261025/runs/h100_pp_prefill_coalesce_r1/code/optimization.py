"""Same-process ABBA PP admission experiment and independent EP reference."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback


def worker(run, cfg):
    q = sys.modules['__main__']
    incoming = json.loads(sys.stdin.readline())
    env = dict(os.environ, **incoming['environment'])
    q.h.authenticate(run, cfg, env, local=True)
    def interrupted(signum, frame):
        raise RuntimeError('Guard/controller requested cleanup')
    signal.signal(signal.SIGTERM, interrupted)
    def watch_parent():
        sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    import pp_ready_send as policy
    policy.create_policy(cfg['policy_path'])
    policy.create_policy(cfg['coalesce_policy_path'])
    key = q.h.read_key(cfg)
    children = []
    failure = None
    points = []
    def health():
        q.h.authenticate(run, cfg, env, local=True)
        for child in children:
            if child['process'].poll() is not None:
                raise RuntimeError(child['arm'] + ' exited')
    try:
        topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True,
                                  text=True, check=True, timeout=20)
        (run/'topology.txt').write_text(topology.stdout)
        q.make_tokens(cfg, 1, 16)
        q.progress(run, cfg, 'parallel_model_loading')
        for arm in ('ep2', 'pp2'):
            children.append(q.launch(run, q.arm_cfg(cfg, arm), arm, env, key))
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs = {arm: pool.submit(q.wait_ready, run, q.arm_cfg(cfg, arm), arm, child, health)
                  for arm, child in zip(('ep2','pp2'), children)}
            capacities = {arm: f.result() for arm, f in fs.items()}
        q.h.save(run/'capacities.json', capacities)
        for row in cfg['optimization_cases']:
            q.progress(run, cfg, 'case_' + row['id'], workload=row)
            n = row['requests']; cached = row['cached_tokens']; fresh = row['new_tokens']
            cold = cached == 0
            tokens = q.make_tokens(cfg, n, fresh if cold else cached)
            for arm in ('pp2','ep2'):
                local = q.arm_cfg(cfg, arm)
                q.set_microbatch(local, arm, key, n)
                def prepare(branch):
                    policy.set_policy(cfg['coalesce_policy_path'], False)
                    if cold:
                        q.b.flush_cache(local, key)
                        return [q.b.packet(local, arm, t, 1, i) for i,t in enumerate(tokens)], None
                    prepared = q.prime(local, arm, key, tokens, health)
                    return q.payloads(local, arm, tokens, fresh, 1, branch), prepared
                def validate(result):
                    if len(result['requests']) != n:
                        raise RuntimeError('Incomplete cohort')
                    if not cold:
                        q.check_cached(result, cached)
                    elif any((x.get('cached_tokens') or 0) != 0 for x in result['requests']):
                        raise RuntimeError('Unexpected cold prefix hit')
                    if any(x.get('output_tokens') != 1 for x in result['requests']):
                        raise RuntimeError('Expected one real generated token per request')
                # Every exact shape is warmed; preparation and warming excluded.
                req, prep = prepare(900)
                validate(q.b.cohort(local, key, req))
                modes = [False,True,True,False] if arm == 'pp2' else [False,False]
                arm_points = []
                for index, candidate in enumerate(modes):
                    q.progress(run, cfg, f'{row["id"]}_{arm}_{index}',
                               workload=row, candidate=candidate)
                    req, prep = prepare(910+index)
                    policy.set_policy(cfg['coalesce_policy_path'], candidate)
                    health()
                    result = q.b.cohort(local, key, req)
                    validate(result)
                    point = dict(case=row, arm=arm, coalescing=candidate, trial=index,
                                 preparation=prep, measured_wait_included=True,
                                 result=result,
                                 new_token_throughput=n*fresh/result['makespan_s'])
                    q.h.save(run/arm/f'metrics.{row["id"]}.{index}.json', point)
                    points.append(point); arm_points.append(point)
                # Repeat the identical branch after a fresh cache reset, to
                # separate scheduler changes from deliberately distinct inputs.
                if arm == 'pp2' and row['id'] == 'short_cached32':
                    outputs=[]
                    for mode in (False,True):
                        req, prep = prepare(999)
                        policy.set_policy(cfg['coalesce_policy_path'], mode)
                        result=q.b.cohort(local,key,req); validate(result)
                        outputs.append([x['output_ids'] for x in result['requests']])
                    agreement=outputs[0]==outputs[1] and all(x is not None for x in outputs[0])
                    q.h.save(run/'pp2/output_agreement.json', dict(passed=agreement,
                             scope='identical cached context and one-token output, admission only'))
                    if not agreement:
                        raise RuntimeError('Candidate output differs on identical inputs')
                if arm == 'pp2' and row['id'] == 'long_cached32':
                    # Diagnostics follow formal samples and never overlap them.
                    for mode in (False, True):
                        req, prep = prepare(980)
                        policy.set_policy(cfg['coalesce_policy_path'], mode)
                        trace_dir=run/arm/'traces'/('coalesced' if mode else 'baseline')
                        trace_dir.mkdir(parents=True)
                        q.b.profile_action(local,key,'/start_profile',dict(
                            output_dir=str(trace_dir),activities=['CPU','GPU'],
                            record_shapes=True,with_stack=False,
                            profile_id='pp-prefill-'+str(int(mode)),merge_profiles=False,
                            detailed_annotations=True))
                        try:
                            result=q.b.cohort(local,key,req)
                        finally:
                            q.b.profile_action(local,key,'/stop_profile')
                        validate(result)
                        q.h.save(trace_dir/'cohort.json',dict(performance_sample=False,
                                 coalescing=mode,result=result,preparation=prep))
            policy.set_policy(cfg['coalesce_policy_path'],False)
        q.h.save(run/'completed.json',dict(passed=True,points=len(points),
                 capacities=capacities,expected_points=cfg['expected_points']))
    except BaseException:
        failure=traceback.format_exc(); (run/'worker_error.txt').write_text(failure)
        raise
    finally:
        evidence=q.h.cleanup(children) if children else dict(cleanup_verified=False)
        q.h.save(run/'worker_cleanup.json',evidence)
        Path(cfg['policy_path']).unlink(missing_ok=True)
        Path(cfg['coalesce_policy_path']).unlink(missing_ok=True)
        if failure is None and not evidence['cleanup_verified']:
            raise RuntimeError('Cleanup did not verify')
    return 0
