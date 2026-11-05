"""Official AgentX coordinator for the copied, Guard-owned E8 launcher."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def flush_endpoints(endpoints, key, http):
    """Use the inherited helper's qualified text-response endpoint exactly."""
    with cf.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        replies = list(pool.map(lambda e: http(e['host'], e['port'],
            '/flush_cache?timeout=60', key, timeout=70), endpoints))
    assert all(x.startswith('Cache flushed.') for x in replies)


def install(namespace):
    def coordinator(run, root, arm):
        import socket
        base, cfg = namespace['load_base'](run, root, socket.gethostname().split('.')[0])
        packet = json.loads(sys.stdin.readline())
        env = dict(os.environ, **packet['environment'])
        child = None

        def interrupted(signum, frame):
            raise InterruptedError('Coordinator stopped by its Guard-owned parent')

        signal.signal(signal.SIGTERM, interrupted)

        def parent_closed():
            sys.stdin.read()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=parent_closed, daemon=True).start()
        base.check(run, cfg, env, True)
        key = base.h.read_key(cfg)
        endpoints = root['endpoints'][arm]
        import agentx_replay as probes
        from agentx_replay import http, request_one
        original_probe_stream = probes.stream
        probes.stream = lambda e,k,p,timeout=600: original_probe_stream(e,k,p,timeout=min(timeout,180))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
        body = tok.encode('def merge_records(items):\n    return sorted(items)\n', add_special_tokens=False)

        def flush():
            flush_endpoints(endpoints, key, http)

        flush()
        p = [e for e in endpoints if e['role'] in ('colocated', 'prefill')]
        d = [e for e in endpoints if e['role'] == 'decode']
        responses = []
        for i, length in enumerate((512, 513, 2048, 2049, 4096, 4097)):
            base.check(run, cfg, env, True)
            responses.append(request_one(p[i % len(p)], d[i % len(d)] if d else None,
                key, ([5012] + body * 512)[:length], 32, i))
        reference_path = run/'evidence/reference.json'
        if not reference_path.exists():
            assert arm == 'M_PP2' and not d, 'A100 mixed PP2 must qualify first'
            base.h.save(reference_path, dict(source_arm=arm, hardware='A100',
                engine_commit=cfg['engine_commit'], output_ids=[r['output_ids'] for r in responses]))
        reference = json.loads(reference_path.read_text())['output_ids']
        passed = [r['output_ids'] for r in responses] == reference
        base.h.save(run/'evidence'/arm/'serial_qualification.json',
            dict(passed=passed, exact_match_reference=passed, responses=responses))
        if not passed:
            raise RuntimeError('Serial output qualification failed; performance withheld')
        # Untimed, single-request qualification supplies initial service priors.
        priors = {}
        for i, response in enumerate(responses):
            target = d[i % len(d)] if d else p[i % len(p)]
            name = target['host'] + '.r' + target['id'].rsplit('.r',1)[1]
            if response.get('tpot_ms') is not None:
                priors.setdefault(name, []).append(response['tpot_ms'])
        base.h.save(run/'evidence'/arm/'serial_routing_prior.json',
            dict(endpoint_tpot_ms=priors, source='Untimed serial qualification; batch1; output32'))
        controls = []
        for target in d:
            info = http(target['host'], target['port'], '/get_server_info', key, timeout=10)
            states = [s.get('experiment_decode_service') for s in info['internal_states']]
            if target['pp'] == 2:
                layout = next(x for x in root['node_configs'][target['host']]['layouts'] if x['id']==arm)
                inst = layout['instances'][int(target['id'].rsplit('.r',1)[1])]
                assert all(s and s['early_proxy']==inst['pp_pd_early_proxy'] and
                    s['hicache_poll_steps']==inst['pp_pd_hicache_poll_steps'] and
                    s['typed_channels']==(3 if inst['pp_pd_early_proxy'] else 0) for s in states)
                if inst['pp_pd_early_proxy']:
                    assert all(s['early_proxy_sends']>0 for s in states)
            controls.append(dict(endpoint=target['id'], states=states))
        base.h.save(run/'evidence'/arm/'decode_controls_qualified.json', dict(passed=True,endpoints=controls))
        dynamic = []
        lengths = (512,513,2048,2049,4096,4097)
        for target in d or p:
            for width in (1,8,3,16,2):
                with cf.ThreadPoolExecutor(max_workers=width) as pool:
                    futures = [pool.submit(request_one, p[j % len(p)] if d else target, target if d else None, key,
                        ([5012]+body*512)[:lengths[j%6]], (7,32,19,1)[j%4], j) for j in range(width)]
                    values = [f.result() for f in futures]
                passed = all(v['output_ids']==reference[j%6][:(7,32,19,1)[j%4]] for j,v in enumerate(values))
                dynamic.append(dict(endpoint=target['id'],offered_width=width,passed=passed,responses=values))
                base.h.save(run/'evidence'/arm/'dynamic_qualification.json',dict(passed=all(v['passed'] for v in dynamic),cases=dynamic))
                if not passed:
                    raise RuntimeError('Dynamic batch output qualification failed; performance withheld')
        probes.stream = original_probe_stream
        from matrix_workload import run_points
        return run_points(namespace,run,root,arm,base,cfg,env,endpoints,key,body,flush,http)
    namespace['coordinator'] = coordinator

    original_controller = namespace['controller']
    def controller(run, root):
        import shlex
        base, first = namespace['load_base'](run, root, root['nodes'][0])
        base.check(run, first, os.environ)
        packet = {'environment': {k: os.environ[k] for k in base.h.GUARD_KEYS}}
        command = [root['agentx_python'], str(run/'code/dataset_preflight.py'), '--run', str(run)]
        base.h.save(run/'task/progress.json', dict(experiment_id=root['id'], task_id='comparison',
            stage='CPU_actual_dataset_preflight_before_model_load', timestamp=time.time(),
            valid_points=len(root.get('imported_arms', {})), expected=root['expected_points']))
        preflight_started = time.monotonic()
        with (run/'dataset_preflight.log').open('wb') as log:
            process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', root['nodes'][0], shlex.join(command)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.stdin.write((json.dumps(packet)+'\n').encode()); process.stdin.flush()
                while process.poll() is None:
                    base.check(run, first, os.environ)
                    if time.monotonic()-preflight_started > 240:
                        raise TimeoutError('Actual dataset CPU preflight timed out')
                    time.sleep(2)
            finally:
                process.stdin.close()
                if process.poll() is None:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.terminate(); process.wait(timeout=15)
        if process.returncode:
            # The GPU controller has not been called. Preserve scientific failure
            # while accurately authorizing teardown of this no-GPU attempt.
            base.h.save(run/'task/progress.json', dict(experiment_id=root['id'],
                task_id='comparison', stage='failed_before_GPU_model_load',
                timestamp=time.time(), valid_points=0, expected=root['expected_points']))
            base.h.save(run/'task/cleanup.json', dict(experiment_id=root['id'],
                task_id='comparison', timestamp=time.time(), passed=True,
                gpu_idle=True, model_processes_exited=True,
                no_gpu_workers_started=True, preflight_returncode=process.returncode,
                source='CPU preflight exited before the GPU controller was called'))
            raise RuntimeError('CPU dataset preflight failed; no GPU model was loaded')
        remaining = root['total_timeout_seconds']-(time.monotonic()-preflight_started)
        return original_controller(run, dict(root, total_timeout_seconds=remaining))
    namespace['controller'] = controller
