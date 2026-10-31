"""Profile only the missing Prefill cohorts; keep the qualified serving path.

One shape warmup and one trace per cell. Prefix preparation is not profiled.
The caller supplies the frozen comparison runner for ownership/cache handling.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
import zlib


def profile_cell(q, run, cfg, arm, key, row, health):
    import pp_ready_send as ready
    q.set_microbatch(cfg, arm, key, row['requests'])
    if arm == 'pp2':
        ready.set_policy(cfg['policy_path'], cfg['prefill_ready_send'])
    cold = row['prefill_cached_context_tokens'] == 0
    length = row['new_prefill_tokens'] if cold else row['cached_context_tokens']
    tokens = q.make_tokens(cfg, row['requests'], length)
    preparations = []

    def prepare(branch):
        if cold:
            q.b.flush_cache(cfg, key)
            return [q.b.packet(cfg, arm, t, 1, i) for i, t in enumerate(tokens)]
        preparations.append(q.prime(cfg, arm, key, tokens, health))
        return q.payloads(cfg, arm, tokens, row['new_prefill_tokens'], 1, branch)

    def validate(result):
        if not cold:
            q.check_cached(result, row['cached_context_tokens'])
        elif any((r.get('cached_tokens') or 0) != 0 for r in result['requests']):
            raise RuntimeError('Cold Prefill must not hit an old cached prefix')
        if len(result['requests']) != row['requests']:
            raise RuntimeError('Missing request in Prefill diagnostic')

    health()
    validate(q.b.cohort(cfg, key, prepare(810)))
    req = prepare(811)
    trace_dir = run / arm / 'traces' / row['id']
    trace_dir.mkdir(parents=True)
    # Manual stop includes EVERY chunk of this Prefill cohort. A fixed small
    # num_steps would truncate the longer cold Prefill and PP pipeline drain.
    q.b.profile_action(cfg, key, '/start_profile', dict(output_dir=str(trace_dir),
        activities=['CPU', 'GPU'], record_shapes=True, with_stack=False,
        profile_id=arm + '-prefill-' + row['id'], merge_profiles=False,
        detailed_annotations=True))
    try:
        result = q.b.cohort(cfg, key, req)
    finally:
        q.b.profile_action(cfg, key, '/stop_profile')
    validate(result)
    deadline = time.monotonic() + 120
    while True:
        health()
        files = sorted(trace_dir.glob('*.trace.json.gz'))
        exports = []
        if len(files) == 2:
            try:
                for p in files:
                    raw = p.read_bytes()
                    # File existence (or HTTP acknowledgement from another
                    # rank) is not completion: require the gzip trailer/CRC
                    # and complete JSON before freezing this export's hash.
                    json.loads(gzip.decompress(raw))
                    exports.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest()))
            except (OSError, EOFError, zlib.error, json.JSONDecodeError):
                exports = []
            if len(exports) == 2:
                break
        if time.monotonic() > deadline:
            raise TimeoutError('Both actual Prefill ranks must export their traces')
        time.sleep(1)
    q.h.save(trace_dir / 'manifest.json', dict(passed=True,
        phase='prefill', performance_sample=False, decode_output_tokens=1,
        case=row, cached_context_tokens=0 if cold else row['cached_context_tokens'],
        new_tokens_per_request=row['new_prefill_tokens'], requests=row['requests'],
        preparation=preparations, all_prefill_chunks=True,
        selected_ready_send=cfg['prefill_ready_send'] if arm == 'pp2' else False,
        profiled_cohort=result,
        files=exports))


def worker(run, cfg, q):
    incoming = json.loads(q.sys.stdin.readline())
    env = dict(os.environ, **incoming['environment'])
    q.h.authenticate(run, cfg, env, local=True)
    def interrupted(signum, frame):
        raise RuntimeError('Guard or controller requested owned cleanup')
    signal.signal(signal.SIGTERM, interrupted)
    def watch_parent():
        q.sys.stdin.read()
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    key = q.h.read_key(cfg)
    import pp_ready_send as ready
    ready.create_policy(cfg['policy_path'])
    children = []
    failure = None
    try:
        topo = subprocess.run(['nvidia-smi', 'topo', '-m'], text=True,
            capture_output=True, timeout=20, check=True)
        (run / 'topology.txt').write_text(topo.stdout)
        devices = subprocess.run(['nvidia-smi',
            '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv'],
            text=True, capture_output=True, timeout=20, check=True)
        (run / 'devices.csv').write_text(devices.stdout)
        def health():
            q.h.authenticate(run, cfg, env, local=True)
            for child in children:
                if child['process'].poll() is not None:
                    raise RuntimeError(child['arm'] + ' exited during Prefill follow-up')
        q.make_tokens(cfg, 1, 16)
        q.progress(run, cfg, 'parallel_model_loading')
        for arm in ('ep2', 'pp2'):
            children.append(q.launch(run, q.arm_cfg(cfg, arm), arm, env, key))
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs = {arm:pool.submit(q.wait_ready, run, q.arm_cfg(cfg, arm), arm, child, health)
                  for arm, child in zip(('ep2', 'pp2'), children)}
            capacities = {arm:f.result() for arm,f in fs.items()}
        if capacities != cfg['reference_capacities']:
            raise RuntimeError('Allocator capacities changed from the paired r7 baseline')
        rows = q.cases(capacities)
        q.h.save(run / 'capacity_matrix.json', dict(logical_capacities=capacities, cases=rows))
        # Same independent GPU pairs as r7. No performance sample overlaps any
        # profiler: this run is ONLY short diagnostics, not a throughput rerun.
        for row in rows:
            q.progress(run, cfg, 'profile_prefill_' + row['id'], workload=row)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs = [pool.submit(profile_cell, q, run, q.arm_cfg(cfg, arm), arm,
                                  key, row, health) for arm in ('ep2', 'pp2')]
                for f in fs:
                    f.result()
        row = q.own_capacity_case(capacities['pp2'])
        q.progress(run, cfg, 'profile_pp_own_capacity', workload=row)
        profile_cell(q, run, q.arm_cfg(cfg, 'pp2'), 'pp2', key, row, health)
        manifests = list(run.glob('*/traces/*/manifest.json'))
        if len(manifests) != 7:
            raise RuntimeError('Expected exactly seven completed Prefill diagnostic cells')
        q.h.save(run / 'completed.json', dict(passed=True, profile_only=True,
            logical_capacities=capacities, cases=rows, extra_case=row,
            manifests=[str(p) for p in manifests]))
    except BaseException:
        failure = traceback.format_exc()
        (run / 'worker_error.txt').write_text(failure)
        raise
    finally:
        evidence = q.h.cleanup(children) if children else dict(cleanup_verified=False,
            reason='no owned model group')
        q.h.save(run / 'worker_cleanup.json', evidence)
        Path(cfg['policy_path']).unlink(missing_ok=True)
        if not evidence['cleanup_verified'] and failure is None:
            raise RuntimeError('Owned node cleanup failed verification')
    return 0
