"""Explicit, leased retirement of the user's idle Qwen CPU weight banks only.

No model/GPU process is launched; KV banks and checkpoint files are untouched.
Uses the registry's documented manual maintenance/recovery integration.
"""
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
import sys
import time

GUARD = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
sys.path.insert(0, str(GUARD))
import cpu_model_cache_daemon as daemon
import gpu_registry as gpu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'maintenance' / ('retirement-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
NODES = ['spartan-gpgpu065', 'spartan-gpgpu167']
ENGINE = ROOT.parents[1] / '.engine-worktrees/glm47-ep16-canonical-profile-20261004'
PYTHON = '/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
EXPECTED_SOURCE = '870269702285d94bdd926a50cc29ad1a20bcf60c'
EXPECTED_JOBS = {'spartan-gpgpu065':'33463294', 'spartan-gpgpu167':'32715983'}
EXPECTED_BYTES = 71903776776


def save(path, value):
    gpu._atomic_json(path, value, mode=0o600)


def main():
    OUT.mkdir(mode=0o700, parents=True)
    source = daemon.DaemonBudgetRegistry().list()
    rows = {r['node']: r for r in source['entries'] if r['node'] in NODES}
    if set(rows) != set(NODES):
        raise RuntimeError('Both exact old bank reservations are required')
    paths = {}
    for node, row in rows.items():
        if (row['source_commit'] != EXPECTED_SOURCE or row['expected_job_id'] != EXPECTED_JOBS[node]
                or row.get('cache_scheme') != 'canonical_full_weight'
                or row.get('status', {}).get('current_bytes') != EXPECTED_BYTES):
            raise RuntimeError('Old bank identity or accounted bytes changed: ' + node)
        spec = {key: row[key] for key in ['node','source_commit','expected_job_id','socket_path','max_bytes','headroom_bytes','cache_scheme']}
        spec.update(engine_root=str(ENGINE), python=PYTHON)
        path = OUT / (node + '.json')
        save(path, spec)
        paths[node] = path
    spec = {'id': OUT.name, 'port': 0, 'max_active_nodes': len(NODES), 'preemptible': False,
            'tasks': [{'id':'retire-weights', 'nodes': NODES}],
            'kind': 'cpu_weight_cache_maintenance', 'gpu_model_launched': False,
            'authorization': 'User 2026-10-22 explicitly permits replacing the two Qwen RAM banks on 065/167 with full GLM banks; preserve disk weights, KV banks and results. No cap increase.'}
    spec_path = OUT / 'spec.json'
    save(spec_path, spec)
    registry = gpu.SharedGpuRegistry()
    receipt_path = OUT / 'gpu_registry_lease.json'
    receipt, public = registry.claim(spec, hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                                     gpu.discover_booked_nodes(), receipt_path, ttl_s=90,
                                     node_limit=len(NODES), exact_node_count=True,
                                     resource_selection={'mode':'user','nodes':len(NODES),'gpus':4*len(NODES)})
    hb = gpu.LeaseHeartbeat(registry, receipt).start()
    evidence = {'gpu_model_launched':False, 'lease_id': public['lease_id'], 'nodes':{}, 'passed':False}
    completed = False
    try:
        # All-node healthy/non-reader check before the first eviction.
        for node in NODES:
            if hb.error or hb.preemption: raise RuntimeError('Maintenance lease lost')
            before = daemon._remote('status', paths[node], None, daemon.DEFAULT_BUDGET_DIR)
            if not before.get('healthy') or before['daemon']['daemon_id'] != rows[node]['daemon_id']:
                raise RuntimeError('Live old daemon differs: ' + node)
            status = before['daemon']['status']
            expected_keys = [entry['key'] for entry in rows[node]['status']['entries']]
            daemon._retirable_entries(status, expected_keys)
            if status['current_bytes'] != EXPECTED_BYTES:
                raise RuntimeError('Live old cache bytes changed: ' + node)
            evidence['nodes'][node] = {'before': before, 'expected_keys': expected_keys}
        save(OUT/'evidence.json', evidence)
        for node in NODES:
            if hb.error or hb.preemption: raise RuntimeError('Maintenance lease lost')
            registry.assert_owned_nodes(receipt, NODES)
            argv = [PYTHON, str(GUARD/'cpu_model_cache_daemon.py'), 'retire-local',
                    '--spec', str(paths[node]), '--budget-dir', str(daemon.DEFAULT_BUDGET_DIR),
                    '--gpu-receipt', str(receipt_path), '--replace-ready-cache']
            for key in evidence['nodes'][node]['expected_keys']:
                argv.extend(['--expected-cache-key', key])
            remote = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,shlex.join(argv)],
                                    text=True,capture_output=True,timeout=600)
            if remote.returncode:
                raise RuntimeError(remote.stderr.strip() or remote.stdout.strip())
            result = json.loads(remote.stdout)
            evidence['nodes'][node]['retirement'] = result
            save(OUT/'evidence.json', evidence)
            if not result.get('passed'): raise RuntimeError('Retirement not proven: ' + node)
        completed = True
        evidence['passed'] = True
    except BaseException as exc:
        evidence['error'] = {'type':type(exc).__name__, 'message':str(exc)}
        raise
    finally:
        hb.stop()
        evidence['ended_at'] = time.time()
        # A maintenance lease launched no GPU/model child. Do not claim GPUs idle.
        evidence['release'] = registry.release(receipt, cleanup_verified=completed,
                        reason='verified exact old CPU daemon retirement; no GPU/model process launched' if completed else 'CPU retirement incomplete; preserve audit')
        save(OUT/'evidence.json', evidence)
        print(json.dumps({'out':str(OUT),'passed':completed,
                          'retired_nodes':[n for n,r in evidence['nodes'].items() if r.get('retirement',{}).get('passed')],
                          'gpu_model_launched':False}))

if __name__ == '__main__':
    main()
