#!/usr/bin/env python3
"""Guard-authenticated, no-model SM80 V4 Attention qualification.

Public entry: worker.py --run ABSOLUTE_DIR. The controller creates and freezes
worker_config.json and dashboard_spec.json; this worker never acquires a lease.
GPU imports happen only after authentication. Each case uses a fresh process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import unittest

HERE = Path(__file__).resolve()
GUARD = HERE.parent / 'guard_candidate' / 'scripts'
SCHEMA = 'glm47-verify-sm80-v1'
TIMEOUT_S = 120
CASES = tuple({'id': f'b4_p{p}_q{q}', 'batch': 4, 'prefix': p, 'new_tokens': q}
              for p, q in ((4096, 3), (16384, 3), (4096, 1)))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def private_json(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Authorization paths must be absolute')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError(f'Expected private UID-owned regular file: {path}')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _guard_api():
    sys.path.insert(0, str(GUARD))
    import gpu_registry
    import model_cache_adapter
    return gpu_registry, model_cache_adapter


def authenticate(run, *, env=None, hostname=None, cgroup_text=None, api=None):
    """Bind a live exact-node lease to normalized contract and frozen inputs."""
    env = os.environ if env is None else env
    run = Path(run)
    if not run.is_absolute() or run.is_symlink() or not run.is_dir():
        raise ValueError('--run must name an existing absolute non-symlink directory')
    run = run.resolve()
    cfg_path = run / 'worker_config.json'
    cfg = private_json(cfg_path)
    if cfg.get('schema') != SCHEMA or cfg.get('run_dir') != str(run):
        raise ValueError('Worker config identity differs')
    if cfg.get('case_timeout_s') != TIMEOUT_S or cfg.get('case_ids') != [c['id'] for c in CASES]:
        raise ValueError('Worker case set/timeout differs from the fixed qualification')
    node = cfg['node']
    if node != (hostname or socket.gethostname()).split('.')[0]:
        raise ValueError('Worker is not on its exact allocated node')
    if not re.fullmatch(r'[0-9]+', str(cfg['job_id'])):
        raise ValueError('Expected booked Slurm job identity')
    if cgroup_text is None:
        cgroup_text = Path('/proc/self/cgroup').read_text()
    if not re.search(r'(?:^|/)job_' + re.escape(str(cfg['job_id'])) + r'(?:/|$)',
                     cgroup_text, flags=re.MULTILINE):
        raise ValueError('Worker cgroup differs from its exact booked Slurm job')
    if Path(cfg['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Python interpreter differs from frozen config')
    if cfg.get('guard_scripts') != str(GUARD):
        raise ValueError('Only the reviewed Guard scripts may authenticate this worker')
    if not isinstance(cfg.get('expected_versions'), dict) or not {
            'torch', 'triton', 'sglang-kernel', 'sglang'}.issubset(cfg['expected_versions']):
        raise ValueError('Reviewed runtime package versions are required')
    if not re.fullmatch(r'[0-3]', cfg.get('cuda_visible_devices', '')):
        raise ValueError('This qualification requires exactly one declared physical GPU')
    spec_path = run / 'dashboard_spec.json'
    spec = private_json(spec_path)
    if spec.get('id') != cfg['id']:
        raise ValueError('Experiment identity differs')
    receipt_path = Path(env['EXPERIMENT_GUARD_LEASE_RECEIPT'])
    binding_path = Path(env['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS'])
    receipt = private_json(receipt_path)
    binding = private_json(binding_path)
    allocation = private_json(Path(env['EXPERIMENT_GUARD_ALLOCATION_FILE']))
    gpu_registry, adapter = _guard_api() if api is None else api
    # Until the separately reviewed policy exists this fails closed, intentionally.
    contract = adapter.validate_contract(spec, launch=True)
    adapter.validate_worker_command(spec, [cfg['python'], str(HERE), '--run', str(run)])
    if contract.get('policy') != 'kernel_validation' or binding.get('policy') != 'kernel_validation':
        raise ValueError('An explicit kernel_validation admission is required')
    if (binding.get('schema_version') != 1 or binding.get('nodes') != {}
            or binding.get('contract') != contract
            or binding.get('contract_sha256') != adapter._json_sha256(contract)
            or binding.get('spec_sha256') != adapter._json_sha256(spec)
            or binding.get('experiment_id') != cfg['id']
            or binding.get('lease_id') != receipt.get('lease_id')
            or binding.get('lease_receipt_sha256') != file_sha(receipt_path)):
        raise ValueError('Kernel admission binding differs')
    spec_sha = file_sha(spec_path)
    registry = gpu_registry.SharedGpuRegistry(receipt['registry_dir'])
    lease = registry.assert_owned(receipt, cfg['id'], spec_sha)
    registry.assert_owned_nodes(receipt, [node])
    if (time.time() - lease['heartbeat_at'] > lease['ttl_s']
            or sorted(lease['nodes']) != [node]
            or binding.get('acquired_nodes') != [node]
            or allocation.get('nodes') != [node]
            or allocation.get('experiment_id') != cfg['id']
            or allocation.get('lease_id') != receipt.get('lease_id')
            or allocation.get('tasks') != {cfg['task_id']: [node]}
            or lease.get('allocation') != allocation['tasks']
            or env.get('EXPERIMENT_GUARD_ACQUIRED_NODES') != node):
        raise ValueError('Live exact-node allocation/heartbeat differs')
    descriptors = {row['path']: row for row in spec['frozen_inputs']}
    required = [str(HERE), str(cfg_path), *cfg['engine_inputs']]
    if not set(required).issubset(descriptors):
        raise ValueError('Worker/config/engine inputs are not frozen')
    for name in required:
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or file_sha(path) != descriptors[name]['sha256']:
            raise ValueError(f'Frozen input changed: {name}')
    return cfg


def environment_info(torch):
    versions = {}
    for name in ('torch', 'triton', 'sglang-kernel', 'sglang', 'pytest'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version.split()[0], 'versions': versions,
            'cuda_runtime': torch.version.cuda,
            'device_name': torch.cuda.get_device_name(0),
            'compute_capability': list(torch.cuda.get_device_capability(0))}


def run_gpu_case(cfg, case, environment):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_visible_devices']
    sys.path.insert(0, str(Path(cfg['engine_root']) / 'python'))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('A100 CUDA unavailable')
    info = environment_info(torch);environment.update(info)
    if info['compute_capability'] != [8,0]:raise RuntimeError('SM80 required')
    if any(info['versions'].get(k) != v for k,v in cfg['expected_versions'].items()):
        raise RuntimeError('Frozen package versions differ')
    actual = subprocess.check_output(['git','-C',cfg['engine_root'],'rev-parse','HEAD'],text=True).strip()
    if actual != cfg['engine_commit']:raise RuntimeError('Frozen source revision differs')
    from bench import run
    result = run(case)
    torch.cuda.synchronize()
    return result


def child_case(run, cfg, case):
    result = {'case': case, 'batch_size': 4, 'num_heads': 24, 'head_dim': 128,
              'page_size': 64, 'model_loaded': False, 'started_at': time.time(),
              'environment': {}}
    try:
        result.update(run_gpu_case(cfg, case, result['environment']))
        result['status'] = 'passed'
    except unittest.SkipTest as error:
        result.update(status='skipped', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(limit=15))
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc(limit=15))
    result['finished_at'] = time.time()
    write_json(run / f'{case["id"]}.json', result)
    return 0 if result['status'] == 'passed' else 1


def collect_child_result(path, case, returncode, *, timed_out=False):
    if timed_out:
        return {'case': case, 'status': 'failed', 'error_type': 'TimeoutExpired',
                'error': f'Case exceeded {TIMEOUT_S}s', 'returncode': returncode}
    if not path.exists():
        return {'case': case, 'status': 'failed', 'error_type': 'ChildProcessError',
                'error': 'Child exited without a case result', 'returncode': returncode}
    result = private_json(path)
    if result.get('case') != case or result.get('status') not in ('passed', 'failed', 'skipped'):
        raise ValueError('Child case result identity differs')
    if returncode != 0 and result['status'] == 'passed':
        result.update(status='failed', error_type='ChildProcessError',
                      error='Passing case exited unsuccessfully')
    result['returncode'] = returncode
    return result


def stop_owned_child(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def wait_owned_case(proc, run):
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, TIMEOUT_S)
        try:
            return proc.wait(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            authenticate(run)  # Fail closed if the supervisor loses its lease.


def worker_identity(cfg):
    # /proc/<pid>/stat field 22 is starttime; comm may contain spaces or ')'.
    start_ticks = int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
    return {'schema': SCHEMA, 'experiment_id': cfg['id'], 'pid': os.getpid(),
            'start_ticks': start_ticks, 'node': cfg['node'], 'job_id': str(cfg['job_id']),
            'started_at': time.time()}


def supervise(run, cfg):
    identity = worker_identity(cfg)
    write_json(run / 'worker_identity.json', identity)
    owned_children = []
    try:
        return _supervise_cases(run, cfg, owned_children)
    finally:
        cleanup_errors = []
        for proc in owned_children:
            try:
                stop_owned_child(proc)
            except Exception as error:
                cleanup_errors.append(type(error).__name__)
        write_json(run / 'worker_exit.json', {
            **identity, 'finished_at': time.time(),
            'owned_children_exited': not cleanup_errors and all(
                proc.poll() is not None for proc in owned_children),
            'children': [{'pid': proc.pid, 'returncode': proc.returncode}
                         for proc in owned_children],
            'cleanup_errors': cleanup_errors,
        })


def _supervise_cases(run, cfg, owned_children):
    results = []
    for case in CASES:
        cfg = authenticate(run)  # Recheck live ownership before every process.
        destination = run / f'{case["id"]}.json'
        if destination.exists():
            raise ValueError('Fresh qualification cannot reuse existing case results')
        env = dict(os.environ, GLM_VERIFY_KERNEL_CASE=case['id'],
                   GLM_VERIFY_KERNEL_PARENT_PID=str(os.getpid()),
                   CUDA_VISIBLE_DEVICES=cfg['cuda_visible_devices'])
        with (run / f'{case["id"]}.log').open('x') as log:
            os.chmod(log.name, 0o600)
            proc = subprocess.Popen([cfg['python'], str(HERE), '--run', str(run)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            owned_children.append(proc)
            timed_out = False
            try:
                wait_owned_case(proc, run)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_owned_child(proc)
            except BaseException:
                stop_owned_child(proc)
                raise
        result = collect_child_result(destination, case, proc.returncode, timed_out=timed_out)
        write_json(destination, result)
        results.append(result)
        print(json.dumps({'case': case['id'], 'status': result['status']}), flush=True)
    summary = {'schema': SCHEMA, 'id': cfg['id'], 'node': cfg['node'],
               'model_loaded': False, 'expected_cases': len(CASES), 'results': results,
               'passed': sum(r['status'] == 'passed' for r in results),
               'failed': sum(r['status'] == 'failed' for r in results),
               'skipped': sum(r['status'] == 'skipped' for r in results),
               'qualification_passed': all(r['status'] == 'passed' for r in results)}
    write_json(run / 'worker_summary.json', summary)
    return 0 if summary['qualification_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    cfg = authenticate(args.run)
    case_id = os.environ.get('GLM_VERIFY_KERNEL_CASE')
    if case_id:
        if os.environ.get('GLM_VERIFY_KERNEL_PARENT_PID') != str(os.getppid()):
            raise ValueError('Case must be an owned direct worker child')
        case = next((c for c in CASES if c['id'] == case_id), None)
        if case is None:
            raise ValueError('Unknown fixed case')
        return child_case(args.run, cfg, case)
    return supervise(args.run, cfg)


if __name__ == '__main__':
    raise SystemExit(main())
