#!/usr/bin/env python3
"""Guard-owned cache-only replica with a supervised CPU benchmark worker.

Path-independent qualified canonical profile with a supervised ephemeral worker.
Existing frozen Qwen and GLM deployments remain untouched.
The read-only dashboard mirrors the own worker's real load-generator counters.
Cleanup reporting failures never interrupt owned process teardown.
No benchmark results or serving logs are retained; cache loader evidence remains.
"""
from __future__ import annotations
import argparse
import ctypes
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.request import Request, urlopen


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def report_io_error(path, exc):
    # stderr is owned by the service journal/log. Never weaken admission or
    # overwrite an existing immutable artifact to conceal storage failures.
    try:
        print(json.dumps(dict(event='cleanup_report_failed', path=str(path),
              error=type(exc).__name__ + ': ' + str(exc))), file=sys.stderr, flush=True)
    except OSError:
        # A full redirected log must not interrupt cleanup either.
        pass


def save_cleanup_report(path, value):
    """Best effort only for cleanup telemetry; launch/admission saves stay strict."""
    try:
        save(path, value)
        return True
    except OSError as exc:
        report_io_error(path, exc)
        return False


def no_benchmark_retention(cfg):
    mode = cfg.get('artifact_retention', 'none')
    if mode != 'none':
        raise ValueError('This runner only supports artifact_retention none')
    benchmark = cfg.get('benchmark')
    if benchmark and benchmark.get('artifact_retention') != 'none':
        raise ValueError('Benchmark must explicitly discard result artifacts')
    return True


@contextlib.contextmanager
def transient_log(cfg, path):
    """Operational child output is discarded; Guard evidence has its own log."""
    no_benchmark_retention(cfg)
    yield subprocess.DEVNULL


class AdmissionLogDrain:
    """Keep only mandatory cache-loader evidence and discard serving output.

    Every loader line (including fallback, malformed or duplicate lines) is
    retained so Guard's unchanged exact-rank/source verifier remains authoritative.
    The log is never truncated or rotated while the binding is live.
    """
    def __init__(self, pipe, path):
        self.pipe, self.path, self.error = pipe, Path(path), None
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        try:
            retained = self.path.stat().st_size if self.path.exists() else 0
            if retained > 1024 * 1024:
                raise RuntimeError("Existing cache evidence exceeded 1 MiB")
            with self.path.open('ab', buffering=0) as evidence:
                while True:
                    line = self.pipe.readline(65537)
                    if len(line) > 65536:
                        raise RuntimeError("Serving output physical line exceeded 64 KiB")
                    if not line:
                        break
                    if b'CPU weight cache load:' in line:
                        retained += len(line)
                        if retained > 1024 * 1024:
                            raise RuntimeError('Cache loader evidence exceeded 1 MiB')
                        evidence.write(line)
                evidence.flush()
                os.fsync(evidence.fileno())
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)

    def finish(self):
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Cache evidence drain did not finish')
        if self.error:
            raise RuntimeError('Cache evidence write failed: ' + self.error)


def ephemeral_progress(cfg, run, node):
    config = read(cfg['benchmark']['config_path'])
    if config.get('artifact_retention') != 'none':
        return {}, None
    root = Path(config['runtime_root']).resolve()
    path = root / 'nodes' / (node + '.json')
    if path.is_symlink() or path.stat().st_size > 32768:
        return {}, None
    raw = read(path)
    if (raw.get('storage_layout') != 'ephemeral_live_v1'
            or raw.get('retention') != 'none' or raw.get('node') != node
            or raw.get('run') != str(Path(run).resolve())
            or not 0 <= time.time() - float(raw.get('timestamp', 0)) <= 20):
        return {}, None
    return raw, path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return dict(pid=pid, parent=int(fields[1]), ticks=int(fields[19]), state=fields[0])
    except (OSError, ValueError, IndexError):
        return None


def track_owned(owned):
    rows = [row for path in Path('/proc').iterdir() if path.name.isdigit()
            if (row := record(int(path.name))) is not None]
    parents = {os.getpid()} | {pid for pid, ticks in owned.items()
                               if (row := record(pid)) and row['ticks'] == ticks}
    changed = True
    while changed:
        changed = False
        for row in rows:
            if row['pid'] not in parents and row['parent'] in parents:
                owned[row['pid']] = row['ticks']
                parents.add(row['pid'])
                changed = True


def live_owned(owned):
    return [pid for pid, ticks in owned.items()
            if (row := record(pid)) and row['ticks'] == ticks and row['state'] != 'Z']


def stop_owned(child, owned):
    for signum, grace in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        deadline = time.monotonic() + grace
        while True:
            track_owned(owned)
            active = live_owned(owned)
            if not active:
                break
            for pid in active:
                current = record(pid)
                if current and current['ticks'] == owned[pid]:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
            if time.monotonic() >= deadline:
                break
            time.sleep(.1)
        if not live_owned(owned):
            break
    if child:
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    for pid in owned:
        if child is None or pid != child.pid:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
    return not live_owned(owned)


def lease_check(receipt_path, run, node):
    receipt_path = Path(receipt_path)
    st = receipt_path.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o777 != 0o600:
        raise RuntimeError('Private user-owned GPU receipt required')
    receipt = read(receipt_path)
    state = read(Path(receipt['registry_dir']) / 'registry.json')
    lease = next((r for r in state['leases'] if r['lease_id'] == receipt['lease_id']), None)
    if not lease or lease['state'] != 'active' or lease['nodes'] != [node]:
        raise RuntimeError('Exact active node lease required')
    if not hmac.compare_digest(hashlib.sha256(receipt['token'].encode()).hexdigest(), lease['token_sha256']):
        raise RuntimeError('GPU lease authentication failed')
    if time.time() - lease['heartbeat_at'] >= lease['ttl_s']:
        raise RuntimeError('GPU lease heartbeat expired')
    if lease.get('preemption_requested'):
        raise InterruptedError('Normal experiment requested this background node')
    spec_path = Path(run) / 'dashboard_spec.json'
    spec = read(spec_path)
    for item in spec.get('frozen_inputs', []):
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Frozen deployment input changed')
    digest = sha(spec_path)
    if digest != lease['spec_sha256'] or digest != receipt['spec_sha256']:
        raise RuntimeError('GPU lease spec changed')
    return receipt, lease


def benchmark_frozen_inputs(cfg):
    """Validate and freeze the benchmark's exact local code and workload inputs."""
    benchmark = cfg.get('benchmark')
    if benchmark is None:
        return []
    if not isinstance(benchmark, dict):
        raise ValueError('benchmark must be an object')
    expected = benchmark.get('frozen_files', {})
    if not isinstance(expected, dict):
        raise ValueError('benchmark.frozen_files must map absolute paths to SHA256 values')
    paths = dict(expected)
    for field in ('worker_script', 'config_path'):
        value = benchmark.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('benchmark.' + field + ' must be an absolute file path')
        paths.setdefault(value, None)
    frozen = []
    for name, digest in sorted(paths.items()):
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Benchmark frozen input must be an existing absolute file: ' + name)
        actual = sha(path)
        if digest is not None and (not isinstance(digest, str)
                or len(digest) != 64 or not hmac.compare_digest(actual, digest)):
            raise ValueError('Benchmark frozen input SHA256 mismatch: ' + name)
        frozen.append(dict(path=str(path), sha256=actual))
    return frozen


def track_worker_owned(child, owned):
    """Only follow descendants of this worker; never the controller's SSH child."""
    rows = [row for path in Path('/proc').iterdir() if path.name.isdigit()
            if (row := record(int(path.name))) is not None]
    parents = {pid for pid, ticks in owned.items()
               if (row := record(pid)) and row['ticks'] == ticks}
    changed = True
    while changed:
        changed = False
        for row in rows:
            if row['pid'] not in parents and row['parent'] in parents:
                owned[row['pid']] = row['ticks']
                parents.add(row['pid'])
                changed = True


def start_benchmark_worker(cfg, args, receipt_path):
    benchmark = cfg.get('benchmark')
    if not benchmark:
        return None, {}, None
    # Recheck both exact receipt and all frozen inputs immediately before spawn.
    lease_check(receipt_path, args.run, args.node)
    benchmark_frozen_inputs(cfg)
    python = benchmark.get('python', cfg['server_python'])
    command = [python, benchmark['worker_script'], '--config', benchmark['config_path'],
               '--run', str(args.run), '--node', args.node]
    with transient_log(cfg, args.run / 'benchmark-worker.log') as log:
        child = subprocess.Popen(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = record(child.pid)
    if identity is None:
        child.wait(timeout=1)
        raise RuntimeError('Benchmark worker exited before its identity was recorded')
    owned = {child.pid: identity['ticks']}
    status = dict(schema_version=1, node=args.node, started_at=time.time(),
                  worker_pid=child.pid, worker_start_ticks=identity['ticks'],
                  state='running', command=command, config_path=benchmark['config_path'],
                  lease_receipt_path=str(receipt_path), exit_code=None)
    try:
        save(args.run / 'benchmark_worker.json', status)
    except BaseException:
        # Spawned worker identity must not be lost if its first report fails.
        stop_benchmark_worker(child, owned, status, args.run)
        raise
    return child, owned, status


def stop_benchmark_worker(child, owned, status, run):
    """Stop only this worker and its proven descendants before model teardown."""
    if child is None:
        return True
    track_worker_owned(child, owned)
    initially_exited = child.poll() is not None
    for signum, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 1)):
        deadline = time.monotonic() + grace
        # The root PID/start-time plus private session prove this process group.
        root = record(child.pid)
        if (root and root['ticks'] == owned.get(child.pid)
                and root['state'] != 'Z'):
            try:
                if os.getpgid(child.pid) == child.pid:
                    os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        while True:
            track_worker_owned(child, owned)
            active = live_owned(owned)
            if not active:
                break
            for pid in active:
                current = record(pid)
                if current and current['ticks'] == owned[pid]:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
        if not live_owned(owned):
            break
    try:
        child.wait(timeout=.2)
    except subprocess.TimeoutExpired:
        pass
    safe = not live_owned(owned)
    status = dict(status or {})
    previous_state = status.get('state')
    status.update(stopped_at=time.time(), exit_code=child.poll(),
                  owned_identities=owned, owned_processes_exited=safe,
                  partial_results=True)
    if previous_state != 'failed':
        status['state'] = 'exited' if initially_exited else ('stopped' if safe else 'cleanup_unverified')
    save_cleanup_report(Path(run) / 'benchmark_worker.json', status)
    return safe


def stop_controller_resources(worker, worker_owned, worker_status, run, proc):
    """Close the model lifetime channel even if worker cleanup/reporting fails."""
    worker_safe = False
    try:
        worker_safe = stop_benchmark_worker(worker, worker_owned, worker_status, run)
    except Exception as exc:
        report_io_error(Path(run) / 'benchmark_worker.json', exc)
    finally:
        if proc:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except OSError as exc:
                report_io_error(Path(run) / 'remote_stdin', exc)
            try:
                proc.wait(timeout=38)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return worker_safe


def benchmark_progress_snapshot(cfg, run, node):
    """Read only this worker's phase counters beneath its configured stage root."""
    benchmark = cfg.get('benchmark')
    if not benchmark:
        return {}
    try:
        raw, path = ephemeral_progress(cfg, run, node)
        if not raw:
            return {}
        phase = raw.get('phase')
        if not isinstance(phase, str) or not phase or Path(phase).name != phase:
            return {}
        names = ('offered_requests', 'dispatched_requests', 'completed_requests',
                 'successful_requests', 'inflight')
        if any(isinstance(raw.get(name), bool) or not isinstance(raw.get(name), int)
               or raw[name] < 0 for name in names):
            return {}
        offered, dispatched, completed, successful, inflight = (raw[name] for name in names)
        if not (successful <= completed <= dispatched <= offered and inflight <= dispatched - completed):
            return {}
        return dict(case=phase, benchmark_phase=phase,
                    benchmark_progress_path=str(path), benchmark_progress_timestamp=raw.get('timestamp'),
                    benchmark_stage=raw.get('status'),
                    benchmark_requests=dispatched, planned_requests=offered,
                    dispatched_requests=dispatched, completed_requests=completed,
                    successful_requests=successful, failed_requests=completed-successful,
                    inflight_requests=inflight,
                    benchmark_target_node_rps=raw.get('target_rps'),
                    benchmark_completed_rps=raw.get('completed_rps'),
                    benchmark_queue_waiting=raw.get('queue_waiting', {}).get(node),
                    benchmark_queue_sample_fresh=raw.get('queue_sample_fresh', {}).get(node),
                    benchmark_queue_policy=raw.get('queue_policy'),
                    benchmark_valid_so_far=raw.get('valid_so_far'),
                    benchmark_reasons=raw.get('reasons', []),
                    progress_percent=100.0*completed/offered if offered else 0.0)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def validate_discovered_model(cfg):
    model = cfg.get('discovered_model')
    if not model:
        raise ValueError('Dynamic runner requires the current discovered cache identity')
    if model['model_root'] != cfg['model_path']:
        raise ValueError('Serving path differs from the discovered current cache')
    for name, expected in model['metadata_sha256'].items():
        if name not in ('config.json', 'model.safetensors.index.json') or sha(Path(cfg['model_path']) / name) != expected:
            raise ValueError('Discovered checkpoint metadata changed: ' + name)
    profile = cfg.get('cache_support_profile')
    if profile not in ('glm4moe-canonical-v1', 'qwen3_5moe-canonical-v1'):
        raise ValueError('Unqualified canonical loader profile')
    entry = cfg.get('server_entry')
    if entry and (not Path(entry).is_absolute() or sha(entry) != cfg.get('server_entry_sha256')):
        raise ValueError('Qualified server entry changed')


def server_entry_args(cfg):
    return [cfg['server_entry']] if cfg.get('server_entry') else ['-m', 'sglang.launch_server']


def prepare(args):
    cfg = read(args.config)
    no_benchmark_retention(cfg)
    validate_discovered_model(cfg)
    benchmark_inputs = benchmark_frozen_inputs(cfg)
    profile = cfg.get('known_node_specs', {}).get(args.node, cfg.get('default_node_spec'))
    if not isinstance(profile, dict) or profile.get('disabled'):
        raise RuntimeError('No enabled cache profile for node')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run, 0o700)
    if (run / 'dashboard_spec.json').exists():
        raise RuntimeError('Use a new immutable deployment attempt directory')
    if 'cache_spec_path' in profile:
        node_spec = read(profile['cache_spec_path'])
        if node_spec['node'] != args.node or str(node_spec['expected_job_id']) != args.job_id:
            raise RuntimeError('Cache profile belongs to another node or booking')
    else:
        node_spec = dict(profile['cache_spec_template'])
        node_spec.update(node=args.node, expected_job_id=args.job_id)
    if cfg.get('cached_weights_only'):
        budget_path = Path(cfg['gpu_registry_dir']).parent / 'cpu_weight_daemons/registry.json'
        entries = read(budget_path)['entries']
        matches = [row for row in entries if row.get('node') == args.node]
        if len(matches) != 1:
            raise RuntimeError('No unique existing CPU weight daemon; cached-only serving skips node')
        existing = matches[0]
        if (existing.get('daemon_id') != cfg['discovered_model']['daemon_id']
                or str(cfg['discovered_model']['expected_job_id']) != args.job_id):
            raise RuntimeError('Discovered cache identity changed; coordinator must rediscover current weights')
        if (existing.get('state') != 'active'
                or str(existing.get('expected_job_id')) != args.job_id
                or existing.get('source_commit') != cfg['engine_commit']
                or existing.get('cache_scheme') != 'canonical_full_weight'):
            raise RuntimeError('Existing weight daemon is incompatible; preserve it and skip node')
        groups = existing.get('status', {}).get('entries', [])
        if not groups or any(row.get('state') != 'ready' for row in groups):
            raise RuntimeError('No complete recorded ready weight bank; cached-only serving skips node')
        for field in ('max_bytes', 'headroom_bytes', 'socket_path'):
            node_spec[field] = existing[field]
    for key, expected in (('engine_root', cfg['engine_root']),
                          ('source_commit', cfg['engine_commit']), ('model_path', cfg['model_path'])):
        if node_spec.get(key) != expected:
            raise RuntimeError('Cache node profile and serving profile differ: ' + key)
    source_commit = subprocess.check_output(['git', '-C', cfg['engine_root'], 'rev-parse', 'HEAD'], text=True).strip()
    if source_commit != cfg['engine_commit']:
        raise RuntimeError('Pinned serving engine commit changed')
    if subprocess.check_output(['git', '-C', cfg['engine_root'], 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Serving engine must be clean')
    if '--enable-hierarchical-cache' in cfg['server_args']:
        raise RuntimeError('This idle profile admits zero HiCache reserve')
    save(run / 'frozen_profile.json', cfg)
    cache_path = run / 'cache_node_spec.json'
    save(cache_path, node_spec)
    ident = 'idle-' + args.node + '-' + run.name
    contract = dict(policy='required', provider='sglang_canonical_full_weight_v1',
                    support_profile=cfg['cache_support_profile'], on_miss=cfg.get('model_cache_on_miss', 'fail'), reuse_only=cfg.get('cached_weights_only',True),
                    node_specs={args.node: dict(path=str(cache_path), sha256=sha(cache_path))},
                    usage_logs={args.node: [dict(id='replica', path=str(run / 'server.log'),
                                                expected_ranks=['TP0', 'TP1', 'TP2', 'TP3'])]},
                    hicache_reserve_bytes_per_node={args.node: 0})
    spec = dict(schema_version=1, id=ident, engine='sglang', template='parallel-tasks-v5',
                protocol='task-events-v2', features=['parallel-tasks', 'task-progress', 'guard'],
                port=cfg.get('dashboard_port_base', 19400) + int(sha_text(ident)[:6], 16) % 10000,
                runtime_dir=str(run / 'dashboard_guard'), gpu_registry_dir=cfg['gpu_registry_dir'],
                resource_priority='background', preemptible=True, gpu_lease_ttl_s=15,
                preemption_stop_grace_s=45, cleanup_wait_s=20, max_active_nodes=1,
                tasks=[dict(id='replica', label=args.node + ' model readiness', root=str(run / 'task'),
                            expected=1, nodes=[args.node], controller_script=str(Path(__file__).resolve()))],
                model_cache=contract,
                frozen_inputs=[dict(path=str(run / 'frozen_profile.json'), sha256=sha(run / 'frozen_profile.json')),
                               dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))])
    spec['frozen_inputs'].extend(benchmark_inputs)
    if cfg.get('server_entry'):
        spec['frozen_inputs'].append(dict(path=cfg['server_entry'], sha256=cfg['server_entry_sha256']))
    recovery = None if cfg.get('cached_weights_only') else profile.get('dead_daemon_recovery')
    if recovery:
        # Recovery is for one frozen old identity. Later attempts reuse the new
        # bank normally and must never target its replacement daemon.
        budget_path = Path(cfg['gpu_registry_dir']).parent / 'cpu_weight_daemons/registry.json'
        current_rows = read(budget_path)['entries']
        recovery = recovery if any(row.get('node') == args.node
                   and row.get('daemon_id') == recovery['daemon_id']
                   and row.get('manifest') == recovery['manifest'] for row in current_rows) else None
    if recovery:
        spec['dead_daemon_recovery'] = [recovery]
        recovery_source = Path(cfg['guard_script']).parent / 'dead_daemon_recovery.py'
        spec['frozen_inputs'].append(dict(path=str(recovery_source), sha256=sha(recovery_source)))
    save(run / 'deployment.json', dict(schema_version=1, deployment_id=ident, node=args.node,
         job_id=args.job_id, source_commit=cfg['engine_commit'], source_tag=cfg['source_tag'],
         engine_root=cfg['engine_root'], model_path=cfg['model_path'], model_revision=cfg['model_revision'],
         attention_tp=4, attention_dp=1, moe_ep=cfg.get('moe_ep', 1), pipeline_parallel=1,
         hicache=False, api_port=cfg['api_port'],
         benchmark_workload=(dict(config_path=cfg['benchmark']['config_path'],
               config_sha256=sha(cfg['benchmark']['config_path']),
               worker_script=cfg['benchmark']['worker_script']) if cfg.get('benchmark') else None),
         status='queued', timestamp=time.time()))
    save(run / 'dashboard_spec.json', spec)
    print(json.dumps(dict(spec_path=str(run / 'dashboard_spec.json'))), flush=True)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def controller(args):
    run = args.run.resolve()
    cfg = read(run / 'frozen_profile.json')
    manifest = read(run / 'deployment.json')
    receipt_path = os.environ['EXPERIMENT_GUARD_LEASE_RECEIPT']
    receipt, lease = lease_check(receipt_path, run, args.node)
    parent = record(os.getppid())
    if not parent or lease['owner_pid'] != parent['pid'] or lease['owner_start_ticks'] != parent['ticks']:
        raise RuntimeError('Controller must be direct Guard child')
    cancelled = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: cancelled.set())
    started, proc, ready, error, safe = time.time(), None, False, None, False
    worker, worker_owned, worker_status, worker_attempted = None, {}, None, False
    worker_cleanup_safe = True
    pid, ticks = os.getpid(), record(os.getpid())['ticks']
    def progress(stage, note):
        counters = benchmark_progress_snapshot(cfg, run, args.node)
        row = dict(timestamp=time.time(), controller_pid=pid,
             process_start_ticks=ticks, stage=stage, phase='ready' if ready else 'loading',
             model_ready=ready, valid_points=int(ready), benchmark_requests=0,
             benchmark_worker_state=worker_status.get('state') if worker_status else None,
             elapsed_s=time.time()-started, note=note)
        row.update(counters)
        if counters:
            row['phase'] = counters['benchmark_stage']
            note += ('; sent=' + str(counters['dispatched_requests'])
                     + ' completed=' + str(counters['completed_requests'])
                     + ' in flight=' + str(counters['inflight_requests'])
                     + ' waiting=' + str(counters['benchmark_queue_waiting'])
                     + ' target RPS=' + str(counters['benchmark_target_node_rps']))
            row['note'] = note
            fields = [('Sent', counters['dispatched_requests']),
                      ('Completed', counters['completed_requests']),
                      ('In flight', counters['inflight_requests']),
                      ('Waiting queue', counters['benchmark_queue_waiting']),
                      ('Queue snapshot fresh', counters['benchmark_queue_sample_fresh']),
                      ('Target RPS per instance', counters['benchmark_target_node_rps']),
                      ('Completed RPS', counters['benchmark_completed_rps']),
                      ('Queue policy', counters['benchmark_queue_policy']),
                      ('Loadgen progress file', counters['benchmark_progress_path'])]
            save(run / 'task/details.json', dict(timestamp=time.time(), summary=note,
                 fields=[dict(label=label, value=value if value is not None else 'unknown')
                         for label, value in fields]))
        save(run / 'task/progress.json', row)
    try:
        progress('running', 'Loading idle-priority model; benchmark awaits user definition')
        command = [cfg.get('remote_python', 'python3'), str(Path(__file__).resolve()), 'remote',
                   '--config', str(Path(args.config).resolve()), '--node', args.node,
                   '--job-id', args.job_id, '--run', str(run)]
        with transient_log(cfg, run / 'remote_worker.log') as log:
            proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                     '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
                                     args.node, shlex.join(command)], stdin=subprocess.PIPE,
                                    stdout=log, stderr=subprocess.STDOUT)
            proc.stdin.write((json.dumps(dict(receipt_path=receipt_path,
                              bindings_path=os.environ['EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS']))+'\n').encode())
            proc.stdin.flush()
            stop_at = read(run / 'attempt.json')['stop_at']
            while proc.poll() is None and not cancelled.wait(1):
                lease_check(receipt_path, run, args.node)
                if time.time() >= stop_at:
                    break
                try:
                    status = read(run / 'remote_status.json')
                    ready = (status.get('ready') is True
                             and time.time() - status.get('timestamp', 0) <= 10)
                    if not ready and manifest.get('status') == 'healthy':
                        manifest.update(status='unhealthy', timestamp=time.time())
                        save(run / 'deployment.json', manifest)
                    if ready:
                        manifest.update(status='healthy', endpoint=f'http://{args.node}:{cfg["api_port"]}',
                                        validation=status.get('validation'), timestamp=time.time())
                        save(run / 'deployment.json', manifest)
                except (OSError, ValueError):
                    ready = False
                    manifest.update(status='unhealthy', timestamp=time.time())
                    save(run / 'deployment.json', manifest)
                if ready and cfg.get('benchmark') and not worker_attempted and not cancelled.is_set():
                    worker_attempted = True
                    try:
                        worker, worker_owned, worker_status = start_benchmark_worker(cfg, args, receipt_path)
                    except BaseException as exc:
                        worker_status = dict(state='failed', node=args.node, timestamp=time.time(),
                                             error=dict(type=type(exc).__name__, message=str(exc)))
                        save(run / 'benchmark_worker.json', worker_status)
                if worker is not None:
                    track_worker_owned(worker, worker_owned)
                    exit_code = worker.poll()
                    if exit_code is not None and worker_status.get('state') == 'running':
                        worker_status.update(state='failed', exit_code=exit_code,
                                             timestamp=time.time(),
                                             error=dict(type='WorkerExited', message='Benchmark worker exited; model remains serving'))
                        save(run / 'benchmark_worker.json', worker_status)
                        # A failed root must not leave a load-generator descendant running.
                        worker_cleanup_safe = stop_benchmark_worker(
                            worker, worker_owned, worker_status, run) and worker_cleanup_safe
                        worker, worker_owned = None, {}
                progress('running', ('Model ready; benchmark worker ' + worker_status.get('state', 'unknown'))
                         if ready and worker_status else 'Model ready; awaiting benchmark' if ready else 'Loading model')
            if proc.poll() is not None and not cancelled.is_set():
                raise RuntimeError(f'Remote worker exited: {proc.returncode}')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        save_cleanup_report(run / 'failure.json', error)
    finally:
        # Always close the model lifetime channel, including after report I/O
        # failure. GPU cleanup remains required and is never inferred from it.
        worker_cleanup_safe = (stop_controller_resources(
            worker, worker_owned, worker_status, run, proc) and worker_cleanup_safe)
        sys.path.insert(0, str(Path(cfg['guard_script']).parent))
        from artifact_visibility import wait_for_stable_json
        evidence = {}
        if proc:
            try:
                evidence, visibility = wait_for_stable_json(run / 'remote_cleanup.json', timeout_s=15)
                save_cleanup_report(run / 'cleanup_visibility.json', visibility)
            except (OSError, ValueError, TimeoutError):
                pass
        safe = (worker_cleanup_safe and evidence.get('owned_processes_exited') is True
                and evidence.get('gpu_idle') is True)
        try:
            progress('complete' if ready else 'failed', 'Deployment stopped; cleanup checked')
        except OSError as exc:
            report_io_error(run / 'task/progress.json', exc)
        save_cleanup_report(run / 'task/cleanup.json', dict(timestamp=time.time(), passed=safe,
             model_processes_exited=evidence.get('owned_processes_exited') is True,
             gpu_idle=evidence.get('gpu_idle') is True,
             benchmark_worker_exited=worker_cleanup_safe, experiment_id=manifest['deployment_id'],
             task_id='replica', evidence_path=str(run / 'remote_cleanup.json')))
        if evidence.get('cache_evidence_complete') is not True and error is None:
            error = dict(type='CacheEvidenceIncomplete', message='Cache loader log did not finish completely')
        manifest.update(status='stopped' if safe else 'cleanup_unverified', timestamp=time.time(), error=error,
                        cache_evidence_complete=evidence.get('cache_evidence_complete') is True)
        save_cleanup_report(run / 'deployment.json', manifest)
    return 0 if safe and evidence.get('cache_evidence_complete') is True else 1


def probe(cfg, stopped, result, expected_model):
    base = f'http://127.0.0.1:{cfg["api_port"]}'
    deadline = time.monotonic() + cfg.get('startup_timeout_s', 1200)
    while not stopped.is_set() and time.monotonic() < deadline:
        try:
            with urlopen(base + '/health', timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            stopped.wait(2)
    else:
        result.update(startup_failed=True)
        return
    try:
        with urlopen(base + '/v1/models', timeout=5) as response:
            models = json.load(response)
        if expected_model not in {row.get('id') for row in models.get('data', [])}:
            raise RuntimeError('Endpoint identity differs from this deployment')
        body = dict(model=expected_model, messages=[dict(role='user', content='Reply with OK.')],
                    max_tokens=32, temperature=0)
        def chat(stream):
            request = Request(base + '/v1/chat/completions',
                              json.dumps(dict(body, stream=stream)).encode(), {'Content-Type': 'application/json'})
            with urlopen(request, timeout=120) as response:
                if not stream:
                    return bool(json.load(response).get('choices'))
                done, chunks = False, 0
                for line in response:
                    if stopped.is_set():
                        return False
                    if line.startswith(b'data: '):
                        if line.strip() == b'data: [DONE]':
                            done = True
                        elif json.loads(line[6:]).get('choices'):
                            chunks += 1
                return done and chunks > 0
        non_stream = chat(False)
        stream = chat(True) if not stopped.is_set() else False
        result.update(ready=non_stream and stream, validation=dict(health=True, models=True,
                      non_stream_chat=non_stream, stream_chat=stream, probe_requests=2,
                      performance_measurement=False), validation_timestamp=time.time())
        if not non_stream or not stream:
            result.update(startup_failed=True)
            return
        failures = 0
        while not stopped.wait(10):
            try:
                if cfg.get('benchmark'):
                    with urlopen(base + '/v1/models', timeout=2) as response:
                        models = json.load(response)
                        health = (response.status == 200 and expected_model in
                                  {row.get('id') for row in models.get('data', [])})
                    check_kind = 'model_identity_no_inference'
                else:
                    with urlopen(base + '/health', timeout=2) as response:
                        health = response.status == 200
                    check_kind = 'health'
                result.update(ready=health, health_checked_at=time.time(),
                              periodic_check=check_kind)
                failures = 0 if health else failures + 1
            except Exception:
                failures += 1
                result.update(ready=False, health_checked_at=time.time())
            if failures >= 3:
                result.update(runtime_unhealthy=True)
                break
    except Exception as exc:
        result.update(ready=False, startup_failed=True, probe_error=type(exc).__name__)


def gpu_cleanup():
    deadline = time.monotonic() + 15
    result = {}
    while True:
        try:
            apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            mem = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5)
            values = [int(row.strip()) for row in mem.stdout.splitlines() if row.strip()]
            idle = apps.returncode == mem.returncode == 0 and not apps.stdout.strip() and len(values) == 4 and max(values) <= 64
            result = dict(gpu_idle=idle, compute_apps=apps.stdout[:4096], memory_mib=values)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            result = dict(gpu_idle=False, error=type(exc).__name__)
        if result['gpu_idle'] or time.monotonic() >= deadline:
            return result
        time.sleep(.5)


def remote(args):
    run = args.run.resolve()
    cfg = read(run / 'frozen_profile.json')
    if socket.gethostname().split('.')[0] != args.node:
        raise RuntimeError('Wrong deployment node')
    packet = json.loads(sys.stdin.readline(65536))
    receipt_path = packet['receipt_path']
    receipt, lease = lease_check(receipt_path, run, args.node)
    sys.path.insert(0, str(Path(cfg['guard_script']).parent))
    from cpu_model_cache_daemon import _job_cgroup
    _job_cgroup(args.job_id)
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('Cannot enable model process subreaper')
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stopped.set())
    poller = select.poll()
    poller.register(sys.stdin.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
    child, owned, error, reason = None, {}, None, 'stopped'
    evidence_drain = None
    status = dict(ready=False)
    try:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('SGLANG_', 'VLLM_', 'DEEPEP_', 'NVSHMEM_', 'HUMMING_', 'NCCL_', 'EXPERIMENT_GUARD_'))
               and key not in ('PYTHONPATH', 'LD_PRELOAD')}
        env.update(cfg['server_env'])
        extra_pythonpath = cfg['server_env'].get('PYTHONPATH')
        env['PYTHONPATH'] = cfg['engine_root'] + '/python' + (':' + extra_pythonpath if extra_pythonpath else '')
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        validate_discovered_model(cfg)
        binding = read(packet['bindings_path'])['nodes'][args.node]
        if (binding.get('daemon_id') != cfg['discovered_model']['daemon_id']
                or binding.get('model_identity', {}).get('group_key') != cfg['discovered_model']['group_key']):
            raise RuntimeError('Admitted cache differs from the discovered current model')
        command = [sys.executable, str(Path(cfg['guard_script']).parent / 'model_cache_adapter.py'),
                   'exec-local', '--bindings', packet['bindings_path'], '--', cfg['server_python'],
                   *server_entry_args(cfg), *[str(a).format(model=cfg['model_path'], port=cfg['api_port']) for a in cfg['server_args']]]
        stop_at = read(run / 'attempt.json')['stop_at']
        if time.time() >= stop_at or (run / 'STOP').exists() or stopped.is_set():
            raise InterruptedError('Deployment cancelled before model spawn')
        lease_check(receipt_path, run, args.node)
        if poller.poll(0) and not os.read(sys.stdin.fileno(), 4096):
            raise InterruptedError('Controller disconnected before model spawn')
        expected_model = read(run / 'deployment.json')['deployment_id']
        command.extend(['--served-model-name', expected_model])
        no_benchmark_retention(cfg)
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        owned[child.pid] = record(child.pid)['ticks']
        evidence_drain = AdmissionLogDrain(child.stdout, run / 'server.log')
        save(run / 'remote_launch.json', dict(timestamp=time.time(), node=args.node, job_id=args.job_id,
             server_pid=child.pid, server_start_ticks=owned[child.pid], command=command,
             lease_id=receipt['lease_id'], cgroup=_job_cgroup(args.job_id, child.pid)))
        threading.Thread(target=probe, args=(cfg, stopped, status, expected_model), daemon=True).start()
        stop_at = read(run / 'attempt.json')['stop_at']
        while child.poll() is None and not stopped.is_set():
            track_owned(owned)
            lease_check(receipt_path, run, args.node)
            if time.time() >= stop_at or (run / 'STOP').exists():
                reason = 'deadline_or_controller_stop'
                break
            if evidence_drain.error:
                raise RuntimeError(evidence_drain.error)
            if status.get('startup_failed') or status.get('runtime_unhealthy'):
                raise RuntimeError('Model readiness probe failed')
            status.update(timestamp=time.time(), node=args.node, server_pid=child.pid)
            save(run / 'remote_status.json', dict(status))
            if poller.poll(1000) and not os.read(sys.stdin.fileno(), 4096):
                reason = 'controller_eof'
                break
        else:
            reason = 'signal' if stopped.is_set() else 'server_exited'
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        reason = 'preempted' if isinstance(exc, InterruptedError) else 'failed'
    finally:
        stopped.set()
        exited = stop_owned(child, owned)
        drain_ok = True
        if evidence_drain is not None:
            try:
                evidence_drain.finish()
            except Exception as exc:
                drain_ok = False
                error = dict(type=type(exc).__name__, message=str(exc))
        evidence = gpu_cleanup()
        evidence['cache_evidence_complete'] = drain_ok
        save_cleanup_report(run / 'remote_cleanup.json', dict(timestamp=time.time(), node=args.node,
             lease_id=receipt['lease_id'], reason=reason, error=error,
             owned_processes_exited=exited, owned_identities=owned, **evidence))
    return 0 if exited and evidence['gpu_idle'] and drain_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'controller', 'remote'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--node', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    return globals()[args.action](args)


if __name__ == '__main__':
    raise SystemExit(main())
