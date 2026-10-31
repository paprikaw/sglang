"""Guard-authenticated model worker; control EOF stops only its owned tree."""
import argparse
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from common import PYTHON, ROOT, GUARD_SCRIPTS, environment, save, server_args
from guard_authorization import authorization_signature, process_ticks, validate_guard


MAX_RUNTIME_S = 1200
ENV_WHITELIST = {
    'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'PYTHONPATH',
    'LD_LIBRARY_PATH', 'LD_PRELOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'MAX_JOBS', 'HUMMING_COMPILER', 'HUMMING_DISABLE_PARALLEL_BUILD',
    'HUMMING_CACHE_DIR', 'HUMMING_TMP_DIR', 'TRITON_CACHE_DIR',
    'SGLANG_ENABLE_JIT_DEEPGEMM', 'SGLANG_HUMMING_USE_F16_ACCUM',
    'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK',
    'SGLANG_DEEPEP_DISABLE_NVLINK_LOW_LATENCY', 'DEEPEP_AH_ZERO_STATIC_RATE',
    'DEEPEP_LL_NVLINK_PAIR_ONLY', 'DEEPEP_LL_PCIE_P2P_MAX_TOKENS',
    'NVSHMEM_IBGDA_NIC_HANDLER', 'NVSHMEM_HCA_LIST', 'NVSHMEM_IB_ADDR_FAMILY',
    'NVSHMEM_IB_ADDR_RANGE', 'NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME',
    'NVSHMEM_DISABLE_P2P', 'NVSHMEM_QP_DEPTH', 'NVSHMEM_DEBUG',
    'NVSHMEM_DEBUG_SUBSYS', 'GLOO_SOCKET_IFNAME',
    'NCCL_SOCKET_IFNAME', 'NCCL_CUMEM_HOST_ENABLE', 'NCCL_IB_HCA',
}


def select_usage_log(cfg, binding, node, arm):
    selected = cfg['cases'][arm]
    log_id = selected['usage_log_ids'][node]
    expected = [item for item in cfg['model_cache']['usage_logs'][node] if item['id']==log_id]
    bound = [item for item in binding['nodes'][node]['usage_logs'] if item['id']==log_id]
    if len(expected)!=1 or len(bound)!=1:
        raise RuntimeError('Exact admitted usage log missing')
    for key in ('id','path','expected_ranks'):
        if bound[0][key]!=expected[0][key]:
            raise RuntimeError('Usage log differs from admission: '+key)
    path=Path(bound[0]['path'])
    assert path.is_absolute()
    return path


def process_record(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'parent': int(fields[1]),
                'ticks': int(fields[19]), 'state': fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def enable_subreaper():
    # Detached grandchildren are reparented here if their model parent exits.
    # They remain identifiable by PID/start ticks, even after setsid().
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        return False


def track_descendants(owner_pid, owner_ticks, owned):
    records = [record for path in Path('/proc').iterdir() if path.name.isdigit()
               if (record := process_record(int(path.name))) is not None]
    parents = {record['pid'] for record in records
               if owned.get(record['pid']) == record['ticks']}
    owner = process_record(owner_pid)
    if owner and owner['ticks'] == owner_ticks:
        parents.add(owner_pid)
    changed = True
    while changed:
        changed = False
        for record in records:
            if record['pid'] not in parents and record['parent'] in parents:
                owned[record['pid']] = record['ticks']
                parents.add(record['pid'])
                changed = True


def alive_owned(owned):
    return [pid for pid, ticks in owned.items()
            if (record := process_record(pid)) and record['ticks'] == ticks
            and record['state'] != 'Z']


def owned_metadata(owned):
    rows = []
    for pid, ticks in sorted(owned.items()):
        record = process_record(pid)
        if record is None or record['ticks'] != ticks:
            continue
        try:
            record['argv_sha256'] = hashlib.sha256(Path(f'/proc/{pid}/cmdline').read_bytes()).hexdigest()
        except OSError:
            record['argv_sha256'] = None
        rows.append(record)
    return rows


def stop_owned(child, owned, owner_pid, owner_ticks):
    track_descendants(owner_pid, owner_ticks, owned)
    for signum, grace_s in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
        deadline = time.monotonic() + grace_s
        # Rescan before every signal round: an owned child may fork or detach
        # while its parent is shutting down. No process-group/name matching.
        while True:
            track_descendants(owner_pid, owner_ticks, owned)
            active = alive_owned(owned)
            if not active:
                break
            for pid in active:
                record = process_record(pid)
                if record is None or record['ticks'] != owned[pid]:
                    continue
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if not alive_owned(owned):
            break
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # Reap adopted descendants without stealing subprocess.Popen's own child.
    for pid in owned:
        if pid != child.pid:
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
    track_descendants(owner_pid, owner_ticks, owned)
    return not alive_owned(owned)


def public_command(command):
    result, redact_next = [], False
    secrets = {'--api-key', '--admin-api-key', '--api-key-file'}
    for value in command:
        value = str(value)
        if redact_next:
            result.append('<redacted>')
            redact_next = False
        elif any(value.startswith(flag + '=') for flag in secrets):
            result.append(value.split('=', 1)[0] + '=<redacted>')
        else:
            result.append(value)
            redact_next = value in secrets
    return result


def gpu_cleanup_query(timeout_s=20):
    """Post-exit/admitted-launch-failure evidence, never a startup preflight."""
    deadline = time.monotonic() + timeout_s
    samples = []
    while True:
        sample = {'timestamp': time.time()}
        for key, command in (
            ('compute_apps', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                              '--format=csv,noheader,nounits']),
            ('gpu_memory', ['nvidia-smi', '--query-gpu=index,memory.used',
                            '--format=csv,noheader,nounits']),
        ):
            try:
                query = subprocess.run(command, capture_output=True, text=True, timeout=5)
                sample[key] = {'returncode': query.returncode, 'stdout': query.stdout[:65536],
                               'stderr': query.stderr[:4096]}
            except (OSError, subprocess.TimeoutExpired) as error:
                sample[key] = {'returncode': -1, 'stdout': '', 'error_type': type(error).__name__}
        apps = sample['compute_apps']
        memory = sample['gpu_memory']
        memory_rows = []
        try:
            for line in memory.get('stdout', '').splitlines():
                if line.strip():
                    gpu, used = line.split(',', 1)
                    memory_rows.append({'gpu': int(gpu), 'used_mib': int(used)})
        except ValueError:
            memory_rows = []
        app_rows = [line for line in apps.get('stdout', '').splitlines()
                    if line.strip() and 'No running processes found' not in line]
        sample['gpu_memory_rows'] = memory_rows
        sample['idle'] = (apps['returncode'] == memory['returncode'] == 0
                          and not app_rows and bool(memory_rows)
                          and all(row['used_mib'] <= 64 for row in memory_rows))
        samples.append(sample)
        if sample['idle'] or time.monotonic() >= deadline:
            return {'passed': sample['idle'], 'post_exit_only': True,
                    'residual_memory_limit_mib_per_gpu': 64, 'samples': samples}
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('profile',))
    parser.add_argument('--node-rank', required=True, type=int, choices=(0,))
    parser.add_argument('--controller-stdin', required=True, action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    node = os.uname().nodename.split('.')[0]
    cfg = json.loads((run / 'config.json').read_text())
    selected = cfg['cases'][args.arm]
    output = run / args.arm
    packet_line = sys.stdin.readline(65537)
    if not packet_line or len(packet_line) > 65536 or not packet_line.endswith('\n'):
        raise RuntimeError('Controller authorization packet missing or oversized')
    packet = json.loads(packet_line)
    payload = packet['payload']
    receipt = validate_guard(run, cfg, payload['guard_env'], run / 'artifact_visibility' / node)
    if not hmac.compare_digest(packet['signature'], authorization_signature(payload, receipt['token'])):
        raise RuntimeError('Controller authorization signature mismatch')
    if (payload['lease_id'] != receipt['lease_id'] or payload['node'] != node
            or payload['arm'] != args.arm or payload['capacity'] != selected['ep_capacity']
            or payload.get('node_rank') != args.node_rank
):
        raise RuntimeError('Controller authorization does not match this worker')
    if not 0 <= args.node_rank < len(selected['nodes']) or selected['nodes'][args.node_rank] != node:
        raise RuntimeError('Configured rank/node allocation does not match this worker')

    owner_pid = os.getpid()
    owner_ticks = process_ticks(owner_pid)
    cancelled = {'signal': 0}
    def interrupted(signum, _frame):
        cancelled['signal'] = signum
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    link = select.poll()
    link.register(sys.stdin.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
    subreaper = enable_subreaper()
    child, owned, rc, reason = None, {}, 1, 'launch_error'
    owned_exited = False
    launch_error = None
    try:
        # All admitted-launch setup belongs inside the cleanup boundary. A
        # binding/usage failure still saves an accurate node_end immediately.
        env = environment(args.arm, args.arm, args.node_rank, run=run)
        assert env.get('RUN_DIR') == str(run)
        binding_path = Path(payload['model_cache_bindings'])
        binding = json.loads(binding_path.read_text())
        if binding.get('policy') != 'expert_layer_only' or binding.get('lease_id') != receipt['lease_id']:
            raise RuntimeError('Expert-layer binding must match this active lease')
        usage_log = output / 'worker.log'
        command = [sys.executable, str(GUARD_SCRIPTS / 'model_cache_adapter.py'),
                   'exec-local', '--bindings', str(binding_path), '--',
                   str(PYTHON), str(ROOT / 'worker.py'), *server_args(args.arm, args.node_rank, run)]
        # No GPU query/import/source/config/help preflight between Guard
        # authentication and this immediate model launch.
        # Append to the Guard-created inode; never replace or truncate its
        # admitted suffix baseline. Inject cache env after environment filtering.
        with usage_log.open('ab', buffering=0) as loader_output:
            child = subprocess.Popen(command, env=env, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=loader_output, stderr=subprocess.STDOUT)
        root_record = process_record(child.pid)
        if root_record:
            owned[child.pid] = root_record['ticks']
        save(output / f'node{args.node_rank}_launch.json', {
            'timestamp': time.time(), 'node': node, 'node_rank': args.node_rank, 'arm': args.arm,
            'parent_id': cfg['id'], 'main_run_dir': str(run),
            'command': public_command(command), 'guard_lease_id': receipt['lease_id'],
            'owner_pid': owner_pid, 'owner_start_ticks': owner_ticks, 'server_pid': child.pid,
            'server_start_ticks': owned.get(child.pid), 'child_subreaper': subreaper,
            'cache_binding_policy': binding['policy'], 'usage_log_path': str(usage_log),
            'environment': {key: env[key] for key in sorted(ENV_WHITELIST) if key in env}})
        deadline = time.monotonic() + MAX_RUNTIME_S
        next_status = 0.0
        while child.poll() is None:
            track_descendants(owner_pid, owner_ticks, owned)
            now = time.monotonic()
            if now >= next_status:
                save(output / f'node{args.node_rank}_running.json', {
                    'timestamp': time.time(), 'node': node, 'node_rank': args.node_rank, 'arm': args.arm,
                    'parent_id': cfg['id'],
                    'owner_pid': owner_pid, 'owner_start_ticks': owner_ticks,
                    'server_pid': child.pid, 'owned_processes': owned_metadata(owned)})
                next_status = now + 2
            if cancelled['signal']:
                rc, reason = 128 + cancelled['signal'], 'signal'
                break
            if (output / 'STOP').exists() or (run / 'STOP').exists():
                rc, reason = 0, 'controller_stop_file'
                break
            if now >= deadline:
                rc, reason = 124, 'runtime_limit'
                break
            events = link.poll(100)
            if events:
                control = os.read(sys.stdin.fileno(), 4096)
                if not control:
                    rc, reason = 0, 'controller_stdin_closed'
                    break
                if control.strip() == b'STOP':
                    rc, reason = 0, 'controller_stop_message'
                    break
        else:
            rc, reason = child.returncode or 1, 'server_exited_before_controller_stop'
    except BaseException as error:
        launch_error = {'error_type': type(error).__name__, 'error': str(error)}
        raise
    finally:
        if child is not None:
            owned_exited = stop_owned(child, owned, owner_pid, owner_ticks)
        else:
            owned_exited = True
        # A setup error on an authenticated/admitted node is a concrete failed
        # launch, not a startup probe. Query actual GPU state after cleanup;
        # never invent idle evidence merely because no child was spawned.
        gpu_query = gpu_cleanup_query(20)
        worker_exited = child is None or child.poll() is not None
        if not worker_exited or not owned_exited or not gpu_query['passed']:
            if rc == 0:
                rc = 3
        save(output / f'node{args.node_rank}_end.json', {
            'timestamp': time.time(), 'node': node, 'node_rank': args.node_rank, 'arm': args.arm,
            'parent_id': cfg['id'],
            'returncode': rc, 'reason': reason, 'signal': cancelled['signal'],
            'model_spawned': child is not None, 'launch_error': launch_error,
            'worker_exited': worker_exited, 'owned_processes_exited': owned_exited,
            'server_returncode': child.returncode if child is not None else None,
            'owned_process_count': len(owned), 'owned_identities': [
                {'pid': pid, 'start_ticks': ticks} for pid, ticks in sorted(owned.items())],
            'gpu_idle': gpu_query['passed'], 'gpu_query': gpu_query})
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
