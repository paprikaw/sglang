import ctypes,hashlib,os,signal,subprocess,time
from pathlib import Path

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
