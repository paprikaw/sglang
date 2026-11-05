"""CPU-only node/NUMA memory evidence at startup and ten-second intervals."""
import json
import os
from pathlib import Path
import threading
import time


def _text(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return 'unavailable:' + type(error).__name__


def collect():
    meminfo = {}
    for line in _text('/proc/meminfo').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Mlocked',
                       'Unevictable', 'SwapFree', 'AnonPages', 'Shmem'):
                meminfo[key] = value.strip()
    relative = _text('/proc/self/cgroup').strip().split(':', 2)[-1]
    cgroup = Path('/sys/fs/cgroup') / relative.lstrip('/')
    parents = [cgroup, *cgroup.parents]
    job = next((p for p in parents if p.name.startswith('job_')), cgroup)
    groups = {}
    for label, path in [('worker', cgroup), ('job', job)]:
        groups[label] = dict(path=str(path), **{name: _text(path / name).strip()
            for name in ('memory.current', 'memory.peak', 'memory.max',
                         'memory.events', 'memory.numa_stat', 'cpuset.mems.effective')})
    numa = {}
    for path in Path('/sys/devices/system/node').glob('node[0-9]*/meminfo'):
        numa[path.parent.name] = '\n'.join(line for line in _text(path).splitlines()
            if any(word in line for word in ('MemTotal:', 'MemFree:', 'MemUsed:',
                                             'Mlocked:', 'Unevictable:')))
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            status = dict(line.split(':', 1) for line in (path/'status').read_text().splitlines()
                          if ':' in line)
            processes[int(path.name)] = (int(status['PPid']), status)
        except (OSError, ValueError, KeyError):
            continue
    owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, (parent, _) in processes.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    rows = []
    for pid in sorted(owned):
        if pid not in processes:
            continue
        _, status = processes[pid]
        policies = {}
        for line in _text(f'/proc/{pid}/numa_maps').splitlines():
            fields = line.split()
            if len(fields) > 1:
                policies[fields[1]] = policies.get(fields[1], 0) + 1
        rows.append(dict(pid=pid, **{key: status.get(key, '').strip() for key in
            ('Name', 'State', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem',
             'VmLck', 'Cpus_allowed_list', 'Mems_allowed_list')}, numa_policies=policies))
    return dict(kind='memory', unix_ns=time.time_ns(), monotonic_s=time.perf_counter(),
                meminfo=meminfo, cgroups=groups, numa=numa, owned_processes=rows)


def start(root, arm):
    stop = threading.Event()
    path = Path(root) / f'memory.{arm}.jsonl'

    def loop():
        while True:
            try:
                row = collect()
            except Exception as error:
                row = dict(kind='memory_error', unix_ns=time.time_ns(),
                           error=type(error).__name__ + ':' + str(error))
            with path.open('a') as out:
                out.write(json.dumps(row, separators=(',', ':')) + '\n')
            if stop.wait(10):
                break

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread
