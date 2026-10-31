"""Read-only node observer. No torch/CUDA imports or cache payload reads."""
import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

RANK_FILE = re.compile(r'_(\d+)_(\d+)(?:_(\d+)_(\d+))?\.bin$')
WEIGHT_LABEL = '/memfd:sglang-cpu-weight-cache (deleted)'
KV_PARENT = Path('/dev/shm/sglang-prefix-bank')


def rank_identity(name):
    match = RANK_FILE.search(name)
    if not match:
        return None
    tp_rank, tp, pp, pp_rank = match.groups()
    return int(tp_rank), int(tp), int(pp or 1), int(pp_rank or 0)


def read_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        values[parts[0].rstrip(':')] = int(parts[1]) * 1024
    return dict(total_bytes=values.get('MemTotal'), available_bytes=values.get('MemAvailable'),
                swap_used_bytes=values.get('SwapTotal', 0) - values.get('SwapFree', 0),
                no_swap=('SwapTotal' in values and 'SwapFree' in values
                         and values.get('SwapTotal') == values.get('SwapFree')
                         and values.get('SwapCached', 0) == 0))


def process_start(pid):
    raw = Path('/proc') / str(pid) / 'stat'
    return int(raw.read_text().rsplit(')', 1)[1].split()[19])


def observe_weight(row, protocol_path, memory):
    if not row:
        return dict(bytes=None, resident_bytes=None, reserved_bytes=None, rss_bytes=None,
                    status='not_registered', banks=[], error='没有权重登记；不能据此认定 CPU 缓存为零')
    reservation = row.get('max_bytes')
    result = dict(bytes=None, resident_bytes=None, reserved_bytes=reservation,
                  rss_bytes=None, status='unverified', banks=[])
    manifest = row.get('manifest') or {}
    pid = manifest.get('pid')
    if not isinstance(pid, int):
        result['error'] = '登记没有进程身份；尚不能验证'
        return result
    try:
        if process_start(pid) != manifest.get('process_start_ticks'):
            result.update(status='identity_mismatch', error='PID 启动身份与登记不同')
            return result
        if (Path('/proc') / str(pid)).stat().st_uid != os.getuid():
            raise ValueError('daemon UID mismatch')
    except FileNotFoundError:
        if Path(row.get('socket_path', '')).exists():
            result.update(status='identity_mismatch', error='旧进程已退出但 socket 仍存在；当前缓存未知')
        else:
            result.update(bytes=None, resident_bytes=None, status='gone',
                          error='登记的进程与 socket 已不存在；不能排除 reader 持有的 memfd，不修改登记')
        return result
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        return result
    try:
        spec = importlib.util.spec_from_file_location('cache_monitor_cpu_protocol', protocol_path)
        if spec is None or spec.loader is None:
            raise ValueError('CPU protocol unavailable')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with module.CpuWeightCacheClient(row['socket_path'], timeout=3) as client:
            status = client.status()  # Never acquire/get/evict/ensure/status-local.
        if any(status.get(key) != value for key, value in {
            'pid': pid, 'start_ticks': manifest['process_start_ticks'],
            'uid': os.getuid(), 'source_commit': row.get('source_commit'),
        }.items()):
            raise ValueError('live daemon identity does not match registry')
        seen = set()
        logical = backed = 0
        for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
            try:
                if os.readlink(fd) != WEIGHT_LABEL:
                    continue
                info = fd.stat()
                identity = (info.st_dev, info.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                logical += info.st_size
                backed += min(info.st_size, info.st_blocks * 512)
            except FileNotFoundError:
                continue
        if logical != status.get('used_bytes'):
            raise ValueError('memfd size and daemon used_bytes differ')
        result.update(bytes=logical, backed_bytes=backed,
                      resident_bytes=backed if memory['no_swap'] else None,
                      memfd_count=len(seen), status='ready',
                      banks=[dict(id=str(row.get('daemon_id', ''))[:12],
                                  model='Canonical full-weight bank', bytes=logical,
                                  scheme=row.get('cache_scheme'), state='ready')])
        for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_bytes'] = int(line.split()[1]) * 1024
        cgroup = (Path('/proc') / str(pid) / 'cgroup').read_text().strip().splitlines()
        for line in cgroup:
            if line.startswith('0::'):
                path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
                memory['cgroup_current_bytes'] = int((path / 'memory.current').read_text())
                maximum = (path / 'memory.max').read_text().strip()
                memory['cgroup_limit_bytes'] = None if maximum == 'max' else int(maximum)
    except Exception as error:
        result.update(status='unverified', error=type(error).__name__ + ': ' + str(error))
    return result


def discover_roots():
    found = []
    if not KV_PARENT.is_dir():
        return found
    for run in KV_PARENT.iterdir():
        if run.is_symlink() or not run.is_dir() or run.stat().st_uid != os.getuid():
            continue
        for bank in run.iterdir():
            if not bank.is_symlink() and bank.is_dir() and bank.stat().st_uid == os.getuid():
                found.append(str(bank))
    return found


def observe_dataset(descriptor, memory, boot_id, seen, deadline):
    root = Path(descriptor['root'])
    result = dict(descriptor)
    result.update(logical_bytes=0, resident_bytes=None, backed_bytes=0, files=0,
                  rank_entries=[], status='missing', observation_complete=True,
                  unclassified_bytes=0, unclassified_files=0)
    try:
        # The UI deliberately observes only user-owned node-local RAM banks.
        if not root.is_absolute() or not root.is_relative_to(Path('/dev/shm')):
            result.update(status='unsupported_storage', logical_bytes=None,
                          error='仅现场统计 /dev/shm；其他存储的 RAM 驻留量未知')
            return result
        if not root.exists():
            return result
        shm = Path('/dev/shm')
        if ('..' in root.parts or root != root.resolve(strict=True)
                or any(p.is_symlink() for p in (root, *root.parents))
                or root.stat().st_dev != shm.stat().st_dev
                or root.stat().st_uid != os.getuid()):
            raise ValueError('bank ownership/path mismatch')
        mounts = Path('/proc/mounts').read_text().splitlines()
        if not any(line.split()[1:3] == ['/dev/shm', 'tmpfs'] for line in mounts):
            raise ValueError('/dev/shm is not a verified tmpfs mount')
        result['status'] = 'observed'
        result['registered_boot_matches'] = (descriptor.get('boot_id') == boot_id
                                             if descriptor.get('boot_id') else None)
        if result['registered_boot_matches'] is False:
            result['layout'] = {}
            result['layer_partition'] = []
            result['error'] = '节点 boot ID 已变；登记层布局不再有效，仅统计当前文件'
        groups = {}
        def scan_error(error):
            raise error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
            dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
            for name in files:
                if time.monotonic() > deadline:
                    result.update(observation_complete=False, status='partial')
                    raise TimeoutError('metadata scan budget reached')
                if not name.endswith('.bin'):
                    continue
                identity = rank_identity(name)
                path = Path(directory) / name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    continue
                if info.st_dev != root.stat().st_dev:
                    raise ValueError('bank has a non-shmem filesystem mount')
                physical = (info.st_dev, info.st_ino)
                if physical in seen:
                    continue
                seen.add(physical)
                if not identity:
                    result['unclassified_bytes'] += info.st_size
                    result['unclassified_files'] += 1
                    result['backed_bytes'] += min(info.st_size, info.st_blocks * 512)
                    continue
                group = groups.setdefault(identity, dict(logical_bytes=0, backed_bytes=0, files=0))
                group['logical_bytes'] += info.st_size
                group['backed_bytes'] += min(info.st_size, info.st_blocks * 512)
                group['files'] += 1
        for identity, counts in sorted(groups.items()):
            tp_rank, tp, pp, pp_rank = identity
            candidates = descriptor.get('layout_candidates', [])
            matching = next((c for c in reversed(candidates)
                             if c.get('layout', {}).get('tp') == tp
                             and c.get('layout', {}).get('pp') == pp
                             and c.get('boot_id') == boot_id
                             and c.get('registry_state') not in ('invalid', 'stale', 'dead')), {})
            layers = matching.get('layer_partition') or result.get('layer_partition') or []
            if result['registered_boot_matches'] is False:
                layers = matching.get('layer_partition') or []
            layer_start = sum(layers[:pp_rank]) if len(layers) == pp else None
            layer_end = layer_start + layers[pp_rank] - 1 if layer_start is not None else None
            result['rank_entries'].append(dict(
                rank=pp_rank * tp + tp_rank, gpu_index=None, tp_rank=tp_rank, pp_rank=pp_rank,
                layer_start=layer_start, layer_end=layer_end, tokens=None,
                resident_bytes=counts['backed_bytes'] if memory['no_swap'] else None,
                **counts))
        shapes = sorted(set((i[1], i[2]) for i in groups))
        if len(shapes) == 1:
            result['layout'] = dict(tp=shapes[0][0], pp=shapes[0][1], ep=None, dp=None)
        elif len(shapes) > 1:
            result['layout'] = {}
            result['layout_note'] = '同目录存在多个 TP/PP 文件布局'
        result.pop('layout_candidates', None)
        result['logical_bytes'] = sum(r['logical_bytes'] for r in result['rank_entries']) + result['unclassified_bytes']
        result['backed_bytes'] += sum(r['backed_bytes'] for r in result['rank_entries'])
        result['files'] = sum(r['files'] for r in result['rank_entries']) + result['unclassified_files']
        result['resident_bytes'] = result['backed_bytes'] if memory['no_swap'] else None
    except FileNotFoundError:
        result.update(status='changing', observation_complete=False, logical_bytes=None)
    except Exception as error:
        result.update(error=type(error).__name__ + ': ' + str(error), observation_complete=False)
        if result['status'] != 'partial':
            result['status'] = 'unverified'
        result['logical_bytes'] = result['resident_bytes'] = None
    return result


def observe_gpus():
    try:
        response = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                   '--format=csv,noheader,nounits'], capture_output=True,
                                  text=True, timeout=3, check=True)
        return [dict(index=int(row[0]), name=row[1].strip(), total_bytes=int(float(row[2])) * 1024**2)
                for row in csv.reader(io.StringIO(response.stdout))]
    except Exception:
        return []


def observe(payload):
    memory = read_memory()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    weight = observe_weight(payload.get('weight_row'), payload['protocol_path'], memory)
    descriptors = {d['root']: d for d in payload.get('datasets', [])}
    for root in discover_roots():
        descriptors.setdefault(root, dict(id=Path(root).parent.name + '/' + Path(root).name,
                                          label=Path(root).parent.name + '/' + Path(root).name,
                                          root=root, layout={}, registered=False))
    seen = set()
    deadline = time.monotonic() + 10
    datasets = [observe_dataset(d, memory, boot_id, seen, deadline) for d in descriptors.values()]
    live = [d for d in datasets if d['status'] not in ('missing', 'unsupported_storage')]
    complete = all(d.get('observation_complete') and d.get('logical_bytes') is not None for d in live)
    return dict(node=socket.gethostname().split('.')[0], status='live', last_probe_at=time.time(),
                memory=memory, weight=weight, gpus=observe_gpus(), active_layouts=[],
                kv=dict(logical_bytes=sum(d['logical_bytes'] for d in live) if complete else None,
                        resident_bytes=sum(d['resident_bytes'] for d in live)
                        if complete and all(d['resident_bytes'] is not None for d in live) else None,
                        files=sum(d['files'] for d in live), datasets=datasets,
                        host_pool_bytes=None,
                        host_pool_note='活跃引擎 HiCache pool 尚无按组件遥测；不把进程 RSS 当成 KV'),
                error=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    parser.parse_args()
    print(json.dumps(observe(json.load(sys.stdin)), ensure_ascii=False))
