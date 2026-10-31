"""Loopback-only live cache dashboard; all collection is observational."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time

BASE = Path('/data/gpfs/projects/punim2715/vllm_workbench')
GUARD = BASE / '.experiment_guard'
KV_REGISTRY = BASE / 'experiment_manager/kv_checkpoints/registry.json'
PROTOCOL = BASE / 'engines/sglang/autoscaling/.engine-worktrees/glm47-canonical-weight-cache-20261003/python/sglang/srt/weight_cache/cpu_protocol.py'
HERE = Path(__file__).resolve().parent


def read_registry(path):
    # No registry class: their constructors/status helpers can write metadata.
    return json.loads(path.read_text()) if path.is_file() else {}


def running_bookings():
    completed = subprocess.run(['squeue', '--me', '--noheader', '--states=RUNNING',
                                '--format=%i|%N'], capture_output=True, text=True,
                               timeout=5, check=True)
    result = {}
    for line in completed.stdout.splitlines():
        job, _, expression = line.strip().partition('|')
        if not expression:
            continue
        if re.fullmatch(r'[A-Za-z0-9._-]+', expression):
            nodes = [expression]
        else:
            response = subprocess.run(['scontrol', 'show', 'hostnames', expression],
                                      capture_output=True, text=True, timeout=3, check=True)
            nodes = response.stdout.splitlines()
        for node in nodes:
            if re.fullmatch(r'[A-Za-z0-9._-]+', node):
                result.setdefault(node, []).append(job)
    return result


def dataset_descriptors(registry):
    """Deduplicate checkpoint aliases by physical node/path, never add nine aliases."""
    entries = registry.get('entries', {})
    entries = entries.values() if isinstance(entries, dict) else entries
    by_node = {}
    for entry in sorted(entries, key=lambda e: e.get('last_verified_at', 0) or 0):
        compatibility = entry.get('compatibility', {})
        instances = compatibility.get('instances', [])
        for location in entry.get('locations', []):
            node, root = location.get('node'), location.get('path')
            if not node or not root:
                continue
            instance = next((i for i in instances if i.get('id') == location.get('instance_id')),
                            next((i for i in instances if node in i.get('nodes', [])), {}))
            existing = by_node.setdefault(node, {}).get(root, {})
            aliases = existing.get('checkpoint_ids', []) + [entry.get('checkpoint_id', '')]
            candidate = dict(layout=dict(tp=instance.get('tp_size'), pp=instance.get('pp_size')),
                             layer_partition=instance.get('layer_partition', []),
                             registry_state=entry.get('state'), boot_id=location.get('boot_id'))
            by_node[node][root] = dict(
                id=Path(root).parent.name + '/' + Path(root).name,
                label=compatibility.get('model', {}).get('served_model_name', 'KV Bank')
                      + ' · ' + Path(root).name,
                root=root, registered=True, checkpoint_ids=sorted(set(aliases)),
                boot_id=location.get('boot_id'),
                layout=dict(tp=instance.get('tp_size'), pp=instance.get('pp_size'), ep=None, dp=None),
                layer_partition=instance.get('layer_partition', []),
                node_rank=instance.get('nodes', []).index(node) if node in instance.get('nodes', []) else None,
                page_size=compatibility.get('cache_format', {}).get('page_size'),
                mem_layout=compatibility.get('cache_format', {}).get('mem_layout'),
                source_verified_at=entry.get('last_verified_at'),
                model=compatibility.get('model', {}).get('served_model_name'),
                layout_candidates=existing.get('layout_candidates', []) + [candidate],
            )
    return by_node


def node_request(node, row, datasets):
    # Only explicit allowlisted metadata crosses SSH; never bindings/tokens.
    weight = None
    if row:
        weight = {k: row.get(k) for k in ('max_bytes', 'socket_path', 'source_commit',
                                         'cache_scheme', 'daemon_id', 'manifest')}
        weight['manifest'] = {k: (row.get('manifest') or {}).get(k)
                              for k in ('pid', 'process_start_ticks')}
    return dict(weight_row=weight, datasets=list(datasets.values()), protocol_path=str(PROTOCOL))


def probe_node(node, payload):
    command = ' '.join(shlex.quote(arg) for arg in ('python3', '-B', str(HERE / 'probe.py')))
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
                               '-o', 'ConnectionAttempts=1', node, command],
                              input=json.dumps(payload), capture_output=True, text=True,
                              timeout=22, check=True)
    result = json.loads(response.stdout)
    if result['node'] != node:
        raise ValueError('remote hostname mismatch')
    return result


def node_placeholder(node, row, datasets, booking, status='unbooked', error=None):
    return dict(node=node, status=status, booking=booking, last_probe_at=None, error=error,
                memory={}, gpus=[], active_layouts=[],
                weight=dict(bytes=None, resident_bytes=None, reserved_bytes=(row or {}).get('max_bytes'),
                            rss_bytes=None, status='unverified', banks=[]),
                kv=dict(logical_bytes=None, resident_bytes=None, files=None,
                        datasets=[dict(d, logical_bytes=None, resident_bytes=None,
                                       rank_entries=[], status='not_live_verified')
                                  for d in datasets.values()]))


def collect_snapshot(refresh_seconds=15):
    errors = []
    try:
        bookings = running_bookings()
    except Exception as error:
        bookings = {}
        errors.append('Slurm 查询失败；本轮不访问节点：' + type(error).__name__)
    try:
        ledger = read_registry(GUARD / 'cpu_weight_daemons/registry.json')
        rows = {r['node']: r for r in ledger.get('entries', [])}
    except Exception as error:
        rows = {}
        errors.append('权重登记读取失败：' + type(error).__name__)
    try:
        datasets = dataset_descriptors(read_registry(KV_REGISTRY))
    except Exception as error:
        datasets = {}
        errors.append('KV 登记读取失败：' + type(error).__name__)
    nodes = {node: node_placeholder(node, rows.get(node), datasets.get(node, {}),
                                   dict(state='RUNNING' if node in bookings else 'NOT_BOOKED',
                                        job_ids=bookings.get(node, [])),
                                   status='probing' if node in bookings else 'unbooked')
             for node in set(rows) | set(datasets) | set(bookings)}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(probe_node, node, node_request(node, rows.get(node),
                                                                  datasets.get(node, {}))): node
                   for node in bookings}
        for future in as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                result['booking'] = nodes[node]['booking']
                nodes[node] = result
            except Exception as error:
                # Do not leak SSH output, environment, or arbitrary exception payloads.
                nodes[node].update(status='unreachable', error='现场查询失败：' + type(error).__name__)
    live = [n for n in nodes.values() if n['status'] == 'live']
    try:
        current = running_bookings()
        for node in live:
            if node['node'] not in current:
                node.update(status='unbooked', error='采样期间预约已结束；本轮不计入在线总量')
                node['booking'] = dict(state='NOT_BOOKED', job_ids=[])
        live = [n for n in nodes.values() if n['status'] == 'live']
    except Exception:
        errors.append('采样后的预约复核失败；保留采样开始时的预约状态')
    def total(component, field):
        known = [n[component].get(field) for n in live]
        observed = [v for v in known if v is not None]
        return sum(observed) if observed else None
    return dict(observed_at=time.time(), refresh_seconds=refresh_seconds, collecting=False,
                errors=errors, readonly=True,
                summary=dict(booked_nodes=len(bookings), live_nodes=len(live),
                             weight_bytes=total('weight', 'bytes'),
                             kv_logical_bytes=total('kv', 'logical_bytes'),
                             kv_resident_bytes=total('kv', 'resident_bytes'),
                             unverified_nodes=len(nodes)-len(live),
                             weight_unknown_nodes=sum(n['weight']['bytes'] is None for n in live),
                             kv_unknown_nodes=sum(n['kv']['resident_bytes'] is None for n in live)),
                nodes=sorted(nodes.values(), key=lambda n: (n['status'] != 'live', n['node'])))


class Monitor:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.snapshot = dict(observed_at=None, refresh_seconds=interval, collecting=True,
                             errors=[], summary={}, nodes=[], readonly=True)
    def run(self):
        while True:
            with self.lock:
                self.snapshot['collecting'] = True
            try:
                result = collect_snapshot(self.interval)
                with self.lock:
                    self.snapshot = result
            except Exception as error:
                with self.lock:
                    self.snapshot.update(collecting=False, errors=['采集失败：' + type(error).__name__])
            self.wake.wait(self.interval)
            self.wake.clear()
    def read(self):
        with self.lock:
            return copy.deepcopy(self.snapshot)


def handler_for(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path in ('/api/status', '/api/health'):
                value = monitor.read() if path == '/api/status' else dict(ok=True, readonly=True)
                body = json.dumps(value, ensure_ascii=False).encode()
                content_type = 'application/json; charset=utf-8'
            else:
                names = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                         '/style.css': ('style.css', 'text/css'),
                         '/static/app.js': ('app.js', 'text/javascript'),
                         '/static/style.css': ('style.css', 'text/css')}
                if path not in names:
                    self.send_error(404)
                    return
                name, mime = names[path]
                body = (HERE / 'static' / name).read_bytes()
                content_type = mime + '; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--collect-once', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.collect_once:
        print(json.dumps(collect_snapshot(args.interval), ensure_ascii=False, indent=2))
        return
    monitor = Monitor(max(10, args.interval))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(monitor))
    port = server.server_address[1]
    receipt = dict(pid=os.getpid(), url=f'http://127.0.0.1:{port}/', readonly=True,
                   started_at=time.time(), source_dir=str(HERE))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    threading.Thread(target=monitor.run, daemon=True).start()
    server.serve_forever()


if __name__ == '__main__':
    main()
