#!/usr/bin/env python3
"""One-shot Spartan A100 booking, hardware inventory, and retention checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import fcntl
import getpass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

SKILL = Path(__file__).resolve().parents[1]
BOOK = Path('/data/gpfs/projects/punim2715/bookings')
STATE = BOOK / 'node_inventory'
GUARD = Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry')
PROFILES = {'deeplearn': ('punim2715', 'gpgpudeeplearn', 700000, 'dlg5'),
            'feit-gpu-a100': ('punim2565', 'feit', 700000, 'avx512'),
            'gpu-a100': ('punim2715', 'publicgpu', 450000, 'avx512'),
            'gpu-a100-preempt': ('punim2715', 'publicgpu', 700000, 'avx512')}
FIELDS = ('id', 'state', 'partition', 'allocated', 'requested', 'nodes', 'cpus', 'gres', 'name', 'end', 'features', 'user')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=45, **kwargs).strip()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load_inventory():
    p = STATE / 'inventory.json'
    return json.loads(p.read_text()) if p.exists() else {'schema_version': 1, 'nodes': {}}


def classify(record):
    nics = record.get('nics', [])
    valid = [n for n in nics if n.get('speed_gtps') and n.get('width')]
    if not valid or len(valid) != len(nics):
        return 'unknown'
    # Conservatively classify mixed adapters or a slower PCIe parent as low.
    for nic in valid:
        links = [nic] + nic.get('ancestors', [])
        if any(not l.get('speed_gtps') or not l.get('width') for l in links):
            return 'unknown'
        if any(l['speed_gtps'] < 16 or l['width'] < 8 for l in links):
            return 'low'
    return 'high'


def exports(db):
    groups = {'high': [], 'low': [], 'unknown': []}
    flat = []
    for node, item in sorted(db['nodes'].items()):
        r = item.get('latest', {})
        group = classify(r)
        item['group'] = group
        groups[group].append({'node': node, 'observed_at': r.get('checked_at'),
                              'last_attempt': item.get('last_attempt')})
        for nic in r.get('nics', []) or [{}]:
            flat.append({'node': node, 'group': group, 'observed_at': r.get('checked_at'),
                         'model': nic.get('model'), 'speed_gtps': nic.get('speed_gtps'),
                         'width': nic.get('width'), 'source': json.dumps(r.get('source', {}), ensure_ascii=False)})
    db['updated_at'] = now()
    save(STATE / 'inventory.json', db)
    save(STATE / 'groups.json', groups)
    if flat:
        tmp = STATE / 'inventory.csv.tmp'
        with tmp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        tmp.replace(STATE / 'inventory.csv')


def record_observation(db, node, observation=None, error=None):
    item = db['nodes'].setdefault(node, {})
    item['last_attempt'] = {'at': now(), 'ok': observation is not None, 'error': error}
    if observation is not None:
        if observation.get('node') != node:
            raise ValueError('Probe hostname mismatch')
        item['latest'] = observation
        with (STATE / 'observations.jsonl').open('a') as log:
            log.write(json.dumps(observation, ensure_ascii=False) + '\n')


def scheduler():
    nodes = {}
    for line in run(['sinfo', '-N', '-h', '-o', '%N|%P|%G|%f']).splitlines():
        name, part, gres, features = line.split('|')
        item = nodes.setdefault(name, {'partitions': [], 'gres': gres, 'features': features})
        if part.rstrip('*') not in item['partitions']:
            item['partitions'].append(part.rstrip('*'))
    rows = []
    raw = run(['squeue', '--me', '-h', '-o', '%i|%T|%P|%N|%n|%D|%C|%b|%j|%e|%f|%u'])
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) != len(FIELDS):
            raise ValueError('Malformed queue; refusing mutation')
        row = dict(zip(FIELDS, fields))
        if row['user'] != getpass.getuser():
            raise ValueError('Scheduler ownership mismatch')
        row['full_a100'] = bool(row['nodes'] == '1' and int(row['cpus']) >= 32
                               and re.search(r'(?:^|,)gres/gpu(?::[Aa]100)?:4(?:,|$)', row['gres'])
                               and (('gpu:A100:4' in nodes.get(row['allocated'], {}).get('gres', ''))
                                    or (row['state'] == 'PENDING' and row['partition'] in PROFILES
                                        and (row['partition'] != 'deeplearn' or 'dlg5' in row['features']))))
        rows.append(row)
    return nodes, rows


def full_running(rows):
    return [r for r in rows if r['full_a100'] and r['state'] == 'RUNNING']


def record_roster(db, nodes):
    for node, metadata in nodes.items():
        if 'gpu:A100:' in metadata['gres'] or 'gpu-a100-mig' in metadata['partitions']:
            db['nodes'].setdefault(node, {})['scheduler_snapshot'] = {**metadata, 'at': now()}


def remote_probe(node):
    if not re.fullmatch(r'spartan-gpgpu\d{3}', node):
        raise ValueError('Unexpected node name')
    source = (SKILL / 'scripts' / 'probe_node.py').read_text()
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', node, 'python3 -'],
                           input=source, text=True, capture_output=True, timeout=55)
        if p.returncode:
            return node, None, p.stderr.strip()[:600]
        record = json.loads(p.stdout)
        if record.get('node') != node:
            raise ValueError('Probe node mismatch')
        record['source']['transport'] = 'ssh_under_running_allocation'
        return node, record, None
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        return node, None, str(e)


def refresh(db, rows):
    current = {}
    targets = sorted({r['allocated'] for r in full_running(rows)})
    with ThreadPoolExecutor(max_workers=4) as pool:
        for node, observation, error in pool.map(remote_probe, targets):
            record_observation(db, node, observation, error)
            if observation:
                current[node] = observation
    exports(db)
    return current


def healthy(record):
    return bool(record and record.get('gpu_query_ok') and len(record.get('gpus', [])) == 4
                and all('A100' in g['name'] for g in record['gpus']))


def idle(record):
    if not healthy(record) or not record.get('gpu_apps_query_ok'):
        return False
    if record.get('gpu_processes') or record.get('cpu_cache_active', True):
        return False
    try:
        return all(float(g['utilization_percent']) == 0 and float(g['memory_mib']) <= 64 for g in record['gpus'])
    except (ValueError, TypeError):
        return False


def guard_nodes():
    x = json.loads((GUARD / 'registry.json').read_text())
    if not isinstance(x.get('leases'), list):
        raise ValueError('Invalid Guard registry')
    return {n for lease in x['leases'] for n in lease['nodes']}


def plan(rows, records, protected, minimum=2, maximum=4):
    active = {r['allocated']: r for r in full_running(rows)}
    usable = {n for n in active if healthy(records.get(n))}
    remaining = set(active)
    actions = []
    candidates = sorted(active, key=lambda n: (active[n]['end'] == 'N/A', active[n]['end'], n))
    for group in ('low', 'high'):
        for node in candidates:
            ceiling = minimum if group == 'low' else maximum
            if len(remaining) <= ceiling:
                break
            if node not in remaining or classify(records.get(node, {})) != group:
                continue
            if node in protected or not idle(records.get(node)) or len((usable & remaining) - {node}) < minimum:
                continue
            actions.append({'node': node, 'job_id': active[node]['id'], 'group': group,
                            'reason': 'low_pcie_above_floor' if group == 'low' else 'running_cap_exceeded'})
            remaining.remove(node)
    return actions, sorted(remaining)


def commitments(rows):
    running = {r['allocated'] for r in full_running(rows)}
    fixed = {r['requested'] for r in rows if r['full_a100'] and r['state'] == 'PENDING' and r['requested']}
    generic = sum(r['full_a100'] and r['state'] == 'PENDING' and not r['requested'] for r in rows)
    return len(running | fixed) + generic


def monitor(args, db):
    nodes, rows = scheduler()
    record_roster(db, nodes)
    records = refresh(db, rows)
    try:
        protected = guard_nodes()
        guard_error = None
    except (OSError, ValueError, KeyError) as e:
        protected = {r['allocated'] for r in full_running(rows)}
        guard_error = str(e)
    actions, survivors = plan(rows, records, protected)
    cancelled, blocked = [], []
    if args.apply:
        for action in actions:
            _, fresh_rows = scheduler()
            fresh = refresh(db, fresh_rows)
            with (GUARD / 'registry.lock').open('a') as guard_lock:
                fcntl.flock(guard_lock, fcntl.LOCK_EX)
                try:
                    new_plan, _ = plan(fresh_rows, fresh, guard_nodes())
                except (OSError, ValueError, KeyError) as e:
                    blocked.append({'action': action, 'reason': str(e)}); continue
                if action not in new_plan:
                    blocked.append({'action': action, 'reason': 'state_or_workload_changed'}); continue
                run(['scancel', action['job_id']])
                cancelled.append(action)
                with (STATE / 'actions.jsonl').open('a') as log:
                    log.write(json.dumps({'at': now(), 'action': 'cancel', **action}) + '\n')
    _, final_rows = scheduler()
    running = full_running(final_rows)
    high = sorted(r['allocated'] for r in running if classify(records.get(r['allocated'], {})) == 'high')
    notes = []
    if len(high) > 2:
        notes.append('超过两台高等级节点；四台以内由用户自行选择是否取消。')
    if len(running) > 4:
        notes.append('运行节点仍超过四台：受保护、未知或健康容量不足的节点没有自动取消。')
    if guard_error:
        notes.append('Guard 状态不可用，自动取消被阻止：' + guard_error)
    failures = {}
    for row in running:
        node = row['allocated']
        if node not in records:
            attempt = db['nodes'].get(node, {}).get('last_attempt', {})
            failures[node] = attempt.get('error') or 'New allocation not yet probed in this run'
    report = {'at': now(), 'mode': 'apply' if args.apply else 'preview', 'running': running,
              'pending': [r for r in final_rows if r['full_a100'] and r['state'] == 'PENDING'],
              'current_groups': {n: classify(r) for n, r in records.items()}, 'probe_errors': failures,
              'protected_nodes': sorted(protected), 'planned_cancellations': actions,
              'cancelled': cancelled, 'blocked': blocked, 'notes': notes,
              'minimum_usable_nodes': 2, 'maximum_running_nodes': 4, 'schedule_installed': False}
    meaningful = {k: report[k] for k in ('running', 'pending', 'current_groups', 'probe_errors', 'protected_nodes', 'notes')}
    # Predicted end times/priorities are not meaningful changes.
    for field in ('running', 'pending'):
        meaningful[field] = [{k: r[k] for k in ('id', 'state', 'allocated', 'requested')} for r in report[field]]
    signature = hashlib.sha256(json.dumps(meaningful, sort_keys=True).encode()).hexdigest()
    old_path = STATE / 'monitor_latest.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    report['signature'] = signature
    report['notify'] = signature != old.get('signature') or bool(cancelled or blocked)
    save(old_path, report)
    if not args.quiet or report['notify']:
        print(json.dumps(report, ensure_ascii=False, indent=2))


def book(args, db):
    inventory, rows = scheduler()
    queued = {r['requested'] for r in rows if r['state'] in {'PENDING', 'CONFIGURING'}}
    allocated = {r['allocated'] for r in rows if r['state'] in {'RUNNING', 'COMPLETING'}}
    eligible = [n for n, meta in inventory.items() if args.partition in meta['partitions']
                and 'gpu:A100:4' in meta['gres'] and n not in queued
                and (args.renew or n not in allocated)]
    if args.mode == 'high':
        eligible = [n for n in eligible if classify(db['nodes'].get(n, {}).get('latest', {})) == 'high']
    if args.node:
        if args.node.isdigit():
            args.node = 'spartan-gpgpu' + args.node.zfill(3)
        if args.node not in eligible:
            raise ValueError('指定节点不在当前可预约的已知组或已有排队预约')
        eligible = [args.node]
    count = args.count
    if count > 4 or count < 1:
        raise ValueError('一次预约数量必须在 1–4 之间')
    running_count = len({r['allocated'] for r in full_running(rows)})
    room = max(0, 4 - running_count)
    if not args.renew and count > room:
        raise ValueError(f'当前已运行 {running_count} 台；目标四台，仅剩 {room} 个运行名额。排队预约单独显示。')
    if args.renew and args.mode != 'high':
        raise ValueError('续约只支持已知高等级节点，不能扩张未知节点容量')
    if args.renew:
        eligible = [n for n in eligible if n in allocated]
    if len(eligible) < count:
        raise ValueError(f'符合条件且没有重复预约的节点只有 {len(eligible)} 台')
    account, qos, memory, feature = PROFILES[args.partition]
    commands = []
    if args.mode == 'high' or args.node:
        choices = eligible[:count]
    else:
        choices = [None] * count
    for node in choices:
        command = ['sbatch', '--parsable', '--partition=' + args.partition, '--account=' + account,
                   '--qos=' + qos, '--constraint=' + feature, '--nodes=1', '--ntasks=1',
                   '--gres=gpu:A100:4', '--cpus-per-task=32', '--mem=' + str(memory),
                   '--time=' + args.time, '--exclusive', '--chdir=' + str(BOOK),
                   '--job-name=A100-node-' + args.mode, '--comment=spartan-node-manager']
        if node:
            command.append('--nodelist=' + node)
        else:
            # Exclude already held/queued nodes; nodelist is NOT a pool of alternatives.
            excluded = sorted(set(inventory) - set(eligible))
            if excluded:
                command.append('--exclude=' + ','.join(excluded))
        if args.test_only:
            command.append('--test-only')
        command.append(str(BOOK / 'A100-node-managed-fullnode.sh'))
        commands.append(command)
        if args.submit or args.test_only:
            result = run(command)
            print(result)
            if args.submit:
                job_id = result.split(';')[0]
                if not job_id.isdigit():
                    raise ValueError('Unexpected sbatch result')
                with (STATE / 'bookings.jsonl').open('a') as log:
                    log.write(json.dumps({'at': now(), 'job_id': job_id, 'node': node,
                                          'mode': args.mode, 'partition': args.partition}) + '\n')
    print(json.dumps({'submitted': args.submit, 'mode': args.mode, 'commands': commands}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status'); sub.add_parser('inventory')
    m = sub.add_parser('monitor'); m.add_argument('--apply', action='store_true'); m.add_argument('--quiet', action='store_true')
    sub.add_parser('record-local')
    b = sub.add_parser('book')
    b.add_argument('--mode', choices=('high', 'discover'), required=True)
    b.add_argument('--partition', choices=tuple(PROFILES), default='deeplearn')
    b.add_argument('--count', type=int, default=1); b.add_argument('--node')
    b.add_argument('--renew', action='store_true'); b.add_argument('--time', default='7-00:00:00')
    actions = b.add_mutually_exclusive_group()
    actions.add_argument('--submit', action='store_true'); actions.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = load_inventory()
        if args.command == 'status':
            _, rows = scheduler()
            print(json.dumps({'running': full_running(rows), 'pending': [r for r in rows if r['full_a100'] and r['state'] == 'PENDING'],
                              'commitments': commitments(rows), 'target_running': 4, 'minimum_usable': 2}, ensure_ascii=False, indent=2))
        elif args.command == 'inventory':
            print(json.dumps({g: [n for n, r in db['nodes'].items() if classify(r.get('latest', {})) == g]
                              for g in ('high', 'low', 'unknown')}, ensure_ascii=False, indent=2))
        elif args.command == 'record-local':
            if not __import__('os').environ.get('SLURM_JOB_ID'):
                raise ValueError('record-local requires a Slurm allocation')
            spec = importlib.util.spec_from_file_location('node_probe', SKILL / 'scripts' / 'probe_node.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            record = module.probe()
            record_observation(db, record['node'], record); exports(db)
            print(record['node'], classify(record))
        elif args.command == 'monitor':
            monitor(args, db)
        else:
            book(args, db)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr); sys.exit(1)
