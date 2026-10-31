#!/usr/bin/env python3
"""Manually inspect or renew the four verified ConnectX-6 A100 bookings.

--status: inspect only (default). --dry-run: preview missing pending bookings.
--book: submit missing bookings once, then exit. --node 065 limits the action.
No background services, automatic renewal inside jobs, or cancellation.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'outputs' / 'connectx6-booking'
TARGETS = tuple('spartan-gpgpu' + n for n in ('065', '167', '168', '169'))
FIELDS = ('id', 'state', 'partition', 'allocated', 'requested', 'nodes', 'cpus', 'gres', 'name')


def run(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT,
                                   timeout=45).strip()


def queue():
    result = run(['squeue', '--me', '-h', '-o', '%i|%T|%P|%N|%n|%D|%C|%b|%j'])
    rows = []
    for line in result.splitlines():
        values = line.split('|')
        if len(values) != len(FIELDS):
            raise ValueError('Malformed scheduler row; refusing to submit: ' + line)
        rows.append(dict(zip(FIELDS, values)))
    return rows


def matching(rows, node):
    return [r for r in rows if r['partition'] == 'deeplearn'
            and r['nodes'] == '1' and int(r['cpus']) >= 32
            and re.search(r'(?:^|,)gres/gpu(?::A100|:a100)?:4(?:,|$)', r['gres'])
            and (r['requested'] == node or r['allocated'] == node)]


def snapshot(rows):
    return {n: matching(rows, n) for n in TARGETS}


def save(name, data):
    path = STATE / name
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def ensure(nodes=TARGETS, dry_run=False):
    with (STATE / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = queue()  # Failure here cannot be interpreted as an empty queue.
        for node in nodes:
            jobs = matching(rows, node)
            # A running booking plus a single pending successor is the maximum
            # automatically requested. Reuse existing exact-node submissions.
            if any(r['state'] in {'PENDING', 'CONFIGURING', 'SUSPENDED'} for r in jobs):
                print(node, '已有排队或启动中的预约，跳过：',
                      ', '.join(r['id'] for r in jobs), flush=True)
                continue
            if any(r['state'] not in {'RUNNING', 'COMPLETING'} for r in jobs):
                raise ValueError('Unrecognized live job state; refusing duplicate')
            argv = ['sbatch', '--parsable', '--nodelist=' + node,
                    '--job-name=A100-CX6-' + node[-3:], '--chdir=' + str(ROOT),
                    '--comment=connectx6-manual-booking',
                    str(ROOT / 'A100-ConnectX6-fullnode.sh')]
            if dry_run:
                print(node, '将提交一次整节点预约：4 张 A100 / 32 CPU / 700000 MiB / 7 天', flush=True)
                continue
            job_id = run(argv).split(';')[0]
            if not job_id.isdigit():
                raise ValueError('Unexpected submission response: ' + job_id)
            with (STATE / 'submissions.jsonl').open('a') as log:
                log.write(json.dumps({'at': stamp(), 'node': node, 'job_id': job_id}) + '\n')
            print(stamp(), 'submitted', node, job_id, flush=True)
            rows = queue()
        save('status.json', {'checked_at': stamp(), 'host': socket.gethostname(),
                             'pid': os.getpid(), 'nodes': snapshot(rows)})


def show_status(nodes, as_json=False):
    rows = queue()
    data = {n: matching(rows, n) for n in nodes}
    if as_json:
        print(json.dumps({'mode': 'manual', 'checked_at': stamp(), 'nodes': data}, indent=2))
        return
    print('手动预约管理（没有后台服务）  ' + stamp())
    for node, jobs in data.items():
        if not jobs:
            print(node + '：无预约')
        for job in jobs:
            labels = {'PENDING': '排队中', 'RUNNING': '运行中', 'CONFIGURING': '启动中',
                      'COMPLETING': '结束中', 'SUSPENDED': '暂停中'}
            print(node + '：' + labels.get(job['state'], job['state']) + '  作业 ' + job['id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--book', '--once', '--submit', dest='book', action='store_true',
                       help='补齐缺少的排队预约一次，然后退出')
    modes.add_argument('--status', action='store_true', help='只查看预约（默认）')
    modes.add_argument('--dry-run', action='store_true', help='预览补约，不提交')
    def node_name(value):
        node = 'spartan-gpgpu' + value if value in ('065', '167', '168', '169') else value
        if node not in TARGETS:
            raise argparse.ArgumentTypeError('节点必须是 065、167、168、169 之一')
        return node
    p.add_argument('--node', type=node_name, help='只检查或补约指定节点，如 065')
    p.add_argument('--json', action='store_true', help='以 JSON 显示状态')
    a = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    nodes = (a.node,) if a.node else TARGETS
    if a.book or a.dry_run:
        ensure(nodes, dry_run=a.dry_run)
    else:
        show_status(nodes, as_json=a.json)


if __name__ == '__main__':
    main()
