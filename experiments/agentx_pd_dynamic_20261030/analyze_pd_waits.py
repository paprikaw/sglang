"""Explain complete P/D episodes with same-server timestamp intervals.

These wall-time intervals are not GPU kernel or isolated RDMA measurements.
P and D overlap, so their intervals must never be added together.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def interval(meta, left, right):
    a, b = meta.get(left), meta.get(right)
    if a is None or b is None:
        return None
    value = (b-a)*1000
    assert value >= -0.001, (left, right, value)
    return max(value, 0)


def mean_defined(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(values) if values else None


def ready_queue_parts(meta, before_forward_ms):
    """Native queue_time excludes bootstrap/transfer; retraction rewrites it."""
    if (before_forward_ms is None or meta.get('queue_time') is None
            or meta.get('num_retractions')):
        return None, None
    ready_ms = meta['queue_time'] * 1000
    assert 0 <= ready_ms <= before_forward_ms + .01, (before_forward_ms, ready_ms)
    before_ready_ms = before_forward_ms - ready_ms
    assert before_ready_ms >= -.01
    return max(0, before_ready_ms), ready_ms


def csv_write(path, rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = read(run/'config.json')
    manifest = read(run/'launch_manifest.json')
    progress = read(run/'task/progress.json')
    clean = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
    posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
    completed_count = len(list(run.glob('results/*/*/*/completed.r*.json')))
    accepted = bool(completed_count == cfg['expected_points']
        and progress['stage'] == 'complete' and clean.get('passed')
        and clean.get('gpu_idle') and posts and read(posts[-1])['passed'])
    requests, episodes, sources = [], [], []
    fields = ('p_before_forward_ms', 'p_forward_to_end_ms', 'p_end_to_completion_ms',
              'p_total_ms', 'd_before_forward_ms', 'd_forward_to_completion_ms',
              'p_cached_tokens', 'p_uncached_tokens',
              'p_before_ready_ms', 'p_ready_queue_ms',
              'd_before_ready_ms', 'd_ready_queue_ms')
    for path in sorted(run.glob('results/pd_*/*/*/formal.r*.json')):
        repeat = path.stem.split('.')[-1]
        marker = path.with_name('completed.'+repeat+'.json')
        if not marker.exists():
            continue
        assert read(marker)['passed']
        data = read(path)
        assert len(data['requests']) == data['completed_requests']
        arm, case, policy = path.relative_to(run).parts[1:4]
        assert read(run/'evidence'/arm/'serial_qualification.json')['exact_match_reference'] is True
        rows = []
        for call in data['requests']:
            p, d = call['prefill_meta'], call['meta_info']
            row = dict(arm=arm, case=case, policy=policy, repeat=repeat,
                root=call['root_id'], conversation=call['conversation_id'], turn=call['turn'],
                p_instance=call['prefill_instance'], p_dp_rank=p.get('dp_rank'),
                d_instance=call['decode_instance'], d_dp_rank=d.get('dp_rank'),
                input_tokens=call['input_tokens'], returned_output_tokens=call['output_tokens'],
                p_before_forward_ms=interval(p, 'request_received_ts', 'forward_entry_time'),
                p_forward_to_end_ms=interval(p, 'forward_entry_time', 'prefill_finished_time'),
                p_end_to_completion_ms=interval(p, 'prefill_finished_time', 'request_finished_ts'),
                p_total_ms=interval(p, 'request_received_ts', 'request_finished_ts'),
                d_before_forward_ms=interval(d, 'request_received_ts', 'forward_entry_time'),
                d_forward_to_completion_ms=interval(d, 'forward_entry_time', 'request_finished_ts'),
                p_cached_tokens=p.get('cached_tokens'),
                p_uncached_tokens=(call['input_tokens']-p['cached_tokens'])
                    if p.get('cached_tokens') is not None else None)
            if all(row[k] is not None for k in fields[:4]):
                assert abs(sum(row[k] for k in fields[:3])-row['p_total_ms']) < .002
            for role, meta in [('p', p), ('d', d)]:
                before, ready = ready_queue_parts(meta, row[role+'_before_forward_ms'])
                row[role+'_before_ready_ms'] = before
                row[role+'_ready_queue_ms'] = ready
                if before is not None:
                    assert abs(before+ready-row[role+'_before_forward_ms']) < .01
            rows.append(row)
        requests.extend(rows)
        episodes.append(dict(arm=arm, case=case, policy=policy, repeat=repeat,
            requests=len(rows), **{k: mean_defined(rows, k) for k in fields},
            **{k+'_observed': sum(r[k] is not None for r in rows) for k in fields}))
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['arm'], row['case'], row['policy'])].append(row)
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    csv_write(out/'pd_wait_requests.csv', requests)
    csv_write(out/'pd_wait_episodes.csv', episodes)
    owner_groups = defaultdict(list)
    for row in requests:
        for role in ('p', 'd'):
            owner_groups[(row['arm'], row['case'], row['policy'], row['repeat'],
                          role, row[role+'_instance'], row[role+'_dp_rank'])].append(row)
    owner_rows = []
    for key, rows in owner_groups.items():
        role = key[4]
        owner_rows.append(dict(zip(('arm','case','policy','repeat','role','instance','dp_rank'), key)))
        owner_rows[-1].update(requests=len(rows),
            input_tokens=sum(r['input_tokens'] for r in rows),
            returned_output_tokens=sum(r['returned_output_tokens'] for r in rows),
            mean_before_forward_ms=mean_defined(rows, role+'_before_forward_ms'),
            mean_before_ready_ms=mean_defined(rows, role+'_before_ready_ms'),
            mean_ready_queue_ms=mean_defined(rows, role+'_ready_queue_ms'),
            p_cached_tokens=sum(r['p_cached_tokens'] for r in rows)
                if role == 'p' and all(r['p_cached_tokens'] is not None for r in rows) else None,
            p_uncached_tokens=sum(r['p_uncached_tokens'] for r in rows)
                if role == 'p' and all(r['p_uncached_tokens'] is not None for r in rows) else None)
    assert sum(r['requests'] for r in owner_rows if r['role'] == 'p') == len(requests)
    assert sum(r['requests'] for r in owner_rows if r['role'] == 'd') == len(requests)
    csv_write(out/'pd_owner_load.csv', owner_rows)
    lines = ['# P/D等待与缓存诊断', '',
        f'源码 `{manifest["engine_commit"]}`。整组验收：**{accepted}**；运行状态 `{progress["stage"]}`。', '',
        '仅使用有完成标记的episode，表中为已完成轮的算术平均；不足三轮或整组未验收时只作诊断。', '',
        '|布局|负载|轮数|P收到→首次forward ms|P首次forward→prefill结束 ms|P结束→请求完成 ms|D收到→首次forward ms|P缓存token/请求|P新算token/请求|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, rows in groups.items():
        def show(k):
            value = mean_defined(rows, k)
            return '缺失' if value is None else f'{value:.3f}'
        values = ['p_before_forward_ms','p_forward_to_end_ms','p_end_to_completion_ms',
                  'd_before_forward_ms','p_cached_tokens','p_uncached_tokens']
        lines.append('|'+ '|'.join([key[0],key[1],f'{len(rows)}/{cfg["repeats"]}']
                     + [show(k) for k in values])+'|')
    if groups:
        lines += ['', '|布局|负载|轮数|P收到→可运行队列 ms|P可运行队列→首次forward ms|D收到→可运行队列 ms|D可运行队列→首次forward ms|',
                  '|---|---|---:|---:|---:|---:|---:|']
        for key, rows in groups.items():
            values = [mean_defined(rows, k) for k in (
                'p_before_ready_ms', 'p_ready_queue_ms', 'd_before_ready_ms', 'd_ready_queue_ms')]
            lines.append('|'+ '|'.join([key[0],key[1],f'{len(rows)}/{cfg["repeats"]}']
                + ['缺失' if v is None else f'{v:.3f}' for v in values])+'|')
    lines += ['', 'P前三个间隔在同一服务器时钟内互斥，且逐请求验证其和等于P收到至完成。'
        'P首次forward至prefill结束包含分块之间的调度等待，不能视为纯GPU计算。'
        'P结束至请求完成包含交接确认和服务端处理，不能视为纯网络传输。'
        'D收到至首次forward包含等待P和本地状态恢复，不能与P时间相加。', '',
        '第二表使用原生queue_time（秒）拆分首次forward前的时间：它只表示进入可运行队列后的等待。'
        'P进入此队列前还可能等待bootstrap或D端容量；D进入此队列前包括状态预分配、等待P、交接和恢复，不能称纯网络传输。'
        '原生wait_queue_entry_time在回退时会重写，因此该拆分排除num_retractions非零的请求；缺失不补零。', '',
        'P缓存字段来自prefill_meta；不使用客户端合并缓存字段推算每个角色的传输字节。'
        '缺失字段不补零；每列有效样本数保存在pd_wait_episodes.csv。', '',
        '按实例和实际Attention-DP rank分组的表用于检查请求分配、输入长度、P新算token和排队差异；'
        '不把它当作Expert rank的实际工作量或GPU利用率。returned_output_tokens包含P生成的首token，不能直接当D执行步数。', '',
        '[每轮均值](pd_wait_episodes.csv) · [逐请求间隔](pd_wait_requests.csv) · '
        '[缓存所属实例的负载](pd_owner_load.csv) · [来源审计](pd_wait_audit.json)']
    (out/'PD_WAITS.md').write_text('\n'.join(lines)+'\n')
    (out/'pd_wait_audit.json').write_text(json.dumps(dict(engine_commit=manifest['engine_commit'],
        run_accepted=accepted, completed_pd_episodes=len(episodes), requests=len(requests),
        sources=sources, same_server_interval_sum_checked=True,
        queue_split_source=str(Path(manifest['engine_root'])/'python/sglang/srt/observability/req_time_stats.py'),
        queue_split_retracted_requests_excluded=True,
        scope='Wall-time diagnosis; no isolated kernel/network timing; P/D intervals overlap'),indent=2)+'\n')
    print(json.dumps(dict(run=run.name, episodes=len(episodes), requests=len(requests), accepted=accepted)))


if __name__ == '__main__':
    main()
