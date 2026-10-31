"""Describe observed Prefill cache/queue imbalance without a causal simulator.

Use P metadata for disaggregated requests and local metadata for colocated
requests. An Attention-DP cache owner is not an Expert-compute rank workload.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from completed_episode_files import completed_formals, gate_sha256
from analyze_pd_waits import interval, ready_queue_parts


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if rows:
        with path.open('w') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    elif path.exists():
        path.unlink()


def mean(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def cv(values):
    return st.pstdev(values)/st.mean(values) if st.mean(values) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    cfg, manifest = read(run/'config.json'), read(run/'launch_manifest.json')
    files, pending = completed_formals(run)
    owners, episodes, sources = [], [], []
    for path in files:
        arm, case, policy = path.relative_to(run).parts[1:4]
        repeat = path.stem.split('.')[-1]
        assert read(run/'evidence'/arm/'serial_qualification.json')['exact_match_reference'] is True
        data = read(path)
        rows = data['requests']
        assert len(rows) == data['completed_requests']
        assert policy == 'sticky_hash', 'This audit assumes conversation-sticky placement'
        role = 'prefill' if any(e['role'] == 'decode' for e in cfg['endpoints'][arm]) else 'colocated'
        group = {(e['id'], rank): [] for e in cfg['endpoints'][arm]
                 if e['role'] == role for rank in range(e['ep'])}
        conversations = defaultdict(set)
        for call in rows:
            meta = call['prefill_meta'] if role == 'prefill' else call['meta_info']
            rank = meta.get('dp_rank')
            if rank is None:
                rank = call['routed_dp_rank']
            owner = (call['prefill_instance'], rank)
            assert owner in group
            cached = meta.get('cached_tokens')
            assert cached is not None and 0 <= cached <= call['input_tokens']
            before = interval(meta, 'request_received_ts', 'forward_entry_time')
            _, ready = ready_queue_parts(meta, before)
            item = dict(root=call['root_id'], conversation=call['conversation_id'],
                is_root=call['root_id'] == call['conversation_id'],
                input_tokens=call['input_tokens'], cached_tokens=cached,
                new_tokens=call['input_tokens']-cached, ready_queue_ms=ready,
                before_forward_ms=before)
            group[owner].append(item)
            conversations[item['conversation']].add(owner)
        assert all(len(x) == 1 for x in conversations.values())
        ident = dict(arm=arm, case=case, policy=policy, repeat=repeat, role=role)
        local = []
        for (instance, rank), calls in sorted(group.items()):
            root = [x for x in calls if x['is_root']]
            child = [x for x in calls if not x['is_root']]
            tokens = sum(x['input_tokens'] for x in calls)
            cached = sum(x['cached_tokens'] for x in calls)
            value = dict(**ident, instance=instance, dp_rank=rank, requests=len(calls),
                root_sessions=len({x['root'] for x in root}), root_calls=len(root),
                child_sessions=len({x['conversation'] for x in child}), child_calls=len(child),
                input_tokens=tokens, cached_tokens=cached,
                new_tokens=sum(x['new_tokens'] for x in calls),
                cached_fraction=cached/tokens if tokens else None,
                root_input_tokens=sum(x['input_tokens'] for x in root),
                root_new_tokens=sum(x['new_tokens'] for x in root),
                child_input_tokens=sum(x['input_tokens'] for x in child),
                child_new_tokens=sum(x['new_tokens'] for x in child),
                mean_ready_queue_ms=mean(x['ready_queue_ms'] for x in calls),
                ready_queue_observed=sum(x['ready_queue_ms'] is not None for x in calls),
                mean_before_forward_ms=mean(x['before_forward_ms'] for x in calls))
            assert value['root_calls']+value['child_calls'] == value['requests']
            assert value['cached_tokens']+value['new_tokens'] == value['input_tokens']
            assert value['root_new_tokens']+value['child_new_tokens'] == value['new_tokens']
            local.append(value)
        assert sum(x['requests'] for x in local) == len(rows)
        assert sum(x['root_sessions'] for x in local) == len({x['root_id'] for x in rows})
        owners.extend(local)
        episodes.append(dict(**ident, cache_owners=len(local), requests=len(rows),
            root_sessions_cv=cv([x['root_sessions'] for x in local]),
            root_input_tokens_cv=cv([x['root_input_tokens'] for x in local]),
            all_input_tokens_cv=cv([x['input_tokens'] for x in local]),
            new_tokens_cv=cv([x['new_tokens'] for x in local]),
            root_new_tokens_cv=cv([x['root_new_tokens'] for x in local])))
        sources.append(dict(path=str(path), sha256=sha(path)))
    progress = read(run/'task/progress.json')
    clean = read(run/'task/cleanup.json') if (run/'task/cleanup.json').exists() else {}
    posts = sorted((run/'dashboard_guard/checks').glob('*-post.json'))
    accepted = bool(len(files) == cfg['expected_points'] and progress['stage'] == 'complete'
                    and clean.get('passed') and clean.get('gpu_idle') and posts and read(posts[-1])['passed'])
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['arm'], row['case'], row['policy'], row['role'])].append(row)
    means = []
    fields = ['root_sessions_cv', 'root_input_tokens_cv', 'all_input_tokens_cv',
              'new_tokens_cv', 'root_new_tokens_cv']
    for key, values in groups.items():
        means.append(dict(zip(['arm','case','policy','role'], key), repeats=len(values),
            cache_owners=values[0]['cache_owners'],
            **{f: st.mean(v[f] for v in values) for f in fields}))
    out = run/'analysis'; out.mkdir(exist_ok=True)
    for name, values in [('cache_owner_pressure.csv', owners),
                         ('cache_owner_pressure_trials.csv', episodes),
                         ('cache_owner_pressure_means.csv', means)]:
        write_csv(out/name, values)
    lines = ['# Prefill缓存位置与等待差异', '',
        f'整组验收：**{accepted}**；完成episode：{len(files)}/{cfg["expected_points"]}。仅纳入通过完成标记的请求。', '',
        '下表每行是一个位置在一轮内的累计绝对token数，等待列为该位置请求的算术平均。'
        '同一conversation的所有调用已验证保持同一位置；分离布局只使用P侧metadata，未拿D合并缓存字段估计独立复用。', '',
        '|布局／负载／轮次|位置|根会话|根调用|子调用|根输入token|根新算token|子新算token|可运行队列等待ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in owners:
        label = f"{row['arm']} / {row['case']} / {row['repeat']}"
        pos = row['instance']+f" / DP{row['dp_rank']}"
        wait = '缺失' if row['mean_ready_queue_ms'] is None else f"{row['mean_ready_queue_ms']:.3f}"
        lines.append(f"|{label}|{pos}|{row['root_sessions']}|{row['root_calls']}|{row['child_calls']}|"
                     f"{row['root_input_tokens']:,}|{row['root_new_tokens']:,}|{row['child_new_tokens']:,}|{wait}|")
    lines += ['', '新算token是输入长度减服务器报告的cached_tokens，不是GPU kernel FLOPs、Expert行数或网络字节。'
        '缓存查询发生时刻、容量、GDN检查点、分支分配和调度都可能影响它；这里描述相关差异，不孤立归因。'
        '请求数或根输入量均衡不保证实际新算token和等待均衡。改变路由会同时改变batch、缓存历史和排队，'
        '不能把逐位置最好结果拼成在线策略收益。', '',
        'trial/means文件另外提供跨位置的总体标准差÷均值（CV）：每轮先算CV，再对已完成轮取算术平均；'
        '不足三轮及整组未验收的数值仅作诊断。它描述分布差异，不是延迟预测器或显著性检验。', '',
        '[逐位置数据](cache_owner_pressure.csv) · [每轮差异](cache_owner_pressure_trials.csv) · '
        '[三轮汇总入口](cache_owner_pressure_means.csv) · [来源](cache_owner_pressure_audit.json)']
    (out/'CACHE_OWNER_PRESSURE.md').write_text('\n'.join(lines)+'\n')
    (out/'cache_owner_pressure_audit.json').write_text(json.dumps(dict(
        engine_commit=manifest['engine_commit'], run_accepted=accepted, completed_episodes=len(files),
        sources=sources, excluded_unmarked_formals=pending,
        gate_sha256=gate_sha256(), analyzer_sha256=sha(Path(__file__)),
        requests_reconciled=True, root_counts_reconciled=True, conversation_affinity_checked=True,
        scope='Observed P/colocated cache-owner work and waiting; not Expert load or isolated causality'), indent=2)+'\n')
    print(json.dumps(dict(run=run.name, episodes=len(files), owners=len(owners), accepted=accepted)))


if __name__ == '__main__':
    main()
