"""Inspect past source-hash prefix opportunities at observed request arrivals.

This is not a GPU residency model or a counterfactual latency prediction. It
uses each completed replay's actual timing and routing, without future calls.
"""
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics as st
from analyze_balanced_matrix import ROOT, RUNS


def source_key(row):
    return row['source_outer_idx'], row.get('source_inner_idx')


def inspect_episode(data, graphs, release):
    source = {}
    for graph in graphs:
        for conv in graph['conversations']:
            for turn in conv['turns']:
                key = graph['id'], source_key(turn)
                assert key not in source
                source[key] = (tuple(turn['hash_ids']), turn['input_tokens'], graph['block_size'])

    @lru_cache(None)
    def overlap(graph_id, a, b):
        left, na, block = source[graph_id, a]
        right, nb, block_b = source[graph_id, b]
        assert block == block_b
        blocks = 0
        for x, y in zip(left, right):
            if x != y:
                break
            blocks += 1
        return min(blocks * block, na, nb)

    requests = data['requests']
    events = []
    for index, row in enumerate(requests):
        assert row['prefill_instance'] == row['decode_instance']
        graph_id = row['root_id'].rsplit(':', 1)[-1]
        assert source[graph_id, source_key(row)][1] == row['input_tokens']
        ready = row['started']
        available = row['finished'] if release == 'response_end' else ready + row['ttft_ms']/1000
        assert available >= ready
        events.extend(((ready, 1, index), (available, 0, index)))
    history = defaultdict(list)
    rows = []
    for _, kind, index in sorted(events):
        row = requests[index]
        root = row['root_id']
        graph_id = root.rsplit(':', 1)[-1]
        owner = row['prefill_instance'], row['routed_dp_rank']
        key = source_key(row)
        if kind == 0:
            history[root].append((key, owner))
            continue
        local = remote = 0
        for previous, where in history[root]:
            n = overlap(graph_id, key, previous)
            remote = max(remote, n)
            if where == owner:
                local = max(local, n)
        assert 0 <= local <= remote <= row['input_tokens']
        rows.append(dict(root_id=root, conversation_id=row['conversation_id'], turn=row['turn'],
            owner=':'.join(map(str, owner)), input_tokens=row['input_tokens'],
            observed_cached_tokens=row['meta_info'].get('cached_tokens', 0) or 0,
            past_same_owner_hash_prefix_tokens=local,
            past_any_owner_hash_prefix_tokens=remote,
            extra_prefix_on_other_owner_tokens=remote-local))
    assert len(rows) == len(requests)
    return rows


def main(out=None):
    output, provenance = [], []
    for name in RUNS:
        run = ROOT/'runs'/name
        cfg = json.loads((run/'config.json').read_text())
        graph_bytes = Path(cfg['graphs_file']).read_bytes()
        graphs = json.loads(graph_bytes)['graphs']
        for p in sorted(run.glob('results/col_*/*/*/formal.r*.json')):
            repeat = p.stem.split('.')[-1]
            marker = p.with_name('completed.'+repeat+'.json')
            if not marker.exists() or not json.loads(marker.read_text())['passed']:
                continue
            raw = p.read_bytes()
            data = json.loads(raw)
            arm, case, policy = p.relative_to(run).parts[1:4]
            for release in ('response_end', 'first_token'):
                output.extend(dict(run=name, arm=arm, case=case, policy=policy,
                                   repeat=repeat, history_available_at=release, **r)
                              for r in inspect_episode(data, graphs, release))
            provenance.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                graphs_path=cfg['graphs_file'], graphs_sha256=hashlib.sha256(graph_bytes).hexdigest()))
    out = Path(out) if out is not None else ROOT/'matrix_analysis'
    if output:
        with (out/'prefix_locality.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    groups = defaultdict(list)
    for row in output:
        groups[row['arm'], row['case'], row['policy'], row['history_available_at']].append(row)
    lines = ['# 历史前缀位置核查（CPU派生指标）', '',
        '仅分析合并实例。按实际请求到达客户端HTTP发送时刻，检查同一root/copy家族里此前请求的hash前缀；'
        '“同owner”是同实例及Attention-DP rank，“任意owner”是该家族其他位置的历史。'
        '分别以响应结束和首token时刻加入历史，不使用未来请求。', '',
        '| 布局 | 负载 | 历史加入时刻 | 轮数 | 实测cached均值 | 同owner历史hash前缀均值 | 任意owner历史hash前缀均值 | 其他owner额外前缀均值 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    fields = ('observed_cached_tokens', 'past_same_owner_hash_prefix_tokens',
              'past_any_owner_hash_prefix_tokens', 'extra_prefix_on_other_owner_tokens')
    for (arm, case, policy, release), rows in groups.items():
        lines.append(f"| {arm} | {case} | {release} | {len({r['repeat'] for r in rows})}/3 | "
                     + ' | '.join(f'{st.mean(r[k] for r in rows):.2f}' for k in fields) + ' |')
    lines += ['', '**这是hash前缀机会，不是可立即复用的真实KV/GDN。** 不模拟容量、驱逐、GDN checkpoint、'
        'chunk写回或服务端排队。首token不是精确cache commit时刻；GPU实际查询也可能晚于HTTP发送，'
        '因此上述值与实测cached的差额不能直接解释为驱逐损失。两个历史时刻也不是严格的上下界。', '',
        '“任意owner”不代表免费传输，也不是family-sticky策略实际执行的结果。改变路由会改变排队、batch和缓存历史；'
        '该表只用于决定是否值得做缓存亲和的实测控制，不能换算成动态路由加速。', '',
        '[逐请求指标](prefix_locality.csv) · [来源哈希](prefix_locality_sources.json)']
    (out/'PREFIX_LOCALITY.md').write_text('\n'.join(lines)+'\n')
    (out/'prefix_locality_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(episodes=len(provenance), analyzed_request_views=len(output))))


if __name__ == '__main__':
    main()
