"""Describe family and SLO differences within accepted, matched static runs.

This is a post-hoc decomposition, not a routing oracle or an isolated per-family
profile. All families experienced the other families' queues and cache pressure.
"""
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent / 'matrix_analysis'
METRICS = ('ttft_ms', 'request_tpot_ms', 'interval_tpot_ms', 'e2e_ms',
           'ready_e2e_ms', 'root_mean_s', 'cached_tokens')
ORDER = ('col_ep4', 'pd_ep4', 'col_pp2', 'col_single')


def atomic(path, text):
    temp = path.with_suffix(path.suffix + '.family.tmp')
    temp.write_text(text)
    temp.replace(path)


def write_csv(name, rows):
    import io
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic(ROOT / name, buffer.getvalue())


def main():
    audit = json.loads((ROOT / 'audit.json').read_text())
    accepted = {r['run'] for r in audit['runs'] if r['run_accepted']}
    sources, trials, identities = [], [], {}
    for source in audit['sources']:
        path = Path(source['path'])
        run = path.parents[4].name
        if run not in accepted:
            continue
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == source['sha256'], path
        episode = json.loads(data)
        arm, case, policy = path.parts[-4:-1]
        repeat = path.stem.split('.')[-1]
        by_family = defaultdict(list)
        for row in episode['requests']:
            family = row['root_id'].rsplit(':', 1)[1]
            by_family[family].append(row)
        for family, calls in by_family.items():
            # Identical calls (including cloned identity and source lengths)
            # must exist in every compared layout and repetition.
            ids = sorted((r['root_id'].split(':', 1)[1],
                          r['conversation_id'].split(':', 1)[1], r['turn'],
                          r['input_tokens'], r['output_tokens']) for r in calls)
            key = case, policy, family
            if key in identities:
                assert identities[key] == ids, key
            identities[key] = ids
            roots = defaultdict(list)
            for row in calls:
                roots[row['root_id']].append(row)
            root_durations = [max(r['finished'] for r in rs) - min(
                r['started'] - r['client_dispatch_lag_ms']/1000 for r in rs)
                for rs in roots.values()]
            intervals = sum(r['output_tokens'] - 1 for r in calls)
            after_first = sum(r['e2e_ms'] - r['ttft_ms'] for r in calls)
            trials.append(dict(run=run, arm=arm, case=case, policy=policy,
                repeat=repeat, family=family, calls=len(calls), roots=len(roots),
                mean_input_tokens=st.mean(r['input_tokens'] for r in calls),
                mean_output_tokens=st.mean(r['output_tokens'] for r in calls),
                ttft_ms=st.mean(r['ttft_ms'] for r in calls),
                request_tpot_ms=st.mean(r['tpot_ms'] for r in calls if r['tpot_ms'] is not None),
                interval_tpot_ms=after_first/intervals,
                e2e_ms=st.mean(r['e2e_ms'] for r in calls),
                ready_e2e_ms=st.mean(r['e2e_ms'] + r['client_dispatch_lag_ms'] for r in calls),
                root_mean_s=st.mean(root_durations),
                cached_tokens=st.mean(r['meta_info'].get('cached_tokens', 0) or 0 for r in calls)))
        sources.append(source)
    groups = defaultdict(dict)
    for row in trials:
        key = row['arm'], row['case'], row['policy'], row['family']
        assert row['repeat'] not in groups[key]
        groups[key][row['repeat']] = row
    groups = {k:v for k,v in groups.items() if set(v) == {'r1', 'r2', 'r3'}}
    means = []
    for (arm, case, policy, family), reps in sorted(groups.items()):
        first = reps['r1']
        means.append(dict(arm=arm, case=case, policy=policy, family=family,
            repeats=3, calls_per_repeat=first['calls'], roots_per_repeat=first['roots'],
            mean_input_tokens=first['mean_input_tokens'],
            mean_output_tokens=first['mean_output_tokens'],
            **{m:st.mean(r[m] for r in reps.values()) for m in METRICS}))
    comparisons = []
    for case, policy, family in sorted({k[1:] for k in groups}):
        arms = [arm for arm in ORDER if (arm, case, policy, family) in groups]
        for left, right in itertools.combinations(arms, 2):
            a, b = groups[left, case, policy, family], groups[right, case, policy, family]
            row = dict(case=case, policy=policy, family=family, reference=left, candidate=right)
            for metric in METRICS:
                deltas = [b[r][metric]-a[r][metric] for r in ('r1','r2','r3')]
                row[metric] = dict(reference=st.mean(v[metric] for v in a.values()),
                    candidate=st.mean(v[metric] for v in b.values()),
                    delta=st.mean(deltas), repeat_deltas=deltas,
                    candidate_lower_repeats=sum(x<0 for x in deltas))
            comparisons.append(row)

    slo = [r for r in csv.DictReader((ROOT/'slo_means.csv').open())
           if r['run'] in accepted and r['repeats']=='3']
    sg = defaultdict(dict)
    for r in slo:
        sg[r['case'],r['policy'],int(r['ttft_limit_ms']),int(r['tpot_limit_ms'])][r['arm']] = r
    slo_comparisons = []
    for (case, policy, ttft, tpot), arms in sorted(sg.items()):
        for left, right in itertools.combinations([a for a in ORDER if a in arms], 2):
            a,b = arms[left],arms[right]
            slo_comparisons.append(dict(case=case, policy=policy,
                ttft_limit_ms=ttft, tpot_limit_ms=tpot, reference=left, candidate=right,
                reference_ready_fraction=float(a['ready_fraction']),
                candidate_ready_fraction=float(b['ready_fraction']),
                reference_ready_goodput_rps=float(a['ready_goodput_rps']),
                candidate_ready_goodput_rps=float(b['ready_goodput_rps']),
                delta_ready_goodput_rps=float(b['ready_goodput_rps'])-float(a['ready_goodput_rps'])))

    text = ['# AgentX：按原始会话家族拆分布局差异', '',
        '仅纳入整组验收通过、三轮完整的同协议布局。每个数是三轮算术平均；C128先对同一家族的16份副本取均值，副本不作为16种独立工作负载。', '',
        '**这是同一混合回放中的事后分解。** 不同家族共享排队与缓存压力；'
        '不能将各家族最佳布局拼接成动态路由收益，也不能把家族ID当作可泛化的在线预测器。'
        '根会话时间仍是压缩回放墙钟，见[时间协议](../TIMING_AUDIT.md)。', '',
        '## EP4与PP2：所有家族，不按胜负筛选', '',
        '| 负载 | 原始家族ID | EP4请求E2E ms | PP2请求E2E ms | PP2更低的重复数 | EP4根会话 s | PP2根会话 s |',
        '|---|---|---:|---:|---:|---:|---:|']
    selected = [r for r in comparisons if r['reference']=='col_ep4' and r['candidate']=='col_pp2']
    for r in selected:
        e,t = r['e2e_ms'],r['root_mean_s']
        text.append(f"| {r['case']} | {r['family']} | {e['reference']:.2f} | {e['candidate']:.2f} | {e['candidate_lower_repeats']}/3 | {t['reference']:.3f} | {t['candidate']:.3f} |")
    labels = {'col_ep4':'EP4', 'pd_ep4':'P/D EP4', 'col_pp2':'PP2', 'col_single':'单卡'}
    available = [arm for arm in ORDER if any(m['arm']==arm for m in means)]
    text += ['', '三轮同方向不等于统计显著；平均值的轻微反转需要结合各轮幅度。'
             '当前家族是预选的8个，不能外推整个AgentX。', '',
             '## 包含单卡的完整已验收基线', '',
             '请求E2E三轮平均，单位ms；最低值仍是混合回放中的事后结果，不能直接用作线上路由标签。', '',
             '| 负载 | 原始家族ID | '+' | '.join(labels[a] for a in available)+' | 最低均值 |',
             '|---|---|'+'---:|'*len(available)+'---|']
    for case, policy, family in sorted({(m['case'],m['policy'],m['family']) for m in means}):
        group = {m['arm']:m for m in means if (m['case'],m['policy'],m['family'])==(case,policy,family)}
        winner = min(group, key=lambda a:group[a]['e2e_ms'])
        text.append(f'| {case} | {family} | '+' | '.join(
            f"{group[a]['e2e_ms']:.2f}" if a in group else '—' for a in available)+f' | {labels[winner]} |')
    text += ['',
             '## 完整SLO网格', '',
             '下表按请求就绪起计TTFT（包括客户端等待），并要求请求平均TPOT达标。'
             '每个目标都使用同一固定布局；较高的达标请求/s只说明该目标下的有限压缩回放结果，'
             '不代表饱和容量或动态机制收益。', '',
             '| 负载 | TTFT上限 ms | TPOT上限 ms/token | '+' | '.join(labels[a]+'达标请求/s' for a in available)+' |',
             '|---|---:|---:|'+'---:|'*len(available)]
    for (case, policy, ttft, tpot), group in sorted(sg.items()):
        text.append(f'| {case} | {ttft} | {tpot} | '+' | '.join(
            f"{float(group[a]['ready_goodput_rps']):.4f}" if a in group else '—' for a in available)+' |')
    text += ['', '[逐家族三轮均值](family_means.csv) · [每轮原值](family_trials.csv) · '
             '[全部布局差值与三轮方向](family_comparisons.json) · '
             '[全部布局SLO差值](family_slo_comparisons.csv) · [文件哈希](family_sources.json)', '']
    write_csv('family_trials.csv', trials)
    write_csv('family_means.csv', means)
    write_csv('family_slo_comparisons.csv', slo_comparisons)
    atomic(ROOT/'family_comparisons.json', json.dumps(comparisons, indent=2)+'\n')
    atomic(ROOT/'family_sources.json', json.dumps(dict(accepted_runs=sorted(accepted),
        sources=sources, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    atomic(ROOT/'FAMILY_TRADEOFFS.md', '\n'.join(text))
    print(json.dumps(dict(accepted_runs=len(accepted), family_points=len(means), comparisons=len(comparisons))))


if __name__ == '__main__':
    main()
