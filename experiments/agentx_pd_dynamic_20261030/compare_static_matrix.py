"""Compare complete, accepted matrix arms using absolute arithmetic means.

Repeated clones are not treated as independent workload families. This script
reports all three repeat differences, without an inferential significance claim.
It reads only saved artifacts and never contacts a running server.
"""
import csv
import itertools
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent / 'matrix_analysis'
METRICS = ('ttft_ms', 'request_tpot_ms', 'interval_tpot_ms', 'e2e_ms',
           'ready_e2e_ms', 'root_mean_s', 'wall_s')
LABELS = {
    'col_ep4': '2×EP4 合并',
    'pd_ep4': 'EP4 P→EP4 D',
    'col_pp2': '4×PP2 合并',
    'col_single': '8×单卡合并',
}


def write_atomic(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main():
    audit = json.loads((ROOT / 'audit.json').read_text())
    accepted = {r['run'] for r in audit['runs'] if r['run_accepted']}
    with (ROOT / 'trials.csv').open() as f:
        trials = list(csv.DictReader(f))
    groups = {}
    for t in trials:
        if t['run'] in accepted:
            groups.setdefault((t['case'], t['policy'], t['arm']), {})[t['repeat']] = t
    groups = {k: v for k, v in groups.items() if set(v) == {'r1', 'r2', 'r3'}}
    comparisons = []
    for case, policy in sorted({k[:2] for k in groups}):
        arms = [a for a in LABELS if (case, policy, a) in groups]
        for reference, candidate in itertools.combinations(arms, 2):
            left = groups[case, policy, reference]
            right = groups[case, policy, candidate]
            for repeat in ('r1', 'r2', 'r3'):
                assert left[repeat]['requests'] == right[repeat]['requests']
            result = dict(case=case, policy=policy, reference=reference, candidate=candidate)
            for metric in METRICS:
                a = [float(left[r][metric]) for r in ('r1', 'r2', 'r3')]
                b = [float(right[r][metric]) for r in ('r1', 'r2', 'r3')]
                deltas = [y - x for x, y in zip(a, b)]
                result[metric] = dict(reference=statistics.mean(a), candidate=statistics.mean(b),
                                      delta=statistics.mean(deltas), repeat_deltas=deltas,
                                      candidate_lower_repeats=sum(d < 0 for d in deltas))
            comparisons.append(result)
    write_atomic(ROOT / 'comparisons.json', json.dumps(dict(
        comparisons=comparisons, accepted_runs=sorted(accepted),
        scope='Only three-repeat complete arms with successful Guard and cleanup. '
              'Candidate minus reference; lower latency and completion time is better. '
              'Three repeats and replicated families do not establish population significance.'), indent=2) + '\n')
    text = ['# AgentX：已验收配置的直接对比', '',
            '只纳入三轮完整、Guard后检查及GPU清理通过的配置。每项是三轮算术平均；差值为右侧配置减左侧配置，负值表示右侧耗时更低。原始绝对值与三轮差值均保留。', '',
            'C8／C128是根会话并发上限。请求平均TPOT与按生成间隔加权TPOT口径不同；下面同时报告，不能只凭其中一个判定P/D有益。', '',
            '| 负载 | 左侧 → 右侧 | TTFT差 ms | 请求平均TPOT差 ms/token | 间隔加权TPOT差 ms/token | 请求E2E差 ms | 根会话完成差 s | 整轮差 s |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:
        text.append(f"| {c['case']} | {LABELS[c['reference']]} → {LABELS[c['candidate']]} | "
                    + ' | '.join(f"{c[m]['delta']:+.3f}" for m in
                                 ('ttft_ms', 'request_tpot_ms', 'interval_tpot_ms',
                                  'e2e_ms', 'root_mean_s', 'wall_s')) + ' |')
    if not comparisons:
        text += ['', '目前尚无两种已完整验收的配置，暂不生成跨布局胜负结论。']
    text += ['', '三轮的波动及每轮配对差值见[全部绝对值和差值](comparisons.json)，'
             '各配置的绝对均值见[主表](REPORT.md)。这里的配对是相同回放重复编号，'
             '并非同时执行或随机交错的实验；三轮方向一致也不等于统计显著。'
             'C128的16份副本不作为16种独立工作负载来估计置信区间。']
    write_atomic(ROOT / 'COMPARISONS.md', '\n'.join(text) + '\n')
    print(json.dumps(dict(accepted_runs=len(accepted), comparisons=len(comparisons))))
    from analyze_family_tradeoffs import main as family_analysis
    family_analysis()
    from analyze_strict_slo_overlap import main as strict_slo_analysis
    strict_slo_analysis()


if __name__ == '__main__':
    main()
