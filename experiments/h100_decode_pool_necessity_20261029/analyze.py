"""Audit complete GPU-hot cohorts and compare the measured layout envelope.

An oracle compares workload-weighted mean metrics with zero transition cost.
Only its makespan column describes elapsed time for serial cohorts.
It is not an online serving or autoscaling throughput measurement.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runs/171_screen_r1/code'))
from artifact_visibility import wait_for_stable_json

METRICS = ('mean_request_tpot_ms', 'mean_request_e2e_ms',
           'mean_request_ttft_ms', 'cohort_makespan_ms', 'output_tokens_per_s')


def read(path):
    return wait_for_stable_json(path, timeout_s=10, poll_s=.025, min_stable_s=.025)[0]


def descriptor(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_run(run):
    cfg = read(run / 'config.json')
    spec = read(run / 'dashboard_spec.json')
    frozen = [dict(**d, matches=descriptor(Path(d['path']))['sha256'] == d['sha256'])
              for d in spec['frozen_inputs']]
    assert all(d['matches'] for d in frozen), 'Frozen source changed'
    cells, rounds, forwards, sources = [], [], [], []
    for layout in cfg['layouts']:
        for case in cfg['workloads']:
            if layout['profile'] != case['profile']:
                continue
            point = run / 'results' / layout['id'] / case['id']
            identity = dict(run=run.name, node=cfg['node'], profile=case['profile'],
                            layout=layout['kind'], arm=layout['id'], case=case['id'],
                            P=case['P'], N=case['requests'], family=case['family'],
                            point_path=str(point))
            if (point / 'infeasible.json').exists():
                value = read(point / 'infeasible.json')
                assert not value['performance_sample']
                cells.append(dict(**identity, status='capacity_infeasible', rounds=0,
                                  required_kv_tokens=value['required_kv_tokens'],
                                  required_resident_requests=value['required_resident_requests'],
                                  actual_min_kv=min(value['actual_admission']['kv_tokens']),
                                  actual_min_running=min(value['actual_admission']['max_running'])))
                sources.append(descriptor(point / 'infeasible.json'))
                continue
            if not (point / 'completed.json').exists():
                continue
            assert read(point / 'completed.json')['passed']
            protocol = read(point / 'protocol.json')
            assert protocol['P'] == case['P'] and protocol['Q'] == 1 and protocol['O'] == 128
            assert protocol['gpu_hot'] and not protocol['hicache'] and not protocol['cpu_kv_restore']
            assert len(protocol['prompt_sha256']) == case['requests']
            round_values, output_sequences = [], []
            for repeat in range(1, cfg['repeats'] + 1):
                path = point / f'formal.r{repeat}.json'
                data = read(path)
                assert data['performance_sample']
                group = data['result']
                requests = group['requests']
                assert len(requests) == case['requests']
                assert all(r['output_tokens'] == 128 and len(r['output_ids']) == 128
                           and r['prompt_tokens'] == case['P'] + 1
                           and r['cached_tokens'] == case['P'] for r in requests)
                assert all((r.get('cached_tokens_details') or {}).get('host', 0) == 0
                           and (r.get('cached_tokens_details') or {}).get('storage', 0) == 0
                           for r in requests)
                for request in requests:
                    expected = (request['events'][-1]['time_s'] - request['events'][0]['time_s']) * 1000 / 127
                    assert abs(expected - request['tpot_ms']) < 1e-7
                values = dict(**identity, repeat=repeat,
                              mean_request_tpot_ms=statistics.mean(r['tpot_ms'] for r in requests),
                              mean_request_e2e_ms=statistics.mean(r['e2e_s'] * 1000 for r in requests),
                              mean_request_ttft_ms=statistics.mean(r['ttft_s'] * 1000 for r in requests),
                              cohort_makespan_ms=group['makespan_s'] * 1000,
                              output_tokens_per_s=group['output_tokens_per_s'])
                rounds.append(values)
                round_values.append(values)
                sources.append(descriptor(path))
                output_sequences.append([r['output_ids'] for r in requests])
                assert len(group['forward_deltas']) == 4
                for rank in group['forward_deltas']:
                    assert any(k.startswith('DECODE:') for k in rank['counts'])
                    assert any(k.startswith('graph_eligible:') for k in rank['counts'])
                    for name, count in rank['counts'].items():
                        forwards.append(dict(**identity, repeat=repeat, replica=rank['replica'],
                                             pp_rank=rank['shape']['pp_rank'], tp_rank=rank['shape']['tp_rank'],
                                             kind_and_batch=name, count=count))
                if layout['pp'] == 2:
                    receipt = group['scheduling_receipt']
                    assert receipt['policy'] == 9 and len(receipt['deltas']) == 4
                    assert all(d['ahead'] > 0 and d['ahead'] == d['consumed'] and d['fast_sent'] > 0
                               for d in receipt['deltas'])
            if layout['pp'] == 2:
                native = read(point / 'native_fixed_group_validation.json')['result']
                native_sequences = [r['output_ids'] for r in native['requests']]
                assert all(sequences == native_sequences for sequences in output_sequences)
            saved = read(point / 'metrics.decode.json')['metrics']
            result = dict(**identity, status='measured', rounds=len(round_values),
                          prompt_hashes_sha256=hashlib.sha256(json.dumps(protocol['prompt_sha256']).encode()).hexdigest())
            for name in METRICS:
                samples = [r[name] for r in round_values]
                result[name] = statistics.mean(samples)
                result[name + '_stdev'] = statistics.stdev(samples) if len(samples) > 1 else 0
                assert abs(result[name] - saved[name]['mean']) < 1e-7
            cells.append(result)
    post = []
    for path in sorted((run / 'dashboard_guard/checks').glob('*-post.json')):
        value = read(path)
        post.append(dict(path=str(path), passed=value.get('passed'),
                         cleanup_verified=value.get('cleanup_verified'),
                         model_cache=value.get('model_cache'), scientific_success=value.get('scientific_success')))
    return cells, rounds, forwards, sources, dict(run=str(run), frozen=frozen, postchecks=post,
                                                progress=read(run / 'task/progress.json'))


def envelope(cells):
    views = []
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell['node'], cell['profile'], cell['case']), []).append(cell)
    for key, group in grouped.items():
        measured = [c for c in group if c['status'] == 'measured']
        if not measured:
            continue
        for metric in ('mean_request_tpot_ms', 'mean_request_e2e_ms', 'cohort_makespan_ms'):
            best = min(measured, key=lambda c: c[metric])
            non_pp = [c for c in measured if c['layout'] not in ('two_pp2', 'pp4')]
            baseline = min(non_pp, key=lambda c: c[metric]) if non_pp else None
            for cell in measured:
                views.append(dict(node=key[0], profile=key[1], case=key[2], P=cell['P'], N=cell['N'],
                                  family=cell['family'], metric=metric, layout=cell['layout'],
                                  absolute_ms=cell[metric], best_tested_layout=best['layout'], best_ms=best[metric],
                                  excess_over_best_percent=100 * (cell[metric] / best[metric] - 1),
                                  best_non_pp_layout=baseline['layout'] if baseline else '',
                                  best_non_pp_ms=baseline[metric] if baseline else '',
                                  measured_layouts=','.join(sorted(c['layout'] for c in measured))))
    return views


def oracle_pairs(cells):
    output = []
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell['node'], cell['profile']), {})[(cell['case'], cell['layout'])] = cell
    for (node, profile), lookup in grouped.items():
        kinds = sorted(set(k for _, k in lookup))
        cases = sorted(set(c for c, _ in lookup))
        if len(kinds) < 3:
            continue
        for left, right in itertools.combinations(cases, 2):
            # Capacity-only gains require a separate, retuned boundary study.
            # Keep this performance proxy on common measured feasible cells.
            if not all((case, kind) in lookup and lookup[(case, kind)]['status'] == 'measured'
                       for case in (left, right) for kind in kinds):
                continue
            for metric in ('mean_request_tpot_ms', 'mean_request_e2e_ms', 'cohort_makespan_ms'):
                def cost(case, kind):
                    cell = lookup[(case, kind)]
                    return cell[metric] if cell['status'] == 'measured' else math.inf
                left_best = min(cost(left, k) for k in kinds)
                right_best = min(cost(right, k) for k in kinds)
                if not math.isfinite(left_best + right_best):
                    continue
                non_pp = [k for k in kinds if k not in ('two_pp2', 'pp4')]
                for tenth in range(1, 10):
                    weight = tenth / 10
                    static = {k: weight * cost(left, k) + (1 - weight) * cost(right, k) for k in kinds}
                    best_static = min(static, key=static.get)
                    if not math.isfinite(static[best_static]):
                        continue
                    oracle = weight * left_best + (1 - weight) * right_best
                    non_pp_oracle = (weight * min(cost(left, k) for k in non_pp)
                                     + (1 - weight) * min(cost(right, k) for k in non_pp)) if non_pp else math.inf
                    output.append(dict(node=node, profile=profile, left=left, right=right,
                                       left_cohort_fraction=weight, metric=metric, static_layout=best_static,
                                       static_weighted_ms=static[best_static], oracle_weighted_ms=oracle,
                                       oracle_time_reduction_percent=100 * (1 - oracle / static[best_static]),
                                       pp_marginal_time_reduction_percent=100 * (1 - oracle / non_pp_oracle)
                                       if math.isfinite(non_pp_oracle) else '',
                                       candidate_layouts=','.join(kinds)))
    return sorted(output, key=lambda r: r['oracle_time_reduction_percent'], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells, rounds, forwards, sources, audits = [], [], [], [], []
    for run in args.run:
        c, r, f, s, a = audit_run(run.resolve())
        cells.extend(c); rounds.extend(r); forwards.extend(f); sources.extend(s); audits.append(a)
    identifiers = [(c['node'], c['profile'], c['case'], c['layout']) for c in cells]
    assert len(identifiers) == len(set(identifiers)), 'Choose one run per duplicate cell explicitly'
    hashes = {}
    for cell in cells:
        if cell['status'] == 'measured':
            hashes.setdefault((cell['P'], cell['N'], cell['family']), set()).add(cell['prompt_hashes_sha256'])
    assert all(len(v) == 1 for v in hashes.values()), 'Cross-layout prompt mismatch'
    views = envelope(cells)
    oracles = oracle_pairs(cells)
    for name, values in [('mean_metrics', cells), ('formal_rounds', rounds), ('forward_histograms', forwards),
                         ('envelope', views), ('oracle_pairs', oracles)]:
        write_csv(out / (name + '.csv'), values)
    save(out / 'audit.json', dict(runs=audits, sources=sources, checks_passed=True,
                                measured_points=sum(c['status'] == 'measured' for c in cells),
                                capacity_cells=sum(c['status'] == 'capacity_infeasible' for c in cells),
                                formal_rounds=len(rounds),
                                oracle_scope='Same-node, same frozen memory profile, all candidates measured feasible; weighted mean metric proxy; zero switching cost. Only makespan describes serial-cohort elapsed time. Not online throughput.'))
    lines = ['# Decode 配置对比：可核对的平均值', '',
             'P 为每请求 GPU 热缓存长度；N 为整个四卡节点同时请求数。Q1，输出 128 token。',
             '下表全部为逐请求、逐轮算术平均。未完成的点不填值，容量不满足不赋予性能数值。', '',
             '| 节点 | 内存档 | P | N | 输入 | 布局 | 轮数 | TPOT ms | 请求 E2E ms | TTFT ms | 整批完成 ms |',
             '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for cell in sorted(cells, key=lambda c: (c['node'], c['profile'], c['P'], c['N'], c['family'], c['layout'])):
        left = f"| {cell['node'][-3:]} | {cell['profile']} | {cell['P']} | {cell['N']} | {cell['family']} | {cell['layout']} | {cell['rounds']} |"
        if cell['status'] == 'measured':
            lines.append(left + ' ' + ' | '.join(f'{cell[m]:.3f}' for m in METRICS[:4]) + ' |')
        else:
            lines.append(left + ' 容量不满足 | — | — | — |')
    lines += ['', 'PP2 的每个正式输出均与同组原生 PP2 的 128-token 输出逐项相等；所有点检查四个 rank 的 Decode 与 Graph 资格计数。',
              '各节点分别比较。该实验使用封闭的同步请求组，无到达流量、SLO goodput 或实际 autoscaling 收益测量。',
              'oracle_pairs.csv 仅在同节点、同内存档内计算零切换成本的串行 cohort 时间模型。它不是在线收益上界的证明。',
              '完整样本见 formal_rounds.csv；容量条件、来源目录及方差见 mean_metrics.csv；哈希和运行状态见 audit.json。', '']
    (out / 'TABLES.md').write_text('\n'.join(lines))
    print(json.dumps(dict(measured_points=sum(c['status'] == 'measured' for c in cells),
                          capacity_cells=sum(c['status'] == 'capacity_infeasible' for c in cells),
                          formal_rounds=len(rounds), output=str(out),
                          largest_proxy_oracle_gains=oracles[:3])))


if __name__ == '__main__':
    main()
