"""Audit paired profiler-off cohorts and independent, position-matched profiles."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def union(spans):
    result = []
    for a, b in sorted(spans):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def audit(run):
    cfg = read(run / 'config.json')
    assert read(run / 'task/progress.json')['stage'] == 'complete'
    assert read(run / 'task/cleanup.json')['passed']
    assert read(run / 'pp2/cleanup.json')['cleanup_verified']
    posts = [read(p) for p in (run / 'dashboard_guard/checks').glob('*-post.json')]
    assert len(posts) == 1 and posts[0]['passed'] and posts[0]['cleanup_verified']
    records = []
    for line in (run / 'supervisor.log').read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'gpu_registry_state' in value:
            records.append(value)
    assert records[-1]['gpu_registry_state'] == 'released' and records[-1]['postcheck'] == 'passed'
    for item in read(run / 'dashboard_spec.json')['frozen_inputs']:
        assert sha(item['path']) == item['sha256'], item
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--skip-profiles', action='store_true')
    a = ap.parse_args()
    run = a.run.resolve()
    cfg = audit(run)
    reader = load('pp_refine_reader', ROOT.parent / 'h100_pp_lookahead_workloads_20261027/analyze.py')
    tools = load('pp_refine_buckets', ROOT.parent / 'h100_pp_expert_fragmentation_20261026/analyze.py').load_analysis()
    vis = load('pp_refine_visibility', Path(cfg['guard_scripts']) / 'artifact_visibility.py')
    output = dict(run=str(run), guard_postcheck='passed', cleanup_verified=True,
                  all_frozen_inputs_verified=True, metrics=[], paired=[], profiles=[], sources=[], visibility=[])
    for case in cfg['workloads']:
        root = run / 'pp2/workloads' / case['id']
        assert read(root / 'completed.json')['passed']
        assert read(root / 'numerical_equivalence.json')['exact_full_output_ids']
        reference = read(root / 'native_output_reference.json')['output_ids']
        by_policy, traces = {}, {}
        for policy in cfg['gap_policies']:
            path = root / f'p{policy}/metrics.decode.json'
            data, receipt = vis.wait_for_stable_json(path, timeout_s=20)
            output['visibility'].append(receipt)
            assert data['valid'] and not data['profiler_enabled']
            assert len(data['rounds']) == cfg['repeats']
            rates, rows, middle = [], [], []
            for group in data['rounds']:
                assert len(group['requests']) == case['requests']
                assert [r['output_ids'] for r in group['requests']] == reference
                assert group['scheduling_receipt']['policy'] == policy
                deltas = group['scheduling_receipt']['counter_deltas']
                assert all(d['ahead_produced'] == d['ahead_consumed'] and d['ahead_produced'] > 0 for d in deltas)
                if policy == 10:
                    assert sum(d['echoes_skipped'] for d in deltas) > 0
                    assert sum(d['produced'] for d in deltas) > 0
                    assert all(d['produced'] == d['retired'] for d in deltas)
                rates.append(case['requests'] * 128 / group['makespan_s'])
                for row in group['requests']:
                    assert row['cached_tokens'] == 4096 and row['prompt_tokens'] == 4097
                    assert row['output_tokens'] == 128
                    assert [e['completion_tokens'] for e in row['events']] == list(range(1, 129))
                    assert abs(row['tpot_ms'] - (row['e2e_s'] - row['ttft_s']) * 1000 / 127) < 1e-6
                    events = row['events']
                    middle.extend((b['time_s'] - a['time_s']) * 1000
                                  for a, b in zip(events, events[1:]) if 9 <= b['completion_tokens'] <= 120)
                rows += group['requests']
            by_policy[policy] = rates
            output['sources'].append(dict(path=str(path), sha256=sha(path), kind='profiler_off_metrics'))
            output['metrics'].append(dict(workload=case['id'], policy=policy, repeats=len(rates),
                throughput_tok_s=st.mean(rates), throughput_stdev=st.stdev(rates),
                throughput_min=min(rates), throughput_max=max(rates), per_round_throughput=rates,
                tpot_ms=st.mean(r['tpot_ms'] for r in rows), middle_itl_ms=st.mean(middle),
                ttft_ms=st.mean(r['ttft_s'] for r in rows) * 1000,
                request_e2e_ms=st.mean(r['e2e_s'] for r in rows) * 1000,
                cohort_ms=st.mean(g['makespan_s'] for g in data['rounds']) * 1000))
            manifest = root / f'p{policy}/traces/decode/manifest.json'
            if not a.skip_profiles and manifest.exists():
                data, receipt = vis.wait_for_stable_json(manifest, timeout_s=20)
                output['visibility'].append(receipt)
                entries = sorted(data['files'], key=lambda f: f['path'])
                assert len(entries) == 2 and not data['performance_sample']
                traces[policy] = [(entry, reader.read_trace(Path(entry['path']), entry['sha256'], case['requests'] // 2))
                                  for entry in entries]
        if 9 in by_policy and 10 in by_policy:
            changes = [(b / a - 1) * 100 for a, b in zip(by_policy[9], by_policy[10])]
            output['paired'].append(dict(workload=case['id'], p10_vs_p9_mean_throughput_pct=
                (st.mean(by_policy[10]) / st.mean(by_policy[9]) - 1) * 100,
                per_pair_throughput_change_pct=changes, positive_pairs=sum(v > 0 for v in changes),
                pairs=len(changes), paired_change_stdev_pct=st.stdev(changes)))
        if traces:
            # Fixed generated-position window, same sequence-length sums on
            # both policies and both slots. All selected graphs stay local.
            local = case['requests'] // 2
            common = set.intersection(*[{g['key'][0] for g in trace['graphs']} for rank_traces in traces.values() for _, trace in rank_traces])
            keys = sorted(common)[20:36]
            assert len(keys) == 16
            for policy, rank_traces in traces.items():
                for rank, (entry, trace) in enumerate(rank_traces):
                    layers = cfg['layouts'][0].get('layer_partition', [24, 24])[rank]
                    graphs = [g for g in trace['graphs'] if g['key'][0] in keys]
                    assert len(graphs) == 32
                    launches = {e['ts']: e for e in trace['events'] if e['name'] == 'cudaGraphLaunch'}
                    correlations = {launches[g['launch']]['args']['correlation'] for g in graphs}
                    spans = defaultdict(list)
                    for event in trace['events']:
                        if event.get('cat') == 'kernel' and event['args'].get('correlation') in correlations:
                            spans[tools.graph_bucket(event['name'])].append((event['ts'], event['ts'] + event['dur']))
                    components = {k: sum(b - a for a, b in union(v)) / (32 * layers) for k, v in spans.items()}
                    cycles = []
                    all_graphs = trace['graphs']
                    for i, g in enumerate(all_graphs[:-2]):
                        if g['key'][0] not in keys:
                            continue
                        middle, nxt = all_graphs[i + 1:i + 3]
                        assert nxt['key'] == (g['key'][0] + local, g['key'][1])
                        assert g['end'] <= middle['start'] and middle['end'] <= nxt['start']
                        graph_ms = (g['end'] - g['start'] + middle['end'] - middle['start']) / 1000
                        gap_ms = (middle['start'] - g['end'] + nxt['start'] - middle['end']) / 1000
                        cycles.append(dict(two_local_graphs_ms=graph_ms, between_graphs_ms=gap_ms,
                                           cycle_ms=(nxt['start'] - g['start']) / 1000))
                    output['profiles'].append(dict(workload=case['id'], policy=policy, rank=rank,
                        selected_sequence_sum_keys=keys, graph_count=32, layers_per_graph=layers,
                        components_us_per_card_layer=components,
                        cycle_mean={k: st.mean(row[k] for row in cycles) for k in cycles[0]}))
                    output['sources'].append(dict(**entry, kind='independent_profile'))
    out = ROOT / 'analysis' / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('run', 'metrics', 'paired', 'profiles')}, indent=2))


if __name__ == '__main__':
    main()
