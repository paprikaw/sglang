"""Recompute old EP/PP profiles at identical generated-position keys."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

from analyze import load, union

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'h100_qwen122b_latest_decode_20261027'


def main():
    reader = load('old_window_reader', ROOT.parent / 'h100_pp_lookahead_workloads_20261027/analyze.py')
    tools = load('old_window_buckets', ROOT.parent / 'h100_pp_expert_fragmentation_20261026/analyze.py').load_analysis()
    traces = {}
    for run in ('ep2_native_r5', 'pp2_lookahead_r6'):
        result = json.loads((OLD / 'analysis' / run / 'results.json').read_text())
        entries = [row['trace'] for row in result['components'] if row['workload'] == 'b32_repeated']
        traces[run] = [(entry, reader.read_trace(Path(entry['path']), entry['sha256'], 16)) for entry in entries]
    ep, pp = [traces[name][0][1]['graphs'] for name in ('ep2_native_r5', 'pp2_lookahead_r6')]
    old_ep_keys = sorted({g['key'][0] for g in ep[20:36]})
    old_pp_keys = sorted({g['key'][0] for g in pp[20:36]})
    common = set.intersection(*[{g['key'][0] for g in trace['graphs']} for values in traces.values() for _, trace in values])
    windows = dict(earlier_pp_window=old_pp_keys, later_common_window=[key for key in old_ep_keys if key in common])
    output = dict(old_ep_keys=old_ep_keys, old_pp_keys=old_pp_keys,
                  old_windows_intersection=sorted(set(old_ep_keys) & set(old_pp_keys)), aligned=[])
    for name, keys in windows.items():
        for run, values in traces.items():
            for rank, (entry, trace) in enumerate(values):
                graphs = [g for g in trace['graphs'] if g['key'][0] in keys]
                pp = run.startswith('pp')
                assert len(graphs) == len(keys) * (2 if pp else 1)
                launches = {e['ts']: e for e in trace['events'] if e['name'] == 'cudaGraphLaunch'}
                correlations = {launches[g['launch']]['args']['correlation'] for g in graphs}
                costs, spans = defaultdict(float), defaultdict(list)
                for e in trace['events']:
                    if e.get('cat') == 'kernel' and e['args'].get('correlation') in correlations:
                        bucket = tools.graph_bucket(e['name'])
                        costs[bucket] += e['dur']
                        spans[bucket].append((e['ts'], e['ts'] + e['dur']))
                layers = 24 if pp else 48
                output['aligned'].append(dict(window=name, keys=keys, run=run, rank=rank, trace=entry,
                    graphs=len(graphs), layers_per_graph=layers,
                    kernel_us_per_layer={k: v / (len(graphs) * layers) for k, v in costs.items()},
                    model_graphs_ms_per_generation=st.mean((g['end'] - g['start']) / 1000 for g in graphs) * (2 if pp else 1)))
    path = ROOT / 'analysis/old_window_audit.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
