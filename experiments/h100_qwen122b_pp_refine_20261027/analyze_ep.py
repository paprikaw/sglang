"""Reuse the accepted EP audit, then add a local-clock cycle/overlap budget."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys

from analyze import load, union

ROOT = Path(__file__).resolve().parent
native = load('refine_native_ep_analysis', ROOT.parent / 'h100_qwen122b_latest_decode_20261027/analyze.py')
native.ROOT = ROOT


def main():
    native.main()
    run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
    result_path = ROOT / 'analysis' / run.name / 'results.json'
    result = json.loads(result_path.read_text())
    reader = load('refine_ep_trace_reader', ROOT.parent / 'h100_pp_lookahead_workloads_20261027/analyze.py')
    tools = load('refine_ep_buckets', ROOT.parent / 'h100_pp_expert_fragmentation_20261026/analyze.py').load_analysis()
    for row in result['components']:
        trace = reader.read_trace(Path(row['trace']['path']), row['trace']['sha256'], row['local_tokens'])
        graphs = trace['graphs'][20:36]
        launches = {e['ts']: e for e in trace['events'] if e['name'] == 'cudaGraphLaunch'}
        correlations = {launches[g['launch']]['args']['correlation'] for g in graphs}
        spans = defaultdict(list)
        for e in trace['events']:
            if e.get('cat') == 'kernel' and e['args'].get('correlation') in correlations:
                bucket = tools.graph_bucket(e['name'])
                if bucket in ('communication.dispatch', 'communication.combine'):
                    kind = 'comm'
                elif not bucket.startswith('communication.'):
                    kind = 'compute'
                else:
                    continue
                spans[kind].append((e['ts'], e['ts'] + e['dur']))
        row['comm_intersection_all_compute_us_per_layer'] = native.intersect(union(spans['comm']), union(spans['compute'])) / (16 * 48)
        row['selected_sequence_sum_keys'] = [g['key'][0] for g in graphs]
        cycles = []
        for i in range(20, 36):
            g, nxt = trace['graphs'][i:i + 2]
            assert nxt['key'][0] == g['key'][0] + row['local_tokens']
            assert g['end'] <= nxt['start']
            cycles.append(dict(model_graph_ms=(g['end'] - g['start']) / 1000,
                               between_graphs_ms=(nxt['start'] - g['end']) / 1000,
                               cycle_ms=(nxt['start'] - g['start']) / 1000))
        row['cycle_mean'] = {k: st.mean(v[k] for v in cycles) for k in cycles[0]}
    result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
