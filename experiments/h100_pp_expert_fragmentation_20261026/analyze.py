"""Recompute metrics and component costs from immutable experiment artifacts."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def load_analysis():
    spec = importlib.util.spec_from_file_location('steady_analysis', ROOT.parent / 'pp_ep_steady_decode_20261026/analyze.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    previous = m.graph_bucket
    m.graph_bucket = lambda name: 'expert.routed_gemm' if 'fused_moe_kernel' in name else previous(name)
    return m


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w') as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader(); w.writerows(rows)


def prefill_components(path, arm, tools):
    with gzip.open(path, 'rt') as h:
        events = [e for e in json.load(h)['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    annotations = [e for e in events if e.get('cat') == 'user_annotation']
    layers = [e for e in annotations if e.get('name', '').startswith('component.layer.')]
    components = [e for e in annotations if e.get('name', '').startswith('component.') and e not in layers]
    li, ci = tools.IntervalIndex(layers), tools.IntervalIndex(components)
    runtime = {e.get('args', {}).get('correlation'): e for e in events
               if e.get('cat') in ('cuda_runtime', 'cuda_driver') and e.get('args', {}).get('correlation') is not None}
    costs = defaultdict(list)
    by_layer = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        launch = runtime.get(event.get('args', {}).get('correlation'))
        if launch is None:
            continue
        layer = li.find(launch['ts'], launch.get('tid'))
        if layer is None:
            continue
        component = ci.find(launch['ts'], launch.get('tid'))
        bucket = tools.label(component) if component else 'other'
        by_layer[id(layer)][bucket].append((event['ts'], event['ts'] + event['dur']))
    for row in by_layer.values():
        for bucket, spans in row.items():
            costs[bucket].append(tools.union_us(spans) / 1000)
    result = dict(actual_layer_invocations=len(layers), scope='Profiled cold Prefill cohort, GPU module scopes, accumulated ms/card; not E2E slices')
    result.update({k: sum(v) for k, v in costs.items()})
    result['attention.total'] = result.get('attention.full', 0) + result.get('attention.linear', 0)
    return result


def analyze():
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    visibility_path = Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py')
    vis_spec = importlib.util.spec_from_file_location('artifact_visibility', visibility_path)
    vis = importlib.util.module_from_spec(vis_spec)
    vis_spec.loader.exec_module(vis)
    paths = [p for p in (ROOT / 'runs').rglob('*.json')
             if ('routing' in p.parts or p.name.startswith(('metrics.', 'validation.'))
                 or p.name in ('cleanup.json', 'completed.json', 'manifest.json'))
             and 'dashboard_guard' not in p.parts]
    def observe(p):
        return vis.wait_for_stable_json(p, timeout_s=15)[1]
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(observe, paths))
    (out / 'visibility_receipts.json').write_text(json.dumps(dict(
        observer=str(visibility_path), observer_sha256=hashlib.sha256(visibility_path.read_bytes()).hexdigest(),
        artifacts=receipts), indent=2))
    tools = load_analysis()
    metrics, components, routes, micro, correctness, failures = [], [], [], [], [], []
    reference = {}
    for run in sorted((ROOT / 'runs').iterdir()):
        if not (run / 'config.json').exists():
            continue
        cfg = json.loads((run / 'config.json').read_text())
        progress = json.loads((run / 'task/progress.json').read_text())
        for arm in cfg['arm_order']:
            for phase in ['prefill', 'decode']:
                p = run / arm / f'metrics.{phase}.json'
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                rounds = data['rounds']
                rows = [r for group in rounds for r in group['requests']]
                expected = 1 if phase == 'prefill' else cfg['output_tokens']
                assert all(r['output_tokens'] == expected for r in rows), p
                cached = sorted({r.get('cached_tokens') or 0 for r in rows})
                assert cached == ([0] if phase == 'prefill' else [cfg['prefix_tokens']]), (p, cached)
                throughput = [sum(r['output_tokens'] for r in g['requests']) / g['makespan_s'] if phase == 'decode'
                              else cfg['requests'] * cfg['prefix_tokens'] / g['makespan_s'] for g in rounds]
                metrics.append(dict(run=run.name, mode=cfg['expert_mode'], arm=arm, phase=phase,
                    guard_stage=progress['stage'], validation_saved=(run / arm / 'validation.probabilities.json').exists(),
                    requests=cfg['requests'], context=0 if phase == 'prefill' else cfg['prefix_tokens'],
                    new_tokens=cfg['prefix_tokens'] if phase == 'prefill' else 1, output_tokens=expected,
                    throughput_tok_s=statistics.mean(throughput), throughput_stdev=statistics.stdev(throughput),
                    cohort_ms=statistics.mean(g['makespan_s'] for g in rounds) * 1000,
                    request_e2e_ms=statistics.mean(r['e2e_s'] for r in rows) * 1000,
                    ttft_ms=statistics.mean(r['ttft_s'] for r in rows) * 1000,
                    tpot_ms=statistics.mean(r['tpot_ms'] for r in rows) if phase == 'decode' else None,
                    repeats=len(rounds), source=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                texts = [r['text'] for r in rounds[0]['requests']]
                stable = sum(r['text'] == texts[i] for group in rounds for i, r in enumerate(group['requests']))
                if run.name == 'r1_native' and arm == 'pp2':
                    reference[phase] = texts
                if phase in reference:
                    matches = sum(a == b for a, b in zip(texts, reference[phase]))
                    correctness.append(dict(run=run.name, arm=arm, phase=phase,
                        equal_texts_to_native_pp=matches, requests=len(texts),
                        repeated_text_matches=stable, repeated_text_total=len(rows)))
            for phase in ['prefill', 'decode']:
                p = run / arm / 'traces' / phase / 'manifest.json'
                if not p.exists():
                    continue
                manifest = json.loads(p.read_text())
                for desc in manifest['files']:
                    trace = Path(desc['path'])
                    assert hashlib.sha256(trace.read_bytes()).hexdigest() == desc['sha256']
                    try:
                        if phase == 'decode':
                            value, detail = tools.trace_components(trace, arm, 'native', 'H100', cfg['selected_forwards'])
                        else:
                            value = prefill_components(trace, arm, tools)
                        components.append({**value, 'run':run.name, 'mode':cfg['expert_mode'], 'arm':arm,
                                           'phase':phase, 'trace':str(trace)})
                    except Exception as error:
                        failures.append(dict(run=run.name, trace=str(trace), error=repr(error)))
        for p in sorted((run / 'pp2/routing').glob('*.json')):
            d = json.loads(p.read_text())
            if p.name.startswith('microbench'):
                for name, value in d['modes'].items():
                    micro.append(dict(run=run.name, pp_rank=d['routing']['pp_rank'], tokens=d['routing']['tokens'],
                        implementation=name, graph_ms=value['graph_median_ms'], **d['numerical'][name], source=str(p)))
            else:
                routes.append(dict(run=run.name, **{k:d[k] for k in ['phase','pp_rank','tokens','layer_id','active_experts','assignments','num_local_experts']},
                                   **{'padded_rows_' + k:v for k,v in d['padded_rows'].items()}, source=str(p)))
    for name, values in [('metrics', metrics), ('components', components), ('routing', routes), ('microbench', micro), ('correctness', correctness)]:
        csv_write(out / (name + '.csv'), values)
    (out / 'analysis_status.json').write_text(json.dumps(dict(metrics=len(metrics), components=len(components), errors=failures), indent=2))
    print(json.dumps(dict(metrics=metrics, correctness=correctness, errors=failures), indent=2))


if __name__ == '__main__':
    analyze()
