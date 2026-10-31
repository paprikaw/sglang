"""Validate longer Decode traces and retain separate execution/timing scopes."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def union_us(intervals):
    total = 0.
    edge = None
    for lo, hi in sorted(intervals):
        total += hi - lo if edge is None or lo > edge else max(0., hi - edge)
        edge = hi if edge is None else max(edge, hi)
    return total


class IntervalIndex:
    def __init__(self, events):
        grouped = defaultdict(list)
        for event in events:
            grouped[event.get('tid')].append(event)
        self.rows = {}
        for tid, values in grouped.items():
            values.sort(key=lambda e: e['ts'])
            starts, prefix_ends = [], []
            edge = float('-inf')
            for event in values:
                starts.append(event['ts'])
                edge = max(edge, event['ts'] + event['dur'])
                prefix_ends.append(edge)
            self.rows[tid] = values, starts, prefix_ends

    def find(self, timestamp, tid):
        if tid not in self.rows:
            return None
        values, starts, ends = self.rows[tid]
        index = bisect_right(starts, timestamp) - 1
        answer = None
        while index >= 0 and ends[index] > timestamp:
            event = values[index]
            if event['ts'] <= timestamp < event['ts'] + event['dur']:
                if answer is None or event['dur'] < answer['dur']:
                    answer = event
            index -= 1
        return answer


def label(event):
    return '.'.join(event['name'].split('.')[1:-1])


def steady_step(event):
    return event is not None and 'step[DECODE bs=4 ' in event['name']


def graph_bucket(name):
    if 'flash::FlashAttnFwd' in name or 'flash::prepare_varlen_num_blocks_kernel' in name:
        return 'attention.core'
    if 'fused_recurrent_gated_delta_rule' in name:
        return 'attention.core'
    if 'deep_gemm::' in name and '(deep_gemm::GemmType)2' in name:
        return 'expert.routed_gemm'
    if 'deep_ep::' in name:
        return ('communication.dispatch' if 'dispatch' in name else
                'communication.combine' if 'combine' in name else 'communication.other')
    if 'nccl' in name.lower():
        return 'communication.nccl'
    return 'other'


def trace_components(path, arm, execution, hardware, selected_count=24):
    with gzip.open(path, 'rt') as handle:
        data = json.load(handle)
    events = [e for e in data['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    annotations = [e for e in events if e.get('cat') == 'user_annotation']
    steps = [e for e in annotations if e.get('name', '').startswith('step[')]
    step_index = IntervalIndex(steps)
    kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    runtime = {e.get('args', {}).get('correlation'): e for e in events
               if e.get('cat') in ('cuda_runtime', 'cuda_driver') and
               e.get('args', {}).get('correlation') is not None}
    layers_expected = 40 if arm == 'ep2' else 20
    graph_launches = [e for e in events if e.get('name') == 'cudaGraphLaunch']
    base = dict(hardware=hardware, arm=arm, execution=execution, rank_trace=path.name,
        layers_per_rank=layers_expected, trace=str(path),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        unit='ms per rank per actual layer invocation; local Decode batch 4')
    selected_steps, selected_graphs = [], []
    samples, attribution = [], {}
    if hardware == 'H100' and execution == 'native':
        by_correlation = defaultdict(list)
        for event in kernels:
            if event.get('cat') == 'kernel':
                by_correlation[event.get('args', {}).get('correlation')].append(event)
        candidates = []
        for launch in graph_launches:
            step = step_index.find(launch['ts'], launch.get('tid'))
            selected = by_correlation.get(launch.get('args', {}).get('correlation'), [])
            if steady_step(step) and selected:
                candidates.append((launch, step, selected))
        candidates.sort(key=lambda row: row[0]['ts'])
        chosen = candidates[4:4 + selected_count]
        if len(chosen) != selected_count:
            raise AssertionError(f'Need {selected_count} steady Graphs after discarding four; got {len(candidates)}: {path}')
        inventory = Counter()
        for launch, step, selected in chosen:
            full = sum('flash::FlashAttnFwdSm90<' in e['name'] for e in selected)
            linear = sum('fused_recurrent_gated_delta_rule' in e['name'] for e in selected)
            assert (full, linear) == (layers_expected // 4, layers_expected * 3 // 4), (path, full, linear)
            assert all(e.get('args', {}).get('graph id') for e in selected)
            costs = defaultdict(list)
            for event in selected:
                costs[graph_bucket(event['name'])].append((event['ts'], event['ts'] + event['dur']))
                inventory[event['name']] += event['dur'] / 1000
            sample = {key: union_us(value) / 1000 / layers_expected for key, value in costs.items()}
            sample['graph_span_per_layer'] = (max(e['ts'] + e['dur'] for e in selected) - min(e['ts'] for e in selected)) / 1000 / layers_expected
            samples.append(sample)
            selected_steps.append(step)
            selected_graphs.append(dict(cpu_start_us=launch['ts'], step=step['name'], **sample))
        scope = 'actual native CUDA Graph; Attention CORE and routed GEMM only; other retains projections/shared/routing'
        attribution = dict(graph_kernel_duration_ms=dict(inventory), actual_graphs=selected_graphs)
    else:
        assert not graph_launches, ('Eager trace unexpectedly contains a CUDA Graph replay', path)
        layers = [e for e in annotations if re.fullmatch(r'(?:component\.)?layer\.\d+', e.get('name', ''))]
        components = [e for e in annotations if e.get('name', '').startswith('component.')
                      and not e['name'].startswith('component.layer.')]
        layer_index = IntervalIndex(layers)
        component_index = IntervalIndex(components)
        by_layer = defaultdict(lambda: defaultdict(list))
        for event in kernels:
            launch = runtime.get(event.get('args', {}).get('correlation'))
            if launch is None:
                continue
            layer = layer_index.find(launch['ts'], launch.get('tid'))
            if layer is None:
                continue
            component = component_index.find(launch['ts'], launch.get('tid'))
            bucket = label(component) if component else 'other'
            by_layer[id(layer)][bucket].append((event['ts'], event['ts'] + event['dur']))
        grouped = defaultdict(list)
        step_events = {}
        for layer in layers:
            step = step_index.find(layer['ts'], layer.get('tid'))
            if steady_step(step):
                grouped[id(step)].append(layer)
                step_events[id(step)] = step
        candidates = sorted(grouped, key=lambda key: step_events[key]['ts'])
        chosen = candidates[4:4 + selected_count]
        if len(chosen) != selected_count:
            raise AssertionError(f'Need {selected_count} steady forwards after discarding four; got {len(candidates)}: {path}')
        layer_rows = []
        for key in chosen:
            selected = grouped[key]
            assert len(selected) == layers_expected, (path, len(selected))
            assert len({int(e['name'].split('.')[-1]) for e in selected}) == layers_expected
            per_forward = defaultdict(list)
            for event in selected:
                values = {bucket: union_us(intervals) / 1000 for bucket, intervals in by_layer[id(event)].items()}
                assert values.get('expert.routed', 0) > 0 and values.get('attention.full', 0) + values.get('attention.linear', 0) > 0, (path, event['name'], values)
                layer_rows.append(dict(layer=int(event['name'].split('.')[-1]),
                    step=step_events[key]['name'], cpu_start_us=event['ts'], **values))
                for bucket in ('attention.full', 'attention.linear', 'expert.routed', 'expert.shared',
                               'communication.dispatch', 'communication.combine', 'other'):
                    per_forward[bucket].append(values.get(bucket, 0))
            sample = {bucket: statistics.mean(values) for bucket, values in per_forward.items()}
            sample['attention.total'] = sample['attention.full'] + sample['attention.linear']
            samples.append(sample)
            selected_steps.append(step_events[key])
        scope = 'native eager module scopes including Attention projections; routed/shared Expert and DeepEP separate'
        if execution == 'eager_diagnostic':
            scope = 'EAGER DIAGNOSTIC on same loaded H100 model; includes Attention projections; not native Graph timing'
        attribution = dict(actual_layer_rows=layer_rows)
    keys = sorted({key for row in samples for key in row})
    result = dict(base, scope=scope, selected_forwards=len(samples), dropped_initial_decode_forwards=4,
        actual_layer_invocations=len(samples) * layers_expected,
        **{key: statistics.mean(row.get(key, 0) for row in samples) for key in keys})
    # PP boundary intervals remain outside model layer means and Graph internals.
    if arm == 'pp2':
        lo = min(e['ts'] for e in selected_steps)
        hi = max(e['ts'] + e['dur'] for e in selected_steps)
        boundary = []
        for event in kernels:
            if 'nccl' not in event.get('name', '').lower():
                continue
            a, b = max(lo, event['ts']), min(hi, event['ts'] + event['dur'])
            if a < b:
                boundary.append((a, b))
        result['pp_boundary_residency_ms_per_selected_forward'] = union_us(boundary) / 1000 / len(samples)
        result['pp_boundary_includes_wait_and_overlap'] = True
        result['pp_boundary_cpu_window_ms'] = (hi - lo) / 1000
    return result, dict(**base, scope=scope, selected_step_labels=[e['name'] for e in selected_steps],
        samples=samples, **attribution)


def analyze_run(run, partial=False):
    cfg = json.loads((run / 'config.json').read_text())
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    rows, metrics = [], []
    for arm in ('ep2', 'pp2'):
        measured = run / arm / 'metrics.decode.json'
        if measured.exists():
            data = json.loads(measured.read_text())
            assert data['valid'] and not data['profiler_enabled'] and not data['diagnostic_eager']
            assert len(data['rounds']) == cfg['repeats']
            flat = [r for group in data['rounds'] for r in group['requests']]
            assert len(flat) == 24
            assert all(r['cached_tokens'] == 4096 and r['prompt_tokens'] == 4097 and r['output_tokens'] == 128 for r in flat)
            rates = [r['output_tokens_per_s'] for r in data['rounds']]
            metrics.append(dict(hardware=cfg['hardware'], arm=arm,
                rounds=len(rates), requests_per_round=8, gpu_cached_context_per_request=4096,
                new_input_per_request=1, output_per_request=128,
                output_tokens_per_s=statistics.mean(rates), throughput_stdev=statistics.stdev(rates),
                tpot_ms=data['tpot_ms']['mean'], ttft_s=data['ttft_s']['mean'],
                throughput_min=min(rates), throughput_max=max(rates),
                actual_gpu_pair='0,1', graph=cfg['graph'], profiler=False))
        manifests = sorted((run / arm / 'traces').glob('*/manifest.json'))
        expected = 6 if cfg['hardware'] == 'H100' else 3
        if not partial:
            assert len(manifests) == expected, (run, arm, len(manifests), expected)
        for manifest in manifests:
            info = json.loads(manifest.read_text())
            assert info['performance_sample'] is False and len(info['files']) == 2
            for item in info['files']:
                path = Path(item['path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
                summary, detail = trace_components(path, arm, info['execution'], cfg['hardware'], cfg['selected_forwards'])
                summary['repeat'] = int(manifest.parent.name.rsplit('_r', 1)[1])
                detail['repeat'] = summary['repeat']
                rows.append(summary)
                save(out / (path.name.removesuffix('.trace.json.gz') + '.json'), detail)
    if metrics:
        csv_write(out / 'native_metrics.csv', metrics)
    if rows:
        csv_write(out / 'components_per_rank.csv', rows)
        groups = []
        excluded = {'layers_per_rank', 'selected_forwards', 'dropped_initial_decode_forwards', 'actual_layer_invocations', 'repeat'}
        fields = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool) and k not in excluded]
        fields = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in excluded})
        for arm, execution in sorted({(r['arm'], r['execution']) for r in rows}):
            selected = [r for r in rows if (r['arm'], r['execution']) == (arm, execution)]
            repeat_means = []
            for repeat in sorted({r['repeat'] for r in selected}):
                rank_rows = [r for r in selected if r['repeat'] == repeat]
                assert len(rank_rows) == 2
                repeat_means.append({k: statistics.mean(r.get(k, 0) for r in rank_rows) for k in fields})
            group = dict(hardware=cfg['hardware'], arm=arm, execution=execution,
                independent_profiles=len(repeat_means), ranks_per_profile=2,
                selected_forwards_per_rank_per_profile=cfg['selected_forwards'],
                scope=selected[0]['scope'], unit=selected[0]['unit'])
            for key in fields:
                values = [r[key] for r in repeat_means]
                group[key] = statistics.mean(values)
                group[key + '.profile_stdev'] = statistics.stdev(values) if len(values) > 1 else 0.
            groups.append(group)
        csv_write(out / 'components.csv', groups)
    save(out / 'validation.json', dict(passed=not partial, partial=partial,
        metrics_cells=len(metrics), profile_rank_rows=len(rows),
        exact_cached_tokens=4096, exact_prompt_tokens=4097, exact_output_tokens=128,
        actual_local_decode_batch=4, selected_forwards_per_rank=cfg['selected_forwards'],
        no_profiler_in_formal_timing=True,
        native_graph_is_separate_from_eager_diagnostic=True,
        full_model_layers=40, limitations=[
            'Component GPU activity can overlap; it is not an additive TPOT decomposition',
            'PP NCCL residency includes waits and may overlap compute',
            'H100 native Graph Attention CORE/routed GEMM has a narrower scope than eager module profiles',
            'H100 eager diagnostics establish module cost under eager execution, not native Graph module timing',
            'A100/H100 retain different native backends and Graph/overlap settings; not hardware-only causality']))
    return dict(run=str(run), metrics=len(metrics), rank_profiles=len(rows), partial=partial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(analyze_run(args.run.absolute(), args.partial)))
