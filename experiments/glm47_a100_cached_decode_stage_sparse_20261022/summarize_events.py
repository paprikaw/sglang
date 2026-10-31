"""Deferred CUDA-event API partitions, with explicit sampling perturbation.

No GPU imports or launches. Communication includes exposed waits, not wire time.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parent / 'glm47_a100_cached_decode_stage_profile_20261022/analysis'
sys.path.insert(0, str(ANALYSIS))
from kineto_summary import _event_ancestors, _exclusive_partition, scope_category, file_receipt


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def bucket(phase):
    if phase == 'attention':
        return 'attention'
    if phase in ('expert.routed', 'expert.shared', 'dense_mlp'):
        return 'expert_ffn'
    if phase.startswith('deep_ep.'):
        return 'moe_communication'
    if phase == 'tp_collective':
        return 'tp_communication'
    if phase == 'pp_transport_api':
        return 'pp_transport_and_wait'
    return 'other'


def partition(sample, model, window=None, extra_indices=()):
    lookup = {row['index']: row for row in sample['events'] + sample.get('cpu', []) if 'index' in row}
    stream = model['start_stream']
    bounds = window or (model['start_ms'], model['end_ms'])
    indices = {model['forward_index'], *extra_indices}
    spans = []
    for row in sample['events']:
        if (row['forward_index'] not in indices or row.get('start_stream') != stream
                or row.get('end_stream') != stream or row.get('duration_ms') is None):
            continue
        if row['end_ms'] <= bounds[0] or row['start_ms'] >= bounds[1]:
            continue
        ancestors = _event_ancestors(row, lookup)
        phase = scope_category(ancestors)
        if phase != 'unknown':
            spans.append((row['start_ms'], row['end_ms'], phase, len(ancestors)))
    phases = _exclusive_partition(spans, bounds)
    components = defaultdict(float)
    for phase, duration in phases.items():
        components[bucket(phase)] += duration
    window_ms = bounds[1] - bounds[0]
    if abs(sum(components.values()) - window_ms) > 1e-4:
        raise ValueError('Forward partition does not close')
    return dict(window_ms=window_ms, phase_ms=phases, component_ms=dict(components))


def summarize_rank(path):
    sample = read(path)
    if not sample['passed'] or sample.get('unresolved_event_indices'):
        raise ValueError('Invalid event receipt')
    if sample.get('profiling_method') != 'deferred_cuda_event_api_intervals':
        raise ValueError('Expected the low-overhead event-only protocol')
    meta = {row['forward_index']: row for row in sample['forwards']}
    models = sorted([row for row in sample['events'] if row['category'] == 'model_forward'],
                    key=lambda row: row['forward_index'])
    selected, control_forwards, cycles = [], [], []
    for pos, model in enumerate(models):
        receipt = meta[model['forward_index']]
        m = receipt['metadata']
        if m['forward_mode'] != '2' or m['batch_size'] != 4:
            continue
        if not receipt['detailed']:
            if 4 <= receipt['extend_index'] <= (27 if sample['arm']=='ep8' else 54):
                control_forwards.append(model['duration_ms'])
            continue
        row = dict(forward_index=model['forward_index'], decode_ordinal=receipt['extend_index'],
                   metadata=m, **partition(sample, model))
        selected.append(row)
        if pos+1 < len(models):
            following = models[pos+1]
            next_meta = meta[following['forward_index']]
            if (next_meta['detailed'] and following['start_stream'] == model['start_stream']
                    and following['end_stream'] == model['end_stream']):
                cycles.append(dict(from_forward_index=model['forward_index'],
                    to_forward_index=following['forward_index'],
                    **partition(sample, model, (model['start_ms'], following['start_ms']))))
    expected = len(sample['detailed_forward_ordinals'])
    if len(selected) != expected or not control_forwards:
        raise ValueError('Insufficient detailed or unannotated steady forwards')
    return dict(arm=sample['arm'], rank=sample['rank'], hostname=sample['hostname'],
        forwards=selected, adjacent_microbatch_cycles=cycles,
        same_cohort_unannotated_steady_forward_mean_ms=statistics.mean(control_forwards),
        selected_forward_mean_ms=statistics.mean(row['window_ms'] for row in selected),
        selected_to_unannotated_forward_ratio=statistics.mean(row['window_ms'] for row in selected)/statistics.mean(control_forwards),
        event_pairs_used=sample['event_pairs_used'], event_pool_size=sample['event_pool_size'],
        added_segment_synchronizations=sample['added_segment_synchronizations'],
        cross_stream_event_count=sum(row.get('start_stream') != row.get('end_stream') for row in sample['events']),
        source=file_receipt(path))


def average_windows(windows):
    if not windows:
        return None
    labels = sorted({key for row in windows for key in row['component_ms']})
    phases = sorted({key for row in windows for key in row['phase_ms']})
    duration = statistics.mean(row['window_ms'] for row in windows)
    values = {key: statistics.mean(row['component_ms'].get(key, 0) for row in windows) for key in labels}
    return dict(sample_count=len(windows), mean_window_ms=duration,
        mean_component_ms=values, component_percent={key: 100*value/duration for key,value in values.items()},
        mean_phase_ms={key: statistics.mean(row['phase_ms'].get(key, 0) for row in windows) for key in phases})


def group(rows):
    return dict(ranks=[row['rank'] for row in rows],
        forward=average_windows([f for row in rows for f in row['forwards']]),
        adjacent_microbatch_cycle=average_windows([f for row in rows for f in row['adjacent_microbatch_cycles']]),
        same_cohort_unannotated_forward_ms=statistics.mean(row['same_cohort_unannotated_steady_forward_mean_ms'] for row in rows),
        selected_to_unannotated_forward_ratio=statistics.mean(row['selected_to_unannotated_forward_ratio'] for row in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('ep8','pp2ep4'))
    a = p.parse_args()
    groups, perturbation = {}, {}
    for arm in ((a.arm,) if a.arm else ('ep8','pp2ep4')):
        ranks = [summarize_rank(a.run/arm/f'attribution_rank{rank}.json') for rank in range(8)]
        for row in ranks:
            save(a.out/f'{arm}_rank{row["rank"]}.json',row)
        groups[arm] = {(f'node{node}' if arm=='ep8' else f'stage{node}'):
                       group([row for row in ranks if row['rank']//4==node]) for node in range(2)}
        groups[arm]['all_rank_mean'] = group(ranks)
        profile=read(a.run/arm/'profile_result.json')
        perturbation[arm] = dict(control_tpot_ms=profile['control']['tpot_s']['mean']*1000,
            event_profile_tpot_ms=profile['tpot_s']['mean']*1000,
            profile_to_control_ratio=profile['profile_to_control_tpot_ratio'])
    result=dict(schema_version=1,groups=groups,perturbation=perturbation,
        denominator='Same-stream model-forward execution/wait API interval, independently per rank/stage; not HTTP TPOT.',
        communication_semantics='DeepEP and TP intervals include exposed synchronization/host-feed waits. PP boundary cycle intervals include upstream readiness/slot scheduling; these are not pure network wire times.',
        cross_rank_sum_used=False,source_run=str(a.run.resolve()),
        analysis_sources=[file_receipt(Path(__file__)),file_receipt(ANALYSIS/'kineto_summary.py')])
    save(a.out/('summary.json' if not a.arm else f'summary_{a.arm}.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
