"""Analyze completed all-rank Decode traces, then average within each stage."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import statistics
from kineto_summary import read_json, summarize_trace, file_receipt


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rank_summary(arm, rank, run, out):
    receipt = run / arm / f'attribution_rank{rank}.json'
    sample = read_json(receipt)
    if sample.get('passed') is not True:
        raise ValueError(f'Failed attribution receipt: {receipt}')
    trace_path = receipt.parent / sample['trace_filename']
    trace = read_json(trace_path)
    data = summarize_trace(trace, sample)
    if not data['forward_mapping']['valid']:
        raise ValueError('Invalid forward ordinal mapping')
    parts = data['decode_components']
    if parts['directly_mapped_forward_count'] != parts['detailed_forward_count']:
        raise ValueError('Incomplete detailed forward mapping')
    if any(f['foreign_overlapping_gpu_events'] for f in parts['forwards']):
        raise ValueError('Foreign GPU work overlapped selected forward: inspect raw trace')
    by_device = data['gpu_devices']
    if not by_device or not all(d['matches_expected_device'] for d in by_device.values()):
        raise ValueError('Unexpected GPU device in trace')
    result = dict(arm=arm, rank=rank, hostname=sample['hostname'],
        forward_mapping=data['forward_mapping'], decode_components=parts,
        api_detail=data['cuda_event_api_detail'],
        sources=dict(attribution=file_receipt(receipt), trace=file_receipt(trace_path)))
    save(out / f'{arm}_rank{rank}.json', result)
    del trace, data
    gc.collect()
    return result


def group_summary(rows):
    forwards = [f for row in rows for f in row['decode_components']['forwards']]
    keys = sorted({k for f in forwards for k in f['partition_ms']})
    average = {key: statistics.mean(f['partition_ms'].get(key, 0) for f in forwards) for key in keys}
    window = statistics.mean(f['window_ms'] for f in forwards)
    categories = sorted({k for f in forwards for k in f['category_duration_sum_ms']})
    category_ms = {key: statistics.mean(f['category_duration_sum_ms'].get(key, 0) for f in forwards) for key in categories}
    fractions = {key: value / window * 100 for key, value in average.items()}
    if abs(sum(fractions.values()) - 100) > 1e-6:
        raise ValueError('Stage fractions do not close')
    api_totals = defaultdict(float)
    layer_windows = 0
    for row in rows:
        for layer in row['api_detail']['per_forward_layer_stream']:
            layer_windows += 1
            for category, ms in layer['exclusive_api_partition_ms'].items():
                api_totals[category] += ms
    return dict(ranks=[row['rank'] for row in rows], rank_count=len(rows),
        sampled_forward_count=len(forwards), request_batch_sizes=sorted({f['metadata']['batch_size'] for f in forwards}),
        mean_gpu_forward_window_ms=window,
        mean_partition_ms=average, partition_percent_of_gpu_forward_window=fractions,
        mean_category_kernel_duration_sum_ms=category_ms,
        mean_layer_api_partition_ms={k: v/layer_windows for k, v in api_totals.items()},
        sampled_layer_windows=layer_windows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=('ep8', 'pp2ep4'))
    a = p.parse_args()
    arms = (a.arm,) if a.arm else ('ep8', 'pp2ep4')
    ranks = {}
    for arm in arms:
        ranks[arm] = []
        for rank in range(8):
            result = rank_summary(arm, rank, a.run, a.out)
            ranks[arm].append(result)
            print(arm, rank, result['decode_components']['directly_mapped_forward_count'], flush=True)
    groups = {}
    for arm in arms:
        groups[arm] = {f'node{node}' if arm == 'ep8' else f'stage{node}':
                       group_summary([r for r in ranks[arm] if r['rank']//4 == node])
                       for node in range(2)}
        groups[arm]['all_rank_mean'] = group_summary(ranks[arm])
    controls = {}
    for arm in arms:
        profile = read_json(a.run / arm / 'profile_result.json')
        controls[arm] = dict(control_tpot_ms=profile['control']['tpot_s']['mean']*1000,
            profile_tpot_ms=profile['tpot_s']['mean']*1000,
            profile_to_control_tpot_ratio=profile['profile_to_control_tpot_ratio'])
    save(a.out / ('summary.json' if len(arms)==2 else f'summary_{a.arm}.json'),
         dict(schema_version=1, groups=groups, perturbation=controls,
              denominator='Same-rank detailed model-forward GPU execution window; not HTTP TPOT',
              cross_rank_sum_used=False, run=str(a.run.resolve())))
    print(json.dumps(dict(groups=groups, perturbation=controls), indent=2))


if __name__ == '__main__':
    main()
