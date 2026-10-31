"""Preserve single-cohort HY3 capacity evidence separately from the H100 audit."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/hy3_stream_cached_measure_v4_20261020')
comparison = json.loads((SOURCE / 'comparison.json').read_text())
assert comparison['repetitions'] == 1
observations = []
sources = {}

def record(path):
    data = path.read_bytes()
    sources[str(path)] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return sources[str(path)]['sha256']

for name in ['comparison.json', 'comparison.csv', 'PROTOCOL.md']:
    record(SOURCE / name)
for metric in comparison['metrics']:
    prefix = metric['cached_context_per_request']
    path = SOURCE / 'runs' / metric['run'] / metric['layout'] / f"p{prefix}_{metric['workload']}_results.json"
    raw = json.loads(path.read_text())
    assert raw['wall_s'] == metric['cohort_wall_s']
    assert raw['request_count'] == len(raw['requests']) == 8
    assert raw['prefix_tokens_per_request'] == prefix
    assert raw['output_tokens_per_request'] == metric['output_per_request']
    assert raw['old_context_constructed_by_model'] is False
    assert raw['hicache_h2d_timed'] is True
    for request in raw['requests']:
        assert request['completion_tokens'] == raw['output_tokens_per_request']
        assert request['token_events'][0][0] == 1
        assert request['token_events'][-1][0] == raw['output_tokens_per_request']
    observations.append({
        'run': metric['run'], 'layout': metric['layout'], 'phase': metric['workload'],
        'requests': 8, 'cached_prefix_per_request': prefix,
        'new_input_per_request': raw['query_tokens_per_request'],
        'output_per_request': raw['output_tokens_per_request'],
        'cohort_wall_ms': raw['wall_s'] * 1000,
        'longest_ttft_ms': max(r['ttft_s'] for r in raw['requests']) * 1000,
        'longest_decode_tail_ms': max(r['token_events'][-1][1] - r['token_events'][0][1] for r in raw['requests']) * 1000 if metric['workload'] == 'decode' else '',
        'source': str(path), 'source_sha256': record(path),
    })
assert len(observations) == 8
with (ROOT / 'hy3_capacity_observations.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(observations[0]))
    writer.writeheader()
    writer.writerows(observations)
(ROOT / 'hy3_sources.json').write_text(json.dumps({
    'sources': list(sources.values()), 'gpu_count': 8, 'gpu': 'A100 80GB',
    'model': comparison['model'], 'attention_tp': comparison['attention_tp'],
    'capacities': comparison['capacities'], 'protocol': comparison['synthetic_context'],
    'repetitions_per_point': 1, 'new_measurements': 0,
    'limitations': ['Synthetic CPU cache; not a quality evaluation', 'Includes H2D and queueing', 'No repeatability estimate'],
}, ensure_ascii=False, indent=2) + '\n')
print(f'Validated {len(observations)} historical HY3 observations against original results; {len(sources)} source hashes.')
