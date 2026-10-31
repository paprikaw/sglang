"""CPU-only validation and GPU attribution for the producer-ready PP test."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ep2_pp2_qwen_a100_20261024'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(run):
    sys.path.insert(0, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
    from artifact_visibility import wait_for_stable_json
    old = load_module('original_trace_analysis', BASE / 'analyze.py')
    intervals = load_module('trace_intervals', ROOT.parent /
                            'ep8_pp2ep4_workload_index_20261024/analyze_qwen_prefill_tension.py')
    root = run / 'pp2'
    output = run / 'analysis'
    output.mkdir(exist_ok=True)
    cfg = json.loads((run / 'config.json').read_text())
    visibility = []
    performance = []
    metrics = {}
    for phase in ('prefill', 'decode'):
        for policy in ('baseline', 'ready'):
            p = root / f'metrics.{phase}.{policy}.json'
            d, receipt = wait_for_stable_json(p, timeout_s=120)
            visibility.append(receipt)
            assert not d['profiler_enabled']
            assert d['valid'] or d.get('diagnostic_only')
            assert len(d['rounds']) == (2 * cfg['repeats'] if phase == 'prefill' else 2)
            requests = [r for g in d['rounds'] for r in g['requests']]
            for g in d['rounds']:
                assert len(g['requests']) == 8
                duration = max(r['finished'] for r in g['requests']) - min(r['started'] for r in g['requests'])
                assert abs(duration - g['makespan_s']) < 1e-8
                for r in g['requests']:
                    assert r['output_tokens'] == (1 if phase == 'prefill' else 128)
                    assert r['prompt_tokens'] == (4096 if phase == 'prefill' else 4097)
                    assert (r['cached_tokens'] or 0) < 64 if phase == 'prefill' else r['cached_tokens'] >= 4032
                rate = (32768 if phase == 'prefill' else 1024) / duration
                actual = g['input_tokens_per_s'] if phase == 'prefill' else g['output_tokens_per_s']
                assert abs(rate - actual) < 1e-6
            row = dict(phase=phase, policy=policy, valid=d['valid'], requests=len(requests), rounds=len(d['rounds']),
                       tokens_per_s=d['input_tokens_per_s'] if phase == 'prefill' else d['output_tokens_per_s'])
            for key in ('ttft_s', 'e2e_s', 'tpot_ms'):
                values = sorted(r[key] for r in requests if r[key] is not None)
                if values:
                    assert abs(statistics.mean(values) - d[key]['mean']) < 1e-8
                    for stat in ('mean', 'median', 'p99'):
                        row[key + '_' + stat] = d[key][stat]
                else:
                    for stat in ('mean', 'median', 'p99'):
                        row[key + '_' + stat] = None
            performance.append(row)
            metrics[phase, policy] = d
    trace_rows = []
    for policy in ('baseline', 'ready'):
        for p in sorted((root / 'traces' / policy).glob('*.trace.json.gz')):
            result = old.trace_summary(p)
            inv = [i for i in result['invocations'] if 'EXTEND bs=1 toks=4096 ' in i['step']]
            assert len(inv) == 192, (p, len(inv))
            with gzip.open(p, 'rt') as f:
                raw = json.load(f)
            ev = [e for e in raw['traceEvents'] if e.get('ph') == 'X' and e.get('dur', 0) > 0]
            kernels = [e for e in ev if e.get('cat') == 'kernel']
            nccl = [(e['ts'], e['ts'] + e['dur']) for e in kernels if 'nccl' in e['name'].lower()]
            scopes = [e for e in ev if e.get('cat') == 'user_annotation'
                      and e['name'].startswith('component.attention.')]
            launches = {e['args']['correlation']: e for e in ev
                        if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
            attention_intervals, projection_intervals = [], []
            for k in kernels:
                launch = launches.get(k.get('args', {}).get('correlation'))
                if launch and intervals.enclosing(scopes, launch['ts'], launch.get('tid')):
                    span = (k['ts'], k['ts'] + k['dur'])
                    attention_intervals.append(span)
                    if '::marlin::Marlin' in k['name']:
                        projection_intervals.append(span)
            rank = int(p.name.split('-PP-')[-1].split('.')[0])
            row = dict(policy=policy, stage=rank, invocations=len(inv),
                attention_ms_per_layer=sum(sum(v for k, v in i['gpu_component_union_ms'].items()
                     if k.startswith('attention.')) for i in inv) / len(inv),
                routed_expert_ms_per_layer=sum(i['gpu_component_union_ms'].get('expert.routed', 0) for i in inv) / len(inv),
                shared_expert_ms_per_layer=sum(i['gpu_component_union_ms'].get('expert.shared', 0) for i in inv) / len(inv),
                projection_ms_per_layer=intervals.duration(projection_intervals) / 1000 / len(inv),
                attention_nccl_overlap_ms=intervals.overlap(attention_intervals, nccl) / 1000,
                nccl_kernel_union_ms=intervals.duration(nccl) / 1000,
                longest_nccl_ms=max([e['dur'] / 1000 for e in kernels if 'nccl' in e['name'].lower()], default=0),
                readiness_wait_calls=sum(1 for e in ev if e.get('cat') == 'user_annotation'
                    and e['name'].startswith('pp.ready_send.wait.')),
                source=str(p), source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            trace_rows.append(row)
            (output / (policy + '-stage' + str(rank) + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    assert len(trace_rows) == 4
    for name, rows in [('metrics.csv', performance), ('profile.csv', trace_rows)]:
        with (output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(artifact_checks_passed=True, prefill_accepted=metrics['prefill','ready']['valid'],
        decode_accepted=metrics['decode','ready']['valid'], no_new_gpu_launch=True, slots=cfg['pp_loop_slots'],
        performance=performance, profiles=trace_rows,
        prefill_throughput_ratio=metrics['prefill', 'ready']['input_tokens_per_s'] /
                                 metrics['prefill', 'baseline']['input_tokens_per_s'],
        prefill_ttft_ratio=metrics['prefill', 'ready']['ttft_s']['mean'] /
                           metrics['prefill', 'baseline']['ttft_s']['mean'],
        visibility=visibility)
    (output / 'validated.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args()
    analyze(args.run.resolve())
