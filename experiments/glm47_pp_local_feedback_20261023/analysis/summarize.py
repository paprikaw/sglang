"""Aggregate saved four-arm performance/parity; never launch an experiment.

Unprofiled timed/repeat cohorts stay separate. The bounded diagnostic profile is
not used for latency or throughput comparisons. Partial runs remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ARMS = ('ep8_mtp_on', 'ep8_mtp_off', 'pp2ep4_mtp_on', 'pp2ep4_mtp_off')


def read(path):
    return json.loads(Path(path).read_text())


def provenance(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stats(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite performance sample')
    return dict(count=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p99=values[math.ceil(.99 * len(values)) - 1],
                p99_estimator='nearest_rank', sample_warning='Eight-request P99 is the sample maximum, not a stable tail estimate' if len(values) == 8 else None)


def index_requests(rows):
    indexed = {row['request_index']: row for row in rows}
    if len(rows) != 8 or set(indexed) != set(range(8)):
        raise ValueError('Eight unique semantic request indices required')
    return indexed


def output_parity(left, right):
    a, b = index_requests(left), index_requests(right)
    details = []
    for index in range(8):
        x, y = a[index], b[index]
        xid, yid = x.get('output_ids'), y.get('output_ids')
        valid_ids = isinstance(xid, list) and isinstance(yid, list) and len(xid) == len(yid) == 64
        ids_equal = valid_ids and xid == yid
        text_equal = isinstance(x.get('output_text'), str) and x.get('output_text') == y.get('output_text')
        first_difference = (next((n for n, (v, w) in enumerate(zip(xid, yid)) if v != w), None)
                            if valid_ids and not ids_equal else None)
        details.append(dict(request_index=index, token_ids_equal=ids_equal,
                            text_equal=text_equal, first_different_output_index=first_difference,
                            output_id_evidence_complete=valid_ids))
    return dict(passed=all(r['token_ids_equal'] and r['text_equal'] for r in details),
                criterion='Exact greedy 64 output token IDs and text, request index matched',
                requests=details,
                caveat='MTP on/off or topology parity is measured, not assumed; a mismatch is not a benchmark throughput result.')


def cohort(path):
    value = read(path)
    if value.get('profile_only') or value.get('profile_only_performance_not_production_measurement'):
        raise ValueError('Profile data cannot be production performance')
    if value.get('cuda_profiler_off_during_production') is not True:
        raise ValueError('Missing profiler-off evidence')
    rows = list(index_requests(value['requests']).values())
    if any(r['completion_tokens'] != 64 or r['first_stream_chunk_tokens'] != 1 for r in rows):
        raise ValueError('Incomplete outputs or buffered first-token boundary')
    if value.get('prefix_tokens_per_request') != 4096 or value.get('actual_cached_tokens_per_request') != 4032:
        raise ValueError('Unexpected semantic/cache workload')
    wall = float(value['wall_s'])
    if wall <= 0:
        raise ValueError('Nonpositive wall time')
    middle = []
    for row in rows:
        # Chunk counts are cumulative. Only exact emitted count witnesses are
        # allowed; never linearly interpolate an MTP multi-token SSE chunk.
        points = dict(row.get('token_events', []))
        if 16 in points and 48 in points:
            middle.append((points[48] - points[16]) / 32)
    result = dict(arm=value['arm'], layout=value['layout'], mtp=value['mtp'],
                  cohort=value.get('cohort'), engine_commit=value.get('engine_commit'),
                  bank_identity_sha256=value.get('bank_identity_sha256'),
                  request_count=8, outputs_per_request=64, wall_s=wall,
                  output_throughput_tok_s=sum(r['completion_tokens'] for r in rows) / wall,
                  ttft_ms=stats(1000 * r['ttft_s'] for r in rows),
                  tpot_ms=stats(1000 * r['tpot_s'] for r in rows),
                  end_to_end_ms=stats(1000 * r['end_to_end_s'] for r in rows),
                  exact_count16_to48_cadence_ms=stats(1000 * v for v in middle),
                  spec_accept_length=stats(r.get('meta', {}).get('spec_accept_length') for r in rows),
                  spec_accept_rate=stats(r.get('meta', {}).get('spec_accept_rate') for r in rows),
                  spec_verify_count_per_request=stats(r.get('meta', {}).get('spec_verify_ct') for r in rows),
                  output_rows=rows, source=provenance(path),
                  boundaries=dict(ttft='One atomic HTTP B8 submission to first token received for each request; includes native HiCache H2D, scheduling and 65-token tail forward',
                                  tpot='(last token received - first token received) / 63 for each request; not kernel cycle time',
                                  e2e='Submission to each request last token received',
                                  bank_import='Persistent semantic source bank -> native CPU HiCache excluded; native CPU HiCache -> GPU included',
                                  middle_cadence='Only exact cumulative output counts16 and48, no MTP interpolation; unavailable if chunks skip either count'))
    return result


def summarize(run):
    run = Path(run)
    arms, evidence = {}, []
    for arm in ARMS:
        entry = dict(status='not_measured', cohorts={}, failures=[])
        for name, filename in (('timed', 'p4096_decode_results.json'), ('repeat', 'p4096_decode_repeat_results.json')):
            path = run / arm / filename
            if not path.exists():
                continue
            try:
                entry['cohorts'][name] = cohort(path)
                evidence.append(provenance(path))
            except (ValueError, KeyError, TypeError) as exc:
                entry['failures'].append(dict(cohort=name, error=str(exc)))
        if len(entry['cohorts']) == 2:
            left, right = (entry['cohorts'][name] for name in ('timed', 'repeat'))
            entry['self_repeat_parity'] = output_parity(left['output_rows'], right['output_rows'])
            entry['status'] = 'measured_repeatable' if entry['self_repeat_parity']['passed'] else 'output_not_repeatable'
        elif entry['cohorts']:
            entry['status'] = 'partial'
        result_path = run / arm / 'arm_result.json'
        if result_path.exists():
            owned = read(result_path)
            entry['controller_passed'] = owned.get('passed')
            entry['controller_failure'] = owned.get('failure')
            evidence.append(provenance(result_path))
        activity = run / arm / 'stage_activity_summary.json'
        if activity.exists():
            entry['activity_summary'] = provenance(activity)
        else:
            entry['activity_summary'] = None
        arms[arm] = entry
    comparisons = []
    available = [arm for arm in ARMS if 'timed' in arms[arm]['cohorts']]
    for left, right in itertools.combinations(available, 2):
        a, b = (arms[arm]['cohorts']['timed'] for arm in (left, right))
        identity = a['bank_identity_sha256'] == b['bank_identity_sha256'] and bool(a['bank_identity_sha256'])
        comparisons.append(dict(left=left, right=right, same_semantic_bank_identity=identity,
                                same_engine_commit=a['engine_commit'] == b['engine_commit'],
                                output_parity=output_parity(a['output_rows'], b['output_rows']),
                                right_over_left_throughput=b['output_throughput_tok_s'] / a['output_throughput_tok_s']))
    # Raw generated tokens/text remain in immutable source results; report only
    # parity evidence and metrics to keep the summary compact and inspectable.
    for arm in arms.values():
        for row in arm['cohorts'].values():
            row.pop('output_rows')
    return dict(schema_version=1, run=str(run.resolve()), arms=arms, cross_arm_comparisons=comparisons,
                workload=dict(model='GLM-4.7-AWQ', requests=8, raw_prefix_per_request=4096,
                              actual_native_cached_per_request=4032, recomputed_tail=64,
                              first_forward_new_tokens_per_request=65, query=1, output=64,
                              semantic=True, synthetic_kv=False, local_graph_batch=4),
                evidence=evidence,
                policy='Timed/repeat cohorts are separate; no profiler results enter performance metrics. Missing arms and failed parity remain explicit.')


def markdown(result):
    lines = ['# GLM-4.7 semantic Decode four-arm comparison', '',
             'B8; P4096 raw context (4032 native CPU-cached +64 recomputed tail); Q1→64. Native CPU HiCache→GPU is timed. Source-bank import and profiler are excluded.', '',
             '| Arm | Cohort | Output tok/s | TTFT mean / median / P99 (ms) | TPOT mean / median / P99 (ms) | E2E mean / median / P99 (ms) |',
             '|---|---|---:|---:|---:|---:|']
    def triplet(row):
        return ' / '.join(f'{row[key]:.3f}' for key in ('mean', 'median', 'p99'))
    for arm, data in result['arms'].items():
        for name, row in data['cohorts'].items():
            lines.append(f"| {arm} | {name} | {row['output_throughput_tok_s']:.3f} | {triplet(row['ttft_ms'])} | {triplet(row['tpot_ms'])} | {triplet(row['end_to_end_ms'])} |")
        if not data['cohorts']:
            lines.append(f'| {arm} | not measured | — | — | — | — |')
    lines += ['', 'P99 is the maximum of only eight request samples. Profile component times are diagnostic, not production TPOT.', '',
              '## Output parity', '', '| Comparison | Same semantic bank | Exact token IDs and text |', '|---|---|---|']
    for row in result['cross_arm_comparisons']:
        lines.append(f"| {row['left']} vs {row['right']} | {row['same_semantic_bank_identity']} | {row['output_parity']['passed']} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    output = args.output or args.run / 'fourway_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    output.with_suffix('.md').write_text(markdown(result))
    print(str(output))


if __name__ == '__main__':
    main()
