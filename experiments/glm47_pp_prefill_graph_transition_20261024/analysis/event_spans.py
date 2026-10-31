"""Validate measured GPU stream spans, never substitute kernel or TPOT estimates.

Same-category nesting is a union. Known nested Attention/TP collectives belong
to communication; other cross-category overlap stays explicitly unassigned.
All timestamps are relative to one Graph on ONE device, not host/rank clocks.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

MEASUREMENT = 'external_cuda_event_stream_spans'
LABELS = {'attention', 'expert_routed', 'expert_shared', 'dense_ffn',
          'deepep_dispatch', 'deepep_combine', 'nccl_tp'}
COMM = {'deepep_dispatch', 'deepep_combine', 'nccl_tp'}
TOL = 0.01  # milliseconds: float timestamp subtraction/elapsed roundoff.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'Invalid finite timestamp: ' + name)
    return float(value)


def partition(span, intervals):
    """Exclusive single-device interval partition with explicit overlap."""
    changes = defaultdict(Counter)
    changes[0.0]; changes[span]
    for value in intervals:
        changes[value['start_ms']][value['label']] += 1
        changes[value['end_ms']][value['label']] -= 1
    points = sorted(changes)
    result = {name: 0.0 for name in (*sorted(LABELS), 'overlap', 'other_graph')}
    counts = Counter()
    for a, b in zip(points, points[1:]):
        counts.update(changes[a])
        if b <= a:
            continue
        active = {label for label, count in counts.items() if count > 0}
        # GroupCoordinator collectives are nested inside Attention.forward.
        if 'nccl_tp' in active:
            active.discard('attention')
        key = next(iter(active)) if len(active) == 1 else 'overlap' if active else 'other_graph'
        result[key] += b - a
    require(abs(sum(result.values()) - span) < TOL, 'Partition fails enclosing-span closure')
    return result


def analyze_replay(row, *, layout, rank, target_pp_partition=(46,46)):
    span = number(row['graph_span_ms'], 'graph_span_ms')
    require(span > 0, 'Graph span must be positive')
    require(row.get('completion_fences') == 1 and row.get('component_fences') == 0
            and row.get('read_before_same_graph_reuse') is True,
            'Missing completion/read-before-reuse proof')
    require(row.get('batch_size') == 4 and len(set(row.get('request_ids', []))) == 4,
            'Actual local B4 is required')
    role = row['role']
    require(role in ('target', 'draft'), 'Unknown Graph role')
    require(len(target_pp_partition)==2 and all(type(x) is int and x>0 for x in target_pp_partition)
            and sum(target_pp_partition)==92,'Invalid frozen PP target partition')
    split=target_pp_partition[0]
    expected = list(range(92)) if layout == 'ep8' else list(range(0, split) if rank < 4 else range(split, 92))
    if role == 'draft':
        expected = [0]  # Glm4MoeModelNextN.decoder uses local layer_id=0.
    require(row.get('layer_ids') == expected, 'Incomplete or different actual layer coverage')
    values = []
    for value in row.get('intervals', []):
        label = value.get('label')
        require(label in LABELS, 'Unknown component label')
        require(value.get('stream') == row.get('anchor_stream'), 'Unqualified cross-stream component')
        a, b = number(value['start_ms'], 'start_ms'), number(value['end_ms'], 'end_ms')
        elapsed = number(value['elapsed_ms'], 'elapsed_ms')
        require(-TOL <= a <= b <= span + TOL and elapsed >= 0,
                'Event interval lies outside enclosing Graph')
        require(abs((b - a) - elapsed) <= TOL, 'Independent event elapsed disagrees with timestamps')
        values.append(dict(value, start_ms=max(0.0, a), end_ms=min(span, b)))
    for layer in expected:
        labels = {x['label'] for x in values if x.get('layer_id') == layer}
        needed = {'attention', 'dense_ffn'} if role == 'target' and layer < 3 else {
            'attention', 'expert_routed', 'expert_shared', 'deepep_dispatch', 'deepep_combine'}
        require(needed <= labels, 'Missing actual component coverage at layer ' + str(layer))
    exclusive = partition(span, values)
    unions = {label: sum(b - a for a, b in _union([
        (x['start_ms'], x['end_ms']) for x in values if x['label'] == label])) for label in LABELS}
    return dict(index=row['index'], segment_index=row['segment_index'], role=role, phase=row['phase'],
        target_ordinal=row['target_ordinal'], layer_count=len(expected), graph_span_ms=span,
        exclusive_ms=exclusive, raw_label_union_ms=unions, interval_count=len(values),
        communication_ms=sum(exclusive[x] for x in COMM),
        expert_ms=sum(exclusive[x] for x in ('expert_routed', 'expert_shared', 'dense_ffn')),
        attention_ms=exclusive['attention'], overlap_ms=exclusive['overlap'],
        other_graph_ms=exclusive['other_graph'])


def _union(rows):
    output = []
    for a, b in sorted(rows):
        if not output or a > output[-1][1]:
            output.append([a, b])
        else:
            output[-1][1] = max(output[-1][1], b)
    return output


def mean_replays(rows):
    if not rows:
        return None
    keys = ('graph_span_ms', 'attention_ms', 'expert_ms', 'communication_ms', 'overlap_ms', 'other_graph_ms')
    return dict(count=len(rows), **{key: mean(x[key] for x in rows) for key in keys},
        exclusive_ms={label: mean(x['exclusive_ms'][label] for x in rows)
                      for label in rows[0]['exclusive_ms']})


def analyze_rank(raw, receipt, *, target_pp_partition=(46,46)):
    rank, layout, mtp = raw['rank'], raw['layout'], raw['mtp']
    require(rank in range(8) and layout in ('ep8', 'pp2ep4'), 'Invalid rank/layout')
    require(raw.get('measurement') == MEASUREMENT and raw.get('cupti_enabled') is False,
            'Wrong measurement protocol')
    require(receipt.get('passed') is True and receipt.get('rank') == rank
            and receipt.get('arm') == raw['arm'] and receipt.get('measurement') == MEASUREMENT,
            'Missing matching event receipt')
    require(raw.get('production_timing_affected') is False and raw.get('diagnostic_cadence_changed') is True,
            'Missing separation from production performance')
    segments = raw['segments']
    require([x['index'] for x in segments] == list(range(len(segments))), 'Segment IDs are not unique and ordered')
    target = [x for x in segments if x['phase'].startswith('target_')]
    expected_phase = 'target_verify' if mtp else 'target_decode'
    require(len(target) == 12 and all(x['phase'] == expected_phase for x in target)
            and [x['target_ordinal'] for x in target] == list(range(1, 13)),
            'Exactly twelve actual matching target forwards are required')
    accepted = raw['accepted']
    require(len(accepted) == 12 and sorted(x['target_segment_index'] for x in accepted)
            == sorted(x['index'] for x in target), 'Missing/duplicate actual committed-output records')
    accepted_by_segment = {x['target_segment_index']: x for x in accepted}
    for segment in target:
        value = accepted_by_segment[segment['index']]
        counts = value.get('output_tokens_by_request', {})
        require(segment.get('batch_size') == 4 and len(set(segment.get('request_ids', []))) == 4,
                'Target request identities are not actual B4')
        require(set(counts) == set(segment['request_ids'])
                and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in counts.values())
                and value.get('output_tokens') == sum(counts.values()) > 0,
                'Invalid actual committed output counts')
    replays = raw['replays']
    require([x['index'] for x in replays] == list(range(len(replays))) and len(replays) <= 160,
            'Unbounded or duplicate replay IDs')
    analyzed = []
    for row in replays:
        require(row['segment_index'] in range(len(segments)), 'Replay without an actual API segment')
        segment = segments[row['segment_index']]
        require(row['target_ordinal'] == segment['target_ordinal'] and row['phase'] == segment['phase']
                and row['request_ids'] == segment['request_ids']
                and row['batch_size'] == segment['batch_size'], 'Replay/API identity mismatch')
        require((row['role'] == 'target') == segment['phase'].startswith('target_'), 'Wrong target/draft assignment')
        analyzed.append(analyze_replay(row, layout=layout, rank=rank,target_pp_partition=target_pp_partition))
    for segment in target:
        rows = [x for x in analyzed if x['role'] == 'target' and x['segment_index'] == segment['index']]
        require(len(rows) == 1, 'Missing or duplicate target Graph replay')
        rows[0]['actual_committed_local_output_tokens'] = accepted_by_segment[segment['index']]['output_tokens']
    draft = [x for x in analyzed if x['role'] == 'draft']
    require(bool(draft) == bool(mtp and (layout == 'ep8' or rank >= 4)), 'Draft execution contradicts real rank/layout/MTP')
    boundaries = raw.get('pp_boundaries', [])
    for value in boundaries:
        require(number(value['elapsed_ms'], 'PP boundary span') >= 0, 'Invalid PP boundary elapsed')
    require(layout == 'pp2ep4' or not boundaries, 'EP8 contains PP boundary operations')
    return dict(rank=rank, passed=True, arm=raw['arm'], layout=layout, mtp=mtp,
        target=mean_replays([x for x in analyzed if x['role'] == 'target']),
        draft=mean_replays(draft), phases={phase: mean_replays([x for x in analyzed if x['phase'] == phase])
            for phase in sorted({x['phase'] for x in analyzed})},
        replays=analyzed, pp_boundary_calls=len(boundaries),
        pp_boundary_api_stream_span_mean_ms=mean(x['elapsed_ms'] for x in boundaries) if boundaries else None,
        note='Per actual Graph GPU stream spans, not a complete pipeline cycle or kernel busy time. '
             'Draft ordinals follow the latest target; they do not prove same-request cycle association. '
             'PP boundary async send completion and pure NIC latency are not measured.')


def analyze_arm(out, ranks=range(8)):
    out = Path(out)
    frozen=json.loads((out.parent/'config.json').read_text())
    partition=frozen['arms'][out.name]['target_pp_partition']
    require(list(ranks) == list(range(8)), 'All eight ranks are mandatory')
    values = []
    for rank in range(8):
        raw = json.loads((out / f'event_spans_rank{rank}.json').read_text())
        receipt = json.loads((out / f'activity_receipt_rank{rank}.json').read_text())
        require(raw.get('rank') == rank, 'Rank file identity mismatch')
        require(raw.get('target_pp_partition')==partition,'Actual profile partition differs from frozen configuration')
        values.append(analyze_rank(raw, receipt,target_pp_partition=partition))
    require(len({(x['arm'], x['layout'], x['mtp']) for x in values}) == 1,
            'Cannot mix arms/layouts/MTP in a profile')
    return dict(schema_version=1, passed=True, measurement=MEASUREMENT, ranks=values,
        diagnostic_cadence_changed=True, production_tpot_partition=False,
        aggregation='Per rank and actual Graph role; never add rank/stage maxima. '
                    'Event/outer-completion overhead is present; uncovered Graph span is not idle.')
