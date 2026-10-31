"""CPU-only, evidence-based CUDA activity attribution for the four-arm study.

Calibration links actual GPU correlation IDs to innermost native CPU launch
scopes. Exact names need independent warmup witnesses; conflicting names stay
unknown. Generic GEMMs may additionally be assigned by an entire held-out-
validated, exact Graph kernel sequence, only on the rank that was calibrated.
No timing values are imported from calibration, and no cross-host clock is used.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

GPU_CATEGORIES = {'kernel', 'cuda_kernel', 'gpu_memcpy', 'gpu_memset'}
RUNTIME_CATEGORIES = {'cuda_runtime', 'cuda_driver', 'hip_runtime'}
PHASES = ('attention', 'expert_routed', 'expert_shared', 'dense_ffn',
          'deepep_dispatch', 'deepep_combine', 'nccl_tp', 'pp_boundary',
          'other_gpu', 'unclassified_gpu', 'copy_or_set', 'marker',
          'overlap', 'gpu_idle')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def provenance(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def arg(event, *names):
    values = {str(k).lower().replace('_', ' '): str(v)
              for k, v in event.get('args', {}).items()}
    return next((values[n.lower().replace('_', ' ')] for n in names
                 if n.lower().replace('_', ' ') in values), None)


def launch_fingerprint(event):
    """Complete observed launch geometry; never infer absent trace fields.

    Generic Triton/ATen names can cover unrelated operators. A fingerprint is
    useful only after independent native launch-scope witnesses agree, on the
    same rank and target/draft role. Stream/graph IDs are deliberately not
    identities: capture/replay use different native streams and graph IDs.
    """
    values = {str(k).lower().replace('_', ' '): v
              for k, v in event.get('args', {}).items()}
    name = event.get('name')
    grid, block = values.get('grid'), values.get('block')
    registers, shared = values.get('registers per thread'), values.get('shared memory')
    if (not isinstance(name, str) or not name
            or not all(isinstance(v, (list, tuple)) and len(v) == 3
                       and all(type(x) is int and x > 0 for x in v) for v in (grid, block))
            or type(registers) is not int or registers < 0
            or type(shared) is not int or shared < 0):
        return None
    return dict(name=name, grid=list(grid), block=list(block),
                registers_per_thread=registers, shared_memory=shared)


def fingerprint_key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def interval(event):
    low, duration = float(event['ts']), float(event.get('dur', 0))
    if duration < 0:
        raise ValueError('Negative trace duration')
    return low, low + duration


class ScopeIndex:
    """Nested stage-cal CPU scopes indexed per actual runtime launch thread."""
    def __init__(self, rows):
        groups = defaultdict(list)
        self.threads = {}
        for index, event in enumerate(rows):
            if event.get('ph') == 'X' and event.get('name', '').startswith('stage-cal:'):
                groups[(str(event.get('pid')), str(event.get('tid')))].append((index, event))
        for thread, items in groups.items():
            items.sort(key=lambda p: (interval(p[1])[0], -interval(p[1])[1]))
            stack, nodes = [], []
            for index, event in items:
                low, high = interval(event)
                while stack and not (nodes[stack[-1]]['low'] <= low and high <= nodes[stack[-1]]['high']):
                    stack.pop()
                nodes.append(dict(index=index, event=event, low=low, high=high,
                                  parent=stack[-1] if stack else None))
                stack.append(len(nodes) - 1)
            self.threads[thread] = ([n['low'] for n in nodes], nodes)

    def containing(self, launch):
        item = self.threads.get((str(launch.get('pid')), str(launch.get('tid'))))
        if item is None:
            return []
        starts, nodes = item
        low, high = interval(launch)
        position = bisect_right(starts, low) - 1
        result = []
        while position >= 0:
            node = nodes[position]
            if node['low'] <= low and high <= node['high'] + .001:
                result.append(node)
            position = node['parent'] if node['parent'] is not None else -1
        return result


def calibration_rows(events):
    """Actual GPU events with unique correlation and innermost launch scope."""
    events = [e for e in events if e.get('ph') == 'X' and 'ts' in e and 'dur' in e]
    scopes, launches = ScopeIndex(events), defaultdict(list)
    for event in events:
        if event.get('cat') in RUNTIME_CATEGORIES:
            correlation = arg(event, 'correlation', 'correlation id')
            if correlation is not None:
                launches[correlation].append(event)
    rows = []
    for event in sorted((e for e in events if e.get('cat') in ('kernel', 'cuda_kernel')),
                        key=lambda e: (float(e['ts']), int(arg(e, 'stream') or 0))):
        correlation = arg(event, 'correlation', 'correlation id')
        calls = launches.get(correlation, [])
        containers = scopes.containing(calls[0]) if len(calls) == 1 else []
        phase = containers[0]['event']['name'].split(':', 1)[1] if containers else 'unclassified_gpu'
        if phase not in PHASES:
            raise ValueError('Unknown explicit calibration phase: ' + phase)
        rows.append(dict(name=event['name'], phase=phase,
                         scope_id=containers[0]['index'] if containers else None,
                         correlation=correlation, unique_launch=len(calls) == 1,
                         launch_fingerprint=launch_fingerprint(event)))
    return rows


def build_calibration(arm_dir):
    """Use actual scoped launch witnesses; import labels/geometry, not timing."""
    arm_dir = Path(arm_dir)
    observations = defaultdict(lambda: dict(phases=set(), warm0=set(), warm1=set(), pp_scopes=set(), count=0))
    fingerprints = {}
    warm_groups, sources = defaultdict(dict), []
    paths = sorted(arm_dir.glob('calibration_rank*_graph*_warm*.json'))
    paths += sorted(arm_dir.glob('calibration_pp_p2p_rank*.json'))
    if not paths:
        raise ValueError('No qualified launch-scope calibration traces')
    reuse = None
    config_path = arm_dir.parent / 'config.json'
    if config_path.is_file():
        cfg = read(config_path)
        if cfg.get('profile_only_followup'):
            source = cfg['profile_source']
            config_evidence = provenance(Path(source['source_run']) / 'config.json')
            if config_evidence['sha256'] != source['source_config_sha256']:
                raise ValueError('Original calibration configuration SHA changed')
            bound = source['arms'][arm_dir.name]['calibrations']
            expected = {str(Path(item['copied_path']).resolve()): item for item in bound}
            if set(map(lambda p: str(p.resolve()), paths)) != set(expected):
                raise ValueError('Unexpected/missing offline calibration copy')
            for path in paths:
                item = expected[str(path.resolve())]
                if provenance(path)['sha256'] != item['sha256'] or provenance(item['path'])['sha256'] != item['sha256']:
                    raise ValueError('Offline calibration/source bytes changed: ' + str(path))
            reuse = dict(source_configuration=config_evidence, original_traces=bound,
                         timing_values_imported=False, same_execution_identity=True)
    for path in paths:
        evidence = provenance(path)
        sources.append(evidence)
        rows = calibration_rows(read(path)['traceEvents'])
        warm = re.fullmatch(r'calibration_rank(\d+)_(target|draft)_graph(\d+)_warm(\d+)\.json', path.name)
        if warm:
            rank, role, graph, index = warm.groups()
            group = (int(rank), role, int(graph))
            warm_groups[group][int(index)] = rows
            witness = path.name
        else:
            witness = path.name
        for row in rows:
            record = observations[row['name']]
            record['phases'].add(row['phase'])
            record['count'] += 1
            if warm and int(index) in (0, 1):
                record['warm' + index].add(group)
            elif row['phase'] == 'pp_boundary' and row['scope_id'] is not None:
                record['pp_scopes'].add((witness, row['scope_id']))
            if warm and row['launch_fingerprint'] is not None:
                key = (int(rank), role, fingerprint_key(row['launch_fingerprint']))
                fingerprint = fingerprints.setdefault(key, dict(
                    fingerprint=row['launch_fingerprint'], phases=set(), warm0=set(), warm1=set(), count=0))
                fingerprint['phases'].add(row['phase'])
                fingerprint['count'] += 1
                if int(index) in (0, 1):
                    fingerprint['warm' + index].add(int(graph))
    accepted, audit = {}, {}
    for name, row in observations.items():
        phases = row['phases']
        repeated = row['warm0'] & row['warm1']
        independent_pp = phases == {'pp_boundary'} and len(row['pp_scopes']) >= 2
        eligible = (len(phases) == 1 and 'unclassified_gpu' not in phases
                    and (bool(repeated) or independent_pp))
        if eligible:
            accepted[name] = next(iter(phases))
        audit[name] = dict(eligible=eligible, phases=sorted(phases), count=row['count'],
                           independent_warm_pairs=[list(x) for x in sorted(repeated)],
                           pp_independent_scope_count=len(row['pp_scopes']))
    sequences, rejected_sequences = [], []
    for (rank, role, graph), warms in sorted(warm_groups.items()):
        # Entire sequence, including labels, must reproduce in held-out warm1.
        train = [(r['name'], r['phase']) for r in warms.get(0, [])]
        heldout = [(r['name'], r['phase']) for r in warms.get(1, [])]
        if train and train == heldout:
            sequences.append(dict(rank=rank, role=role, graph=graph,
                                  names=[r[0] for r in train], phases=[r[1] for r in train],
                                  validation='Entire ordered kernel-name/phase sequence reproduced in independent warm1'))
        else:
            rejected_sequences.append(dict(rank=rank, role=role, graph=graph,
                                           train_kernels=len(train), heldout_kernels=len(heldout),
                                           reason='Missing warm or ordered sequence/phase mismatch'))
    fingerprint_audit, accepted_fingerprints = [], []
    for (rank, role, _), row in sorted(fingerprints.items()):
        pairs = row['warm0'] & row['warm1']
        eligible = len(row['phases']) == 1 and 'unclassified_gpu' not in row['phases'] and bool(pairs)
        entry = dict(rank=rank, role=role, fingerprint=row['fingerprint'],
                     eligible=eligible, phases=sorted(row['phases']), count=row['count'],
                     independent_warm_graph_pairs=sorted(pairs))
        fingerprint_audit.append(entry)
        if eligible:
            accepted_fingerprints.append(dict(rank=rank, role=role, fingerprint=row['fingerprint'],
                                             phase=next(iter(row['phases'])),
                                             independent_warm_graph_pairs=sorted(pairs)))
    return dict(schema_version=1, exact_names=accepted, audit=audit, sequences=sequences,
                launch_fingerprints=accepted_fingerprints, launch_fingerprint_audit=fingerprint_audit,
                rejected_sequences=rejected_sequences, sources=sources, offline_reuse= reuse,
                method='GPU correlation -> unique runtime launch -> innermost CPU scope; warm0 train and warm1 held-out. Generic names need a complete exact sequence or a complete exact launch fingerprint (name/grid/block/registers/shared-memory) independently witnessed on the same rank and target/draft role. Conflicts and missing geometry stay unknown. No name-substring rules; no imported calibration timing.')


def subsequence_positions(haystack, needle):
    """KMP exact contiguous matches; full Graph kernels may have API wrappers."""
    if not needle:
        return []
    prefix, index = [0] * len(needle), 0
    for i in range(1, len(needle)):
        while index and needle[index] != needle[i]:
            index = prefix[index - 1]
        if needle[index] == needle[i]:
            index += 1
        prefix[i] = index
    result, index = [], 0
    for i, value in enumerate(haystack):
        while index and needle[index] != value:
            index = prefix[index - 1]
        if needle[index] == value:
            index += 1
        if index == len(needle):
            result.append(i + 1 - index)
            index = prefix[index - 1]
    return result


def assign_phases(kernels, calibration, rank, role, *, use_launch_fingerprints=True):
    """Return evidence assignment; conflicting sequence/name claims stay unknown."""
    names = [e['name'] for e in kernels]
    claims = [set() for _ in kernels]
    sequence_matches = []
    for sequence in calibration.get('sequences', []):
        if sequence['rank'] != rank or sequence['role'] != role:
            continue
        positions = subsequence_positions(names, sequence['names'])
        for start in positions:
            for offset, phase in enumerate(sequence['phases']):
                if phase != 'unclassified_gpu':
                    claims[start + offset].add(phase)
        if positions:
            sequence_matches.append(dict(graph=sequence['graph'], positions=positions,
                                         kernel_count=len(sequence['names'])))
    result = []
    fingerprint_phases = defaultdict(set)
    if use_launch_fingerprints:
        for row in calibration.get('launch_fingerprints', []):
            if row['rank'] == rank and row['role'] == role:
                fingerprint_phases[fingerprint_key(row['fingerprint'])].add(row['phase'])
    for event, name, phases in zip(kernels, names, claims):
        exact = calibration['exact_names'].get(name)
        if exact:
            phases.add(exact)
        fingerprint = launch_fingerprint(event)
        if fingerprint is not None:
            phases.update(fingerprint_phases.get(fingerprint_key(fingerprint), ()))
        result.append(next(iter(phases)) if len(phases) == 1 else 'unclassified_gpu')
    return result, sequence_matches


def partition(events, start, end):
    """Disjoint wall-clock attribution; same-phase overlap is unioned once."""
    if end <= start:
        raise ValueError('Non-positive GPU window')
    changes = defaultdict(Counter)
    for event in events:
        low, high = interval(event)
        low, high = max(start, low), min(end, high)
        if high > low:
            phase = event['phase']
            if phase not in PHASES:
                raise ValueError('Unknown partition phase ' + phase)
            changes[low][phase] += 1
            changes[high][phase] -= 1
    changes[start]; changes[end]
    active, last, totals = Counter(), start, Counter({p: 0. for p in PHASES})
    for point, delta in sorted(changes.items()):
        phases = {phase for phase, n in active.items() if n > 0}
        category = 'gpu_idle' if not phases else next(iter(phases)) if len(phases) == 1 else 'overlap'
        totals[category] += point - last
        active.update(delta)
        last = point
    duration = end - start
    if abs(sum(totals.values()) - duration) > max(.001, duration * 1e-9):
        raise ValueError('Activity partition is not exhaustive')
    groups = dict(attention=('attention',), expert_routed=('expert_routed',),
                  expert_shared=('expert_shared',), dense_ffn=('dense_ffn',),
                  communication=('deepep_dispatch', 'deepep_combine', 'nccl_tp', 'pp_boundary'),
                  other_gpu=('other_gpu', 'copy_or_set', 'marker'),
                  unclassified_gpu=('unclassified_gpu',), overlap=('overlap',), gpu_idle=('gpu_idle',))
    return dict(start_us=start, end_us=end, duration_ms=duration / 1000,
                ms={phase: totals[phase] / 1000 for phase in PHASES},
                percent={phase: 100 * totals[phase] / duration for phase in PHASES},
                exclusive_group_ms={group: sum(totals[p] for p in phases) / 1000 for group, phases in groups.items()},
                exclusive_group_percent={group: 100 * sum(totals[p] for p in phases) / duration for group, phases in groups.items()},
                method='Disjoint per-device wall-clock partition; cross-category concurrency is overlap, not double-counted. Idle means no traced GPU activity, not proven pipeline bubble.')


def marker_segments(receipt, gpu_events):
    marker = receipt['marker_kernel_name']
    observed = sorted([e for e in gpu_events if e.get('cat') in ('kernel', 'cuda_kernel') and e['name'] == marker],
                      key=lambda e: float(e['ts']))
    signature = receipt.get('same_session_marker_signature')
    if signature is not None:
        if (signature.get('name') != marker or signature.get('observed_proof_count', 0) < 2
            or not receipt.get('bootstrap', {}).get('same_session_marker_proofs')):
            raise ValueError('Invalid same-session marker proof signature')
        proofs = [p for p in signature.get('proofs', []) if p.get('actual_gpu_kernel_count') == 1]
        keys = {(str(p.get('correlation')), str(p.get('device')), str(p.get('stream'))) for p in proofs}
        if len(keys) != len(proofs) or len(keys) != signature['observed_proof_count']:
            raise ValueError('Duplicate or incomplete same-session marker proof identity')
        excluded = [e for e in observed if (arg(e, 'correlation', 'correlation id'), arg(e, 'device'), arg(e, 'stream')) in keys]
        if len(excluded) != len(keys):
            raise ValueError('Actual proof kernels do not match all attested correlation/device/stream identities')
        # Exclude only specifically CPU-scope-correlated proofs, never the first
        # three by order/name, and preserve strict 2-markers-per-segment audit.
        observed = [e for e in observed if e not in excluded]
        if observed and max(float(e['ts']) + float(e.get('dur', 0)) for e in excluded) > float(observed[0]['ts']):
            raise ValueError('Marker proof overlaps the formal model profile window')
    segments = receipt['segments']
    if len(observed) != 2 * len(segments):
        raise ValueError(f'Marker mismatch: {len(observed)} actual, {2 * len(segments)} required')
    result = []
    for i, meta in enumerate(segments):
        first, last = observed[2 * i:2 * i + 2]
        if arg(first, 'stream') != arg(last, 'stream'):
            raise ValueError('Coarse marker pair changed GPU streams')
        # Native cudaStream_t in receipt is not CUPTI stream ID. Validate its
        # stability separately, without pretending the two ID spaces coincide.
        if meta['start_stream'] != meta['end_stream']:
            raise ValueError('Receipt marker pair changed native streams')
        low, high = interval(first)[0], interval(last)[1]
        if high <= low:
            raise ValueError('Non-positive GPU segment')
        result.append(dict(meta, gpu_start_us=low, gpu_end_us=high))
    return result


def describe(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def accepted_normalization(receipt, segments):
    targets = [s for s in segments if s['phase'].startswith('target_')]
    cycle_segments = targets[:-1]
    rows = {r['target_segment_index']: r for r in receipt['accepted']}
    matches = [rows.get(s['index']) for s in cycle_segments]
    if not cycle_segments or any(r is None for r in matches):
        return dict(valid=False, reason='Actual committed output advances missing for a selected complete cycle')
    counts = [r['output_tokens'] for r in matches]
    if any(n <= 0 for n in counts):
        return dict(valid=False, reason='Nonpositive actual committed token count')
    by_request = [n for row in matches for n in row['output_tokens_by_request'].values()]
    return dict(valid=True, complete_local_microbatch_cycles=len(cycle_segments),
                actual_committed_output_tokens=sum(counts),
                committed_tokens_per_request_per_cycle=describe(by_request),
                normalization='Per-device work divided by actual local committed output tokens; not request TPOT. PP has two alternating B4 slots, each target forward advances only one slot.')


def analyze_rank(arm_dir, rank, calibration):
    arm_dir = Path(arm_dir)
    receipt_path = arm_dir / f'activity_receipt_rank{rank}.json'
    receipt = read(receipt_path)
    if not receipt['passed']:
        raise ValueError('Failed activity receipt: ' + str(receipt['errors']))
    trace_path = arm_dir / f'activity_trace_rank{rank}.json'
    events = [e for e in read(trace_path)['traceEvents']
              if e.get('ph') == 'X' and e.get('cat') in GPU_CATEGORIES and 'ts' in e and 'dur' in e]
    devices = {arg(e, 'device') for e in events}
    if devices != {str(rank % 4)}:
        raise ValueError('Wrong/ambiguous rank-local GPU activity device: ' + str(devices))
    segments = marker_segments(receipt, events)
    kernels = sorted((e for e in events if e.get('cat') in ('kernel', 'cuda_kernel')),
                     key=lambda e: float(e['ts']))
    # CUDA-only Graph replays have no inner CPU scopes. Attribute fresh actual
    # kernel names or an entire independently validated same-rank sequence.
    for event in events:
        event['phase'] = ('marker' if event.get('name') == receipt['marker_kernel_name']
                          else calibration['exact_names'].get(event.get('name'), 'unclassified_gpu')
                          if event.get('cat') in ('kernel', 'cuda_kernel') else 'copy_or_set')
        event['phase_before_launch_fingerprint'] = event['phase']
    for segment in segments:
        local = [e for e in kernels if segment['gpu_start_us'] <= float(e['ts']) < segment['gpu_end_us']
                 and e['name'] != receipt['marker_kernel_name']]
        role = 'target' if segment['phase'].startswith('target_') else 'draft'
        assigned, matches = assign_phases(local, calibration, rank, role)
        before, _ = assign_phases(local, calibration, rank, role, use_launch_fingerprints=False)
        for event, phase, baseline in zip(local, assigned, before):
            event['phase'] = phase
            event['phase_before_launch_fingerprint'] = baseline
        segment['graph_sequence_matches'] = matches
        segment['activity'] = partition(events, segment['gpu_start_us'], segment['gpu_end_us'])
    targets = [s for s in segments if s['phase'].startswith('target_')]
    if len(targets) < 2:
        raise ValueError('No complete post-Extend target cycle')
    start, end = targets[0]['gpu_start_us'], targets[-1]['gpu_start_us']
    window = partition(events, start, end)
    baseline_window = partition([dict(e, phase=e['phase_before_launch_fingerprint']) for e in events], start, end)
    coverage = {}
    for label, field in (('before_launch_fingerprint', 'phase_before_launch_fingerprint'), ('after_launch_fingerprint', 'phase')):
        unknown_count, unknown_sum = 0, 0.
        for event in kernels:
            low, high = interval(event)
            clipped = max(0., min(end, high) - max(start, low))
            if clipped and event[field] == 'unclassified_gpu':
                unknown_count += 1
                unknown_sum += clipped / 1000
        coverage[label] = dict(unknown_kernel_count=unknown_count,
                               unknown_kernel_duration_sum_ms=unknown_sum,
                               exclusive_unknown_wall_ms=(baseline_window if label.startswith('before') else window)['ms']['unclassified_gpu'])
    cycles = len(targets) - 1
    window['per_local_microbatch_cycle_ms'] = {k: v / cycles for k, v in window['ms'].items()}
    actual = accepted_normalization(receipt, segments)
    window['per_committed_local_output_token_ms'] = ({k: v / actual['actual_committed_output_tokens']
                                                      for k, v in window['ms'].items()}
                                                     if actual['valid'] else None)
    cadence = [(b['gpu_start_us'] - a['gpu_start_us']) / 1000 for a, b in zip(targets, targets[1:])]
    slots = defaultdict(list)
    for s in targets:
        slots[tuple(sorted(s['request_ids']))].append(s['gpu_start_us'])
    same_slot = [(b-a)/1000 for values in slots.values() for a, b in zip(values, values[1:])]
    phases = defaultdict(list)
    for segment in segments:
        phases[segment['phase']].append(segment['activity'])
    per_phase = {phase: dict(count=len(rows), duration_ms=describe([r['duration_ms'] for r in rows]),
                            component_mean_ms={p: statistics.fmean(r['ms'][p] for r in rows) for p in PHASES})
                 for phase, rows in phases.items()}
    unknown = Counter()
    for event in events:
        if event['phase'] == 'unclassified_gpu':
            low, high = interval(event)
            unknown[event['name']] += max(0, min(end, high) - max(start, low)) / 1000
    return dict(rank=rank, arm=receipt['arm'], layout=receipt['layout'], mtp=receipt['mtp'],
                topology=receipt['topology'], steady_window=window,
                attribution_coverage=coverage,
                actual_output_normalization=actual, per_forward_phase=per_phase,
                observed_microbatch_start_cadence_ms=describe(cadence),
                observed_same_request_slot_start_cadence_ms=describe(same_slot),
                unique_request_slots=len(slots), segments=segments,
                unclassified_kernel_duration_sum_ms=dict(unknown.most_common()),
                sources=[provenance(receipt_path), provenance(trace_path)],
                warnings=['Profile cadence includes profiler/marker overhead; use unprofiled cohort for TTFT/TPOT/E2E.',
                          'Dispatch/combine/collective kernels can include wait/polling; their time is not wire-transfer time.',
                          'nccl_tp denotes calibrated GroupCoordinator collective API GPU activity (TP expected from this layout); per-call group IDs are not separately recorded.',
                          'Per-stage or per-rank maxima cannot be summed into request TPOT; clocks are host-local.',
                          'Generic kernels require either a complete same-rank Graph sequence match or a complete same-rank/role launch fingerprint agreed by both native scoped warmups; conflicts and partial fingerprints stay unknown.'])


def analyze_arm(arm_dir, ranks=range(8)):
    calibration = build_calibration(arm_dir)
    rows = [analyze_rank(arm_dir, rank, calibration) for rank in ranks]
    groups = []
    for group in (0, 1):
        local = [r for r in rows if r['rank'] // 4 == group]
        if not local:
            continue
        label = ('stage' if local[0]['layout'] == 'pp2ep4' else 'attention_dp') + str(group)
        groups.append(dict(group=label, ranks=[r['rank'] for r in local],
                           per_rank_cycle_ms={p: describe([r['steady_window']['per_local_microbatch_cycle_ms'][p]
                                                           for r in local]) for p in PHASES},
                           warning='Mean/min/max over rank-local measurements, not an additive critical path.'))
    return dict(schema_version=1, arm=Path(arm_dir).name, calibration=calibration,
                ranks=rows, rank_groups=groups,
                aggregate_policy='Do not sum stage/rank component maxima. Request metrics come only from separate unprofiled cohorts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm_dir', type=Path)
    parser.add_argument('--ranks', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze_arm(args.arm_dir, [int(x) for x in args.ranks.split(',')])
    output = args.output or args.arm_dir / 'stage_activity_summary.json'
    save(output, result)
    print(str(output))


if __name__ == '__main__':
    main()
