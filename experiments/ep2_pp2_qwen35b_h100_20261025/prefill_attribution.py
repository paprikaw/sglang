"""Attribute eager Prefill GPU kernels by their actual CPU launch scopes.

Kernel cumulative times are not exclusive critical-path shares. Preserve PP
NCCL residence separately; it can include waiting and profiler control work.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

COMPONENTS = ('attention_full_ms', 'attention_linear_ms', 'routed_expert_ms',
              'shared_expert_ms', 'dispatch_ms', 'combine_ms', 'deepep_other_ms',
              'other_kernel_ms', 'nccl_residence_ms')


def union_ms(intervals):
    end = -float('inf')
    total = 0.
    for a, b in sorted(intervals):
        total += max(0., b - max(a, end))
        end = max(end, b)
    return total / 1000


def owners(events):
    scopes = defaultdict(list)
    queries = defaultdict(list)
    for e in events:
        if e.get('ph') != 'X' or 'ts' not in e:
            continue
        thread = (e.get('pid'), e.get('tid'))
        name, cat = e.get('name', ''), e.get('cat', '')
        if cat == 'user_annotation' and (name.startswith('component.') or name.startswith('step[')):
            scopes[thread].append(e)
        if cat in ('cuda_runtime', 'cuda_driver', 'cpu_op', 'user_annotation'):
            queries[thread].append(e)
    by_correlation, by_external = {}, {}
    for thread, rows in queries.items():
        ranges = sorted(scopes[thread], key=lambda e:e['ts'])
        active, pos = [], 0
        for e in sorted(rows, key=lambda e:e['ts']):
            ts = e['ts']
            while pos < len(ranges) and ranges[pos]['ts'] <= ts + .001:
                active.append(ranges[pos]); pos += 1
            active = [s for s in active if s['ts'] + s.get('dur',0) >= ts - .001]
            components = [s for s in active if s['name'].startswith('component.')]
            owner = min(components, key=lambda s:s['dur'])['name'] if components else ''
            step = next((s['name'] for s in reversed(active) if s['name'].startswith('step[')), '')
            item = (owner, step)
            args = e.get('args', {})
            if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in args:
                by_correlation[args['correlation']] = item
            if 'External id' in args and e.get('cat') in ('cpu_op', 'user_annotation'):
                by_external[args['External id']] = item
    return by_correlation, by_external


def bucket(name, owner):
    if name.startswith('ncclDevKernel'):
        return 'nccl_residence_ms'
    if 'deep_ep::' in name:
        return 'dispatch_ms' if 'dispatch' in name else 'combine_ms' if 'combine' in name else 'deepep_other_ms'
    for prefix, label in [('component.attention.full.', 'attention_full_ms'),
                          ('component.attention.linear.', 'attention_linear_ms'),
                          ('component.expert.routed.', 'routed_expert_ms'),
                          ('component.expert.shared.', 'shared_expert_ms')]:
        if owner.startswith(prefix):
            return label
    return 'other_kernel_ms'


def is_decode_kernel(args, step):
    return bool(args.get('graph id', 0)) or 'DECODE' in step


def inspect_trace(path, manifest, arm):
    raw = path.read_bytes()
    events = json.loads(gzip.decompress(raw))['traceEvents']
    correlation, external = owners(events)
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    assert kernels
    all_steps = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name','').startswith('step[')]
    steps = [e for e in all_steps if 'EXTEND' in e['name']]
    assert steps and all(any(mode in e['name'] for mode in ('EXTEND','DECODE','IDLE'))
                         for e in all_steps), (path,[e['name'] for e in all_steps])
    fresh = sum(int(re.search(r'c_sq=(\d+)', e['name']).group(1)) for e in steps)
    expected = manifest['requests'] * manifest['new_tokens_per_request'] // (2 if arm == 'ep2' else 1)
    assert fresh == expected, (path, fresh, expected)
    counts = Counter(e['name'].rsplit('.',1)[0] for e in events
                     if e.get('cat') == 'user_annotation' and e.get('name','').startswith('component.'))
    full = 10 if arm == 'ep2' else 5
    # Empty/idle padded steps are retained in step records but have no layer
    # execution. Count actual full/linear invocations rather than guessing from
    # request concurrency or total trace duration.
    assert counts['component.attention.full'] > 0
    assert counts['component.attention.linear'] == 3 * counts['component.attention.full']
    assert counts['component.attention.full'] % full == 0
    costs = Counter({k:0. for k in COMPONENTS})
    inventory = defaultdict(lambda:Counter())
    intervals = defaultdict(list)
    missing = Counter()
    incidental = Counter()
    incidental_owners = Counter()
    outside = Counter()
    for e in kernels:
        args = e.get('args', {})
        owner, step = correlation.get(args.get('correlation'), external.get(args.get('External id'), ('','')))
        if is_decode_kernel(args, step):
            # Native EP overlap may run an extra Decode before consuming the
            # completion result of an output-one Prefill. Do not misclassify it
            # as Prefill compute; preserve it as a separate diagnostic cost.
            # A replay launch can sit under a surrounding model/layer scope.
            # Graph identity and the actual Decode step take precedence over
            # that CPU wrapper; never reclassify these kernels as Prefill.
            incidental_owners[owner] += 1
            incidental[e['name']] += e['dur'] / 1000
            continue
        if not owner and 'EXTEND' not in step and not e['name'].startswith('ncclDevKernel'):
            outside[e['name']] += e['dur'] / 1000
            continue
        label = bucket(e['name'], owner)
        costs[label] += e['dur'] / 1000
        inventory[label][e['name']] += e['dur'] / 1000
        intervals[label].append((e['ts'],e['ts']+e['dur']))
        if not owner:
            missing[e['name']] += 1
    assert costs['attention_full_ms'] > 0 and costs['attention_linear_ms'] > 0 and costs['routed_expert_ms'] > 0
    if arm == 'ep2':
        assert costs['dispatch_ms'] > 0 and costs['combine_ms'] > 0
    else:
        assert sum(costs[k] for k in ('dispatch_ms','combine_ms','deepep_other_ms')) == 0
    stage = '' if arm == 'ep2' else int(path.name.split('-PP-')[1].split('.')[0])
    row = dict(case=path.parent.name, arm=arm, stage=stage, rank_trace=path.name,
               requests=manifest['requests'], cached_tokens_per_request=manifest['cached_context_tokens'],
               fresh_tokens_per_request=manifest['new_tokens_per_request'], fresh_tokens_on_rank=fresh,
               actual_model_forwards=counts['component.attention.full']//full,
               kernel_sum_ms=sum(costs.values()), incidental_decode_kernel_ms=sum(incidental.values()),
               incidental_decode_steps=sum('DECODE' in e['name'] for e in all_steps),
               outside_model_kernel_ms=sum(outside.values()), **costs)
    row['attention_ms'] = costs['attention_full_ms'] + costs['attention_linear_ms']
    row['expert_compute_ms'] = costs['routed_expert_ms'] + costs['shared_expert_ms']
    row['deepep_ms'] = sum(costs[k] for k in ('dispatch_ms','combine_ms','deepep_other_ms'))
    proof = dict(trace=str(path), sha256=hashlib.sha256(raw).hexdigest(), steps=[e['name'] for e in steps],
                 scope_counts=dict(counts), kernel_categories={k:dict(v) for k,v in inventory.items()},
                 unscoped_kernels=dict(missing), active_union_ms={k:union_ms(v) for k,v in intervals.items()},
                 incidental_decode_steps=[e['name'] for e in all_steps if 'DECODE' in e['name']],
                 incidental_decode_kernel_ms=dict(incidental), outside_model_kernel_ms=dict(outside),
                 incidental_decode_cpu_owners=dict(incidental_owners),
                 classification_unit='sum of GPU kernel durations for entire single Prefill cohort per rank')
    return row, proof
