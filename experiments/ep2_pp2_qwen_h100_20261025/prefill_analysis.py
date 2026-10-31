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
    full = 12 if arm == 'ep2' else 6
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
    outside = Counter()
    for e in kernels:
        args = e.get('args', {})
        owner, step = correlation.get(args.get('correlation'), external.get(args.get('External id'), ('','')))
        if args.get('graph id', 0) or 'DECODE' in step:
            # Native EP overlap may run an extra Decode before consuming the
            # completion result of an output-one Prefill. Do not misclassify it
            # as Prefill compute; preserve it as a separate diagnostic cost.
            assert not owner.startswith('component.'), (e['name'], owner)
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
                 classification_unit='sum of GPU kernel durations for entire single Prefill cohort per rank')
    return row, proof


def analyze(run):
    sys.path.insert(0, '/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
    from artifact_visibility import wait_for_stable_json
    complete, _ = wait_for_stable_json(run/'completed.json',timeout_s=120)
    cleanup, _ = wait_for_stable_json(run/'worker_cleanup.json',timeout_s=120)
    assert complete['passed'] and complete['profile_only'] and cleanup['cleanup_verified']
    post = [(p,json.loads(p.read_text())) for p in (run/'dashboard_guard/checks').glob('*-post.json')]
    post = [(p,d) for p,d in post if d.get('passed') and d.get('scientific_success') and d.get('cleanup_verified')]
    assert post, 'Guard postcheck has not passed'
    out = run/'analysis'; out.mkdir(exist_ok=True)
    rows, proofs, export_reconciliation = [], [], []
    for manifest_path in sorted(run.glob('*/traces/*/manifest.json')):
        manifest, _ = wait_for_stable_json(manifest_path,timeout_s=60)
        assert manifest['passed'] and not manifest['performance_sample'] and manifest['phase']=='prefill'
        arm = manifest_path.parents[2].name
        reqs = manifest['profiled_cohort']['requests']
        assert len(reqs)==manifest['requests']
        assert all(r['output_tokens']==1 for r in reqs)
        assert all(r['prompt_tokens']==manifest['cached_context_tokens']+manifest['new_tokens_per_request'] for r in reqs)
        assert all((r['cached_tokens'] or 0) >= manifest['cached_context_tokens']-64 for r in reqs)
        for item in manifest['files']:
            path = Path(item['path'])
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != item['sha256']:
                # The original snapshot caught one rank's just-created empty
                # export. Preserve that manifest; accept only this explicitly
                # proven empty-file race after verified producer shutdown.
                assert item['sha256'] == hashlib.sha256(b'').hexdigest()
                assert manifest_path.stat().st_mtime < path.stat().st_mtime < (run/'worker_cleanup.json').stat().st_mtime
                export_reconciliation.append(dict(path=str(path), manifest=str(manifest_path),
                    original_empty_file_sha256=item['sha256'], completed_export_sha256=observed,
                    bytes=path.stat().st_size, manifest_mtime=manifest_path.stat().st_mtime,
                    completed_export_mtime=path.stat().st_mtime,
                    reason='Rank export completed after an empty filename became visible; original evidence unchanged'))
            row, proof = inspect_trace(path, manifest, arm)
            assert proof['sha256']==observed==hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row);proofs.append(proof)
    assert len(rows)==14
    fields = [*COMPONENTS,'kernel_sum_ms','attention_ms','expert_compute_ms','deepep_ms',
              'incidental_decode_kernel_ms','incidental_decode_steps','outside_model_kernel_ms']
    groups = []
    for case in ('moderate','near_capacity_b8','near_capacity_b16','pp_own_capacity_b16'):
        for arm,stage in [('ep2',''),('pp2',0),('pp2',1)]:
            selected = [r for r in rows if (r['case'],r['arm'],r['stage'])==(case,arm,stage)]
            if not selected: continue
            row = {k:selected[0][k] for k in ('case','arm','stage','requests','cached_tokens_per_request','fresh_tokens_per_request')}
            row['ranks']=len(selected)
            for f in fields: row[f]=statistics.mean(r[f] for r in selected)
            groups.append(row)
    for name,data in [('per_rank.csv',rows),('comparison.csv',groups)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'validation.json').write_text(json.dumps(dict(passed=True,unit='ms/kernel cumulative per full Prefill cohort per rank',
        guard_postcheck=str(post[-1][0]),trace_rows=len(rows),traces=proofs,
        export_reconciliation=export_reconciliation),indent=2)+'\n')
    print(json.dumps(dict(passed=True,rows=groups),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run.resolve())
