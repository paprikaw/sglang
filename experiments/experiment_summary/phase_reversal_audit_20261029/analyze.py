"""Audit existing, accepted EP/PP cohorts. CPU only; no arithmetic means."""
from pathlib import Path
import csv
import hashlib
import json

W = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
M = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
OUT = Path(__file__).resolve().parent
sources = {}
rows = []

def record(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, path
    sources[str(path)] = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
    return raw

def read(path, expected=None):
    return json.loads(record(path, expected))

def add(path, *, series, model, phase, workload, layout, policy, expected=None):
    d = read(path, expected)
    assert d['valid'] and not d['profiler_enabled'], path
    for idx, r in enumerate(d['rounds']):
        qs = r['requests']
        b = len(qs)
        elapsed = max(q['finished'] for q in qs) - min(q['started'] for q in qs)
        assert abs(elapsed-r['makespan_s']) < 1e-5, path
        outputs = {q['output_tokens'] for q in qs}
        cached = {q['cached_tokens'] for q in qs}
        assert len(outputs) == len(cached) == 1, path
        o = next(iter(outputs)); p = next(iter(cached))
        q = qs[0]['prompt_tokens'] - p
        assert all(x['prompt_tokens']-p == q for x in qs), path
        row = dict(series=series, model=model, phase=phase, workload=workload,
                   layout=layout, policy=policy, round=idx, B=b, cached_P=p,
                   new_Q=q, output_O=o, makespan_ms=elapsed*1000,
                   throughput_tokens_s=(b*q if phase == 'prefill' else b*o)/elapsed,
                   longest_decode_tail_ms=None, request0_decode_tail_ms=None,
                   longest_request_ms=max(x['e2e_s'] for x in qs)*1000,
                   source=str(path), source_sha256=sources[str(path)]['sha256'])
        if phase == 'decode':
            assert o == 128 and q == 1, path
            if series != '122b_historical_1025':
                assert p == 4096, path
            tails = []
            for x in qs:
                ev = x['events']
                assert [e['completion_tokens'] for e in ev] == list(range(1,129)), path
                tail = (ev[-1]['time_s']-ev[0]['time_s'])*1000
                assert abs(tail-(x['e2e_s']-x['ttft_s'])*1000) < 1e-4
                tails.append(tail)
            row['longest_decode_tail_ms'] = max(tails)
            row['request0_decode_tail_ms'] = tails[0]
        rows.append(row)

# Prefill: all accepted main policies, common 8192-token budget.
root = W/'experiments/h100_ep_prefill_overlap_20261027'
d = read(root/'report/results.json')
expected = {x['path']: x['sha256'] for x in d['source_descriptors']}
for m in d['metrics']:
    p = Path(d['run'])/m['arm']/'workloads'/m['workload']/f"metrics.prefill.p{m['policy']}.json"
    add(p, series='35b_prefill_8192', model='35B BF16', phase='prefill',
        workload=m['workload'], layout=m['arm'], policy=str(m['policy']), expected=expected[str(p)])

# Decode: fastest native EP family versus PP lookahead; retain all measured arms.
for dirname, series, model in [
    ('h100_ep_decode_overlap_20261027','35b_decode_latest','35B BF16'),
    ('h100_qwen122b_latest_decode_20261027','122b_decode_latest','122B FP8'),
]:
    d = read(W/'experiments'/dirname/'report/results.json')
    for m in d['metrics']:
        add(m['source'], series=series, model=model, phase='decode',
            workload=m['workload'], layout=m['arm'], policy=m['variant'], expected=m['source_sha256'])

# Independent earlier, accepted lookahead rerun, kept as a separate series.
root = W/'experiments/h100_pp_lookahead_workloads_20261027/runs/lookahead_r2'
for work in ['b8_repeated','b16_repeated','b32_repeated','b32_diverse']:
    for arm, policy in [('ep2',0),('pp2',9)]:
        p = root/arm/'workloads'/work/f'metrics.decode.p{policy}.json'
        add(p, series='35b_decode_lookahead_r2', model='35B BF16', phase='decode',
            workload=work, layout=arm, policy=str(policy))

# Newer 122B PP refinement and its fresh EP reference.
root = W/'experiments/h100_qwen122b_pp_refine_20261027'
d = read(root/'report/results.json')
for m in d['metrics']:
    arm = 'ep2' if m['run'].startswith('ep_') else 'pp2'
    base = root/'runs'/m['run']/arm/'workloads'/m['workload']
    path = base/'metrics.decode.json' if arm == 'ep2' else base/f"p{m['policy']}"/'metrics.decode.json'
    add(path, series='122b_decode_refinement', model='122B FP8', phase='decode',
        workload=m['workload'], layout=arm, policy=f"{m['run']}:p{m['policy']}")

# Historical 122B P/D example: not a latest-versus-latest crossover.
root = W/'experiments/ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7'
for arm in ['ep2','pp2']:
    for work in ['moderate','near_capacity_b8','near_capacity_b16']:
        for phase in ['prefill','decode']:
            add(root/arm/f'metrics.{work}.{phase}.json', series='122b_historical_1025',
                model='122B FP8', phase=phase, workload=work, layout=arm, policy='historical')

# A100 is a separate eager/native-backend cohort, never merged with H100.
root = W/'experiments/a100_pp_ep_regression_20261027/runs/a10035_nvlink_r4'
for arm in ['ep2','pp2']:
    for path in sorted((root/arm/'workloads').glob('*/metrics.*.json')):
        if path.name.startswith('metrics.prefill'):
            phase='prefill'
        elif path.name.startswith('metrics.decode'):
            phase='decode'
        else:
            continue
        policy=path.stem.rsplit('.p',1)[-1] if '.p' in path.stem else '0'
        add(path, series='35b_a100_eager',model='35B BF16 A100',phase=phase,
            workload=path.parent.name,layout=arm,policy=policy)

# Four-GPU runs split into two simultaneous two-GPU groups: retain their boundary.
p = W/'experiments/h100_five_layouts_hicache_20261028/two_gpu_three_layouts_analysis/per_group_rounds.csv'
group_rows = list(csv.DictReader(record(p).decode().splitlines()))
for r in group_rows:
    record(r['source'], r['source_sha256'])

def write_csv(name, data):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)

write_csv('formal_rounds.csv', rows)
write_csv('hicache_group_rounds.csv', group_rows)

grouped = {}
for r in rows:
    key = tuple(r[k] for k in ['series','phase','workload','layout','policy'])
    grouped.setdefault(key, []).append(r)
summaries=[]
for key, rs in grouped.items():
    out=dict(zip(['series','phase','workload','layout','policy'],key),n=len(rs))
    for metric in ['makespan_ms','longest_decode_tail_ms','request0_decode_tail_ms','throughput_tokens_s']:
        vals=[r[metric] for r in rs if r[metric] is not None]
        out[metric+'_first']=vals[0] if vals else None
        out[metric+'_min']=min(vals) if vals else None
        out[metric+'_max']=max(vals) if vals else None
    summaries.append(out)
write_csv('ranges.csv', summaries)
(OUT/'sources.json').write_text(json.dumps({'no_new_gpu_measurements':True,
    'aggregation':'fixed first formal round and observed ranges; no arithmetic means',
    'decode_tail_definition':'max over requests of SSE time(output128)-SSE time(output1); 127 output intervals, excludes TTFT',
    'sources':list(sources.values())},ensure_ascii=False,indent=2)+'\n')

for s in summaries:
    if s['series'] in ['35b_decode_lookahead_r2','35b_a100_eager']: continue
    if s['series']=='35b_prefill_8192' and s['policy'] not in ['0','3']:continue
    if s['series'].endswith('decode_latest') and s['layout']=='ep2' and s['policy']!='native':continue
    print(s['series'],s['workload'],s['phase'],s['layout'],s['policy'],
          'cohort_ms',round(s['makespan_ms_min'],2),round(s['makespan_ms_max'],2),
          'decode_tail_ms',*(round(s['longest_decode_tail_ms_'+x],2) if s['longest_decode_tail_ms_'+x] is not None else None for x in ['min','max']))
print('VALIDATED',len(rows),'formal cohorts;',len(group_rows),'historical group rounds;',len(sources),'source hashes')
