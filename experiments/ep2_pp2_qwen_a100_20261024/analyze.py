"""Attribute actual Kineto GPU activity through launch correlation IDs.

CPU annotation durations and GPU kernel durations are deliberately separate.
Unassigned kernels remain explicit; CPU waits are not relabelled as network.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def union_us(intervals):
    end = None
    total = 0.
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi-lo
        else:
            total += max(0., hi-end)
        end = hi if end is None else max(end, hi)
    return total


def enclosing(events, timestamp, tid=None):
    candidates = [e for e in events if e['ts'] <= timestamp < e['ts']+e['dur']
                  and (tid is None or e.get('tid') == tid)]
    return min(candidates, key=lambda e:e['dur']) if candidates else None


def trace_summary(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    events = [e for e in data['traceEvents'] if e.get('ph') == 'X' and e.get('dur',0) > 0]
    # Kineto also emits mirrored GPU user annotations with the SAME names.
    # Those are not extra model invocations or CPU launch scopes. Counting
    # them doubles the layer denominator and halves per-layer GPU means.
    annotations = [e for e in events if e.get('cat') == 'user_annotation']
    scopes = [e for e in annotations if e.get('name','').startswith('component.')]
    layers = [e for e in annotations if e.get('name','').startswith('layer.')]
    steps = [e for e in annotations if e.get('name','').startswith('step[')]
    runtime = {}
    for e in events:
        if e.get('cat') in ('cuda_runtime','cuda_driver'):
            corr = e.get('args',{}).get('correlation')
            if corr is not None:
                runtime[corr] = e
    by_external = {e.get('args',{}).get('External id'):e for e in events
                   if e.get('cat') in ('cpu_op','user_annotation')
                   and e.get('args',{}).get('External id') is not None}
    # Each individual layer invocation remains inspectable, including changing
    # microbatch sizes and profile edges. Do not merge unlike invocations.
    invocations = {id(e):dict(layer=int(e['name'].split('.')[1]),
        cpu_start_us=e['ts'], cpu_scope_ms=e['dur']/1000,
        step=(enclosing(steps,e['ts'],e.get('tid')) or {}).get('name','unknown'),
        component_intervals=defaultdict(list), cpu_components=defaultdict(float),
        kernel_names=Counter()) for e in layers}
    for e in scopes:
        parent=enclosing(layers,e['ts'],e.get('tid'))
        if parent:
            label='.'.join(e['name'].split('.')[1:-1])
            invocations[id(parent)]['cpu_components'][label]+=e['dur']/1000
    all_gpu = []
    classified = defaultdict(list)
    unmatched = Counter()
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):
            continue
        args=e.get('args',{})
        launch=runtime.get(args.get('correlation'))
        if launch is None:
            launch=by_external.get(args.get('External id'))
        label='unassigned'
        layer=None
        if launch:
            selected=enclosing(scopes,launch['ts'],launch.get('tid'))
            layer=enclosing(layers,launch['ts'],launch.get('tid'))
            if selected:
                label='.'.join(selected['name'].split('.')[1:-1])
        if label=='unassigned' and 'nccl' in e.get('name','').lower():
            label='communication.nccl_outside_component'
        interval=(e['ts'],e['ts']+e['dur'])
        all_gpu.append(interval)
        classified[label].append(interval)
        if label=='unassigned':unmatched[e.get('name','')]+=1
        if layer:
            row=invocations[id(layer)]
            row['component_intervals'][label].append(interval)
            row['kernel_names'][label]+=1
    rows=[]
    for row in invocations.values():
        intervals=row.pop('component_intervals')
        row['gpu_component_union_ms']={k:union_us(v)/1000 for k,v in intervals.items()}
        row['gpu_active_union_ms']=union_us([x for v in intervals.values() for x in v])/1000
        row['kernel_counts']=dict(row.pop('kernel_names'))
        row['cpu_component_scope_ms']=dict(row.pop('cpu_components'))
        rows.append(row)
    means=[]
    for layer,step in sorted({(r['layer'],r['step']) for r in rows}):
        selected=[r for r in rows if r['layer']==layer and r['step']==step]
        phases={k for r in selected for k in r['gpu_component_union_ms']}
        cpu_phases={k for r in selected for k in r['cpu_component_scope_ms']}
        means.append(dict(layer=layer,step=step,invocations=len(selected),
            cpu_scope_mean_ms=statistics.mean(r['cpu_scope_ms'] for r in selected),
            gpu_active_mean_ms=statistics.mean(r['gpu_active_union_ms'] for r in selected),
            **{k:statistics.mean(r['gpu_component_union_ms'].get(k,0) for r in selected) for k in phases},
            **{'cpu.'+k:statistics.mean(r['cpu_component_scope_ms'].get(k,0) for r in selected) for k in cpu_phases}))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method='GPU correlation -> actual CPU launch -> tightest enclosing component annotation',
        units='ms/rank/layer invocation; GPU unions can overlap between components',
        total_gpu_active_union_ms=union_us(all_gpu)/1000,
        component_gpu_union_ms={k:union_us(v)/1000 for k,v in classified.items()},
        unassigned_kernel_names=dict(unmatched),layer_means=means,invocations=rows,
        trace_window_us=[min(x[0] for x in all_gpu),max(x[1] for x in all_gpu)] if all_gpu else None)


def main(run):
    output=run/'analysis'
    output.mkdir(exist_ok=True)
    summary=[]
    csv_rows=[]
    for arm in ['ep2','pp2']:
        for phase in ['prefill','decode']:
            for path in sorted((run/arm/'traces'/phase).glob('*.trace.json.gz')):
                row=trace_summary(path)
                row.update(arm=arm,workload=phase)
                target=output/(path.name.replace('.trace.json.gz','')+'.json')
                target.write_text(json.dumps(row,indent=2)+'\n')
                summary.append({k:v for k,v in row.items() if k not in ('invocations','layer_means')})
                csv_rows.extend(dict(arm=arm,workload=phase,trace=path.name,**r) for r in row['layer_means'])
    (output/'trace_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if csv_rows:
        with (output/'per_layer.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=sorted({k for r in csv_rows for k in r}))
            writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(dict(traces=len(summary),layer_rows=len(csv_rows),output=str(output))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    main(ap.parse_args().run)
