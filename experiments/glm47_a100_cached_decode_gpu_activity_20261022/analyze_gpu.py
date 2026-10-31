"""Calibrated CUDA-only device timeline, bounded by real GPU marker kernels."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'glm47_a100_cached_decode_stage_profile_20261022/analysis'))
from kineto_summary import arg,interval,union_length,file_receipt


def read(path):return json.loads(path.read_text())
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def group(phase):
    if phase=='attention':return 'attention'
    if phase=='expert_ffn':return 'expert_ffn'
    if phase.startswith('deep_ep.') or phase in ('tp_communication','pp_communication'):return 'communication'
    return phase


def partition(gpu,window):
    changes=defaultdict(Counter);phase_intervals=defaultdict(list);counts=Counter()
    for e in gpu:
        low=max(window[0],e['interval'][0]);high=min(window[1],e['interval'][1])
        if high<=low:continue
        phase=e['phase'];label=group(phase)
        changes[low][label]+=1;changes[high][label]-=1
        phase_intervals[phase].append((low,high));counts[phase]+=1
    changes[window[0]];changes[window[1]]
    active=Counter();totals=defaultdict(float);last=window[0]
    for point in sorted(changes):
        labels=[key for key,count in active.items() if count>0]
        label=labels[0] if len(labels)==1 else 'overlap' if labels else 'no_traced_gpu_activity'
        totals[label]+=point-last;active.update(changes[point]);last=point
    length=window[1]-window[0]
    if abs(sum(totals.values())-length)>1e-4:raise ValueError('Timeline partition does not close')
    return dict(window_ms=length/1000,component_ms={k:v/1000 for k,v in totals.items()},
        phase_union_ms={k:union_length(v)/1000 for k,v in phase_intervals.items()},event_counts=dict(counts))


def align_markers(markers,models,tolerance_ms=.5):
    """Check actual device intervals; allow one incomplete trace-edge pair only.

    CUPTI can omit a marker at collection startup. No interior marker may be
    guessed or interpolated. Independent event durations AND relative starts
    must match every complete pair, and the solution must be unique.
    """
    candidates=[]
    for model_skip in (0,1):
        for marker_skip in (0,1):
            count=min(len(models)-model_skip,(len(markers)-marker_skip)//2)
            model_tail=len(models)-model_skip-count
            marker_tail=len(markers)-marker_skip-2*count
            if count<2 or model_tail>1 or marker_tail>1:continue
            if model_skip+model_tail>1 or marker_skip+marker_tail>1:continue
            pairs=[];errors=[];offsets=[]
            for i in range(count):
                event=models[model_skip+i];low,high=markers[marker_skip+2*i:marker_skip+2*i+2]
                envelope=(interval(high)[1]-interval(low)[0])/1000
                errors.append(abs(envelope-event['duration_ms']))
                offsets.append(interval(low)[0]/1000-event['start_ms'])
                pairs.append((event,low,high))
            spread=max(offsets)-min(offsets)
            if max(errors)<=tolerance_ms and spread<=tolerance_ms:
                candidates.append(dict(pairs=pairs,matched=count,max_duration_error_ms=max(errors),
                    relative_start_offset_spread_ms=spread,omitted_model_edges=[model_skip,model_tail],
                    omitted_marker_edges=[marker_skip,marker_tail]))
    if not candidates:raise ValueError('No independently validated GPU-marker/event alignment')
    longest=max(c['matched'] for c in candidates);best=[c for c in candidates if c['matched']==longest]
    if len(best)!=1:raise ValueError('Ambiguous GPU-marker/event alignment')
    return best[0]


def correlate_markers(markers,models,runtime,sample,trace_base_ns):
    """Actual CUDA correlation IDs bind markers to their native host forward.

    The wall/perf anchor is within one process only. It never aligns independent
    rank GPU clocks. GPU timestamps define measured windows after this join.
    """
    calls=defaultdict(list)
    for row in runtime:
        key=arg(row,'correlation','correlation id')
        if key is not None:calls[key].append(row)
    def cpu_ts(ns):return (sample['wall_origin_ns']+(ns-sample['host_origin_ns'])-trace_base_ns)/1000
    assigned=defaultdict(list)
    for marker in markers:
        rows=calls.get(arg(marker,'correlation','correlation id'),[])
        if len(rows)!=1:raise ValueError('Marker lacks unique actual CUDA launch correlation')
        launch=interval(rows[0])[0]
        matches=[e for e in models if cpu_ts(e['cpu_start_ns'])-10<=launch<=cpu_ts(e['cpu_end_ns'])+10]
        if len(matches)!=1:raise ValueError('Marker launch is not in one actual forward CPU interval')
        assigned[matches[0]['forward_index']].append(marker)
    pairs=[];edges=[];errors=[]
    for i,event in enumerate(models):
        found=sorted(assigned[event['forward_index']],key=lambda m:interval(m)[0])
        if len(found)!=2:
            if i not in (0,len(models)-1) or len(found)!=1:
                raise ValueError('Interior forward has incomplete GPU markers')
            edges.append(event['forward_index']);continue
        low,high=found
        envelope=(interval(high)[1]-interval(low)[0])/1000
        error=abs(envelope-event['duration_ms']);errors.append(error)
        if error>max(5,event['duration_ms']*.03):raise ValueError('Correlated marker/event duration grossly disagrees')
        pairs.append((event,low,high))
    if len(edges)>1:raise ValueError('More than one incomplete collection-edge forward')
    return dict(pairs=pairs,matched=len(pairs),omitted_edge_forward_ids=edges,
        max_duration_error_ms=max(errors),method='GPU correlation ID -> unique actual CUDA launch -> same-process recorded forward CPU bounds; GPU clocks remain independent')


def rank_summary(run,arm,rank,calibration):
    path=run/arm/f'attribution_rank{rank}.json';sample=read(path)
    if not sample['passed'] or sample.get('profiling_method')!='cuda_activity_exact_name_calibration':
        raise ValueError('Invalid CUDA-only receipt')
    marker=read(run/arm/f'marker_signature_rank{rank}.json')
    if marker['kernel_name'] in calibration['all_observed_kernel_names']:
        raise ValueError('Marker was also present in unmodified model execution')
    trace_path=run/arm/sample['trace_filename'];trace=read(trace_path)
    expected_device=str(sample['cuda_device']);gpu=[];markers=[];unknown_names=Counter()
    for e in trace['traceEvents']:
        if e.get('cat') not in ('kernel','cuda_kernel','gpu_memcpy','gpu_memset') or e.get('ph')!='X':continue
        device=arg(e,'device','device id')
        if device is not None and device!=expected_device:raise ValueError('Unexpected GPU device')
        if e['name']==marker['kernel_name']:
            markers.append(e);continue
        phase=calibration['exact_names'].get(e['name'],'unclassified_gpu') if e['cat'] in ('kernel','cuda_kernel') else 'copy_or_set'
        if phase=='unclassified_gpu':unknown_names[e['name']]+=1
        gpu.append(dict(interval=interval(e),phase=phase,name=e['name']))
    markers.sort(key=lambda e:interval(e)[0])
    models=sorted([e for e in sample['events'] if e['category']=='model_forward'],key=lambda e:e['cpu_start_ns'])
    if len({arg(m,'stream') for m in markers})!=1:raise ValueError('Markers crossed compute streams')
    alignment=correlate_markers(markers,models,
        [e for e in trace['traceEvents'] if e.get('cat')=='cuda_runtime' and e.get('ph')=='X'],
        sample,int(trace['baseTimeNanoseconds']))
    meta={f['forward_index']:f for f in sample['forwards']};forwards=[];marker_error=[]
    for event,low,high in alignment['pairs']:
        envelope=(interval(high)[1]-interval(low)[0])/1000
        error=abs(envelope-event['duration_ms']);marker_error.append(error)
        receipt=meta[event['forward_index']]
        if not receipt['detailed']:continue
        if error>.5:raise ValueError(f'Selected GPU marker / event duration mismatch: {error} ms')
        if receipt['metadata']['batch_size']!=4 or receipt['metadata']['forward_mode']!='2':raise ValueError('Not B4 Decode')
        window=(interval(low)[1],interval(high)[0])
        forwards.append(dict(forward_index=event['forward_index'],decode_ordinal=receipt['extend_index'],
            metadata=receipt['metadata'],gpu_start_us=window[0],gpu_end_us=window[1],
            cuda_event_forward_ms=event['duration_ms'],**partition(gpu,window)))
    if len(forwards)!=len(sample['detailed_forward_ordinals']):raise ValueError('Incomplete selected Decode steps')
    # Exact adjacent microbatch cycles include inter-forward PP boundary work.
    first=forwards[0]['gpu_start_us'];last=forwards[-1]['gpu_start_us']
    cycles=len(forwards)-1;steady=partition(gpu,(first,last))
    return dict(arm=arm,rank=rank,hostname=sample['hostname'],forwards=forwards,
        steady_cycle=dict(cycle_count=cycles,mean_cycle_ms=steady['window_ms']/cycles,
            mean_component_ms={k:v/cycles for k,v in steady['component_ms'].items()},
            component_percent={k:100*v/steady['window_ms'] for k,v in steady['component_ms'].items()},
            mean_phase_union_ms={k:v/cycles for k,v in steady['phase_union_ms'].items()}),
        mean_forward_gpu_window_ms=statistics.mean(f['window_ms'] for f in forwards),
        max_marker_event_difference_ms=max(marker_error),marker_kernel_count=len(markers),
        expected_forward_markers=2*len(models),post_forward_marker_count=len(markers)-2*len(models),
        marker_alignment={k:v for k,v in alignment.items() if k!='pairs'},
        unclassified_kernel_names=dict(unknown_names),
        source=dict(receipt=file_receipt(path),trace=file_receipt(trace_path),marker=file_receipt(run/arm/f'marker_signature_rank{rank}.json')))


def aggregate(rows):
    keys=sorted({k for row in rows for k in row['steady_cycle']['mean_component_ms']})
    phases=sorted({k for row in rows for k in row['steady_cycle']['mean_phase_union_ms']})
    cycle=statistics.mean(row['steady_cycle']['mean_cycle_ms'] for row in rows)
    ms={k:statistics.mean(row['steady_cycle']['mean_component_ms'].get(k,0) for row in rows) for k in keys}
    return dict(ranks=[row['rank'] for row in rows],mean_microbatch_cycle_ms=cycle,
        mean_component_ms=ms,component_percent={k:v/cycle*100 for k,v in ms.items()},
        mean_phase_union_ms={k:statistics.mean(row['steady_cycle']['mean_phase_union_ms'].get(k,0) for row in rows) for k in phases},
        mean_forward_gpu_window_ms=statistics.mean(row['mean_forward_gpu_window_ms'] for row in rows),
        max_marker_event_difference_ms=max(row['max_marker_event_difference_ms'] for row in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=('ep8','pp2ep4'));a=p.parse_args()
    calibration=read(ROOT/'calibration.json');groups={};perturbation={}
    for arm in ((a.arm,) if a.arm else ('ep8','pp2ep4')):
        rows=[rank_summary(a.run,arm,rank,calibration['by_arm'][arm]) for rank in range(8)]
        for row in rows:save(a.out/f'{arm}_rank{row["rank"]}.json',row)
        groups[arm]={(f'node{n}' if arm=='ep8' else f'stage{n}'):aggregate([r for r in rows if r['rank']//4==n]) for n in range(2)}
        groups[arm]['all_rank_mean']=aggregate(rows)
        profile=read(a.run/arm/'profile_result.json')
        perturbation[arm]=dict(control_tpot_ms=profile['control']['tpot_s']['mean']*1000,
            gpu_profile_tpot_ms=profile['tpot_s']['mean']*1000,whole_cohort_ratio=profile['profile_to_control_tpot_ratio'])
    result=dict(schema_version=1,groups=groups,perturbation=perturbation,
        denominator='Independent per-device steady microbatch GPU timeline, including GPU idle and overlap; adjacent actual GPU marker starts. Not a decomposition of HTTP TPOT.',
        method='CUDA activities only, exact-name CPU-scope calibration with held-out checks. No inner phase timers or CPU operator trace. Unknown kernels, idle, overlap and copies remain explicit.',
        communication='GPU communication activity can include polling/waiting; PP boundary work is included when present in the steady cycle. Not pure wire time.',
        source_run=str(a.run.resolve()),calibration=file_receipt(ROOT/'calibration.json'),
        analysis_source=file_receipt(Path(__file__)),cross_rank_clock_sum=False)
    save(a.out/('summary.json' if not a.arm else f'summary_{a.arm}.json'),result)
    print(json.dumps(dict(groups=groups,perturbation=perturbation),indent=2))


if __name__=='__main__':main()
