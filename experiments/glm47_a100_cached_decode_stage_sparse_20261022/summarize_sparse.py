"""Estimate per-step layer-body shares from rotating low-overhead layer samples.

All layers are observed. Only within-rank layer means are summed; concurrent
rank/stage clocks are never summed. Independent envelope samples expose local
scope overhead rather than hiding it in the whole-cohort TPOT average.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from summarize_events import partition, read, save, file_receipt, bucket


def average_parts(rows):
    phases=sorted({k for row in rows for k in row['phase_ms']})
    components=sorted({k for row in rows for k in row['component_ms']})
    return dict(window_ms=statistics.mean(row['window_ms'] for row in rows),
        phase_ms={k:statistics.mean(row['phase_ms'].get(k,0) for row in rows) for k in phases},
        component_ms={k:statistics.mean(row['component_ms'].get(k,0) for row in rows) for k in components})


def rank_summary(path):
    sample=read(path)
    if not sample['passed'] or sample.get('unresolved_event_indices'):
        raise ValueError('Invalid event receipt')
    if sample.get('profiling_method')!='sparse_deferred_cuda_event_api_intervals':
        raise ValueError('Wrong sampling protocol')
    rank,arm=sample['rank'],sample['arm']
    expected=set(range(92)) if arm=='ep8' else set(range(rank//4*46,rank//4*46+46))
    repeats=1 if arm=='ep8' else 2
    detailed,controls=defaultdict(list),defaultdict(list)
    for event in sample['events']:
        if event['category'] not in ('layer','layer_envelope_control'):
            continue
        if event['start_stream']!=event['end_stream'] or event.get('duration_ms') is None:
            raise ValueError('Layer envelope crossed streams')
        layer=int(event['label'])
        if event['category']=='layer':
            detailed[layer].append(dict(**partition(sample,event),
                forward_index=event['forward_index'],decode_ordinal=event['extend_index']))
        else:
            controls[layer].append(event['duration_ms'])
    if set(detailed)!=expected or set(controls)!=expected:
        raise ValueError('Incomplete model layer coverage')
    if any(len(detailed[i])!=repeats or len(controls[i])!=repeats for i in expected):
        raise ValueError('Wrong per-layer repetition count')
    layers={str(layer):dict(**average_parts(rows),samples=rows,
        control_envelope_ms=statistics.mean(controls[layer])) for layer,rows in sorted(detailed.items())}
    totals=defaultdict(float);phase_totals=defaultdict(float)
    for layer in layers.values():
        for k,v in layer['component_ms'].items():totals[k]+=v
        for k,v in layer['phase_ms'].items():phase_totals[k]+=v
    layer_sum=sum(layer['window_ms'] for layer in layers.values())
    envelope_sum=sum(layer['control_envelope_ms'] for layer in layers.values())
    meta={r['forward_index']:r for r in sample['forwards']}
    models=sorted([e for e in sample['events'] if e['category']=='model_forward'],key=lambda e:e['forward_index'])
    selected,unselected,cycles=[],[],[]
    for pos,event in enumerate(models):
        receipt=meta[event['forward_index']];m=receipt['metadata']
        if m['forward_mode']!='2' or m['batch_size']!=4 or receipt['extend_index']<3:
            continue
        (selected if receipt['detailed'] else unselected).append(event['duration_ms'])
        if pos+1>=len(models) or not receipt['detailed']:continue
        following=models[pos+1];n=meta[following['forward_index']]
        if (not n['detailed'] or following['start_stream']!=event['end_stream']
                or following['start_ms']<event['end_ms']):continue
        gap=partition(sample,event,(event['end_ms'],following['start_ms']))
        cycles.append(dict(forward_ms=event['duration_ms'],
            cycle_ms=following['start_ms']-event['start_ms'],
            outside_forward_ms=gap['window_ms'],
            pp_transport_and_readiness_ms=gap['component_ms'].get('pp_transport_and_wait',0),
            other_outside_forward_ms=gap['component_ms'].get('other',0)))
    if not selected or not unselected or not cycles:
        raise ValueError('Insufficient steady step perturbation evidence')
    return dict(arm=arm,rank=rank,hostname=sample['hostname'],
        layer_count=len(expected),samples_per_layer=repeats,layers=layers,
        estimated_layer_body_ms=layer_sum,
        estimated_component_ms=dict(totals),estimated_phase_ms=dict(phase_totals),
        control_envelope_layer_body_ms=envelope_sum,
        local_component_scope_to_envelope_ratio=layer_sum/envelope_sum,
        selected_forward_ms=statistics.mean(selected),unselected_forward_ms=statistics.mean(unselected),
        selected_to_unselected_forward_ratio=statistics.mean(selected)/statistics.mean(unselected),
        steady_cycle={k:statistics.mean(r[k] for r in cycles) for k in cycles[0]},
        steady_cycle_sample_count=len(cycles),
        source=file_receipt(path))


def group(rows):
    duration=statistics.mean(row['estimated_layer_body_ms'] for row in rows)
    labels=sorted({k for row in rows for k in row['estimated_component_ms']})
    phases=sorted({k for row in rows for k in row['estimated_phase_ms']})
    ms={key:statistics.mean(row['estimated_component_ms'].get(key,0) for row in rows) for key in labels}
    layer_count={row['layer_count'] for row in rows}
    if len(layer_count)!=1:raise ValueError('Cannot mix stage layer counts')
    return dict(ranks=[r['rank'] for r in rows],layer_count=next(iter(layer_count)),
        estimated_layer_body_ms=duration,estimated_component_ms=ms,
        estimated_component_percent={key:value/duration*100 for key,value in ms.items()},
        estimated_phase_ms={key:statistics.mean(row['estimated_phase_ms'].get(key,0) for row in rows) for key in phases},
        control_envelope_layer_body_ms=statistics.mean(row['control_envelope_layer_body_ms'] for row in rows),
        local_component_scope_to_envelope_ratio=statistics.mean(row['local_component_scope_to_envelope_ratio'] for row in rows),
        selected_to_unselected_forward_ratio=statistics.mean(row['selected_to_unselected_forward_ratio'] for row in rows),
        selected_forward_ms=statistics.mean(row['selected_forward_ms'] for row in rows),
        unselected_forward_ms=statistics.mean(row['unselected_forward_ms'] for row in rows),
        steady_cycle={key:statistics.mean(row['steady_cycle'][key] for row in rows) for key in rows[0]['steady_cycle']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('ep8','pp2ep4'))
    a=p.parse_args();groups={};perturbation={}
    for arm in ((a.arm,) if a.arm else ('ep8','pp2ep4')):
        rows=[rank_summary(a.run/arm/f'attribution_rank{rank}.json') for rank in range(8)]
        for row in rows:save(a.out/f'{arm}_rank{row["rank"]}.json',row)
        groups[arm]={(f'node{node}' if arm=='ep8' else f'stage{node}'):
                     group([row for row in rows if row['rank']//4==node]) for node in range(2)}
        groups[arm]['all_rank_mean']=group(rows)
        profile=read(a.run/arm/'profile_result.json')
        perturbation[arm]=dict(control_tpot_ms=profile['control']['tpot_s']['mean']*1000,
            sampled_tpot_ms=profile['tpot_s']['mean']*1000,
            whole_cohort_ratio=profile['profile_to_control_tpot_ratio'])
    result=dict(schema_version=1,groups=groups,perturbation=perturbation,
        denominator='Estimated local model-layer body: sum each unique layer mean, then average ranks. EP8=92 layers, each PP stage=46 layers.',
        cache='P81920 per request, B8; GPU-hot stable Decode only. Q1 entry, HiCache H2D and warmup excluded.',
        method='Sparse deferred CUDA-event API intervals. Components include exposed wait/host-feed time; not pure GPU kernel or network wire fractions. All layers sampled, independent envelope-only controls retained; no rescaling or correction applied.',
        cross_rank_or_stage_clock_sum=False,source_run=str(a.run.resolve()),
        analysis_sources=[file_receipt(Path(__file__)),file_receipt(Path(__file__).with_name('summarize_events.py'))])
    save(a.out/('summary.json' if not a.arm else f'summary_{a.arm}.json'),result)
    print(json.dumps(dict(groups=groups,perturbation=perturbation),indent=2))


if __name__=='__main__':main()
