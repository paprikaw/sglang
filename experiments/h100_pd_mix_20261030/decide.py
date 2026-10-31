"""Compare measured configurations, not fastest samples or cross-node averages."""
import argparse
import csv
import json
from pathlib import Path

PP={'two_pp2','mixed'}
def records(p):
    rows=list(csv.DictReader(p.open()))
    for r in rows:
        for k,v in list(r.items()):
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows
def candidate(r):
    return '/'.join([r['layout'],r['route']]+([f"mb{int(r['pp_microbatch'])}",f"policy{int(r['pp_policy'])}"] if r['layout'] in PP else []))
def public(r,field):
    return dict(candidate=candidate(r),value=r[field],run=r['run'],source=r.get('source'),
                layout=r['layout'],route=r['route'])
def write_csv(p,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);a=ap.parse_args();root=a.analysis
    all_rows=records(root/'full_metrics.csv');all_slo=records(root/'slo_sensitivity.csv')
    assert all(r['rounds']==3 for r in all_rows)
    # Retain invalid traces in the full tables, but never relax the predeclared
    # client-arrival threshold or mix selected repeats into a replacement point.
    excluded=[r for r in all_rows if not r['arrival_valid']]
    rows=[r for r in all_rows if r['arrival_valid']]
    slo=[r for r in all_slo if r['arrival_valid']]
    for old in excluded:
        replacements=[r for r in rows if r['run']!=old['run'] and all(
            r[k]==old[k] for k in ('node','layout','case','route','pp_policy','pp_microbatch'))]
        assert replacements, 'Invalid point requires a complete valid replacement: '+old['source']
    decisions=[];slo_comparison=[]
    for node,case in sorted(set((r['node'],r['case']) for r in rows if r['kind']=='mixed_arrivals')):
        group=[r for r in rows if r['node']==node and r['case']==case]
        for metric in ('all_mean_ttft_ms','all_mean_e2e_ms','D_mean_tpot_ms','makespan_ms'):
            nonpp=min((r for r in group if r['layout'] not in PP),key=lambda r:r[metric])
            pp=min((r for r in group if r['layout'] in PP),key=lambda r:r[metric])
            decisions.append(dict(node=node,case=case,metric=metric,nonpp_candidate=candidate(nonpp),
                nonpp_value=nonpp[metric],pp_candidate=candidate(pp),pp_value=pp[metric],
                pp_improvement_pct=(1-pp[metric]/nonpp[metric])*100,
                nonpp_source=nonpp['source'],pp_source=pp['source']))
    for node,case,ttft,tpot in sorted(set((r['node'],r['case'],r['ttft_slo_ms'],r['tpot_slo_ms']) for r in slo)):
        g=[r for r in slo if (r['node'],r['case'],r['ttft_slo_ms'],r['tpot_slo_ms'])==(node,case,ttft,tpot)]
        nonpp=max((r for r in g if r['layout'] not in PP),key=lambda r:r['goodput_req_s'])
        pp=max((r for r in g if r['layout'] in PP),key=lambda r:r['goodput_req_s'])
        sno=[r for r in g if r['layout'] not in PP and r['route']=='least_outstanding']
        original=max(sno,key=lambda r:r['goodput_req_s'])
        slo_comparison.append(dict(node=node,case=case,ttft_slo_ms=ttft,tpot_slo_ms=tpot,
            nonpp_candidate=candidate(nonpp),nonpp_goodput=nonpp['goodput_req_s'],nonpp_good_fraction=nonpp['good_fraction'],
            pp_candidate=candidate(pp),pp_goodput=pp['goodput_req_s'],pp_good_fraction=pp['good_fraction'],
            pp_minus_nonpp_goodput=pp['goodput_req_s']-nonpp['goodput_req_s'],
            pp_improvement_pct=(pp['goodput_req_s']/nonpp['goodput_req_s']-1)*100 if nonpp['goodput_req_s']>0 else None,
            original_nonpp_goodput=original['goodput_req_s'],
            nonpp_phase_isolation_measured=any(r['route']=='phase_pools' and r['layout'] not in PP for r in g)))
    write_csv(root/'winner_comparison.csv',decisions);write_csv(root/'slo_comparison.csv',slo_comparison)
    result=dict(scope='35B BF16, four H100 GPUs per candidate, cold full-request sticky placement; '
        'uniform finite synthetic arrivals, not a sustained-rate capacity benchmark or online resizing.',
        accepted_runs=sorted({r['run'] for r in rows}),measured_points=len(rows),
        formal_rounds=sum(int(r['rounds']) for r in rows),
        formal_requests=sum(int(r['requests']*r['rounds']) for r in rows),
        recorded_points=len(all_rows),recorded_formal_rounds=sum(int(r['rounds']) for r in all_rows),
        recorded_formal_requests=sum(int(r['requests']*r['rounds']) for r in all_rows),
        excluded_points=[dict(run=r['run'],source=r['source'],reason='At least one repeat had client arrival lag >= 100 ms') for r in excluded],
        controlled_cases=sorted({r['case'] for r in rows if r['qualification']=='controlled'}),
        actual_resizing_measured=False,phase_state_transfer_measured=False,
        fixed_schema_lookahead_used=False,
        e2e_pp_wins=[r for r in decisions if r['metric']=='all_mean_e2e_ms' and r['pp_improvement_pct']>0],
        ttft_pp_wins=[r for r in decisions if r['metric']=='all_mean_ttft_ms' and r['pp_improvement_pct']>0],
        slo_pp_improvements=[r for r in slo_comparison if r['pp_minus_nonpp_goodput']>.001],
        slo_pp_improvements_above_10pct=[r for r in slo_comparison if r['pp_improvement_pct'] is not None and r['pp_improvement_pct']>=10],
        slo_zero_baseline_pp_positive=[r for r in slo_comparison if r['nonpp_goodput']==0 and r['pp_goodput']>0],
        node_means_combined=False)
    (root/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:len(v) if isinstance(v,list) and k.endswith(('wins','improvements','10pct','positive')) else v
                      for k,v in result.items()},indent=2))

if __name__=='__main__':main()
