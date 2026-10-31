"""Offline family comparison; no model launch and no raw-artifact mutation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/experiments/experiment_summary/pd_mix_pool_20261030/full_metrics.csv')

def main():
    with SOURCE.open() as f:rows=list(csv.DictReader(f))
    primary=[r for r in rows if r['run'] in ('011_mixed_r1','171_mixed_r1') and
             r['kind']=='mixed_arrivals' and r['route']=='least_outstanding' and r['arrival_valid']=='True']
    out=[]
    for node,case in sorted({(r['node'],r['case']) for r in primary}):
        group=[r for r in primary if (r['node'],r['case'])==(node,case)]
        for metric in ('all_mean_e2e_ms','makespan_ms','D_mean_tpot_ms','all_mean_ttft_ms'):
            local=min((r for r in group if r['layout'] in ('four_ep1','two_pp2')),key=lambda r:float(r[metric]))
            ep=min((r for r in group if r['layout'] in ('ep4','two_ep2')),key=lambda r:float(r[metric]))
            out.append(dict(node=node,case=case,metric=metric,lower_is_better=True,
                ep1_layout=local['layout'],ep1_mean_ms=float(local[metric]),
                ep_layout=ep['layout'],ep_mean_ms=float(ep[metric]),
                ep1_source=local['source'],ep_source=ep['source']))
    assert len(out)==48
    assert all(r['ep1_mean_ms']<r['ep_mean_ms'] for r in out if r['metric']=='all_mean_e2e_ms')
    with (ROOT/'family_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    manifest=dict(id='qwen35b_h100_pd_disaggregation_scope_20261030',status='complete',
        kind='offline evidence reinterpretation and proposed protocol',new_gpu_experiment=False,
        model='Qwen3.6-35B-A3B BF16',gpus_per_candidate=4,gpu='H100 80GB',
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        engine_and_launch_provenance=str(SOURCE.parent/'manifest.json'),
        scope='Existing P=0 whole-request sticky mixture traces, three complete repeats per point; not real PD disaggregation',
        comparisons=len(out),all_e2e_ep1_wins=True,
        phase_transfer_measured=False,dynamic_ratio_measured=False,
        interpretation='PP1 and pure PP belong to the EP=1 family. Internal PP-versus-single results do not falsify EP=1/EP>1 dynamic allocation.',
        historical_snapshot_unchanged=True)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'comparisons':len(out),'new_gpu_experiment':False}))

if __name__=='__main__':main()
