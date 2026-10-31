"""Recompute every timing from rank evidence; preserve both batch boundaries."""
import argparse
import csv
import json
from pathlib import Path
from common import save,native_for
from geometry import aggregate
from guard_authorization import visible_json


def analyze(run):
    cfg=json.loads((run/'config.json').read_text());points=[];flat=[]
    for phase in cfg['phases']:
        records=[visible_json(run/phase['id']/f'result_rank{rank}.json',run/'artifact_visibility/analysis',30)
                 for rank in range(phase['world'])]
        native=native_for(run,phase['model'])
        assert all(r['passed'] and r['buffer_explicitly_destroyed'] and not r['model_loaded']
                   and not r['expert_compute'] and r['native_sha256']==native['sha256'] for r in records)
        for i,work in enumerate(phase['workloads']):
            rows=[r['points'][i] for r in records]
            for row in rows:
                assert row['id']==work['id'] and row['passed'] and not row['cuda_graph']
                assert row['shape']['owner_assignments']==[work['tokens']*8//phase['world']]*phase['world']
                assert row['correctness_before']['passed'] and row['correctness_after_changed_payload']['passed']
                if work['global_batch'] in (8,4096):
                    assert row['distinct_expert_validation']['passed']
                    assert len(row['skewed_changed_payload_checks'])==4
                    assert all(x['passed'] for x in row['skewed_changed_payload_checks'])
                for sample in row['samples']:
                    assert abs(sample['pair_cuda_ms']-sample['dispatch_cuda_ms']-sample['combine_cuda_ms'])<.02
                    if phase['ce_enabled']:
                        delta=sample['ce_counter_delta']
                        assert delta[0]==2 and delta[1]==4 and delta[2]>0 and delta[4:]==[1,1]
                    else:assert sample['ce_counter_delta'] is None
            metrics={key:aggregate([r['samples'] for r in rows],key) for key in
                     ('pair_cuda_ms','dispatch_cuda_ms','combine_cuda_ms','pair_cpu_wall_ms')}
            point=dict(phase=phase['id'],model=phase['model'],transport=phase['transport'],world=phase['world'],
                       nodes=phase['nodes'],cohort=work['cohort'],global_batch=work['global_batch'],
                       call_unique_tokens=work['tokens'],offset=work['offset'],
                       source_tokens_per_rank=rows[0]['source_tokens_per_rank'],
                       assignments_per_gpu=rows[0]['correctness_before']['received_assignments'],
                       active_experts=rows[0]['shape']['active_experts'],route_sha256=rows[0]['shape']['route_sha256'],
                       native_sha256=native['sha256'],capacity=phase['capacity'],cuda_graph=False,
                       skip_local_drain=phase['skip_local_drain'],metrics=metrics)
            points.append(point)
            for iteration in range(3):
                flat.append(dict(**{k:v for k,v in point.items() if k!='metrics'},iteration=iteration,
                     **{key:val['samples'][iteration] for key,val in metrics.items()}))
    assert len(points)==280
    # Transport ablation: every field determining data and rank work must match.
    for row in points:
        if row['transport']!='rdma':continue
        twin=next(x for x in points if x['phase']==row['phase'].removesuffix('rdma')+'ce'
                  and x['cohort']==row['cohort'] and x['global_batch']==row['global_batch'])
        for key in ['native_sha256','source_tokens_per_rank','call_unique_tokens','route_sha256','assignments_per_gpu','capacity','cuda_graph']:
            assert row[key]==twin[key],(row['phase'],key)
    save(run/'comparison.json',dict(passed=True,points=points,
         aggregation='max of aligned ranks per iteration, then median/mean over 3 iterations',
         scope='Synthetic identity-expert communication only; no Attention/GEMM/model/KV',
         limitations=['No request TTFT/TPOT; no full PP scheduler or stage boundary',
             'Independent segment rank maxima are not additive',
             'Different source sizes, ownership, QPs and topology between widths are intentional',
             'Hy3 CE is a new H4096 shape specialization; GLM reuses c340',
             'Three samples are descriptive, not a service tail distribution']))
    with (run/'full_metrics.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(passed=True,points=len(points),aligned_samples=len(flat))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    analyze(p.parse_args().run.resolve())
