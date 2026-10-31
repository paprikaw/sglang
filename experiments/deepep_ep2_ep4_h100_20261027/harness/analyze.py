"""Aligned per-sample critical rank; never sum independently maximized phases."""
import argparse
import csv
import statistics as st
from pathlib import Path
import json
from common import save
from guard_authorization import visible_json


def analyze(run,out=None):
    out=out or run
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((run/'config.json').read_text());rows=[];raw=[]
    receipts=run/'artifact_visibility/analysis'
    for phase in cfg['phases']:
        ranks=[visible_json(run/phase['id']/f'result_rank{r}.json',receipts) for r in range(phase['world'])]
        assert all(r['passed'] and r['buffer_explicitly_destroyed'] for r in ranks)
        for wi,work in enumerate(phase['workloads']):
            points=[r['points'][wi] for r in ranks]
            assert all(p['id']==work['id'] and p['passed'] and len(p['samples'])==cfg['iterations'] for p in points)
            assert len({p['shape']['global_route_sha256'] for p in points})==1
            ds=[];cs=[];totals=[]
            for iteration in range(cfg['iterations']):
                samples=[p['samples'][iteration] for p in points]
                assert all(s['iteration']==iteration for s in samples)
                critical=max(range(phase['world']),key=lambda r:samples[r]['pair_cuda_ms'])
                sample=samples[critical]
                assert abs(sample['pair_cuda_ms']-sample['dispatch_cuda_ms']-sample['combine_cuda_ms'])<1e-4
                ds.append(sample['dispatch_cuda_ms']*1000)
                cs.append(sample['combine_cuda_ms']*1000)
                totals.append(sample['pair_cuda_ms']*1000)
                for rank,s in enumerate(samples):
                    raw.append(dict(phase=phase['id'],round=phase['round'],layout=phase['layout'],
                        world=phase['world'],tokens=work['tokens'],mode='graph' if work['cuda_graph'] else 'eager',
                        rank=rank,critical_rank=critical,**s))
            rows.append(dict(phase=phase['id'],round=phase['round'],layout=phase['layout'],
                world=phase['world'],tokens=work['tokens'],local_tokens=work['tokens']//phase['world'],
                mode='graph' if work['cuda_graph'] else 'eager',samples=len(totals),
                dispatch_us=st.median(ds),combine_us=st.median(cs),total_us=st.median(totals),
                total_mean_us=st.mean(totals),total_stdev_us=st.stdev(totals),
                total_min_us=min(totals),total_max_us=max(totals),
                route_sha256=points[0]['shape']['global_route_sha256'],
                global_input_bytes=work['tokens']*phase['hidden']*2,
                global_remote_assignments=points[0]['shape']['global_remote_assignments']))
    for name,data in [('full_metrics.csv',rows),('raw_samples.csv',raw)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for tokens in cfg['token_sweep']:
        assert len({r['route_sha256'] for r in rows if r['tokens']==tokens})==1
    assert len(rows)==len(cfg['phases'])*9
    save(out/'comparison.json',dict(passed=True,rows=len(rows),samples=len(raw),
        model_loaded=False,expert_compute=False,attention_compute=False,
        global_routes_identical_across_EP2_EP4=True,
        timing='median of sample-aligned rank-max D+C CUDA events; components from that same critical rank',
        event_scope='includes completion hooks, waiting and eager host feed; not measured wire time',
        correctness='exact counts/source IDs/payload/weighted identity; graph changed-payload checks',
        formal_points=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args();analyze(a.run,a.out)
