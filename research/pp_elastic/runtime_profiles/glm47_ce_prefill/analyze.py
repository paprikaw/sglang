"""Summarize one compact formal batch per arm and workload, preserving raw samples."""
import argparse
import csv
import math
import statistics
from pathlib import Path
from common import ARMS,cfg,save,visible_json

def nearest(values,p):
    values=sorted(values);return values[max(0,math.ceil(len(values)*p)-1)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);run=ap.parse_args().run.resolve();config=cfg(run)
    rows=[];transport_evidence=[]
    for arm in config['arms']:
        result=visible_json(run/arm/'benchmark_result.json',run/'artifact_visibility/analyze',60)
        assert result['passed'] and result['formal_trials']==config['benchmark']['formal_trials']
        assert result['formal_requests']==sum(w['concurrency'] for w in config['workloads'].values())*result['formal_trials']
        if config['cases'][arm]['transport']=='ce':
            for rank in range(8):
                before=visible_json(run/arm/f'ce_after_validation_rank{rank}.json',run/'artifact_visibility/analyze',30)
                after=visible_json(run/arm/f'ce_final_rank{rank}.json',run/'artifact_visibility/analyze',30)
                delta=[b-a for a,b in zip(before['counters']['counters'],after['counters']['counters'])]
                assert before['passed'] and after['passed'] and after['after_formal_requests']
                assert delta[0]>2 and delta[1]>4 and delta[2]>0 and delta[4]>1 and delta[5]>1
                transport_evidence.append(dict(arm=arm,rank=rank,ce_delta=delta,
                    scope='Model initialization/validation and warmup plus formal requests after the standalone communication check; not per-request timing.'))
        for batch in result['results']:
            if batch['warmup'] or batch.get('diagnostic'):continue
            requests=batch['requests'];assert len(requests)==4 and all(r['passed'] for r in requests)
            row=dict(arm=arm,workload=batch['workload'],trial=batch['trial'],request_count=len(requests),batch_wall_s=batch['wall_s'],
                input_tokens_per_s=batch['input_tokens_per_s'],output_tokens_per_s=batch['output_tokens_per_s'])
            for source,label in [('ttft_s','ttft'),('tpot_s','tpot'),('e2e_s','e2e')]:
                values=[r[source] for r in requests if r[source] is not None]
                for metric,fn in [('mean',statistics.mean),('median',statistics.median),('p99',lambda v:nearest(v,.99))]:
                    row[f'{label}_{metric}_ms']=fn(values)*1000 if values else None
            rows.append(row)
    dest=run/'compare';dest.mkdir(exist_ok=True)
    with (dest/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(dest/'summary.json',dict(passed=True,rows=rows,scope='one formal C4 batch per workload/arm; P99 descriptive over4 requests',
        ce_transport_evidence=transport_evidence,
        raw_request_artifacts='Each arm/formal_*_request*.json and SSE stream artifacts',config_id=config['id']))
    save(run/'completed_run.json',dict(passed=True,arms=list(config['arms']),workloads=list(config['workloads']),formal_trials=1,summary_path=str(dest/'summary.csv')))
    print(str(dest/'summary.csv'))

if __name__=='__main__':main()
