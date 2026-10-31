"""Summarize transfer-integrity receipts without exposing token payloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pp_transfer_audit import compare_transfers


def summarize(run):
    rows=[];outputs={};partitions={}
    for epoch in range(1,7):
        row=json.loads((run/'pp2'/f'qualification.epoch{epoch}.json').read_text())
        stages=[json.loads((run/'pp2/transfer_audit'/f'epoch{epoch}.pp{rank}.json').read_text())
                for rank in (0,1)]
        comparison=compare_transfers(stages)
        if not comparison['passed'] or comparison != row['audit']:
            raise RuntimeError('Saved comparison does not reproduce')
        stage_groups=[]
        for stage in stages:
            groups=Counter()
            for batch in stage['batches']:
                reqs=batch['requests']
                # Exclude initialization and draining. The exact native forward
                # mode remains in the source receipt; count real steady B4 here.
                if len(reqs)==4 and all(8 <= r['generated'] <= 112 for r in reqs):
                    groups[','.join(str(r['marker']) for r in reqs)]+=1
            stage_groups.append(dict(pp_rank=stage['pp_rank'],
                captured_bytes=stage['captured_bytes'],steady_b4_groups=dict(groups)))
        sequence=[r['output_ids'] for r in row['cohort']['requests']]
        outputs[epoch]=sequence
        partitions[epoch]=[sorted(s['steady_b4_groups']) for s in stage_groups]
        rows.append(dict(epoch=epoch,mode=row['mode'],**comparison,stages=stage_groups,
            performance_sample=False,output_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest()))
    baseline=[r['epoch'] for r in rows if r['mode']=='baseline']
    early=[r['epoch'] for r in rows if r['mode']=='early_activation']
    def agreements(epochs):
        return [dict(first=a,second=b,all_requests_equal=outputs[a]==outputs[b],
            equal_requests=sum(x==y for x,y in zip(outputs[a],outputs[b])),
            same_steady_ordered_groups=partitions[a]==partitions[b])
            for i,a in enumerate(epochs) for b in epochs[i+1:]]
    cfg=json.loads((run/'config.json').read_text())
    return dict(run=str(run),model=cfg['model_path'],passed=all(r['passed'] for r in rows),
        matched_messages=sum(r['matched_messages'] for r in rows),rows=rows,
        baseline_repeatability=agreements(baseline),early_repeatability=agreements(early),
        caution='This establishes exact sender/consumer bytes and records native microbatches. '
                'It is not a generation quality evaluation or a proof of bitwise inference invariance.',
        source_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((run/'pp2').glob('qualification.epoch*.json'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.run.absolute())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','matched_messages','baseline_repeatability','early_repeatability')}))
