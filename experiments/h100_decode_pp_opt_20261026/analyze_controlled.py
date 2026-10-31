"""Keep fixed-group confirmation separate from the earlier random-group pilot."""
import csv
import hashlib
import json
from pathlib import Path

from analyze import profile
from pp_transfer_audit import compare_transfers

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'analysis/fixed_groups';out.mkdir(parents=True,exist_ok=True)
    metrics=[];profiles=[];audits=[];sources=[]
    cache_path=out/'profile_cache.json'
    code_hash=hashlib.sha256((HERE/'analyze.py').read_bytes()).hexdigest()
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('analysis_sha256')!=code_hash:cache={}
    profile_cache=cache.get('profiles',{})
    for model in ('35b','122b'):
        run=HERE/'runs'/(model+'_fixed_groups_r3')
        if not (run/'task/cleanup.json').exists():continue
        cleanup=json.loads((run/'task/cleanup.json').read_text())
        if not cleanup['passed']:raise RuntimeError('Cleanup not verified')
        prompt_sets=[json.loads((run/arm/'prompt_fingerprints.json').read_text()) for arm in ('ep2','pp2')]
        if prompt_sets[0]!=prompt_sets[1] or prompt_sets[0]['prefix_tokens']!=4096 or len(prompt_sets[0]['sha256'])!=8:
            raise RuntimeError('EP/PP semantic input prefixes differ')
        for arm in ('ep2','pp2'):
            for path in sorted((run/arm).glob('metrics.decode.*.json')):
                d=json.loads(path.read_text())
                metrics.append(dict(model=model,arm=arm,mode=path.stem.split('.')[-1],
                    source=str(path),rounds=len(d['rounds']),output_tokens_per_s=d['output_tokens_per_s'],
                    ttft_ms=d['ttft_s']['mean']*1000,tpot_ms=d['tpot_ms']['mean'],
                    e2e_ms=d['e2e_s']['mean']*1000))
            for directory in sorted((run/arm/'traces').iterdir()):
                manifest=json.loads((directory/'manifest.json').read_text())
                if len(manifest['files'])!=2:raise RuntimeError('Missing rank trace')
                for path in sorted(directory.glob('*.trace.json.gz')):
                    key=str(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    previous=profile_cache.get(key)
                    if previous is None or previous['sha256']!=digest:
                        previous=dict(mode=directory.name,**profile(path,arm,model))
                        profile_cache[key]=previous
                    profiles.append(previous)
        transfers=[]
        for epoch in (1,2):
            receipts=[json.loads((run/'pp2/transfer_audit'/f'epoch{epoch}.pp{rank}.json').read_text())
                      for rank in (0,1)]
            row=json.loads((run/'pp2'/f'qualification.epoch{epoch}.json').read_text())
            result=compare_transfers(receipts)
            if not result['passed'] or result!=row['audit']:raise RuntimeError('Payload audit did not reproduce')
            sequences=[[tuple(r['marker'] for r in b['requests']) for b in receipt['batches']]
                       for receipt in receipts]
            if sequences[0]!=sequences[1]:raise RuntimeError('Stage batch sequences are not aligned')
            transfers.append(dict(epoch=epoch,mode=row['mode'],**result,
                exact_stage_batch_sequence_equal=True,actual_forwards_per_stage=len(sequences[0])))
        agreement=json.loads((run/'pp2/output_agreement.json').read_text());agreement.pop('outputs_by_mode')
        ep_agreement=json.loads((run/'ep2/output_agreement.json').read_text());ep_agreement.pop('outputs_by_mode')
        admissions=[]
        for p in sorted((run/'pp2').glob('cohort.formal*.json')):
            row=json.loads(p.read_text())
            admissions.append(dict(mode=row['mode'],**row['result']['fixed_group_receipt']))
        audits.append(dict(model=model,cleanup=cleanup,pp_output_agreement=agreement,
            ep_output_agreement=ep_agreement,payload_audits=transfers,formal_admissions=admissions,
            same_semantic_prefix_fingerprints=True,prefix_tokens_per_request=4096,
            correctness_scope='Exact tested transmitted bytes; equal native groups checked in separate diagnostics; '
                              'formal per-request output equality reported without hiding differences',
            production_promoted=False))
        for relative in ('config.json','dashboard_spec.json','task/cleanup.json',
                         'pp2/output_agreement.json','ep2/output_agreement.json',
                         'ep2/prompt_fingerprints.json','pp2/prompt_fingerprints.json'):
            path=run/relative;sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name,rows in (('metrics',metrics),('profiles',profiles)):
        if rows:
            with (out/(name+'.csv')).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'audit.json').write_text(json.dumps(dict(runs=audits,sources=sources),indent=2)+'\n')
    cache_path.write_text(json.dumps(dict(analysis_sha256=code_hash,profiles=profile_cache),indent=2)+'\n')
    print(json.dumps(dict(models=[r['model'] for r in audits],metrics=len(metrics),profiles=len(profiles))))


if __name__=='__main__':main()
