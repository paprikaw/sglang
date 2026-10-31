"""Recompute matched measurements, kernel profiles and correctness evidence."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'h100_pp_expert_fragmentation_20261026'
OUT=ROOT/'analysis'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    old=module('fragment_analysis',OLD/'analyze.py')
    tools=old.load_analysis()
    previous=tools.graph_bucket
    tools.graph_bucket=lambda name:'expert.layout_adapter' if name=='_metadata' or name.startswith('_metadata<') else previous(name)
    vis_path=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py')
    vis=module('fair_visibility',vis_path)
    probabilities=module('fair_probabilities',OLD/'validate_probabilities.py')
    metrics,rounds,requests,components,details,audits,routing,qualifications,receipts=[],[],[],[],[],[],[],[],[]
    accepted=[]
    for run in sorted((ROOT/'runs').iterdir()):
        if not (run/'config.json').exists():continue
        cfg=read(run/'config.json');progress=read(run/'task/progress.json')
        if progress['stage']=='queued':
            audits.append(dict(run=run.name,launched=False,stage='prepared_only'));continue
        audit=dict(run=run.name,launched=True,stage=progress['stage'])
        for line in (run/'supervisor.log').read_text().splitlines():
            try:item=json.loads(line)
            except ValueError:continue
            if isinstance(item,dict) and 'gpu_registry_state' in item:
                audit.update({k:item.get(k) for k in ['experiment_exit_code','postcheck','gpu_registry_state']})
        cleanup=read(run/'task/cleanup.json')
        assert cleanup['passed'] and cleanup['model_processes_exited'],run
        assert audit.get('gpu_registry_state')=='released',audit
        for item in read(run/'launch_manifest.json')['private_adapters']:
            assert sha(item['path'])==item['sha256'],item
        audit.update(cleanup_verified=True,private_source_hashes_verified=True)
        audits.append(audit)
        if progress['stage']!='complete' or audit.get('postcheck')!='passed':continue
        accepted.append(run)
        for arm in cfg['arm_order']:
            cleanup=read(run/arm/'cleanup.json')
            assert cleanup['cleanup_verified'] and cleanup['process_groups_exited']
            for phase in ['prefill','decode']:
                path=run/arm/f'metrics.{phase}.json'
                data,receipt=vis.wait_for_stable_json(path,timeout_s=30);receipts.append(receipt)
                throughputs=[]
                allrows=[]
                for i,group in enumerate(data['rounds']):
                    assert len(group['requests'])==cfg['requests']
                    count=cfg['output_tokens'] if phase=='decode' else 1
                    cached=cfg['prefix_tokens'] if phase=='decode' else 0
                    for j,row in enumerate(group['requests']):
                        assert row['output_tokens']==count and (row.get('cached_tokens') or 0)==cached,(path,j)
                        allrows.append(row)
                        requests.append(dict(run=run.name,mode=cfg['expert_mode'],arm=arm,phase=phase,
                            round=i,request=j,cached_tokens=cached,output_tokens=count,
                            e2e_ms=row['e2e_s']*1000,ttft_ms=row['ttft_s']*1000,tpot_ms=row.get('tpot_ms')))
                    total=cfg['requests']*(cfg['output_tokens'] if phase=='decode' else cfg['prefix_tokens'])
                    rate=total/group['makespan_s'];throughputs.append(rate)
                    rounds.append(dict(run=run.name,mode=cfg['expert_mode'],arm=arm,phase=phase,
                        round=i,throughput_tok_s=rate,cohort_ms=group['makespan_s']*1000))
                metrics.append(dict(run=run.name,mode=cfg['expert_mode'],arm=arm,phase=phase,
                    throughput_tok_s=st.mean(throughputs),throughput_stdev=st.stdev(throughputs),
                    throughput_median=st.median(throughputs),repeats=len(throughputs),
                    cohort_ms=st.mean(g['makespan_s'] for g in data['rounds'])*1000,
                    e2e_ms=st.mean(r['e2e_s'] for r in allrows)*1000,
                    ttft_ms=st.mean(r['ttft_s'] for r in allrows)*1000,
                    tpot_ms=st.mean(r['tpot_ms'] for r in allrows) if phase=='decode' else None,
                    context_tokens=cached,new_tokens=1 if phase=='decode' else cfg['prefix_tokens'],
                    output_tokens=count,requests=cfg['requests'],source=str(path),sha256=sha(path)))
                manifest_path=run/arm/'traces'/phase/'manifest.json'
                manifest,receipt=vis.wait_for_stable_json(manifest_path,timeout_s=30);receipts.append(receipt)
                assert len(manifest['files'])==2
                for entry in manifest['files']:
                    path=Path(entry['path']);assert sha(path)==entry['sha256']
                    if phase=='decode':
                        value,detail=tools.trace_components(path,arm,'native','H100',cfg['selected_forwards'])
                        details.append({**detail,'run':run.name,'arm':arm,'phase':phase,'trace':str(path)})
                    else:value=old.prefill_components(path,arm,tools)
                    components.append({**value,'run':run.name,'mode':cfg['expert_mode'],'arm':arm,'phase':phase,'trace':str(path)})
            path=run/arm/'validation.probabilities.json'
            _,receipt=vis.wait_for_stable_json(path,timeout_s=30);receipts.append(receipt)
            if arm=='ep2':
                for path in sorted((run/arm/'ep_kernel_qualification').glob('*.json')):
                    item,receipt=vis.wait_for_stable_json(path,timeout_s=30);receipts.append(receipt)
                    assert item['passed'] and all(t['finite'] and t['relative_l2']<=.015 and t['metadata_coverage'] for t in item['tests'])
                    qualifications.append(dict(run=run.name,source=str(path),**item))
                assert len([q for q in qualifications if q['run']==run.name])==2
                for path in sorted((run/arm/'routing').glob('*.json')):
                    item,receipt=vis.wait_for_stable_json(path,timeout_s=30);receipts.append(receipt)
                    assert item['numerical']['finite'] and item['numerical']['relative_l2']<=.015
                    routing.append(dict(run=run.name,arm=arm,source=str(path),**item))
    summary,positions=[],[]
    for arm in ['ep2','pp2']:
        candidates=[r for r in accepted if arm in read(r/'config.json')['arm_order']]
        native=[r for r in candidates if read(r/'config.json')['expert_mode']=='native']
        if not native:continue
        ref=native[-1]/arm/'validation.probabilities.json'
        reference=probabilities.samples(ref)
        for run in candidates:
            for i,sample in enumerate(probabilities.samples(run/arm/'validation.probabilities.json')):
                item,points=probabilities.compare(reference[0],sample,f'{run.name}.{arm}.r{i}')
                summary.append(item);positions+=points
    for name,rows in [('metrics',metrics),('rounds',rounds),('per_request',requests),('components',components),('probability_positions',positions)]:
        table(OUT/(name+'.csv'),rows)
    for name,data in [('run_audit',audits),('visibility_receipts',receipts),('profile_details',details),
                      ('ep_routing',routing),('ep_qualifications',qualifications),('probability_validation',summary)]:
        save(OUT/(name+'.json'),data)
    print(json.dumps(dict(accepted=[r.name for r in accepted],metrics=metrics,audits=audits),indent=2))


if __name__=='__main__':main()
