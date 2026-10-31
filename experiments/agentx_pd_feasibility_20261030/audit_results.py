"""Independent read-only qualification of actual run artifacts."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
GUARD=ROOT.parent/'h100_small_batch_layout_verify_20261029/guard_bundle_posix_v1/skills/llm-experiment-guard/scripts'
sys.path.insert(0,str(GUARD))
from artifact_visibility import wait_for_stable_json
from model_cache_adapter import _verify_usage_log

def read(p):return wait_for_stable_json(p,timeout_s=15,poll_s=.05,min_stable_s=.05)[0]

def audit(run):
    spec=read(run/'dashboard_spec.json');cfg=read(run/'config.json')
    changes=[x['path'] for x in spec['frozen_inputs'] if hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()!=x['sha256']]
    points=[];data={};tiers={}
    for cp in sorted(run.glob('results/*/*/completed.json')):
        out=cp.parent;done=read(cp);assert done['passed'] and done['repeats']==3
        rounds=[read(out/f'formal.r{i}.json') for i in (1,2,3)]
        assert all(len(x['requests'])==40 for x in rounds)
        for x in rounds:
            r=[r for r in x['requests'] if not r['primer']];assert len(r)==32
            for k in ('ttft_ms','tpot_ms','e2e_ms'):assert abs(statistics.mean(v[k] for v in r)-x['means'][k])<1e-8
            assert all(len(v['output_ids'])==v['output_tokens'] for v in x['requests'])
        points.append(dict(arm=out.parent.name,case=out.name,means={k:statistics.mean(x['means'][k] for x in rounds) for k in rounds[0]['means']},
            wall_samples_s=[x['wall_s'] for x in rounds],samples={k:[x['means'][k] for x in rounds] for k in rounds[0]['means']}))
        data[(out.parent.name,out.name)]=rounds
        allrows=[r for x in rounds for r in x['requests'] if not r['primer']]
        side='prefill_meta' if out.parent.name=='pd' else 'meta_info'
        tiers[out.parent.name+'_'+out.name]={k:sum((r[side].get('cached_tokens_details') or {}).get(k,0) for r in allrows) for k in ('device','host','storage')}
    equality=[]
    if cfg.get('baseline_reference_run'):
        ref=Path(cfg['baseline_reference_run']);reference_cfg=read(ref/'config.json')
        for key in ('node','model_path','engine_commit','family','profile','mem_fraction_static','hicache_gb_per_rank'):
            assert cfg[key]==reference_cfg[key], ('changed baseline condition',key)
        assert Path(cfg['screening_windows']).read_bytes()==Path(reference_cfg['screening_windows']).read_bytes()
        data[('colocated','c1')]=[read(ref/f'results/colocated/c1/formal.r{i}.json') for i in (1,2,3)]
    if ('pd','c1') in data and ('colocated','c1') in data:
        for ri,(left,right) in enumerate(zip(data[('colocated','c1')],data[('pd','c1')]),1):
            lm={(x['session_id'],x['turn']):x for x in left['requests']}
            for r in right['requests']:
                l=lm[(r['session_id'],r['turn'])];assert l['input_sha256']==r['input_sha256']
                a,b=l['output_ids'],r['output_ids'];match=0
                for x,y in zip(a,b):
                    if x!=y:break
                    match+=1
                equality.append(dict(repeat=ri,session_id=r['session_id'],turn=r['turn'],
                    exact=a==b,matching_prefix=match,output_tokens=len(a)))
    loaders=[]
    bindings_path=run/'dashboard_guard/model_cache_bindings.json'
    if points:
        bindings=read(bindings_path)
        completed_arms={x['arm'] for x in points}
        for node,binding in bindings['nodes'].items():
            for use in binding['usage_logs']:
                if any('.'+arm+'.' in use['id'] for arm in completed_arms):
                    receipt=_verify_usage_log(use,binding['group_key'],bindings['admitted_at'])
                    receipt['node']=node
                    loaders.append(receipt)
        assert len(loaders)==2*len(completed_arms), 'Missing replica loader audit'
    evidence=[]
    for f in run.glob('evidence/*/cleanup.json'):evidence.append(dict(path=str(f),verified=read(f).get('cleanup_verified')))
    result=dict(run=run.name,frozen_inputs_match=not changes,changed_inputs=changes,complete_points=points,
        completed_arm_loader_checks=loaders,
        loader_verifier=dict(path=str(GUARD/'model_cache_adapter.py'),sha256=hashlib.sha256((GUARD/'model_cache_adapter.py').read_bytes()).hexdigest()),
        cache_hit_tokens=tiers,long_serial_output_comparisons=equality,
        long_serial_all_exact=all(x['exact'] for x in equality) if equality else None,cleanup=evidence,
        qualification='Completed colocated arms independently audited only. Whole Guard run remains failed due to later PD startup; its receipts are not modified. No PD performance claim without short and long serial output validation; concurrent layouts can batch differently.')
    (run/'analysis_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('run','frozen_inputs_match','complete_points','cache_hit_tokens','long_serial_all_exact')}),flush=True)

if __name__=='__main__':
    for run in sorted((ROOT/'runs').glob('*hicache*')):audit(run)
