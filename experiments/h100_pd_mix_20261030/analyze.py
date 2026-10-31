"""Recompute all published metrics from complete raw traces, preserving node identity."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
VIS=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/artifact_visibility.py')
spec=importlib.util.spec_from_file_location('visibility',VIS)
visibility=importlib.util.module_from_spec(spec);spec.loader.exec_module(visibility)

def read(p):
    value,_=visibility.wait_for_stable_json(p,timeout_s=30,poll_s=.05,min_stable_s=.05)
    return value
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def mean(xs):return statistics.mean(xs) if xs else None
def percentile(xs,p):
    if not xs:return None
    s=sorted(xs);k=(len(s)-1)*p;i=int(k);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(k-i)
def csv_write(p,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'analysis');a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    metrics=[];rounds=[];slo=[];audit=[];hist=[]
    for name in a.runs:
        run=(ROOT/'runs'/name).resolve();cfg=read(run/'config.json');frozen=read(run/'dashboard_spec.json')['frozen_inputs']
        altered=[d['path'] for d in frozen if hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()!=d['sha256']]
        if altered:raise RuntimeError('Frozen inputs changed: '+str(altered))
        runaudit=dict(run=name,node=cfg['node'],frozen_inputs_verified=len(frozen),complete_points=0,
                      expected_points=cfg['expected_points'],arrival_invalid_points=[],pp_qualification=[])
        for path in run.glob('evidence/*/controlled_qualification.json'):
            q=read(path);left,right=q['records']
            assert left['policy']==0 and right['policy']==1
            assert len(left['result']['requests'])==len(right['result']['requests'])
            agreement=True
            for x,y in zip(left['result']['requests'],right['result']['requests']):
                assert x['index']==y['index'] and x['prompt_sha256']==y['prompt_sha256']
                assert x['replica']==y['replica'] and x['cached_tokens']==y['cached_tokens']==0
                assert x['prompt_tokens']==y['prompt_tokens'] in (256,8192)
                assert x['output_tokens']==y['output_tokens'] in (8,64)
                agreement &= x['output_ids']==y['output_ids']
            assert agreement==q['exact_output_agreement']
            for x,y in zip(left['admission_receipts'],right['admission_receipts']):
                assert x['epoch']==1 and y['epoch']==2 and x['ordered_markers']==y['ordered_markers']
            assert q['selected_policy']==(1 if agreement else 0)
            runaudit['pp_qualification'].append(dict(layout=path.parent.name,exact_output_agreement=agreement,selected_policy=q['selected_policy']))
        for completed in sorted(run.glob('results/*/*/completed.json')):
            done=read(completed);p=completed.parent;protocol=read(p/'protocol.json');case=protocol['case'];arm=p.parent.name
            assert done['passed'] and done['formal_rounds']==cfg['repeats']
            per=[];slorows=[]
            for repeat in range(1,cfg['repeats']+1):
                packet=read(p/f'formal.r{repeat}.json');assert packet['performance_sample']
                g=packet['result'];rs=g['requests'];assert len(rs)==case['requests']
                for r,expect in zip(rs,protocol['requests']):
                    assert r['index']==expect['index'] and r['kind']==expect['kind']
                    assert r['prompt_sha256']==expect['prompt_sha256'] and r['prompt_tokens']==expect['Q']
                    assert r['output_tokens']==expect['O']==len(r['output_ids']) and r['cached_tokens']==0
                    assert r['planned_arrival_s']==expect['arrival_s']
                    assert abs(r['client_arrival_lag_ms']-(r['started']-r['scheduled'])*1000)<1e-6
                    assert abs(r['e2e_s']-(r['finished']-r['started']))<1e-8
                    if r['output_tokens']>1:
                        assert abs(r['tpot_ms']-(r['e2e_s']-r['ttft_s'])*1000/(r['output_tokens']-1))<1e-6
                duration=max(r['finished'] for r in rs)-g['origin']
                assert abs(duration-g['makespan_s'])<1e-8
                assert len(g['forward_deltas'])==4
                for d in g['forward_deltas']:
                    assert any(k.startswith('EXTEND:') and v>0 for k,v in d['counts'].items())
                    if case['kind']=='mixed_arrivals':assert any(k.startswith('DECODE:') and v>0 for k,v in d['counts'].items())
                if cfg.get('controlled_qualification') and protocol['pp_policy']==1:
                    assert g['pp_early_send_deltas'] and all(v>0 for v in g['pp_early_send_deltas'])
                assert g['arrival_valid']==(max(r['client_arrival_lag_ms'] for r in rs)<100)
                identity=dict(run=name,node=cfg['node'],layout=arm,case=case['id'],route=protocol['route'],
                    pp_policy=protocol['pp_policy'],pp_microbatch=cfg['pp_stream_microbatch'],
                    qualification='controlled' if cfg.get('controlled_qualification') else 'free_arrival',kind=case['kind'],repeat=repeat)
                m=dict(identity,requests=len(rs),input_tokens=sum(r['prompt_tokens'] for r in rs),
                    output_tokens=sum(r['output_tokens'] for r in rs),makespan_ms=duration*1000,
                    request_per_s=len(rs)/duration,input_tokens_per_s=g['input_tokens_per_s'],
                    output_tokens_per_s=g['output_tokens_per_s'],
                    client_lag_max_ms=max(r['client_arrival_lag_ms'] for r in rs),
                    client_lag_p95_ms=percentile([r['client_arrival_lag_ms'] for r in rs],.95),arrival_valid=g['arrival_valid'])
                for cls in ('all','P','D'):
                    sub=[r for r in rs if cls=='all' or r['kind']==cls]
                    m[f'{cls}_requests']=len(sub)
                    for metric,field,mul in [('ttft_ms','ttft_s',1000),('e2e_ms','e2e_s',1000),('tpot_ms','tpot_ms',1)]:
                        v=[r[field]*mul for r in sub if r[field] is not None]
                        m[f'{cls}_mean_{metric}']=mean(v)
                        m[f'{cls}_p95_{metric}']=percentile(v,.95)
                per.append(m);rounds.append(m)
                for d in g['forward_deltas']:
                    for key,value in d['counts'].items():hist.append(dict(identity,replica=d['replica'],
                        pp_rank=d['shape']['pp_rank'],tp_rank=d['shape']['tp_rank'],counter=key,count=value))
                if case['kind']=='mixed_arrivals':
                    for ttft in cfg['slo_sensitivity']['ttft_ms']:
                        for tpot in cfg['slo_sensitivity']['tpot_ms']:
                            good=sum(r['ttft_s']*1000<=ttft and r['tpot_ms']<=tpot for r in rs)
                            slorows.append(dict(identity,ttft_slo_ms=ttft,tpot_slo_ms=tpot,
                                good_requests=good,good_fraction=good/len(rs),goodput_req_s=good/duration))
            arrival_valid=all(m['arrival_valid'] for m in per)
            meta={k:per[0][k] for k in ('run','node','layout','case','route','pp_policy','pp_microbatch','qualification','kind')}
            row=dict(meta,rounds=len(per),arrival_valid=arrival_valid,source=str(p))
            exclude=set(meta)|{'repeat','arrival_valid'}
            for k in per[0]:
                v=[m[k] for m in per if m[k] is not None]
                if k not in exclude and v:
                    row[k]=mean(v)
                    if k.endswith(('_ms','_s')):row[k+'_stdev']=statistics.stdev(v)
            metrics.append(row)
            for ttft in cfg['slo_sensitivity']['ttft_ms']:
                for tpot in cfg['slo_sensitivity']['tpot_ms']:
                    group=[r for r in slorows if r['ttft_slo_ms']==ttft and r['tpot_slo_ms']==tpot]
                    if group:slo.append(dict(meta,ttft_slo_ms=ttft,tpot_slo_ms=tpot,arrival_valid=arrival_valid,
                        goodput_req_s=mean([x['goodput_req_s'] for x in group]),good_fraction=mean([x['good_fraction'] for x in group])))
            runaudit['complete_points']+=1
            if not arrival_valid:runaudit['arrival_invalid_points'].append(str(p))
        audit.append(runaudit)
    csv_write(out/'full_metrics.csv',metrics);csv_write(out/'formal_rounds.csv',rounds)
    csv_write(out/'slo_sensitivity.csv',slo);csv_write(out/'forward_histograms.csv',hist)
    save(out/'audit.json',audit)
    lines=['# P/D 混合负载：完整原始轮次的算术平均','',
           '每节点分别比较；单位均为绝对毫秒。未通过客户端到达偏差检查的点不作为受控到达结论。原始请求、全部轮次和各自源目录保留。', '']
    for node in sorted({m['node'] for m in metrics}):
        lines+=['## '+node,'','| run | workload | 布局／路由 | PP policy／微批上限 | 平均 TTFT | 平均 TPOT | 平均 E2E | 完整流量完成时间 | 到达有效 |',
                '|---|---|---|---:|---:|---:|---:|---:|---|']
        for m in sorted((m for m in metrics if m['node']==node),key=lambda x:(x['case'],x['layout'],x['route'])):
            vals=[m.get(k) for k in ['all_mean_ttft_ms','all_mean_tpot_ms','all_mean_e2e_ms','makespan_ms']]
            pp_label=(str(m['pp_policy'])+'/'+str(m['pp_microbatch'])) if m['layout'] in ('mixed','two_pp2') else '—'
            lines.append('| '+m['run']+' | '+m['case']+' | '+m['layout']+'/'+m['route']+' | '+pp_label+' | '+
                ' | '.join('—' if v is None else f'{v:.2f}' for v in vals)+' | '+str(m['arrival_valid'])+' |')
        lines+=['']
    (out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(points=len(metrics),formal_rounds=len(rounds),requests=sum(r['requests'] for r in rounds),
                         source_audit=audit,out=str(out)),indent=2))

if __name__=='__main__':main()
