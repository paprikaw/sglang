"""CPU-only component accounting. No fitting, interpolation, or sample averages."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
MANAGER = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')
EXPERT = MANAGER/'h100_four_model_expert_surface_20261029'
COMM = MANAGER/'h100_ep_token_comm_surface_20261029'
TOKENS = [8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
MODELS = [('glm47','GLM-4.7',160,4),('hy3','HY3',192,4),
          ('qwen35b','35B',256,1),('qwen122b','122B',256,1)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with p.open() as stream:
        return list(csv.DictReader(stream))


def save(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def pieces(n,limit=2048):
    return [min(limit,n-start) for start in range(0,n,limit)]


def active(n,ep,total):
    # Exactly the measured deterministic routing: eight expert groups, one
    # expert per group and token, consecutive IDs modulo total_experts/8.
    return (8//ep)*min(n,total//8)


def route_check():
    count=0
    for _,_,total,_ in MODELS:
        for ep in (1,2,4,8):
            for n in (1,2,4,8,16,32,64,128,1024,2048):
                for offset in (0,17):
                    per_rank=[{} for _ in range(ep)]
                    for token in range(offset,offset+n):
                        for group in range(8):
                            expert=group*(total//8)+token%(total//8)
                            rank=expert//(total//ep)
                            per_rank[rank][expert]=per_rank[rank].get(expert,0)+1
                    for rows in per_rank:
                        assert len(rows)==active(n,ep,total)
                        assert sum(rows.values())==8*n//ep
                        assert max(rows.values())-min(rows.values())<=1
                    count+=1
    return dict(passed=True,checked_route_patterns=count,
                offsets=[0,17],routing='eight groups; consecutive original token IDs',
                verified=['same active expert count on every rank',
                          'same assignment row count on every rank','balanced rows across active experts'])


def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    raw_g=read_csv(EXPERT/'absolute_all_events.csv')
    raw_c=read_csv(COMM/'all_critical_events.csv')
    selected_g=read_csv(EXPERT/'absolute_single_event.csv')
    selected_c=read_csv(COMM/'absolute_event01.csv')
    meta=json.loads((EXPERT/'manifest.json').read_text())
    comm_cfg=json.loads((COMM/'config.json').read_text())
    assert json.loads((COMM/'independent_audit.json').read_text())['passed']
    assert meta['gpu_release']['cleanup_verified']
    assert comm_cfg['global_chunk_limit']==2048
    g={}
    cg={}
    compact_g=[]
    compact_c=[]
    model_index={m[0]:i for i,m in enumerate(MODELS)}
    for r in raw_g:
        model,rep,mode,t,e=r['model'],int(r['replica']),r['mode'],int(r['T_assignment_rows']),int(r['E_active'])
        key=(model,rep,mode,t,e)
        values=[float(r[f'event_{i:02d}_ms']) for i in range(1,31)]
        assert key not in g
        g[key]=values
        compact_g.append([model_index[model],rep,0 if mode=='cuda_graph' else 1,t,e,[round(x,6) for x in values]])
    for r in raw_c:
        key=(r['shape'],int(r['round']),'cuda_graph' if r['mode']=='graph' else 'eager',int(r['ep']),int(r['global_original_tokens']))
        cg.setdefault(key,[None]*30)[int(r['event_1based'])-1]=float(r['total_ms'])
    for (model,rep,mode,ep,n),values in cg.items():
        assert len(values)==30 and all(x is not None for x in values)
        compact_c.append([model_index[model],rep,0 if mode=='cuda_graph' else 1,ep,n,[round(x,6) for x in values]])
    assert len(g)==1584 and len(cg)==624
    for r in selected_g:
        key=(r['model'],int(r['replica']),r['mode'],int(r['T_assignment_rows']),int(r['E_active']))
        assert float(r['latency_ms'])==g[key][0] and int(r['event_1based'])==1
    for r in selected_c:
        key=(r['shape'],int(r['round']),'cuda_graph' if r['mode']=='graph' else 'eager',int(r['ep']),int(r['global_original_tokens']))
        assert float(r['total_ms'])==cg[key][0] and int(r['event_1based'])==1

    def gb(model,mode,t,e):
        vals=[x for rep in (1,2) for x in g.get((model,rep,mode,t,e),[])]
        return (min(vals),max(vals)) if vals else None

    def cb(model,mode,ep,n):
        if ep==1:return (0.,0.)
        vals=[x for rep in (1,2) for x in cg.get((model,rep,mode,ep,n),[])]
        return (min(vals),max(vals)) if vals else None

    cases=[]
    rows=[]
    case_number=0
    for mi,(model,label,total,min_ep) in enumerate(MODELS):
        candidates=[(8,4)] if min_ep==4 else [(2,1),(4,1),(8,1),(8,4)]
        for ep,low in candidates:
            pp=ep//low
            alt=f'PP{pp}' if low==1 else f'PP{pp}×EP{low}'
            for n in TOKENS:
                # Ideal saturated-pipeline accounting: pp microbatches, N/pp
                # original tokens each. Per-device layer work matches after
                # factoring pp fewer layers per stage against pp microbatches.
                ep_plan=[(q*8//ep,active(q,ep,total)) for q in pieces(n)]
                local_n=n//pp
                pp_plan=([(local_n*8,active(local_n,1,total))] if low==1 else
                         [(q*8//low,active(q,low,total)) for q in pieces(local_n)])
                case=dict(id=case_number,model=mi,ep=ep,low=low,pp=pp,n=n,
                    alternative=alt,ep_plan=ep_plan,pp_plan=pp_plan,
                    ep_calls=len(ep_plan),pp_compute_calls=len(pp_plan),
                    pp_comm_calls=0 if low==1 else len(pp_plan),
                    pp_group_tokens=local_n,
                    same_compute_rows_per_call=ep_plan[0][0]==pp_plan[0][0] and len(ep_plan)==len(pp_plan),
                    multi_call=n>2048)
                cases.append(case)
                for mode in ('cuda_graph','eager'):
                    eb=[gb(model,mode,t,e) for t,e in ep_plan]
                    pb=[gb(model,mode,t,e) for t,e in pp_plan]
                    ce=cb(model,mode,ep,n)
                    cp=cb(model,mode,low,local_n)
                    missing=[]
                    for side,plan,bounds in [('EP',ep_plan,eb),('PP',pp_plan,pb)]:
                        for (t,e),bound in zip(plan,bounds):
                            if bound is None:missing.append(f'{side}:T{t}/E{e}')
                    if ce is None:missing.append(f'EP{ep}:N{n} communication')
                    if cp is None:missing.append(f'EP{low}:N{local_n} communication')
                    interval=None
                    if not missing:
                        ge_lo,ge_hi=sum(x[0] for x in eb),sum(x[1] for x in eb)
                        gp_lo,gp_hi=sum(x[0] for x in pb),sum(x[1] for x in pb)
                        margin_lo=(ce[0]-cp[1])-(gp_hi-ge_lo)
                        margin_hi=(ce[1]-cp[0])-(gp_lo-ge_hi)
                        interval=[margin_lo,margin_hi]
                        robust='PP' if margin_lo>0 else 'EP' if margin_hi<0 else 'boundary'
                    else:robust='missing'
                    for grepeat in (1,2):
                        for crepeat in (1,2):
                            row=dict(case_id=case_number,model=model,mode=mode,N_original=n,gpus=ep,
                                ep_layout=f'EP{ep}',pp_layout=alt,compute_process=grepeat,comm_round=crepeat,
                                event_1based=1,missing='; '.join(dict.fromkeys(missing)),
                                classification_all_samples=robust,
                                margin_min_ms=None if interval is None else interval[0],
                                margin_max_ms=None if interval is None else interval[1],
                                ep_compute_calls=len(ep_plan),pp_compute_calls=len(pp_plan),
                                matched_rows_per_call=case['same_compute_rows_per_call'])
                            if not missing:
                                ge=sum(g[(model,grepeat,mode,t,e)][0] for t,e in ep_plan)
                                gp=sum(g[(model,grepeat,mode,t,e)][0] for t,e in pp_plan)
                                ce1=cg[(model,crepeat,mode,ep,n)][0]
                                cp1=0 if low==1 else cg[(model,crepeat,mode,low,local_n)][0]
                                delta=gp-ge;saved=ce1-cp1;margin=saved-delta
                                assert abs((ge+ce1)-(gp+cp1)-margin)<1e-9
                                assert margin_lo-1e-9<=margin<=margin_hi+1e-9
                                row.update(ep_compute_ms=ge,ep_comm_ms=ce1,pp_compute_ms=gp,pp_comm_ms=cp1,
                                    extra_compute_ms=delta,saved_comm_ms=saved,pp_extra_budget_ms=margin,
                                    ep_component_total_ms=ge+ce1,pp_component_total_ms=gp+cp1)
                            rows.append(row)
                    case.setdefault('status',{})[mode]=dict(classification=robust,interval=interval,missing=missing)
                case_number+=1
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/'component_estimates_event01.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    payload=dict(models=[dict(id=m[0],label=m[1],experts=m[2],min_ep=m[3],
        kernel=meta['model'][m[0]]['backend']) for m in MODELS],tokens=TOKENS,cases=cases,
        compute=compact_g,communication=compact_c,
        units='ms',event_samples=30,rounds=2,aggregation='none',
        kind='component estimate under ideal saturated pipeline, not measured PP/EP end-to-end')
    (ROOT/'data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
    save(ROOT/'validation.json',dict(passed=True,routing=route_check(),
        source_selected_events_exactly_verified=True,raw_expert_events=1584*30,raw_communication_events=624*30,
        cases=len(cases),component_rows=len(rows),arithmetic='EP total - PP total = saved communication - extra compute',
        missing_points_interpolated=False,sample_average=False,round_average=False))
    source_paths=[EXPERT/'absolute_all_events.csv',EXPERT/'absolute_single_event.csv',EXPERT/'manifest.json',
        COMM/'all_critical_events.csv',COMM/'absolute_event01.csv',COMM/'manifest.json',COMM/'config.json',
        COMM/'independent_audit.json']
    save(ROOT/'sources.json',dict(inputs=[dict(path=str(p),sha256=sha(p)) for p in source_paths],
        generator=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        timestamp=datetime.now(ZoneInfo('Australia/Melbourne')).isoformat(),
        no_new_gpu_measurements=True,minimum_ep=dict(glm47=4,hy3=4,qwen35b=1,qwen122b=1)))
    for model,label,_,_ in MODELS:
        print(label)
        for ep,low in ([(8,4)] if model in ('glm47','hy3') else [(2,1),(4,1),(8,1),(8,4)]):
            selected=[r for r in rows if r['model']==model and r['gpus']==ep and
                r['pp_layout']==(f'PP{ep}' if low==1 else 'PP2×EP4') and r['mode']=='cuda_graph' and
                r['compute_process']==1 and r['comm_round']==1]
            print(f'EP{ep} vs '+('purePP' if low==1 else 'PP2EP4'),
                [(r['N_original'],r['classification_all_samples'],None if r['missing'] else round(r['pp_extra_budget_ms'],4)) for r in selected])
    return payload,rows


if __name__=='__main__':
    build()
