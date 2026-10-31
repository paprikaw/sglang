"""Re-extract a common Expert-GEMM scope from saved profiles; CPU only."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
SPEC=importlib.util.spec_from_file_location('steady_analysis',EXP/'pp_ep_steady_decode_20261026/analyze.py')
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)
IntervalIndex,union_us=MOD.IntervalIndex,MOD.union_us


def write_csv(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def extract(path,hardware,model,phase,arm):
    with gzip.open(path,'rt')as handle: data=json.load(handle)
    events=[e for e in data['traceEvents']if e.get('ph')=='X'and e.get('dur',0)>0]
    annotations=[e for e in events if e.get('cat')=='user_annotation']
    steps=[e for e in annotations if e.get('name','').startswith('step[')]
    step_index=IntervalIndex(steps)
    kernels=[e for e in events if e.get('cat')=='kernel']
    cpu={e.get('args',{}).get('correlation'):e for e in events
         if e.get('cat')in('cuda_runtime','cuda_driver')and e.get('args',{}).get('correlation')is not None}
    layers_per_rank=(40 if model=='35B'else 48)//(2 if arm=='pp2'else 1)
    inventory=Counter();selected_steps={};by_layer=defaultdict(lambda:defaultdict(list));layer_objects={}
    def valid_step(step):
        return step is not None and (step['name'].startswith('step[EXTEND')if phase=='prefill'
            else step['name'].startswith('step[DECODE bs=4 '))
    graph=hardware=='H100'and phase=='decode'
    if graph:
        by_correlation=defaultdict(list)
        for kernel in kernels:by_correlation[kernel.get('args',{}).get('correlation')].append(kernel)
        for launch in events:
            if launch.get('name')!='cudaGraphLaunch':continue
            step=step_index.find(launch['ts'],launch.get('tid'))
            if not valid_step(step):continue
            gpu=by_correlation.get(launch.get('args',{}).get('correlation'),[])
            if not gpu:continue
            full=sum('flash::FlashAttnFwdSm90<'in e['name']for e in gpu)
            linear=sum('fused_recurrent_gated_delta_rule'in e['name']for e in gpu)
            assert (full,linear)==(layers_per_rank//4,layers_per_rank*3//4),(path,full,linear)
            selected_steps[id(step)]=step
            for e in gpu:
                name=e['name']
                bucket='expert_gemm'if'deep_gemm::'in name and '(deep_gemm::GemmType)2'in name else (
                    'expert_exchange'if'deep_ep::'in name and('dispatch'in name or'combine'in name)else None)
                if bucket:
                    by_layer[id(step)][bucket].append((e['ts'],e['ts']+e['dur']))
                    if bucket=='expert_gemm':inventory[name]+=1
        count=len(selected_steps)*layers_per_rank
    else:
        layers=[e for e in annotations if re.fullmatch(r'(?:component\.)?layer\.\d+',e.get('name',''))]
        layer_index=IntervalIndex(layers)
        components=[e for e in annotations if e.get('name','').startswith('component.')and not e['name'].startswith('component.layer.')]
        component_index=IntervalIndex(components)
        for layer in layers:
            step=step_index.find(layer['ts'],layer.get('tid'))
            if valid_step(step):
                layer_objects[id(layer)]=layer
                selected_steps[id(step)]=step
        for e in kernels:
            launch=cpu.get(e.get('args',{}).get('correlation'))
            if launch is None:continue
            layer=layer_index.find(launch['ts'],launch.get('tid'))
            if layer is None or id(layer)not in layer_objects:continue
            scope=component_index.find(launch['ts'],launch.get('tid'))
            name=e['name'];scope_name=scope['name']if scope else ''
            if 'expert.routed'in scope_name and ('marlin'in name.lower()if hardware=='A100'else'deep_gemm::'in name):
                bucket='expert_gemm';inventory[name]+=1
            elif 'deep_ep::'in name and ('dispatch'in name or'combine'in name):
                bucket='expert_exchange'
            else:continue
            by_layer[id(layer)][bucket].append((e['ts'],e['ts']+e['dur']))
        count=len(layer_objects)
    assert count>0 and selected_steps and inventory,(path,count)
    totals={key:sum(union_us(values.get(key,[]))for values in by_layer.values())/1000
        for key in ['expert_gemm','expert_exchange']}
    lo=min(e['ts']for e in selected_steps.values());hi=max(e['ts']+e['dur']for e in selected_steps.values())
    boundary=[]
    if arm=='pp2':
        for event in kernels:
            if'nccl'not in event['name'].lower():continue
            a,b=max(lo,event['ts']),min(hi,event['ts']+event['dur'])
            if a<b:boundary.append((a,b))
    result=dict(hardware=hardware,model=model,phase=phase,arm=arm,trace=str(path),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected_forwards=len(selected_steps),
        actual_layer_invocations=count,layers_per_rank=layers_per_rank,
        expert_gemm_ms_per_layer=totals['expert_gemm']/count,
        expert_exchange_ms_per_layer=totals['expert_exchange']/count,
        pp_communication_ms_per_layer=union_us(boundary)/1000/count,
        expert_gemm_ms_per_card_cohort=totals['expert_gemm'],
        expert_exchange_ms_per_card_cohort=totals['expert_exchange'],
        pp_communication_ms_per_card_window=union_us(boundary)/1000,
        unit='GPU ms/card/actual layer invocation; PP communication is NCCL residency including wait')
    return result,dict(**result,expert_kernel_inventory=dict(inventory))


def main():
    out=ROOT/'data';out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for arm,run in [('ep2','qwen122_ep2_pp2_r7'),('pp2','qwen122_pp2_two_slots_r8')]:
        folder=EXP/'ep2_pp2_qwen_a100_20261024/runs'/run/arm/'traces/decode'
        jobs.extend((p,'A100','122B','decode',arm)for p in sorted(folder.glob('*.trace.json.gz')))
    for arm in ['ep2','pp2']:
        folder=EXP/'ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7'/arm/'traces/moderate'
        jobs.extend((p,'H100','122B','decode',arm)for p in sorted(folder.glob('*.trace.json.gz')))
        for model,base,subfolder in [
            ('35B',EXP/'ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3','moderate_prefill'),
            ('122B',EXP/'ep2_pp2_qwen_h100_20261025/runs/h100_prefill_components_r1','moderate')]:
            folder=base/arm/'traces'/subfolder
            jobs.extend((p,'H100',model,'prefill',arm)for p in sorted(folder.glob('*.trace.json.gz')))
    assert len(jobs)==16,len(jobs)
    rows=[]
    for index,job in enumerate(jobs,1):
        row,detail=extract(*job);rows.append(row)
        (out/f'rank_{index:02d}.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(progress=f'{index}/{len(jobs)}',**{k:row[k]for k in ['hardware','model','phase','arm','actual_layer_invocations','expert_gemm_ms_per_layer','expert_exchange_ms_per_layer','pp_communication_ms_per_layer']})),flush=True)
    write_csv(out/'components_per_rank.csv',rows)
    grouped=[]
    for hw,model,phase,arm in sorted({(r['hardware'],r['model'],r['phase'],r['arm'])for r in rows}):
        selected=[r for r in rows if(r['hardware'],r['model'],r['phase'],r['arm'])==(hw,model,phase,arm)]
        assert len(selected)==2
        grouped.append(dict(hardware=hw,model=model,phase=phase,arm=arm,
            **{k:statistics.mean(r[k]for r in selected)for k in selected[0]if k.endswith('_per_layer')or k.endswith('_cohort')or k.endswith('_window')}))
    write_csv(out/'components.csv',grouped)
    (out/'extraction_validation.json').write_text(json.dumps(dict(passed=True,raw_traces=16,
        cells=8,expert_scope='routed Expert GEMM kernels only, selected using CPU module correlation or actual native Graph',
        normalization='actual layer calls; equal rank mean',pp_communication='NCCL GPU residency within selected CPU window, includes wait',
        gpu_experiments_launched=False),indent=2)+'\n')


if __name__=='__main__':main()
