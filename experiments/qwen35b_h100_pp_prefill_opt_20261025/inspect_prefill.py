"""Postprocess independently sampled PP traces using the existing classifier."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def inspect(run):
    source=Path(__file__).parent.parent/'ep2_pp2_qwen35b_h100_20261025/prefill_attribution.py'
    spec=importlib.util.spec_from_file_location('qualified_prefill_classifier',source)
    classifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(classifier)
    rows=[];proofs=[];timelines=[]
    for directory in sorted((run/'pp2/traces').iterdir()):
        sample=json.loads((directory/'cohort.json').read_text())
        manifest=dict(requests=32,cached_context_tokens=89984,new_tokens_per_request=512)
        for path in sorted(directory.glob('*.trace.json.gz')):
            row,proof=classifier.inspect_trace(path,manifest,'pp2')
            row['policy']=sample.get('chunk_tokens',directory.name)
            rows.append(row);proofs.append(proof)
            with gzip.open(path,'rt') as handle:events=json.load(handle)['traceEvents']
            steps=sorted((x for x in events if x.get('cat')=='user_annotation'
                          and x.get('name','').startswith('step[EXTEND')),key=lambda x:x['ts'])
            launches={}
            for e in events:
                if e.get('cat') not in ('cuda_runtime','cuda_driver') or 'correlation' not in e.get('args',{}):continue
                matched=[(i,s) for i,s in enumerate(steps)
                         if (s.get('pid'),s.get('tid'))==(e.get('pid'),e.get('tid'))
                         and s['ts']<=e['ts']<=s['ts']+s['dur']]
                if matched:launches[e['args']['correlation']]=min(matched,key=lambda x:x[1]['dur'])[0]
            kernels=defaultdict(list)
            for e in events:
                if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in launches:
                    kernels[launches[e['args']['correlation']]].append(e)
            detail=[]
            for i,s in enumerate(steps):
                k=kernels[i]
                if not k:raise RuntimeError('Missing actual GPU work for a Prefill forward')
                detail.append(dict(index=i,requests=int(re.search(r'bs=(\d+)',s['name']).group(1)),
                    new_tokens=int(re.search(r'c_sq=(\d+)',s['name']).group(1)),
                    cpu_begin_us=s['ts'],cpu_duration_ms=s['dur']/1000,
                    gpu_begin_us=min(x['ts'] for x in k),
                    gpu_end_us=max(x['ts']+x['dur'] for x in k)))
            timelines.append(dict(policy=row['policy'],stage=row['stage'],steps=detail,trace=str(path)))
    out=run/'analysis';out.mkdir(exist_ok=True)
    with (out/'prefill_components.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'profile_inventory.json').write_text(json.dumps(dict(
        classifier=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        rows=rows,proofs=proofs,timelines=timelines),indent=2)+'\n')
    text=['# PP Prefill 分阶段诊断','',
          '单位为毫秒／每卡／整批 32 请求，历史缓存每条 89,984 token，新输入每条 512 token，输出 1 token。'
          '这是正式性能之后的独立 profiler，不混入吞吐统计。Attention 包含模块内投影；'
          'Expert 分路由和共享模块。内核累计时间可以重叠，不能相加当作请求耗时；'
          'NCCL 驻留包含等待，不能解释成纯载荷传输。','',
          '| 策略 | stage | 模型 forward 次数 | Attention | 路由 Expert | 共享 Expert | NCCL 驻留 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['policy']} | {r['stage']} | {r['actual_model_forwards']} | "
                    f"{r['attention_ms']:.2f} | {r['routed_expert_ms']:.2f} | {r['shared_expert_ms']:.2f} | {r['nccl_residence_ms']:.2f} |")
    text+=['','## 实际每次 forward 的新增输入组成','']
    for t in timelines:
        text.append(f"- {t['policy']}，stage {t['stage']}："+' + '.join(str(s['new_tokens']) for s in t['steps'])+' 个新 token。')
    text+=['','[逐卡完整 CSV](prefill_components.csv) · [逐内核归属、时间线与哈希](profile_inventory.json)','']
    (out/'PROFILE.md').write_text('\n'.join(text))
    print(json.dumps(dict(profile_rows=len(rows),report=str(out/'PROFILE.md'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();inspect(a.run.resolve())
