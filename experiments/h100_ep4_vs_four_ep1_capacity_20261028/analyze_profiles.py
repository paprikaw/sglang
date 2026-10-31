"""Separate profiler diagnostics: correlate actual Graph GPU kernels."""
import argparse,collections,csv,gzip,hashlib,json,re,statistics
from pathlib import Path

def analyze(run):
    rows=[];inventory=[]
    for path in sorted((run/'results').glob('*/traces/decode/**/*.gz')):
        with gzip.open(path,'rt') as f:events=json.load(f)['traceEvents']
        kernels=[e for e in events if e.get('cat')=='kernel']
        launches=[e for e in events if e.get('name')=='cudaGraphLaunch']
        steps=[e for e in events if e.get('name','').startswith('step[DECODE')]
        samples=[];names=collections.Counter();counts=collections.Counter()
        for launch in launches:
            selected=[e for e in kernels if e.get('args',{}).get('correlation')==launch.get('args',{}).get('correlation')]
            if not selected:continue
            assert all(e.get('args',{}).get('graph id') for e in selected)
            enclosing=[e for e in steps if (e.get('pid'),e.get('tid'))==(launch.get('pid'),launch.get('tid')) and e['ts']<=launch['ts']<=e['ts']+e.get('dur',0)]
            step=min(enclosing,key=lambda e:e['dur'])['name'] if enclosing else ''
            batch=re.search(r'\bbs=(\d+)',step)
            def total(fn):return sum(e['dur'] for e in selected if fn(e['name']))/1000
            samples.append(dict(batch=int(batch[1]) if batch else None,
                graph_gpu_span_ms=(max(e['ts']+e['dur'] for e in selected)-min(e['ts'] for e in selected))/1000,
                kernel_sum_ms=total(lambda n:True),
                routed_gemm_sum_ms=total(lambda n:n=='fused_moe_kernel' or 'deep_gemm::' in n and '(deep_gemm::GemmType)2' in n),
                deepep_sum_ms=total(lambda n:'deep_ep::' in n),
                attention_full_sum_ms=total(lambda n:'FlashAttn' in n),
                gdn_recurrent_sum_ms=total(lambda n:'fused_recurrent_gated_delta_rule' in n)))
            for e in selected:names[e['name']]+=e['dur']/1000;counts[e['name']]+=1
        case=path.relative_to(run/'results').parts[0]
        row=dict(case=case,trace=str(path),graph_replays=len(samples),actual_batch_sizes=sorted(set(s['batch'] for s in samples if s['batch'] is not None)))
        for k in ['graph_gpu_span_ms','kernel_sum_ms','routed_gemm_sum_ms','deepep_sum_ms','attention_full_sum_ms','gdn_recurrent_sum_ms']:
            row[k]=statistics.mean(s[k] for s in samples) if samples else None
        rows.append(row);inventory.append(dict(trace=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),samples=samples,kernel_duration_ms=dict(names),kernel_counts=dict(counts)))
    out=run/'analysis';out.mkdir(exist_ok=True)
    if rows:
        with (out/'decode_graph.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'decode_graph.json').write_text(json.dumps(dict(rows=rows,inventory=inventory,performance_sample=False,
        limitations=['Separate profiler samples, not formal TPOT.','Only correlated actual Graph GPU kernels; excludes first-token Extend.','GPU kernel sums may overlap and communication kernels include waiting; sums are not exclusive critical-path shares.','Routed GEMM includes Triton fused_moe_kernel or native DeepGEMM grouped GEMM; excludes activation, routing, packing, shared Expert.']),indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run)
