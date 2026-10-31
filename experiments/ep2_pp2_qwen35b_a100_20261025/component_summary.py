"""Select actual Prefill/Decode invocations and retain whole-window PP boundaries."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from trace_analysis import union_us


def csv_write(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def summarize(run):
    out=run/'analysis';rows=[];boundaries=[]
    for arm in ('ep2','pp2'):
        for phase in ('prefill','decode'):
            directory=run/arm/'traces'/phase
            manifest=json.loads((directory/'manifest.json').read_text())
            assert not manifest['performance_sample'] and len(manifest['files'])==2
            files=[]
            for item in manifest['files']:
                path=Path(item['path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
                files.append(path)
            for path in files:
                derived=out/path.name.replace('.trace.json.gz','.json')
                d=json.loads(derived.read_text());assert d['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                selected=[r for r in d['invocations'] if ('DECODE' if phase=='decode' else 'EXTEND') in r['step']]
                assert selected
                layer_count=40 if arm=='ep2' else 20
                assert len({r['layer'] for r in selected})==layer_count
                assert len(selected)%layer_count==0
                if phase=='decode':assert all('bs=4 ' in r['step'] for r in selected)
                else:assert all('bs=1 toks=4096 ' in r['step'] for r in selected)
                names=('attention.full','attention.linear','expert.routed','expert.shared','communication.dispatch','communication.combine')
                row=dict(arm=arm,phase=phase,rank=path.name,actual_layer_invocations=len(selected),
                    actual_forward_count=len(selected)//layer_count,unit='ms per rank per actual layer invocation',
                    **{k:statistics.mean(r['gpu_component_union_ms'].get(k,0) for r in selected) for k in names})
                row['attention.total']=row['attention.full']+row['attention.linear']
                rows.append(row)
                if arm=='pp2':
                    with gzip.open(path,'rt') as f:trace=json.load(f)
                    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'nccl' in e.get('name','').lower()]
                    assert kernels and all('SendRecv' in e['name'] for e in kernels)
                    boundaries.append(dict(phase=phase,rank=path.name,unit='ms/rank for WHOLE separate profile window',
                        sendrecv_kernel_count=len(kernels),kernel_union_ms=union_us([(e['ts'],e['ts']+e['dur']) for e in kernels])/1000,
                        includes_gpu_wait=True,includes_initial_extend=phase=='decode',
                        interpretation='NCCL residency overlaps compute; not pure transfer time and not added to layer means'))
    means=[]
    for arm in ('ep2','pp2'):
        for phase in ('prefill','decode'):
            ranks=[r for r in rows if (r['arm'],r['phase'])==(arm,phase)];assert len(ranks)==2
            means.append(dict(arm=arm,phase=phase,rank_count=2,unit=ranks[0]['unit'],
                **{k:statistics.mean(r[k] for r in ranks) for k in names+('attention.total',)}))
    csv_write(out/'components_per_rank.csv',rows);csv_write(out/'components.csv',means)
    csv_write(out/'pp_boundary_residency.csv',boundaries)
    (out/'profile_validation.json').write_text(json.dumps(dict(passed=True,complete_exports=8,
        rank_rows=8,whole_model_coverage=True,decode_actual_batch=4,
        decode_forward_counts={r['rank']:r['actual_forward_count'] for r in rows if r['phase']=='decode'},
        limitations=['Independent profiler samples are not formal throughput or TPOT',
            'EP has 3 actual Decode forwards/rank; PP has 2/rank after excluding initial Extend',
            'Attention includes projections within original model annotation; full/linear weighted by actual layer invocations',
            'Component unions may overlap; PP SendRecv residency includes wait and overlaps compute',
            'Per-layer local microbatch/routing differs between EP and PP; not a same-route isolated Expert test']),indent=2)+'\n')
    print(json.dumps(dict(components=means,boundaries=boundaries),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    summarize(p.parse_args().run.resolve())
