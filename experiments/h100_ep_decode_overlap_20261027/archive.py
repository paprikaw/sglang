"""Compact, checksummed, idempotently indexed Decode experiment snapshot."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
ID='qwen35b_h100_ep_decode_overlap_20261027'
DEST=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def index(p,text):
    a,b=f'<!-- begin {ID} -->',f'<!-- end {ID} -->';block=a+'\n'+text+'\n'+b
    old=p.read_text();new=old[:old.index(a)]+block+old[old.index(b)+len(b):] if a in old else old.rstrip()+'\n\n'+block+'\n'
    if old!=new:p.write_text(new)
def main():
    d=read(ROOT/'report/results.json');assert all(r['audit']['guard_and_cleanup_verified'] for r in d['runs'])
    DEST.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'report').iterdir():
        if p.is_file():copy(p,DEST/'report'/p.name)
    for p in ROOT.glob('*.py'):copy(p,DEST/'scripts'/p.name)
    for name in ['gpu_qualification.json','source_cpu_qualification.json','cpu_tests.log']:
        copy(ROOT/name,DEST/'qualification'/name)
    raw={s['path']:s for r in d['runs'] for s in r['sources']}
    for r in d['runs']:
        p=ROOT/'analysis'/Path(r['run']).name/'results.json';copy(p,DEST/'analysis'/p.parent.name/p.name)
    patterns=['config.json','dashboard_spec.json','launch_manifest.json','preparation.json','diagnosis.json',
        'cpu_qualification.json','decode_overlap_cpu_qualification.json','kv_checkpoint_discovery.json','gpu_qualification.json',
        'task/*.json','dashboard_guard/checks/*.json','code/*.py','*/qualification/*.json','*/ep_kernel_qualification/*.json',
        '*/cleanup.json','*/completed.json','*/workloads/*/*.json','*/workloads/*/overlap_qualification/*.json',
        '*/workloads/*/traces/decode/manifest.json']
    attempts=[]
    for run in sorted((ROOT/'runs').iterdir()):
        paths=set()
        for pattern in patterns:paths.update(run.glob(pattern))
        for p in sorted(paths):
            assert p.name!='admin.key';copy(p,DEST/'runs'/run.name/p.relative_to(run))
            raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='run_evidence')
        attempts.append(dict(run=str(run),progress=read(run/'task/progress.json'),
            included_in_comparison=str(run) in [r['run'] for r in d['runs']],
            diagnosis=read(run/'diagnosis.json') if (run/'diagnosis.json').exists() else None))
    cfg=d['runs'][0]['config']
    for name in ['decode_overlap.py','prefill_overlap.py']:
        p=Path(cfg['engine_root'])/'python/sglang/srt/layers/moe'/name
        copy(p,DEST/'source'/name);raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='implementation')
    for parent,name in [('h100_pp_lookahead_workloads_20261027','analyze.py'),('h100_pp_expert_fragmentation_20261026','analyze.py')]:
        p=ROOT.parent/parent/name;copy(p,DEST/'analysis_dependencies'/parent/name)
        raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='analysis_dependency')
    save(DEST/'raw_sources.json',list(raw.values()))
    old=read(DEST/'manifest.json') if (DEST/'manifest.json').exists() else {}
    manifest=dict(id=ID,title='H100 fixed-policy low-latency Decode EP overlap versus native EP and PP lookahead',
        status='complete',created_at=old.get('created_at',datetime.now(timezone.utc).isoformat()),source_root=str(ROOT),
        engine=dict(worktree=cfg['engine_root'],commit=cfg['engine_commit'],source_tag=cfg['engine_tag']),
        gpu_qualification=read(ROOT/'gpu_qualification.json'),metrics=d['metrics'],components=d['components'],
        comparisons=d['comparisons'],output_stability=d['output_stability'],attempts=attempts,
        raw_sources='raw_sources.json',report='report/REPORT.md')
    save(DEST/'manifest.json',manifest)
    results='; '.join(f"{r['workload']}: overlap vs native {r['throughput_gain_pct']:+.2f}% throughput" for r in d['comparisons'])
    description=('Same H100178 NVLink GPU0/1, full Qwen3.6-35B-A3B BF16. '
        'B8/B16/B32 repeated and B32 diverse, real GPUcached4096 + new1 -> output128. '
        'Static per-process Decode CUDA Graph policies, identical generic Expert kernel family. '
        'Native EP2, two-chunk overlapping EP2, B32 serial control, and PP2 with two fixed slots and policy9 lookahead. '
        'Five profiler-off cohorts after three warmups; independent two-rank profiles with actual communication/GEMM intersections. '
        'Attention/GDN whole, shared-expert overlap ON; no Prefill DeepGEMM SM budget carried into Decode. '
        'Every EP variant/workload validates both ranks all40 real-weight MoE layers. '
        'EP long-sequence cross-cohort variation is recorded, including native controls; PP fixed-group sequences must agree. '
        '23 CPU tests, exact cache/output/stream counts, actual graph replay receipts, frozen hashes, canonical CPU weight-cache '
        'postcheck, owned cleanup and lease release audited. Failed EP sequence-stability attempt and PP schema-warmup attempt excluded. '+results+'.')
    text=f'### {ID}\n\n{description}\n\n[Report]({DEST}/report/REPORT.md) · [Figure]({DEST}/report/decode_overlap.png)\n\nSnapshot: {DEST}'
    (DEST/'summary.md').write_text('# H100 Decode EP overlap\n\n'+description+'\n\n[Report](report/REPORT.md)\n')
    hashes=[f'{sha(p)}  {p.relative_to(DEST)}' for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (DEST/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
    for line in hashes:
        h,n=line.split('  ',1);assert sha(DEST/n)==h
    index(Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),text)
    index(ROOT.parents[1]/'experiments/experiment_summary/README.md',text)
    print(json.dumps(dict(archive=str(DEST),files=len(hashes),comparisons=d['comparisons']),indent=2))
if __name__=='__main__':main()
