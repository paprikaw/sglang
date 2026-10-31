"""Preserve small reproducible 122B artifacts and idempotent report indexes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
ID = 'qwen122b_h100_latest_ep_pp_decode_20261027'
DEST = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def copy(p,q): q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def index(p,text):
    a,b=f'<!-- begin {ID} -->',f'<!-- end {ID} -->'
    block=a+'\n'+text+'\n'+b
    old=p.read_text()
    new=old[:old.index(a)]+block+old[old.index(b)+len(b):] if a in old else old.rstrip()+'\n\n'+block+'\n'
    if new!=old:p.write_text(new)

def main():
    d=read(ROOT/'report/results.json')
    assert len(d['metrics'])==12 and all(r['audit']['guard_and_cleanup_verified'] for r in d['runs'])
    DEST.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'report').iterdir():
        if p.is_file():copy(p,DEST/'report'/p.name)
    for p in ROOT.glob('*.py'):copy(p,DEST/'scripts'/p.name)
    for name in ['gpu_qualification.json','source_cpu_qualification.json','native_cache_cpu_tests_r2.log']:
        copy(ROOT/name,DEST/'qualification'/name)
    raw={s['path']:s for r in d['runs'] for s in r['sources']}
    for r in d['runs']:
        p=ROOT/'analysis'/Path(r['run']).name/'results.json'
        copy(p,DEST/'analysis'/p.parent.name/p.name)
        copy(p.parent/'visibility_receipts.json',DEST/'analysis'/p.parent.name/'visibility_receipts.json')
    patterns=['config.json','dashboard_spec.json','launch_manifest.json','preparation.json','diagnosis.json',
        'cpu_qualification.json','decode_overlap_cpu_qualification.json','kv_checkpoint_discovery.json','gpu_qualification.json',
        'task/*.json','dashboard_guard/checks/*.json','code/*.py','*/qualification/*.json',
        '*/cleanup.json','*/completed.json','*/workloads/*/*.json','*/workloads/*/overlap_qualification/*.json',
        '*/workloads/*/traces/decode/manifest.json']
    attempts=[]
    for run in sorted((ROOT/'runs').iterdir()):
        if not run.is_dir():continue
        paths=set()
        for pattern in patterns:paths.update(run.glob(pattern))
        for p in sorted(paths):
            assert p.name!='admin.key'
            copy(p,DEST/'runs'/run.name/p.relative_to(run))
            raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='run_evidence')
        attempts.append(dict(run=str(run),progress=read(run/'task/progress.json'),
            included_in_comparison=str(run) in [r['run'] for r in d['runs']],
            diagnosis=read(run/'diagnosis.json') if (run/'diagnosis.json').exists() else None))
    cfg=d['runs'][0]['config'];source=Path(cfg['engine_root'])
    for name in ['decode_overlap.py','prefill_overlap.py']:
        p=source/'python/sglang/srt/layers/moe'/name
        copy(p,DEST/'source'/name)
        raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='implementation')
    dependencies=[('h100_pp_lookahead_workloads_20261027','analyze.py'),
                  ('h100_pp_expert_fragmentation_20261026','analyze.py'),
                  ('pp_ep_steady_decode_20261026','analyze.py')]
    for parent,name in dependencies:
        p=ROOT.parent/parent/name;copy(p,DEST/'analysis_dependencies'/parent/name)
        raw[str(p)]=dict(path=str(p),sha256=sha(p),kind='analysis_dependency')
    save(DEST/'raw_sources.json',list(raw.values()))
    old=read(DEST/'manifest.json') if (DEST/'manifest.json').exists() else {}
    branch=subprocess.check_output(['git','-C',str(source),'branch','--show-current'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    assert not dirty
    manifest=dict(id=ID,title='122B FP8 H100 EP2 communication overlap versus PP2 lookahead Decode',
        status='complete',created_at=old.get('created_at',datetime.now(timezone.utc).isoformat()),source_root=str(ROOT),
        engine=dict(name='sglang',worktree=str(source),branch=branch,commit=cfg['engine_commit'],
                    source_tag=cfg['engine_tag'],dirty=False,old_implementation=False),
        model=dict(path=cfg['model_path'],revision=cfg['model_revision'],precision=cfg['precision'],layers=48),
        deployment=dict(node=cfg['node'],slurm_job=33646188,gpus=[0,1],link='NVLink',
            launch_manifests=[r['run']+'/launch_manifest.json' for r in d['runs']],gateway_route=None,
            retired_after_measurement=True),
        workload=dict(protocol=cfg['protocol'],formal_cohorts_per_point=5,requests=[8,16,32],
                      prefix_tokens_per_request=4096,new_tokens_per_request=1,output_tokens_per_request=128),
        gpu_qualification=read(ROOT/'gpu_qualification.json'),metrics=d['metrics'],components=d['components'],
        comparisons=d['comparisons'],output_stability=d['output_stability'],attempts=attempts,
        raw_sources='raw_sources.json',report='report/REPORT.md')
    save(DEST/'manifest.json',manifest)
    results='; '.join(f"{r['workload']}: PP vs overlapping EP {r['pp_vs_overlap_ep_pct']:+.2f}%, EP overlap vs native {r['ep_overlap_vs_native_pct']:+.2f}% throughput" for r in d['comparisons'])
    description=('2×H100178 GPU0/1 NVLink, complete Qwen3.5-122B-A10B native FP8/DeepGEMM. '
        'B8/B16/B32 repeated and B32 diverse; identical prior prompts, real GPUcached4096 + new1 → output128. '
        'EP2 AttentionDP2/TP1 and 48 layers/rank versus PP2 24+24 layers and two fixed B/2 slots. '
        'Both BF16 KV/residuals and FP32 GDN; BF16-only Triton kernel substitution is inapplicable to this model. '
        'Common native redundant-prefix-lock skip and 1.875GiB KV/rank. '
        'EP native/shared-expert overlap and optional two-chunk MoE overlap versus PP policy9 early send/lookahead. '
        'Five profiler-off cohorts/point, independent rank profiles, 768 real-weight layer comparisons, '
        '23 CPU ownership/admission tests plus 12 native state-cache tests. '
        'Exact input/cache/output/Graph checks, canonical CPU weight-loader receipts and owned cleanup audited. '
        'Exploratory failures and prepared-only configurations remain excluded and preserved. '+results+'.')
    text=f'### {ID}\n\n{description}\n\n[Report]({DEST}/report/REPORT.md) · [Figure]({DEST}/report/decode_comparison.png)\n\nSnapshot: {DEST}'
    (DEST/'summary.md').write_text('# 122B FP8 EP2 / PP2 Decode\n\n'+description+'\n\n[Report](report/REPORT.md)\n')
    hashes=[f'{sha(p)}  {p.relative_to(DEST)}' for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (DEST/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
    for line in hashes:
        h,n=line.split('  ',1);assert sha(DEST/n)==h
    index(Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),text)
    index(ROOT.parents[1]/'experiments/experiment_summary/README.md',text)
    print(json.dumps(dict(archive=str(DEST),files=len(hashes),comparisons=d['comparisons']),indent=2))

if __name__=='__main__':main()
