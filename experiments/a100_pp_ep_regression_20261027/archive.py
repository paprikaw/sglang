"""Archive validated foreground results without credentials or large traces."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
EXPERIMENT_ID='qwen35b_a100_nvlink_regression_20261027'
ARCHIVE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/EXPERIMENT_ID


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def register(path,text):
    s=path.read_text();begin=f'<!-- begin {EXPERIMENT_ID} -->';end=f'<!-- end {EXPERIMENT_ID} -->'
    block=begin+'\n'+text.strip()+'\n'+end
    if begin in s:
        x=s.index(begin);y=s.index(end,x)+len(end);s=s[:x]+block+s[y:]
    else:s=s.rstrip()+'\n\n'+block+'\n'
    path.write_text(s)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);a=ap.parse_args()
    run=a.run.absolute();result=read(ROOT/'analysis'/run.name/'result.json');cfg=read(run/'config.json')
    assert len(result['metrics'])==15 and len(result['components'])==30 and result['audit']['guard_postcheck']
    assert all(r['passed'] for r in result['output_agreement'])
    records=[]
    for line in (run/'supervisor.log').read_text().splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    assert any(r.get('gpu_registry_state')=='released' and r.get('postcheck')=='passed' for r in records)
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    def copy(p,dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(p)==sha(dest),f'Immutable archive differs: {dest}'
        else:shutil.copyfile(p,dest)
    for folder in ('analysis','report'):
        for p in (ROOT/folder/run.name).glob('*'):
            if p.is_file():copy(p,ARCHIVE/folder/p.name)
    for p in ROOT.glob('*.py'):copy(p,ARCHIVE/'tools'/p.name)
    benchmark=ROOT.parent/'experiment_summary/benchmark'
    for name in ('build_benchmark.py','REPORT_TEMPLATE.md'):
        copy(benchmark/name,ARCHIVE/'benchmark_extension_tools'/name)
    compact=[run/n for n in ['config.json','launch_manifest.json','dashboard_spec.json','topology.txt',
        'topology_qualification.json','input_compatibility.json','task/progress.json','task/cleanup.json','cache/node.json','supervisor.log']]
    compact+=list((run/'code').glob('*.py'))+list((run/'dashboard_guard/checks').glob('*-post.json'))
    for arm in cfg['arm_order']:
        compact += [run/arm/'cleanup.json',run/arm/'completed.json']
        compact += list((run/arm/'qualification').glob('*.json'))
        for c in cfg['workloads']:
            root=run/arm/'workloads'/c['id']
            compact += [root/'protocol.json',root/'completed.json',root/'output_agreement.json',root/'server_info.json']
            compact += list(root.glob('metrics.*.json'))+list(root.glob('qualification.*.json'))
            compact += list((root/'traces').glob('*/manifest.json'))
    for p in compact:
        assert p.is_file() and p.name!='admin.key',p
        copy(p,ARCHIVE/'run'/p.relative_to(run))
    raw=[]
    for p in run.rglob('*'):
        if p.is_file() and (p.name.endswith('.trace.json.gz') or p.name.startswith('cohort.') or
                           p.parent.name=='transfer_audit' or p.name in ('controller.log','ep2.log','pp2.log')):
            raw.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    (ARCHIVE/'raw_sources.json').write_text(json.dumps(raw,indent=2)+'\n')
    attempts=[]
    reasons={'a10035_nvlink_r1':'Public GLM-only Guard rejected Qwen profile before acquisition',
             'a10035_nvlink_r2':'ANSI topology-parser failure before model launch; physical pair was NV12',
             'a10035_nvlink_r3':'Smaller GDN pool lost exact GPU-prefix residency during B32 warmup; B8 diagnostic excluded'}
    for name,reason in reasons.items():
        previous=ROOT/'runs'/name
        if not previous.exists():continue
        p=previous/'task/progress.json';c=previous/'task/cleanup.json'
        attempts.append(dict(run=str(previous),reason=reason,used_for_final_performance=False,
            progress=read(p) if p.exists() else None,cleanup=read(c) if c.exists() else None,
            completed_metric_sources=[dict(path=str(p),sha256=sha(p)) for p in previous.glob('*/workloads/*/metrics.*.json')]))
    (ARCHIVE/'diagnostic_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    (ARCHIVE/'summary.md').write_text(f'''# A100 NVLink EP2/PP2 regression

Qwen3.6-35B-A3B BF16 on spartan-gpgpu069 physical GPU0/1, verified NV12. Attention DP2/TP1 + EP2 versus static two-slot PP2/TP1/EP1. Shared native Triton Expert kernels; DeepEP normal BF16, eager execution. Three Decode workloads (B8 repeated, B32 repeated/diverse), three cold Prefill workloads (B8 Q128/Q4096, B32 Q4096 diverse). Five profiler-off cohorts/configuration and independent profiles: 15 timing points, 30 rank profiles. Decode PP0 and PP9 preserve complete generated output sequences and actual payload hashes. Required canonical weight-cache postcheck, owned cleanup and lease release pass.

[Readable report](report/REPORT.md) · [Throughput figure](report/throughput.png) · [Metrics](analysis/metrics.csv) · [Components](analysis/components.csv).

Initial attempts stopped before model launch (GLM-only Guard profile; ANSI topology-parser issue). A smaller GDN-state budget then failed the exact-prefix check during B32 warmup; its completed B8 data remain diagnostic and are not pooled. The final run uses the validated state-retention budget for both layouts. GPU-native A100/H100 execution paths differ and are recorded; hardware-only causal attribution is not claimed.
''')
    created_at=read(ARCHIVE/'manifest.json')['created_at'] if (ARCHIVE/'manifest.json').exists() else datetime.now(timezone.utc).isoformat()
    manifest=dict(id=EXPERIMENT_ID,title='A100 NVLink EP2 versus native and lookahead PP2 regression points',
        created_at=created_at,status='complete',implementation=read(run/'launch_manifest.json'),
        deployment=dict(node=cfg['node'],job_id=cfg['job_id'],physical_gpus=[0,1],link='NV12',retired=True),
        workload=cfg['workloads'],protocol=cfg['protocol'],engine_parameters=cfg,
        key_results=result['comparisons'],audit=result['audit'],raw_sources='raw_sources.json',
        cross_layout_output_agreement=result['cross_layout_output_agreement'],
        source_root=str(ROOT),artifacts=[dict(path=str(p.relative_to(ARCHIVE)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(ARCHIVE.rglob('*')) if p.is_file() and p.name not in ('manifest.json','SHA256SUMS')])
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ARCHIVE/'SHA256SUMS').write_text('\n'.join(f'{sha(p)}  {p.relative_to(ARCHIVE)}' for p in sorted(ARCHIVE.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')+'\n')
    headline='; '.join(f"{r['workload']} p{r['policy']}: PP/EP {r['pp_throughput']/r['ep_throughput']:.3f}x" for r in result['comparisons'])
    register(Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),f'''### {EXPERIMENT_ID}

Snapshot: {ARCHIVE}

- Qwen35B BF16; A100069 GPU0/1 verified NV12; SG8702697022. EP2 AttentionDP2/TP1 versus PP2/EP1/TP1,20 layers/stage,two B/2 slots. Shared native Triton Expert, DeepEP normal BF16; Graph/overlap schedule OFF.
- B8/B32 repeated and B32 diverse GPUcached4096+new1->128 Decode; cold B8 Q128/Q4096 and B32 diverse Q4096->1 Prefill. Five unprofiled cohorts/configuration,3warmups,15timing points and30rank profiles. {headline}.
- PP0/PP9 full-output equality, actual grouping and payload hashes, live lookahead receipts, independent eager Expert/D+C/Attention attribution; cross-layout output agreement reported separately. Canonical weight-cache/Guard postcheck, owned cleanup and lease release pass. Initial pre-model failures excluded and preserved in workspace.
- Report, metrics/components/comparisons CSV, PNG/PDF, frozen code/config/native descriptors, raw trace hashes and SHA256SUMS. Latest A100 hardware-native path, not a GPU-only controlled ablation of H100 kernels.
''')
    register(ROOT.parent/'experiment_summary/README.md',f'''## A100 NVLink 双卡回归补测（2026-10-27）

[报告]({ARCHIVE}/report/REPORT.md) · [吞吐图]({ARCHIVE}/report/throughput.png) · [指标]({ARCHIVE}/analysis/metrics.csv) · [Expert／Dispatch／Combine／Attention]({ARCHIVE}/analysis/components.csv)

Qwen35B BF16，A100069物理GPU0/1，租约内确认NV12。EP2与PP2同原生Triton Expert；3组Decode、3组Prefill，每配置5轮，另列PP0/PP9调度比较。15个计时点与30个rank profile，完整输出/通信载荷/分组校验，Guard清理和释放通过。{headline}。
''')
    print(json.dumps(dict(archive=str(ARCHIVE),files=len(list(ARCHIVE.rglob('*')))),indent=2))


if __name__=='__main__':main()
