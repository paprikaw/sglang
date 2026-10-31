"""Archive compact immutable evidence and index original large/raw artifacts."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
OUT=WS/'experiments/experiment_summary/agentx_pd_feasibility_20261030'
DATA=WS/'datasets/agentx_20261030'
ID='qwen35b_h100_agentx_pd_feasibility_20261030'
DEST=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    copied=[];refs=[]
    def reference(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    def copy(p,target):
        q=DEST/target;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        copied.append(dict(source=str(p),snapshot=target,sha256=reference(q)['sha256']))
    for name in ('REPORT.md','metrics.csv'):copy(OUT/name,name)
    copy(ROOT/'final_findings.json','final_findings.json')
    for p in sorted(ROOT.glob('*.py')):copy(p,'scripts/'+p.name)
    copy(ROOT/'PROTOCOL.md','PROTOCOL.md')
    copy(ROOT/'screening_windows.json','screening_windows.json')
    for p in sorted(ROOT.glob('*-statistics.json')):copy(p,'dataset/'+p.name)
    for p in sorted(ROOT.glob('*-sessions.csv')):copy(p,'dataset/'+p.name)
    copy(DATA/'manifest.json','dataset/download_manifest.json')
    copy(DATA/'harness_source.json','dataset/harness_source.json')
    conditions=[]
    for run in sorted((ROOT/'runs').iterdir()):
        cfg=json.loads((run/'config.json').read_text())
        conditions.append(dict(id=run.name,**{k:cfg.get(k) for k in ('node','model_path','engine_commit','family','mem_fraction_static','hicache_gb_per_rank','baseline_reference_run')}))
        for p in run.glob('analysis_audit.json'):copy(p,'runs/'+run.name+'/'+p.name)
        for p in run.glob('superseded.json'):copy(p,'runs/'+run.name+'/'+p.name)
        targets=[run/'config.json',run/'dashboard_spec.json',run/'launch_manifest.json']
        targets+=list(run.glob('results/**/*.json'))+list(run.glob('evidence/*/cleanup.json'))+list(run.glob('dashboard_guard/checks/*.json'))
        targets+=list(run.glob('cache/*.log'))+list(run.glob('code/*.py'))
        for p in sorted(set(targets)):
            if p.exists():refs.append(reference(p))
    (DEST/'original_artifacts.json').write_text(json.dumps(refs,indent=2)+'\n')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest.update(canonical_snapshot=str(DEST),source_root=str(ROOT),original_dataset_root=str(DATA),
        conditions=conditions,artifact_inventory=copied,original_artifact_index=reference(DEST/'original_artifacts.json'),
        gpu_budget=dict(nodes=['spartan-gpgpu011','spartan-gpgpu171'],gpus_per_node=4,gpu='H100 80GB',comparison_scope='Each qualification comparison planned within one node; baseline families on different nodes.'),
        data_scope='Official full and 256k datasets downloaded; custom short no-subagent screening only; no official AgentX score.',
        final_findings=str(DEST/'final_findings.json'),report=str(DEST/'REPORT.md'))
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reg=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker='<!-- begin '+ID+' -->'
    if marker not in reg.read_text():
        entry=f'''\n{marker}
### {ID}

Snapshot: {DEST}

- AgentX v1.0 official full and 256k repositories downloaded at fixed revisions, all original files preserved and JSONL LFS SHA256 checked. Full 393 sessions / 98,827 calls; 256k 393 / 68,266; original hashed block data, not prompt text. Official harness saved for inspection only.
- Qwen3.6-35B-A3B BF16, engine 0f2f1990994774472b1089d2b0682f0bf926a6b4, 4 H100 GPUs per node. Required HiCache enabled, 32 decimal GB host per rank, static memory fraction0.95, no artificial GPU KV token cap; session/DP affinity retained.
- Completed colocated baselines: 2xEP2 on011 and 2xPP2 on171, C1/C8 x3 rounds:4 points,480 formal requests,384 later turns,48,486 mean GPU cached tokens/later request; host hits0. EP2 mean TTFT/TPOT C1=134.38/5.99ms,C8=163.84/7.64ms; PP2=111.56/5.88ms and109.42/8.93ms. Different nodes and bounded no-subagent windows: no causal layout ranking, full AgentX score or saturation claim.
- Actual PD qualification blocked: after isolated Mooncake dependency supplied, both EP2 and PP2 Decode reject GDN/Mamba radix prefix caching. Zero PD performance points and no dynamic allocation gain established. Cache was not disabled to bypass the requirement; engine source unchanged.
- Original whole-run Guard postchecks remain failed; completed colocated arms separately audited for frozen hashes, all expected CPU-bank loader ranks, output lengths, arithmetic means and owned-process cleanup. Cancelled GPU-only/dependency trials retained and excluded. Both nodes returned to existing backfill.

Report: {OUT/'REPORT.md'}

<!-- end {ID} -->
'''
        with reg.open('a') as f:f.write(entry)
    print(DEST)
    print('snapshot_files',len(copied),'original_artifact_references',len(refs))

if __name__=='__main__':main()
