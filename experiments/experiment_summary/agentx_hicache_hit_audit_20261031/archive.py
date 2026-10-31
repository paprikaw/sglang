"""Archive only this cache-forensics topic; preserve other running topics."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
IDENTIFIER = 'qwen35b_h100_agentx_hicache_hit_audit_20261031'
STORE = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / IDENTIFIER
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    if source.suffix == '.md':
        def absolute(match):
            link = match.group(1)
            if link.startswith(('/', '#')) or re.match(r'\w+://', link):
                return match.group(0)
            return '](' + str((source.parent/link).resolve()) + ')'
        target.write_text(re.sub(r'\]\(([^)]+)\)',absolute,source.read_text()))
    else:
        shutil.copyfile(source,target)
    return dict(source=str(source),source_sha256=sha(source),snapshot=target.name,snapshot_sha256=sha(target))


def update_registry(entry):
    with REGISTRY.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        original=stream.read()
        start,end=f'<!-- begin {IDENTIFIER} -->',f'<!-- end {IDENTIFIER} -->'
        if start in original:
            left,rest=original.split(start,1)
            _,right=rest.split(end,1)
            updated=left+entry+right
        else:
            updated=original.rstrip()+'\n\n'+entry+'\n'
        stream.seek(0)
        stream.write(updated)
        stream.truncate()


def main():
    audit=json.loads((HERE/'audit.json').read_text())
    assert audit['cpu_only'] and len(audit['verified_episodes'])==18
    assert audit['request_count']==26163 and audit['paired_c128_count']==8208
    STORE.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in ('.py','.md','.csv','.json','.log'):
            artifacts.append(copy(source,STORE/source.name))
    artifacts.append(copy(HERE/'REPORT.md',STORE/'summary.md'))
    manifest=dict(id=IDENTIFIER,title='AgentX total HiCache hit differences across colocated EP4, PP2 and single-GPU pools',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete_cpu_forensic_analysis_no_gpu_launch',
        implementation=dict(engine='SGLang E5 historical frozen engine',
            worktree=str(HERE.parents[2]/'.engine-worktrees/agentx-gdn-pd-hicache-r5-20261031'),
            commit=audit['engine_commit'],source_tag='serving/sglang/qwen35b/agentx-gdn-pd-candidate-r5-20261031',
            source_file_sha256=audit['source_files'],runtime='Frozen run adapters included by hash; engine not edited',
            replay='V3, fixed 18 accepted colocated episodes; later PD/E7/E8/V4 excluded'),
        deployment=dict(historical_only=True,launch_sources=[s for s in audit['source_files'] if s['path'].endswith('launch_manifest.json')]),
        model='Qwen3.6-35B-A3B, BF16 weights and KV, FP32 GDN',
        testbed=dict(nodes=['spartan-gpgpu011','spartan-gpgpu171'],gpus=8,gpu_type='H100 80GB',
            layouts=['2xEP4 Attention TP1/DP4','4xPP2 TP1','8xsingle TP1']),
        workload=dict(name='AgentX selected eight source families',tracks=18,calls_per_copy=171,
            roots=[8,128],copies_per_family=[1,16],repeats=3,accepted_calls=26163,
            replay='Finite closed loop, sticky conversation owner, cache-isolated clones, fixed source lengths',
            initial_cache='Native flush before each formal episode; retained during episode'),
        parameters=audit['cache_policy'],
        provenance=str(HERE/'audit.json'),artifacts=artifacts,key_results=audit['summary'],
        limitations=audit['limits'])
    (STORE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    entry=f'''<!-- begin {IDENTIFIER} -->
### {IDENTIFIER}

Snapshot: {STORE}

CPU-only E5/V3 colocated cache audit: 18 accepted episodes, 26163 requests, 8208 paired C128 identities. GPU+host accounting verified per request; explicit HiCache size overrides ratio, and shared budget splits into unequal KV/GDN capacities. EP4/PP2 gap localized to subagent calls and two hot owners per repeat. Exact miss reasons unavailable; capacity/state eviction and scheduling not causally separated. No engine edits or GPU launch.

Report: {HERE/'REPORT.md'}

<!-- end {IDENTIFIER} -->'''
    update_registry(entry)
    index=HERE.parent/'README.md'
    with index.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        original=stream.read()
        marker=f'<!-- {IDENTIFIER} -->'
        if marker not in original:
            anchor='| AgentX workload 画像与回放方法 |'
            assert original.count(anchor)==1
            line=f'| AgentX：HiCache 总命中为何不同 | 同一 E5/V3 的 18 轮逐请求核对；32 GB 实际拆分、KV/GDN 容量、子请求热点与整段前缀丢失 | [专项核查](agentx_hicache_hit_audit_20261031/REPORT.md) {marker} |\n'
            updated=original.replace(anchor,line+anchor,1)
            stream.seek(0)
            stream.write(updated)
            stream.truncate()
    print(json.dumps(dict(snapshot=str(STORE),artifacts=len(artifacts),index=str(index))))


if __name__=='__main__':
    main()
