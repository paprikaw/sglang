"""Archive only public provenance, small metrics, and code after verified cleanup."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
ID='qwen35b_h100_pd_mix_20261030'

def read(p):return json.loads(p.read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def desc(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--human',type=Path,required=True);ap.add_argument('--snapshot',type=Path,required=True)
    a=ap.parse_args();human=a.human.resolve();snapshot=a.snapshot.resolve()
    assert (human/'REPORT.md').is_file()
    runs=[]
    for name in a.runs:
        run=ROOT/'runs'/name;cfg=read(run/'config.json');spec=read(run/'dashboard_spec.json')
        progress=read(run/'task/progress.json');cleanup=read(run/'task/cleanup.json')
        posts=[read(p) for p in (run/'dashboard_guard/checks').glob('*-post*.json')]
        assert len(posts)==1 and posts[0]['passed'] and posts[0]['scientific_success'],name
        assert cleanup['passed'] and cleanup['gpu_idle'] and progress['stage']=='complete',name
        assert progress['valid_points']==cfg['expected_points'],name
        for d in spec['frozen_inputs']:assert desc(Path(d['path']))['sha256']==d['sha256'],d['path']
        dst=human/'run_manifests'/name;dst.mkdir(parents=True,exist_ok=True)
        for rel in ('config.json','dashboard_spec.json','launch_manifest.json','kv_checkpoint_discovery.json',
                    'topology.txt','task/cleanup.json','task/progress.json'):
            p=run/rel;target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        for p in (run/'dashboard_guard/checks').glob('*-post*.json'):shutil.copyfile(p,dst/p.name)
        for p in (run/'code').glob('*.py'):
            target=dst/'code'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        for p in (run/'evidence').glob('*/*.json'):
            if p.name in ('variable_batch_qualification.json','controlled_qualification.json','actual_admission.json','cleanup.json','server_info.json'):
                target=dst/'evidence'/p.relative_to(run/'evidence');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        raw=[desc(p) for p in sorted(run.glob('results/*/*/formal.r*.json'))]
        save(dst/'raw_trace_sources.json',raw)
        runs.append(dict(id=name,node=cfg['node'],job=cfg['job_id'],source=str(run),
            config=desc(run/'config.json'),points=cfg['expected_points'],formal_rounds=len(raw),
            pp_microbatch=cfg['pp_stream_microbatch'],controlled_qualification=cfg.get('controlled_qualification',False),
            phase_isolation=cfg.get('phase_isolation',False),guard_postpassed=True,
            cleanup_verified=True,usage_logs=[desc(Path(u['path'])) for r in cfg['layouts'] for u in r['usage_logs']]))
    for p in ROOT.glob('*.md'):shutil.copyfile(p,human/p.name)
    for p in ROOT.glob('*.py'):
        target=human/'analysis_code'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    state_path=WS/'experiments/cluster_backfill_20261020/state.json';state=read(state_path)
    nodes={node:state['nodes'][node] for node in sorted({r['node'] for r in runs})}
    assert all(d['state']=='benchmark' and d['reason']=='verified_live_worker' for d in nodes.values()),nodes
    recovery=dict(observed_utc=datetime.now(timezone.utc).isoformat(),source=str(state_path),
                  coordinator_timestamp=state['timestamp'],nodes=nodes,foreground_cleanup_verified=True)
    save(human/'RESOURCE_RECOVERY.json',recovery)
    cfg=read(ROOT/'runs'/a.runs[0]/'config.json')
    manifest=dict(id=ID,title='Qwen35B Prefill/Decode mixture and resident EP/PP pool opportunity',
        created_at=datetime.now(timezone.utc).isoformat(),report_date_local='2026-10-30',status='complete',
        implementation=dict(engine='sglang',worktree=cfg['engine_root'],commit=cfg['engine_commit'],
            immutable_source_tag=cfg['engine_tag'],dirty_at_freeze=False,runtime_adapters='run_manifests/*/code'),
        model=dict(path=cfg['model_path'],weights='BF16',kv='BF16',gdn='FP32'),
        testbed=dict(gpu='H100 80GB',gpus_per_candidate=4,nodes=list(nodes),cross_node_layouts=False),
        workload=dict(source='Deterministic unique-token synthetic prompts; frozen request generation and arrival rules',
            cache='P=0 cold prompts, flush before each complete trace; no HiCache/CPU restore',
            protocol='Full request sticky placement; three complete profiler-off repeats; arithmetic means',
            resizing=False,phase_state_migration=False),
        deployment=dict(temporary_benchmark=True,gateway_modified=False,serving_endpoint_retained=False),
        runs=runs,prepared_but_not_launched=['011_controlled_r2'],
        reused_evidence=str(human/'EXISTING_EVIDENCE.md'),resource_recovery=desc(human/'RESOURCE_RECOVERY.json'),
        decision=read(human/'decision.json'),artifacts=dict(report=str(human/'REPORT.md'),snapshot=str(snapshot),
            metrics=str(human/'full_metrics.csv'),raw_trace_sources='run_manifests/*/raw_trace_sources.json'))
    save(human/'manifest.json',manifest)
    if snapshot.exists():assert read(snapshot/'manifest.json')['id']==ID
    shutil.copytree(human,snapshot,dirs_exist_ok=True)
    inventory=[]
    for p in sorted(snapshot.rglob('*')):
        if p.is_file() and p.name!='snapshot_inventory.json':
            relative=p.relative_to(snapshot);source=human/relative
            assert desc(p)['sha256']==desc(source)['sha256']
            assert p.name not in ('admin.key','lease.json','binding.json') and p.suffix!='.bin'
            inventory.append(dict(path=str(relative),sha256=desc(p)['sha256'],bytes=p.stat().st_size))
    save(snapshot/'snapshot_inventory.json',dict(id=ID,verified_files=len(inventory),files=inventory))
    print(json.dumps(dict(snapshot=str(snapshot),verified_files=len(inventory),runs=runs),indent=2))

if __name__=='__main__':main()
