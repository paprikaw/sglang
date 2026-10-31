"""Freeze a two-model, same-native CE/RDMA communication-only experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import (ROOT,WORKSPACE,DEEP,DEEP_COMMIT,DEEP_TAG,DEFAULT_RUN,
                    NODES,GUARD,GUARD_SCRIPTS,PYTHON,NATIVE_SO,NATIVE_SHA,
                    TRACE_SHIM,TRACE_SHIM_SHA,CAPACITY,save,sha)
from geometry import TOKENS,MODELS,validate_grid,cases
from ce_support import geometry
import build_native


def config(experiment_id):
    phases=[]
    for model,(hidden,experts) in MODELS.items():
        for location in ('local065','local167','cross8'):
            nodes=list(NODES) if location=='cross8' else [NODES[int(location=='local167')]]
            placement=[dict(rank=r,node=node,gpu=gpu)
                       for r,(node,gpu) in enumerate((n,g) for n in nodes for g in range(4))]
            for mode in ('rdma','ce'):
                world=len(placement)
                phases.append(dict(id=f'{model}_{location}_{mode}',model=model,hidden=hidden,
                     experts=experts,transport=mode,ce_enabled=mode=='ce',world=world,
                     nodes=nodes,placement=placement,capacity=CAPACITY,pair_only=True,
                     pcie_threshold=CAPACITY if mode=='ce' else 0,
                     dispatch_dedup=mode=='ce',matched_peer_order=mode=='ce',
                     skip_local_drain=mode=='ce' and world==4,local_notify=mode=='ce',
                     port=38101+len(phases),workloads=list(cases(world))))
    return dict(id=experiment_id,nodes=list(NODES),phases=phases,capacity=CAPACITY,
                warmup=3,iterations=3,cuda_graph=False,dtype='bfloat16',topk=8,
                model_loaded=False,attention_compute=False,expert_compute=False,
                token_sweep=list(TOKENS),deep_ep=dict(worktree=str(DEEP),commit=DEEP_COMMIT,tag=DEEP_TAG),
                native=dict(glm=dict(path=str(NATIVE_SO),sha256=NATIVE_SHA),
                            hy3='same host/other CUDA objects; frozen H4096 CE shape specialization'),
                timing='dispatch + receive hook + combine + receive hook; aligned rank max then median of 3',
                expected_nic=dict(vendor='0x15b3',device='0x101d',model='ConnectX-6 Dx'))


def cpu_tests():
    result=validate_grid()
    for experts in (160,192):
        for hidden in (4096,5120):
            for world in (4,8):
                g=geometry(world,CAPACITY,experts,hidden)
                assert g['frame_bytes'] >= g['header_bytes']+CAPACITY*(hidden*2+16+80)
                assert g['extra_bytes']%128==0
    assert len(list(cases(8)))==10 and len(list(cases(4)))==30
    result['ce_geometry_cases']=8
    return result


def prepare(run):
    run=run.resolve();assert not run.exists(),'Use a fresh immutable run ID'
    assert sha(NATIVE_SO)==NATIVE_SHA and sha(TRACE_SHIM)==TRACE_SHIM_SHA
    assert subprocess.check_output(['git','-C',str(DEEP),'rev-parse','HEAD'],text=True).strip()==DEEP_COMMIT
    assert not subprocess.check_output(['git','-C',str(DEEP),'status','--porcelain'],text=True).strip()
    tests=cpu_tests();cfg=config(run.name)
    staging=run.with_name('.'+run.name+'.prepare.'+str(os.getpid()))
    staging.mkdir()
    try:
        harness=staging/'frozen/harness'
        shutil.copytree(ROOT,harness,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        save(staging/'config.json',cfg);save(staging/'cpu_tests.json',tests)
        final_harness=run/'frozen/harness';controller=final_harness/'controller.py'
        command=['python3',str(controller),'--run',str(run)]
        frozen=[dict(path=str(final_harness/p.relative_to(harness)),sha256=sha(p))
                for p in sorted(harness.rglob('*')) if p.is_file()]
        frozen.append(dict(path=str(run/'config.json'),sha256=sha(staging/'config.json')))
        nvlib=Path(PYTHON).parents[1]/'lib/python3.12/site-packages/nvidia/nvshmem/lib'
        dependencies=[p for p in build_native.inputs() if p!=ROOT/'hy3_internode_ll.cu']
        dependencies += [TRACE_SHIM,nvlib/'libnvshmem_host.so.3',nvlib/'nvshmem_transport_ibgda.so.3']
        frozen += [dict(path=str(p.resolve()),sha256=sha(p)) for p in dict.fromkeys(dependencies)]
        assert len({r['path'] for r in frozen})==len(frozen)
        contract=dict(policy='communication_only',benchmark='deepep_identity_expert',
                      reason='Single-layer shapes, synthetic identity return only; CE and RDMA separate; no model or Expert/Attention compute',
                      entrypoint=dict(path=str(controller),sha256=sha(harness/'controller.py')),command=command)
        spec=dict(schema_version=1,id=cfg['id'],title='A100 GLM / Hy3 single-layer CE versus RDMA token sweep',
             engine='deepep',protocol='task-events-v2',template='parallel-tasks-v5',
             features=['parallel-tasks','task-progress','task-details','gpu-tab','guard'],
             gpu_telemetry=dict(enabled=False,poll_interval_s=10,timeout_s=10),
             port=24869,max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=90,
             runtime_dir=str(run/'dashboard_guard'),
             gpu_registry_dir=str(WORKSPACE.parents[2]/'.experiment_guard/gpu_registry'),
             sources=[dict(worktree=str(DEEP),commit=DEEP_COMMIT,tag=DEEP_TAG,role='deepep_base_source')],
             frozen_inputs=frozen,model_cache=contract,
             workload=dict(kind='synthetic_identity_expert_communication',model_loaded=False,expert_compute=False),
             tasks=[dict(id='local',label='Two models / two transports / two batch boundaries',
                         root=str(run/'task'),expected=len(cfg['phases']),nodes=list(NODES),candidate_nodes=list(NODES),
                         controller_script=str(controller),progress_file='progress.json',cleanup_file='cleanup.json')])
        save(staging/'frozen_inputs.json',dict(files=frozen));save(staging/'dashboard_spec.json',spec)
        save(staging/'task/progress.json',dict(timestamp=time.time(),stage='queued',valid_points=0,
             note='两种模型尺寸；每档3次；CE/RDMA；整批/半批与相同单次 token；无模型计算'))
        staging.rename(run)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise
    sys.path.insert(0,str(GUARD_SCRIPTS))
    import model_cache_adapter
    model_cache_adapter.validate_contract(spec,launch=True)
    subprocess.run(['python3',str(GUARD_SCRIPTS/'backfill_trigger.py')],check=True)
    done=subprocess.run(['python3',str(GUARD),'prepare','--spec',str(run/'dashboard_spec.json')],capture_output=True,text=True)
    save(run/'dashboard_prepare.json',dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr))
    assert done.returncode==0,done.stdout+done.stderr
    launch=['python3',str(GUARD),'run','--spec',str(run/'dashboard_spec.json'),
            '--resource-choice','auto','--exit-after-run','--',*command]
    save(run/'launch.json',launch)
    return dict(prepared=True,run=str(run),phases=len(cfg['phases']),formal_points=sum(len(p['workloads']) for p in cfg['phases']),launch=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    print(json.dumps(prepare(p.parse_args().run)))
