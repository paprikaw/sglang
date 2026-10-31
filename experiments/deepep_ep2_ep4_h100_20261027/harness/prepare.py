"""Freeze one booked H100 node, EP2/EP4 equal-global-data communication test."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import ROOT,WORKSPACE,DEEP,DEEP_COMMIT,DEEP_TAG,NATIVE_SO,NATIVE_SHA,PYTHON,TRACE_SHIM,TRACE_SHIM_SHA,GUARD,GUARD_SCRIPTS,save,sha,environment
from geometry import TOKENS,workloads,validate_grid


def prepare(run,node,job):
    run=run.resolve();assert not run.exists(),'Use a fresh run ID'
    assert sha(NATIVE_SO)==NATIVE_SHA and sha(TRACE_SHIM)==TRACE_SHIM_SHA
    assert subprocess.check_output(['git','-C',str(DEEP),'rev-parse','HEAD'],text=True).strip()==DEEP_COMMIT
    assert not subprocess.check_output(['git','-C',str(DEEP),'status','--porcelain'],text=True).strip()
    tests=validate_grid()
    assert int(environment([0,1])["NVSHMEM_QP_DEPTH"]) >= (1024+1)*2
    phases=[]
    definitions=[('ep2_nvlink',[0,1],True),('ep4_nvlink',[0,1,2,3],True),
                 ('ep4_rdma',[0,1,2,3],False),('ep2_rdma',[0,2],False)]
    for repeat in (1,2):
        for label,gpus,nvlink in (definitions if repeat==1 else definitions[::-1]):
            phases.append(dict(id=f'r{repeat}_{label}',round=repeat,layout=label,model='glm',
                hidden=5120,experts=160,world=len(gpus),nodes=[node],
                placement=[dict(rank=r,node=node,gpu=g) for r,g in enumerate(gpus)],
                capacity=1024,allow_nvlink=nvlink,pair_only=False,pcie_threshold=0,
                transport='all_local_NVLink' if nvlink else 'all_remote_RDMA',
                ce_enabled=False,skip_local_drain=False,port=18481+len(phases),workloads=workloads()))
    cfg=dict(id=run.name,nodes=[node],booked_job_id=job,phases=phases,warmup=10,iterations=30,
        capacity=1024,nvshmem_qp_depth=4096,token_sweep=list(TOKENS),dtype='bfloat16',hidden=5120,experts=160,topk=8,
        model_loaded=False,expert_compute=False,attention_compute=False,
        native=dict(path=str(NATIVE_SO),sha256=NATIVE_SHA),
        source_A100_reference='experiments/deepep_ep2_ep4_a100_20261027/runs/a100065_ep2_ep4_r4/analysis_v1',
        topology_policy='H100 fully connected NVLink enabled for EP2/EP4; pair_only disabled; RDMA-only controls separate',
        comparison='same global source tokens/IDs/Expert routes/hidden/dtype/topk; local rows halve from EP2 to EP4',
        graph_policy='small 8/16/32 both eager and Graph; large256/1024/2048 eager; all use LL BF16 to isolate message size')
    run.mkdir(parents=True)
    harness=run/'frozen/harness'
    shutil.copytree(ROOT,harness,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(run/'config.json',cfg);save(run/'cpu_tests.json',tests)
    controller=harness/'controller.py';command=['python3',str(controller),'--run',str(run)]
    dependencies=[NATIVE_SO,TRACE_SHIM,*DEEP.glob('deep_ep/*.py')]
    nvlib=Path(PYTHON).parents[1]/'lib/python3.12/site-packages/nvidia/nvshmem/lib'
    dependencies += [nvlib/'libnvshmem_host.so.3',nvlib/'nvshmem_transport_ibgda.so.3']
    frozen=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [*sorted(harness.glob('*.py')),run/'config.json',*dependencies]]
    assert len({f['path'] for f in frozen})==len(frozen)
    spec=dict(schema_version=1,id=cfg['id'],title='H100 DeepEP EP2 to EP4: equal total data, small and large messages',
        engine='deepep',protocol='task-events-v2',template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','guard'],
        gpu_telemetry=dict(enabled=False,poll_interval_s=10,timeout_s=10),
        port=24911,max_active_nodes=1,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        gpu_registry_dir=str(WORKSPACE.parents[2]/'.experiment_guard/gpu_registry'),
        sources=[dict(worktree=str(DEEP),commit=DEEP_COMMIT,tag=DEEP_TAG,role='deepep_base_source')],
        frozen_inputs=frozen,
        model_cache=dict(policy='communication_only',benchmark='deepep_identity_expert',
            reason='Synthetic identity return only; no model, Expert GEMM, Attention or KV cache',
            entrypoint=dict(path=str(controller),sha256=sha(controller)),command=command),
        workload=dict(kind='synthetic_identity_expert_communication',model_loaded=False,expert_compute=False),
        tasks=[dict(id='local',label='EP2/EP4; two orders; 6 sizes; Graph small-message control',
            root=str(run/'task'),expected=len(phases),nodes=[node],candidate_nodes=[node],
            controller_script=str(controller),progress_file='progress.json',cleanup_file='cleanup.json')])
    save(run/'frozen_inputs.json',dict(files=frozen));save(run/'dashboard_spec.json',spec)
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='queued',valid_points=0,note='72 communication points; no model loaded'))
    sys.path.insert(0,str(GUARD_SCRIPTS))
    import model_cache_adapter
    model_cache_adapter.validate_contract(spec,launch=True)
    result=subprocess.run(['python3',str(GUARD),'prepare','--spec',str(run/'dashboard_spec.json')],capture_output=True,text=True)
    save(run/'dashboard_prepare.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stdout+result.stderr
    launch=['python3',str(GUARD),'run','--spec',str(run/'dashboard_spec.json'),
            '--resource-choice','auto','--exit-after-run','--',*command]
    save(run/'launch.json',launch)
    return dict(prepared=True,run=str(run),phases=len(phases),points=len(phases)*9,launch=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--node',default='spartan-gpgpu178');p.add_argument('--job',default='33646188')
    a=p.parse_args();print(json.dumps(prepare(a.run,a.node,a.job)))
