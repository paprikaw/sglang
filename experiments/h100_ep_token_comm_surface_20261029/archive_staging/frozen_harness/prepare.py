"""Freeze a four-shape EP2/4/8 communication grid on existing H100 bookings."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import (ROOT,WORKSPACE,DEEP,DEEP_COMMIT,DEEP_TAG,NATIVE_SO,NATIVE_SHA,
    PYTHON,TRACE_SHIM,TRACE_SHIM_SHA,GUARD,GUARD_SCRIPTS,NODES,JOBS,save,sha,environment)
from geometry import TOKENS,GLOBAL_CHUNK,SHAPES,workloads,validate_grid


def prepare(run):
    run=run.resolve()
    assert not run.exists(), 'Use a fresh run ID'
    assert sha(NATIVE_SO)==NATIVE_SHA and sha(TRACE_SHIM)==TRACE_SHIM_SHA
    assert subprocess.check_output(['git','-C',str(DEEP),'rev-parse','HEAD'],text=True).strip()==DEEP_COMMIT
    assert not subprocess.check_output(['git','-C',str(DEEP),'status','--porcelain'],text=True).strip()
    tests=validate_grid()
    assert int(environment([0,1])['NVSHMEM_QP_DEPTH']) == 4096
    assert int(environment([0,1])['NVSHMEM_QP_DEPTH']) >= (GLOBAL_CHUNK//2+1)*2
    phases=[]
    groups=[]
    for repeat in (1,2):
        for model in (SHAPES if repeat==1 else SHAPES[::-1]):
            local=[]
            for world,node in ((2,NODES[0]),(4,NODES[1])) if repeat==1 else ((4,NODES[0]),(2,NODES[1])):
                phase=dict(id=f'r{repeat}_{model["id"]}_ep{world}',round=repeat,
                    layout=f'ep{world}',model=model['id'],label=model['label'],hidden=model['hidden'],
                    experts=model['experts'],world=world,nodes=[node],
                    placement=[dict(rank=r,node=node,gpu=r) for r in range(world)],
                    capacity=GLOBAL_CHUNK//world,allow_nvlink=True,pair_only=False,pcie_threshold=0,
                    transport='intra_node_NVLink',ce_enabled=False,skip_local_drain=False,
                    port=19801+len(phases),workloads=workloads())
                phases.append(phase)
                local.append(phase['id'])
            cross=dict(id=f'r{repeat}_{model["id"]}_ep8',round=repeat,layout='ep8',
                model=model['id'],label=model['label'],hidden=model['hidden'],experts=model['experts'],
                world=8,nodes=list(NODES),placement=[dict(rank=r,node=NODES[r//4],gpu=r%4) for r in range(8)],
                capacity=GLOBAL_CHUNK//8,allow_nvlink=True,pair_only=False,pcie_threshold=0,
                transport='intra_node_NVLink_plus_inter_node_RDMA',ce_enabled=False,skip_local_drain=False,
                port=19801+len(phases),workloads=workloads())
            phases.append(cross)
            # Disjoint node-local worlds run concurrently; EP8 then owns both nodes.
            groups.extend([local,[cross['id']]])
    cfg=dict(id='h100_ep_token_comm_surface_20261029_'+run.name,nodes=list(NODES),booked_jobs=dict(zip(NODES,JOBS)),
        phases=phases,phase_groups=groups,cpus_per_node_step=64,warmup=10,iterations=30,token_sweep=list(TOKENS),
        dtype='bfloat16',topk=8,global_chunk_limit=GLOBAL_CHUNK,nvshmem_qp_depth=4096,
        shapes=list(SHAPES),model_loaded=False,expert_compute=False,attention_compute=False,
        native=dict(path=str(NATIVE_SO),sha256=NATIVE_SHA),
        token_definition='total original unique input tokens across the EP group; top8 expansion excluded from this count',
        comparison='identical global original tokens, expert routes and 2048-token global chunk boundaries across EP2/4/8',
        graph_policy='each point has independent eager and CUDA Graph timing of the complete identical D/C sequence',
        presentation=dict(selected_event_1based=1,aggregation='none',rank_selection='max aligned total span; D/C from that same rank',rounds_separate=True),
        chunking_reason='qualified NVSHMEM QP_DEPTH4096 bounds source capacity to1024 per EP2 rank; larger queue16384 failed IBRC QP creation; keep the same global2048-token chunks for every degree')
    run.mkdir(parents=True)
    harness=run/'frozen/harness'
    shutil.copytree(ROOT,harness,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(run/'config.json',cfg)
    save(run/'cpu_tests.json',tests)
    controller=harness/'controller.py'
    command=['python3',str(controller),'--run',str(run)]
    nvlib=Path(PYTHON).parents[1]/'lib/python3.12/site-packages/nvidia/nvshmem/lib'
    dependencies=[NATIVE_SO,TRACE_SHIM,*DEEP.glob('deep_ep/*.py'),nvlib/'libnvshmem_host.so.3',nvlib/'nvshmem_transport_ibgda.so.3']
    frozen=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [*sorted(harness.glob('*.py')),run/'config.json',*dependencies]]
    assert len({f['path'] for f in frozen})==len(frozen)
    spec=dict(schema_version=1,id=cfg['id'],title='H100 EP degree versus global tokens: four activation geometries, absolute raw milliseconds',
        engine='deepep',protocol='task-events-v2',template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','guard'],
        gpu_telemetry=dict(enabled=False,poll_interval_s=10,timeout_s=10),port=25047,
        max_active_nodes=2,cleanup_wait_s=90,runtime_dir=str(run/'dashboard_guard'),
        gpu_registry_dir=str(WORKSPACE.parents[2]/'.experiment_guard/gpu_registry'),
        sources=[dict(worktree=str(DEEP),commit=DEEP_COMMIT,tag=DEEP_TAG,role='deepep_base_source')],
        frozen_inputs=frozen,
        model_cache=dict(policy='communication_only',benchmark='deepep_identity_expert',
            reason='Synthetic BF16 identity return; no model weights, Expert GEMM, Attention or KV cache',
            entrypoint=dict(path=str(controller),sha256=sha(controller)),command=command),
        workload=dict(kind='synthetic_identity_expert_communication',model_loaded=False,expert_compute=False),
        tasks=[dict(id='local',label='EP2/4/8, 4 shapes, 13 sizes, eager/Graph, 2 distinct rounds',root=str(run/'task'),
            expected=len(phases),nodes=list(NODES),candidate_nodes=list(NODES),controller_script=str(controller),
            progress_file='progress.json',cleanup_file='cleanup.json')])
    save(run/'frozen_inputs.json',dict(files=frozen))
    save(run/'dashboard_spec.json',spec)
    save(run/'task/progress.json',dict(timestamp=time.time(),stage='queued',valid_points=0,
        note=f'{len(phases)*len(workloads())} communication points; event1/raw, no averages'))
    sys.path.insert(0,str(GUARD_SCRIPTS))
    import model_cache_adapter
    model_cache_adapter.validate_contract(spec,launch=True)
    result=subprocess.run(['python3',str(GUARD),'prepare','--spec',str(run/'dashboard_spec.json')],capture_output=True,text=True)
    save(run/'dashboard_prepare.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stdout+result.stderr
    launch=['python3',str(GUARD),'run','--spec',str(run/'dashboard_spec.json'),'--resource-choice','auto','--exit-after-run','--',*command]
    save(run/'launch.json',launch)
    return dict(prepared=True,run=str(run),phases=len(phases),points=len(phases)*len(workloads()),launch=launch)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().run)))
