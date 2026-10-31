"""CPU qualification + exact freeze only; root launches through Guard later."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from common import GUARD,MODES,PYTHON,DEEP,NATIVE,NATIVE_SHA,SHIM,save,sha


def prepare(run):
    run=Path(run).resolve()
    if run.exists():raise FileExistsError('Immutable run already exists')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        env=env,capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    if sha(NATIVE)!=NATIVE_SHA:raise RuntimeError('Real v7 native changed')
    # No CUDA import, node probing, model loading or GPU work in preparation.
    run.mkdir(parents=True)
    nodes=['spartan-gpgpu065','spartan-gpgpu167']
    cfg=dict(id=f'{ROOT.name}-{run.name}',nodes=nodes,groups={'lifecycle':{'nodes':nodes}},
        python=PYTHON,master_port=30923,worker_timeout_s=60,modes=list(MODES),
        local_tokens=12,world=8,hidden=5120,experts=160,topk=8,capacity=256,dtype='BF16',
        local_experts=20,num_qps_per_rank=20,registered_gpus=8,actual_gpus=8,
        graph_count=1,measured_graph_replays=3,model_loaded=False,expert_compute=False,attention_compute=False,
        purpose='Qualify fresh external GPU timing event spans in captured DeepEP dispatch/combine, no CUPTI collector')
    save(run/'config.json',cfg)
    save(run/'cpu_tests.json',dict(passed=True,gpu_execution=False,stdout=tests.stdout,stderr=tests.stderr))
    controller=ROOT/'runtime/controller.py'
    command=['python3',str(controller),'--run',str(run)]
    dependencies=list((DEEP/'deep_ep').rglob('*.py'))+[NATIVE,SHIM]
    source_common=ROOT.parent/'glm47_semantic_bank_fourway_20261023_v7/runtime/common.py'
    frozen=list((ROOT/'runtime').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+dependencies+[
        Path(__file__),run/'config.json',run/'cpu_tests.json',source_common]
    frozen=sorted(set(p.resolve() for p in frozen))
    spec=dict(schema_version=1,id=cfg['id'],title='DeepEP EP8 Graph: captured external GPU timing events',
        engine='deepep',protocol='task-events-v2',template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard'],
        port=54153,max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=60,
        runtime_dir=str(run/'dashboard_guard'),gpu_telemetry=dict(enabled=True,poll_interval_s=10,timeout_s=10),
        workload=dict(kind='synthetic_identity_expert_communication',model_loaded=False,expert_compute=False),
        model_cache=dict(policy='communication_only',benchmark='deepep_identity_expert',
            reason='Pure BF16 identity dispatch/combine with captured external timing events, no model/Attention/Expert compute and no CUPTI',
            entrypoint=dict(path=str(controller),sha256=sha(controller)),command=command),
        frozen_inputs=[dict(path=str(p),sha256=sha(p)) for p in frozen],
        tasks=[dict(id='lifecycle',label='DeepEP fresh per-replay dispatch/combine GPU timing events',
            root=str(run/'task'),expected=len(MODES),nodes=nodes,controller_script=str(controller))])
    sys.path.insert(0,str(GUARD.parent))
    import model_cache_adapter
    model_cache_adapter.validate_contract(spec,launch=True)
    model_cache_adapter.validate_command(spec,command)
    save(run/'dashboard_spec.json',spec)
    result=subprocess.run(['python3',str(GUARD),'prepare','--spec',str(run/'dashboard_spec.json')],
        env=env,capture_output=True,text=True)
    save(run/'dashboard_prepare.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    print(json.dumps(dict(prepared=True,run=str(run),spec=str(run/'dashboard_spec.json'),command=command,
        cpu_tests='passed',frozen_inputs=len(frozen),gpu_launched=False,nodes=nodes,modes=list(MODES))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    prepare(p.parse_args().run)
