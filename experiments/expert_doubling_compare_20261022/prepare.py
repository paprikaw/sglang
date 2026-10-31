"""Freeze identical doubling grids under the existing exact one-layer contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parent
GUARD=ROOT/'guard'
GUARD_TAG='experiment/expert-doubling-contract-20261022-cpu-validated'
PLAN=dict(schema_version=1,tokens=[2**p for p in range(3,14)],
          active_experts=[2**p for p in range(6)],modes=['eager','cuda_graph'],
          samples_per_mode=30,routing='balanced',token_semantics='local_token_expert_rows')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def prepare(label,node):
    import importlib.util
    folder=ROOT/label
    module=importlib.util.spec_from_file_location('common',folder/'common.py')
    common=importlib.util.module_from_spec(module);module.loader.exec_module(common)
    run=folder/'runs/r1'
    if run.exists():raise RuntimeError('Run already exists; never overwrite evidence')
    old=ROOT.parent/('hy3_expert_surface_20261022/runs/r2' if label=='hy3'
                     else 'glm47_expert_weight_footprint_20261022/runs/r1')
    selection=json.loads((old/'selection.json').read_text())
    assert selection['model_path']==str(common.MODEL)
    assert selection['config_sha256']==sha(common.MODEL/'config.json')
    assert selection['index_sha256']==sha(common.MODEL/'model.safetensors.index.json')
    for filename,record in selection['source_files'].items():
        path=common.MODEL/filename;st=path.stat()
        assert st.st_size==record['size'] and st.st_mtime_ns==record['mtime_ns']
        with path.open('rb')as f:
            n=struct.unpack('<Q',f.read(8))[0];header=f.read(n)
        assert hashlib.sha256(header).hexdigest()==record['header_sha256']
    run.mkdir(parents=True);(run/'profile/task').mkdir(parents=True)
    save(run/'selection.json',selection);save(run/'measurement_plan.json',PLAN)
    sources=[dict(worktree=str(common.SG),commit=common.SG_COMMIT,tag=common.SG_TAG),
             dict(worktree=str(GUARD),commit=git(GUARD,'rev-parse','HEAD'),tag=GUARD_TAG)]
    for source in sources:
        assert git(source['worktree'],'rev-parse','HEAD')==source['commit']
        assert git(source['worktree'],'rev-parse',source['tag']+'^{commit}')==source['commit']
        assert git(source['worktree'],'cat-file','-t',source['tag'])=='tag'
        assert not git(source['worktree'],'status','--porcelain','--untracked-files=no')
    oldspec=json.loads((old/'dashboard_spec.json').read_text())
    paths=list(folder.glob('*.py'))+[ROOT/'prepare.py',run/'selection.json',run/'measurement_plan.json']
    # Retain the already validated external AWQ loader/reference and all kernel inputs.
    for desc in oldspec['frozen_inputs']:
        p=Path(desc['path'])
        if p.is_relative_to(common.SG) or not p.is_relative_to(ROOT.parent):
            paths.append(p)
    frozen=[dict(path=str(p),sha256=sha(p))for p in sorted(set(paths))]
    def desc(p):return next(x for x in frozen if x['path']==str(p))
    count=sum(e<=t for t in PLAN['tokens']for e in PLAN['active_experts'])*len(PLAN['modes'])
    contract=dict(policy='expert_layer_only',reason='User-requested common doubling T/E grid; same exact real layer inventory, one GPU, no model or bank mutation',
        benchmark=('hy3_bf16_doubling_experts' if label=='hy3'else'glm47_awq_doubling_experts'),
        model_path=str(common.MODEL),layer=selection['layer'],expert_ids=selection['expert_ids'],
        selection=desc(run/'selection.json'),measurement_plan=desc(run/'measurement_plan.json'),
        entrypoint=desc(folder/'controller_r2.py'),worker=desc(folder/'worker.py'),
        command=['python3',str(folder/'controller_r2.py'),'--run',str(run)],
        worker_command=[common.PYTHON,str(folder/'worker.py'),'--run',str(run)])
    with socket.socket()as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ident=f'{label}_expert_doubling_20261022_r1'
    spec=dict(schema_version=1,id=ident,title=f'{label.upper()} real expert layer: common doubling T/E grid',
        engine='sglang',protocol='task-events-v2',template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard'],
        workload=dict(kind='hy3_bf16_expert_layer_compute'if label=='hy3'else'glm47_awq_expert_layer_compute',model_loaded=False,
                      expert_compute=True,attention_compute=False,communication=False,gpu_count=1),
        gpu_telemetry=dict(enabled=True,poll_interval_s=10,timeout_s=10),port=port,max_active_nodes=1,
        gpu_lease_ttl_s=90,cleanup_wait_s=60,runtime_dir=str(run/'dashboard_guard'),
        gpu_registry_dir='/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry',
        sources=sources,model_cache=contract,frozen_inputs=frozen,
        tasks=[dict(id='profile',label=f'{label}: 63 shapes, 2 timing modes, 30 samples',
                    root=str(run/'profile/task'),expected=count,nodes=[node],controller_script=str(folder/'controller_r2.py'))])
    config=dict(id=ident,nodes=[node],groups=dict(profile=dict(nodes=[node])),
                cases=dict(profile=dict(nodes=[node],ep_capacity=8192)),model_cache=contract,expected_points=count)
    save(run/'config.json',config)
    spec['frozen_inputs'].append(dict(path=str(run/'config.json'),sha256=sha(run/'config.json')))
    save(run/'dashboard_spec.json',spec)
    save(run/'profile/task/progress.json',dict(timestamp=time.time(),stage='waiting',valid_points=0,expected_points=count,note='Frozen; no GPU measurement yet'))
    print(json.dumps(dict(run=str(run),rows=count,selected_weight_bytes=selection['total_tensor_bytes'],port=port)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('hy3','glm'),required=True)
    p.add_argument('--node',default='spartan-gpgpu065');a=p.parse_args();prepare(a.model,a.node)
