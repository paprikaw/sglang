"""Select only real HY3 expert weights and freeze a bounded compute run."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

from common import ROOT, SG, SG_COMMIT, SG_TAG, MODEL, PYTHON, save, sha

GUARD=ROOT/'guard'
GUARD_TAG='experiment/hy3-expert-surface-contract-20261022-cpu-validated'
NODE='spartan-gpgpu065'


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def selection():
    names=[f'model.layers.1.mlp.experts.{e}.{p}.weight' for e in range(48)
           for p in ('gate_proj','up_proj','down_proj')]
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    result=dict(model_path=str(MODEL),layer=1,expert_ids=list(range(48)),
                config_sha256=sha(MODEL/'config.json'),
                index_sha256=sha(MODEL/'model.safetensors.index.json'),tensors={},source_files={})
    for filename in sorted({index[n] for n in names}):
        path=MODEL/filename;before=path.stat()
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];raw=f.read(length)
            header=json.loads(raw);base=8+length
            result['source_files'][filename]=dict(size=before.st_size,mtime_ns=before.st_mtime_ns,
                                                   header_sha256=hashlib.sha256(raw).hexdigest())
            for name in sorted(n for n in names if index[n]==filename):
                item=header[name];lo,hi=item['data_offsets'];left=hi-lo
                f.seek(base+lo);digest=hashlib.sha256()
                while left:
                    chunk=f.read(min(left,4*1024**2))
                    if not chunk:raise RuntimeError('Truncated tensor: '+name)
                    digest.update(chunk);left-=len(chunk)
                result['tensors'][name]=dict(file=filename,dtype=item['dtype'],shape=item['shape'],
                                            bytes=hi-lo,sha256=digest.hexdigest())
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result['total_tensor_bytes']=sum(v['bytes'] for v in result['tensors'].values())
    assert result['total_tensor_bytes']==1811939328 and len(result['tensors'])==144
    return result


def main(run):
    if run.exists():raise RuntimeError('Immutable run already exists')
    run.mkdir(parents=True);(run/'profile/task').mkdir(parents=True)
    save(run/'selection.json',selection())
    sources=[dict(worktree=str(SG),commit=SG_COMMIT,tag=SG_TAG),
             dict(worktree=str(GUARD),commit=git(GUARD,'rev-parse','HEAD'),tag=GUARD_TAG)]
    for source in sources:
        root=source['worktree']
        assert git(root,'rev-parse','HEAD')==source['commit']
        assert git(root,'rev-parse',source['tag']+'^{commit}')==source['commit']
        assert git(root,'cat-file','-t',source['tag'])=='tag'
        assert not git(root,'status','--porcelain','--untracked-files=no')
    paths=list(ROOT.glob('*.py'))+[run/'selection.json',MODEL/'config.json',MODEL/'model.safetensors.index.json']
    paths += [SG/'python'/p for p in (
        'sglang/srt/layers/moe/moe_runner/base.py',
        'sglang/srt/layers/moe/moe_runner/triton.py',
        'sglang/srt/layers/moe/moe_runner/bf16_deepep.py',
        'sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py',
        'sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py',
        'sglang/srt/layers/moe/token_dispatcher/deepep.py',
        'sglang/kernels/ops/moe/fused_moe_triton_kernels.py')]
    frozen=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))]
    def desc(p):return next(v for v in frozen if v['path']==str(p))
    contract=dict(policy='expert_layer_only',
        reason='User-requested HY3 T-by-active-expert surface; exact one-layer BF16 selection, no model, bank or timed communication',
        benchmark='hy3_bf16_local_experts',model_path=str(MODEL),layer=1,expert_ids=list(range(48)),
        selection=desc(run/'selection.json'),entrypoint=desc(ROOT/'controller_r2.py'),
        worker=desc(ROOT/'worker.py'),
        command=['python3',str(ROOT/'controller_r2.py'),'--run',str(run)],
        worker_command=[PYTHON,str(ROOT/'worker.py'),'--run',str(run)])
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    ident='hy3_expert_surface_20261022_'+run.name
    spec=dict(schema_version=1,id=ident,title='HY3 BF16 expert compute: T/E surface and matched EP8/PP2xEP4 local work',
        engine='sglang',protocol='task-events-v2',template='parallel-tasks-v5',
        features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard'],
        workload=dict(kind='hy3_bf16_expert_layer_compute',model_loaded=False,expert_compute=True,
                      attention_compute=False,communication=False,gpu_count=1),
        gpu_telemetry=dict(enabled=True,poll_interval_s=10,timeout_s=10),port=port,max_active_nodes=1,
        gpu_lease_ttl_s=90,cleanup_wait_s=60,runtime_dir=str(run/'dashboard_guard'),
        gpu_registry_dir='/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry',
        sources=sources,model_cache=contract,frozen_inputs=frozen,
        tasks=[dict(id='profile',label='32 T/E shapes + matched compute/layout controls; 112 rows',
                    root=str(run/'profile/task'),expected=112,nodes=[NODE],controller_script=str(ROOT/'controller_r2.py'))])
    config=dict(id=ident,nodes=[NODE],groups=dict(profile=dict(nodes=[NODE])),
                cases=dict(profile=dict(nodes=[NODE],ep_capacity=8192)),model_cache=contract,expected_points=112)
    save(run/'config.json',config)
    spec['frozen_inputs'].append(dict(path=str(run/'config.json'),sha256=sha(run/'config.json')))
    save(run/'dashboard_spec.json',spec)
    save(run/'profile/task/progress.json',dict(timestamp=time.time(),stage='waiting',valid_points=0,
         expected_points=112,note='Frozen; GPU measurement has not started'))
    print(json.dumps(dict(run=str(run),port=port,selected_GiB=1811939328/2**30,sources=sources),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/r1')
    main(p.parse_args().run.resolve())
