"""CPU-only preparation: real tensor selection, source freeze, bounded contract."""
import argparse,ast,hashlib,json,os,shutil,socket,struct,subprocess,time
from pathlib import Path
from common import ROOT,WORKSPACE,ENGINE,PYTHON,GUARD_SCRIPTS,MODELS,PLAN,save,sha

OLD=WORKSPACE/'experiments/expert_doubling_compare_20261022'
LEGACY=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/harness')
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def bootstrap():
    guard=ROOT/'guard'
    if guard.exists():raise RuntimeError('Guard snapshot already exists')
    subprocess.run(['git','clone','--no-hardlinks','--quiet',str(OLD/'guard'),str(guard)],check=True)
    shutil.copyfile(ROOT/'h100_surface_policy.py',GUARD_SCRIPTS/'h100_surface_policy.py')
    p=GUARD_SCRIPTS/'expert_layer_policy.py';s=p.read_text()
    s=s.replace("def validate(spec):\n", "def validate(spec):\n    if spec.get('model_cache', {}).get('benchmark') == 'h100_four_model_expert_surface_v1':\n        import h100_surface_policy\n        return h100_surface_policy.validate(spec)\n",1)
    s=s.replace("def verify_result(contract, binding):\n", "def verify_result(contract, binding):\n    if contract.get('benchmark') == 'h100_four_model_expert_surface_v1':\n        import h100_surface_policy\n        return h100_surface_policy.verify_result(contract, binding)\n",1);p.write_text(s)
    p=GUARD_SCRIPTS/'model_cache_adapter.py';s=p.read_text()
    old='        env["CUDA_VISIBLE_DEVICES"] = "0"\n        os.execvpe(command[0], command, env)'
    new='''        if contract.get("benchmark") == "h100_four_model_expert_surface_v1":
            import h100_surface_policy
            env["CUDA_VISIBLE_DEVICES"] = h100_surface_policy.worker_gpu(
                contract, node, env.get("SURFACE_WORKER_ID"), env.get("CUDA_VISIBLE_DEVICES"))
        else:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        os.execvpe(command[0], command, env)'''
    assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
    # Retain the tested PID/start-tick ownership and post-exit GPU evidence.
    source=OLD/'hy3/model_node_r2.py';tree=ast.parse(source.read_text());names={
      'process_record','enable_subreaper','track_descendants','alive_owned','owned_metadata','stop_owned','gpu_cleanup_query'}
    functions=[ast.get_source_segment(source.read_text(),n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    (ROOT/'remote_safety.py').write_text('import ctypes,hashlib,os,signal,subprocess,time\nfrom pathlib import Path\n\n'+'\n\n'.join(functions)+'\n')
    print('BOOTSTRAP_READY',flush=True)
def selection(name,m):
    model=Path(m['path']);cfg=json.loads((model/'config.json').read_text());idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    prefix=('model.language_model' if name.startswith('qwen') else 'model')+f".layers.{m['layer']}.mlp.experts."
    names=[]
    if name=='qwen35b':names=[prefix+'gate_up_proj',prefix+'down_proj']
    else:
        fields=('qweight','qzeros','scales') if name=='glm47' else ('weight','weight_scale_inv') if name=='qwen122b' else ('weight',)
        names=[f'{prefix}{e}.{p}.{field}' for e in range(m['resident']) for p in ('gate_proj','up_proj','down_proj') for field in fields]
    headers={};files={};tensors={};old={}
    if name in ('hy3','glm47'):
        old=json.loads((OLD/('glm' if name=='glm47' else 'hy3')/'runs/r1/selection.json').read_text())['tensors']
    for key in names:
        file=idx[key]
        if file not in headers:
            path=model/file;st=path.stat()
            with path.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
            headers[file]=json.loads(raw)
            files[file]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,header_sha256=hashlib.sha256(raw).hexdigest())
        h=headers[file][key];tensors[key]=dict(file=file,dtype=h['dtype'],shape=h['shape'],bytes=h['data_offsets'][1]-h['data_offsets'][0])
        if key in old:tensors[key]['sha256']=old[key]['sha256']
    return dict(model_path=str(model),layer=m['layer'],resident=m['resident'],hidden=m['hidden'],intermediate=m['intermediate'],
       expert_ids=list(range(m['resident'])),active_experts=m['experts'],config_sha256=sha(model/'config.json'),
       index_sha256=sha(model/'model.safetensors.index.json'),source_files=files,tensors=tensors,total_tensor_bytes=sum(x['bytes'] for x in tensors.values()))
def prepare(attempt):
    run=ROOT/'runs'/attempt
    if run.exists():raise RuntimeError('Never overwrite an existing attempt')
    run.mkdir(parents=True);(run/'profile').mkdir()
    # Read-only scheduling inventory: no new allocations and no per-node probes.
    live=subprocess.check_output(['squeue','--me','--states=RUNNING','--noheader','--format=%i|%j|%N|%b'],text=True)
    jobs={line.split('|')[2].strip():line.split('|')[0].strip() for line in live.splitlines() if 'H100' in line}
    nodes=sorted(jobs)
    assert len(nodes)==2,('Expected the two current H100 bookings',jobs)
    selected={name:selection(name,m) for name,m in MODELS.items()}
    save(run/'selection.json',selected);save(run/'measurement_plan.json',PLAN)
    workers=[]
    for rep,node in enumerate(nodes):
        for index,name in enumerate(MODELS):
            gpu=(index+2*rep)%4
            workers.append(dict(id=f'{name}_rep{rep+1}',model=name,node=node,gpu=gpu,replica=rep+1))
    save(run/'workers.json',workers)
    expected=sum(sum(e<=t for t in PLAN['tokens'] for e in m['experts'])*2 for m in MODELS.values())
    config=dict(id=f'h100_expert_surface_4models_20261029_{attempt}',nodes=nodes,jobs=jobs,workers=workers,expected_per_node=expected,models=MODELS,plan=PLAN)
    save(run/'config.json',config)
    inputs=list(ROOT.glob('*.py'))+[run/x for x in ['selection.json','measurement_plan.json','workers.json','config.json']]
    inputs.extend([LEGACY/x for x in ['profile_entry.py','profile_utils.py','common.py']])
    for s in selected.values():inputs.extend(Path(s['model_path'])/f for f in ['config.json','model.safetensors.index.json'])
    # Freeze the actual operator/quantization/activation sources as well as Git identity.
    for relative in ['python/sglang/srt/layers/moe/moe_runner/deep_gemm.py','python/sglang/srt/layers/moe/moe_runner/humming.py','python/sglang/srt/layers/quantization/humming.py','python/sglang/srt/layers/quantization/fp8_kernel.py']:
        p=ENGINE/relative
        if p.is_file():inputs.append(p)
    inputs.extend(p for p in (ENGINE/'python/sglang/srt/layers/deep_gemm_wrapper').rglob('*.py'))
    frozen=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(set(inputs))]
    desc=lambda p:next(x for x in frozen if x['path']==str(p.resolve()))
    c=dict(policy='expert_layer_only',reason='User explicitly requests four real single Expert layers, one H100 per worker, all available H100s in parallel; no complete model or CPU/KV bank mutations',
      benchmark='h100_four_model_expert_surface_v1',selection=desc(run/'selection.json'),measurement_plan=desc(run/'measurement_plan.json'),workers=desc(run/'workers.json'),
      entrypoint=desc(ROOT/'controller.py'),worker=desc(ROOT/'worker.py'),command=['python3',str(ROOT/'controller.py'),'--run',str(run)],worker_command=[PYTHON,str(ROOT/'worker.py'),'--run',str(run)])
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    sources=[]
    for source,tag in [(ENGINE,'experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated'),(ROOT/'guard','experiment/h100-four-model-expert-surface-20261029-cpu-validated')]:
        commit=git(source,'rev-parse','HEAD')
        assert git(source,'rev-parse',tag+'^{commit}')==commit
        assert not git(source,'status','--porcelain','--untracked-files=no')
        sources.append(dict(worktree=str(source),commit=commit,tag=tag))
    spec=dict(schema_version=1,id=config['id'],title='H100: GLM4.7 / HY3 / Qwen35B / Qwen122B Expert T-by-E surfaces',engine='sglang',protocol='task-events-v2',template='parallel-tasks-v5',
       features=['parallel-tasks','task-progress','task-details','gpu-tab','gpu-telemetry','guard'],
       workload=dict(kind='h100_four_model_expert_layer_compute',model_loaded=False,expert_compute=True,attention_compute=False,communication=False,gpu_count=8),
       gpu_telemetry=dict(enabled=True,poll_interval_s=10,timeout_s=10),port=port,max_active_nodes=2,gpu_lease_ttl_s=90,cleanup_wait_s=60,
       runtime_dir=str(run/'dashboard_guard'),gpu_registry_dir='/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry',
       resource_priority='normal',sources=sources,model_cache=c,frozen_inputs=frozen,
       tasks=[dict(id=node,label=f'{node}: four single-GPU models',root=str(run/'tasks'/node),nodes=[node],expected=expected,controller_script=str(ROOT/'controller.py')) for node in nodes])
    save(run/'dashboard_spec.json',spec)
    for node in nodes:save(run/'tasks'/node/'progress.json',dict(timestamp=time.time(),stage='waiting',valid_points=0,expected_points=expected))
    save(ROOT/'latest_run.json',dict(run=str(run),port=port,expected_cells=expected*2))
    print(json.dumps(dict(run=str(run),nodes=nodes,gpus=8,cells=expected*2,port=port)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',action='store_true');p.add_argument('--attempt',default='r1');a=p.parse_args()
    bootstrap() if a.bootstrap else prepare(a.attempt)
