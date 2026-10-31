"""Separate PP2 and PP4 lanes after the EP lane completes."""
import ast,copy,json,shutil
import prepare
previous=prepare.ROOT/'runs/pp_171_r5'
for kind,version in [('pp','r6'),('ep','pp4_r1')]:
    run=prepare.prepare(kind,version)
    for p in (previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
    shutil.copy2(prepare.ROOT/'dynamic_pp_schema.py',run/'code/dynamic_pp_schema.py')
    shutil.copy2(prepare.ROOT/'dynamic_schema_validation.json',run/'dynamic_schema_validation.json')
    cfg=json.loads((run/'config.json').read_text())
    if kind=='pp':cfg['layouts']=[r for r in cfg['layouts'] if r['id']!='pp4']
    else:
        row=copy.deepcopy(next(r for r in json.loads((previous/'config.json').read_text())['layouts'] if r['id']=='pp4'))
        row['usage_logs']=[dict(id=cfg['id']+'.pp4.r0',path=str(run/'cache/pp4.r0.log'),expected_ranks=[f'PP{i} TP0' for i in range(4)])]
        cfg['layouts']=[row];cfg['model_cache']['on_miss']='build'
    cfg['expected_points']=10*len(cfg['layouts'])
    cfg['model_cache']['usage_logs']={cfg['node']:[u for row in cfg['layouts'] for u in row['usage_logs']]}
    cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000,pp_mem_fraction_static=.92,pp_decode_graph_max_bs=16,ep_control_mem_fraction_static=.94)
    cfg['protocol']+=' PP2 and PP4 run concurrently on nodes171/178 after EP completes. PP static memory fraction0.92 reserves native GDN/NCCL workspace; Decode Graph maxB16. PP2 typed dynamic CPU shape headers are retained independently until Gloo completion, with no blocking CPU send-wait in native CUDA retirement. Source-verified management reply List[bool].'
    prepare.save(run/'config.json',cfg)
    spec=json.loads((run/'dashboard_spec.json').read_text());spec['model_cache']=cfg['model_cache'];spec['tasks'][0]['expected']=cfg['expected_points']
    spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
    for p in [*(run/'code').glob('*.py'),run/'dynamic_schema_validation.json']:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
    for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
    prepare.save(run/'dashboard_spec.json',spec)
    print('Frozen parallel PP lane',run,cfg['node'],cfg['expected_points'])
