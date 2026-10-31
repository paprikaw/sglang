from pathlib import Path
import ast,json,shutil
import prepare
run=prepare.prepare('pp','r4')
previous=prepare.ROOT/'runs/pp_171_r3'
for p in (previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
shutil.copy2(prepare.ROOT/'dynamic_pp_schema.py',run/'code/dynamic_pp_schema.py')
shutil.copy2(prepare.ROOT/'dynamic_schema_validation.json',run/'dynamic_schema_validation.json')
p=run/'code/pp_optim.py';s=p.read_text().replace('  gap_schedule.install()','  import dynamic_pp_schema\n  gap_schedule.install_schema=dynamic_pp_schema.install_schema\n  gap_schedule.install()');p.write_text(s)
cfg=json.loads((run/'config.json').read_text())
cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000)
cfg['protocol']+=' PP2 dynamic CPU shape headers on three asynchronous Gloo channels retain exact native CUDA payloads on the three NCCL channels; fixes fixed-schema failure on actual batch shapes. Source-verified management API uses List[bool].'
prepare.save(run/'config.json',cfg)
spec=json.loads((run/'dashboard_spec.json').read_text())
spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
for p in [*(run/'code').glob('*.py'),run/'dynamic_schema_validation.json']:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
prepare.save(run/'dashboard_spec.json',spec)
print('Frozen dynamic-shape PP retry',run)
