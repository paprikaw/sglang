"""Reserve measured PP workspace and capture only the actual microbatch sizes."""
import ast,json,shutil
import prepare
run=prepare.prepare('pp','r5')
for p in (prepare.ROOT/'runs/pp_171_r4/code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
shutil.copy2(prepare.ROOT/'dynamic_schema_validation.json',run/'dynamic_schema_validation.json')
p=run/'code/runner.py';s=p.read_text()
old="'--mem-fraction-static',str(cfg['mem_fraction_static'])"
assert old in s;s=s.replace(old,"'--mem-fraction-static',str(0.92 if pp>1 else cfg['mem_fraction_static'])")
old="'--cuda-graph-max-bs-decode',str(running)"
assert old in s;s=s.replace(old,"'--cuda-graph-max-bs-decode',str(min(16,running) if pp>1 else running)")
p.write_text(s)
cfg=json.loads((run/'config.json').read_text())
cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000,
           pp_mem_fraction_static=0.92,pp_decode_graph_max_bs=16,ep_control_mem_fraction_static=0.94)
cfg['protocol']+=' PP uses mem_fraction_static=0.92 to reserve native GDN copy-on-write and NCCL workspace; observed prior 0.94 OOM at a 480 MiB native COW temporary. PP Graph capture max B16 matches actual maximum microbatch; EP calibration retains 0.94. Dynamic CPU shape headers on separate Gloo channels retain exact native CUDA payloads. Management reply List[bool].'
prepare.save(run/'config.json',cfg)
spec=json.loads((run/'dashboard_spec.json').read_text())
spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
for p in [*(run/'code').glob('*.py'),run/'dynamic_schema_validation.json']:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
prepare.save(run/'dashboard_spec.json',spec)
print('Frozen PP workspace/capture retry',run)
