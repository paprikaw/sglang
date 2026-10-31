"""Freeze new retries using the source-verified List[bool] control API."""
from pathlib import Path
import ast,json,shutil
import prepare

native=prepare.WS/'.engine-worktrees/glm47-ep16-canonical-profile-20261004/python/sglang/srt/managers/tokenizer_control_mixin.py'
tree=ast.parse(native.read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='set_internal_state')
assert ast.unparse(method.returns)=='List[bool]'
assert 'return [res.updated for res in responses]' in ast.unparse(method)

def acknowledged(reply):
    return isinstance(reply,list) and bool(reply) and all(r is True for r in reply)

cases=[([True],True),([True]*4,True),([True,False],False),([],False),([1],False),({'updated':True},False)]
for reply,expected in cases:assert acknowledged(reply) is expected
prepare.save(prepare.ROOT/'management_api_validation.json',dict(passed=True,native_source=str(native),native_sha256=prepare.sha(native),signature='List[bool]',cases=len(cases)))

for kind,version,previous in [('ep','r4','ep_178_r3'),('pp','r3','pp_171_r2')]:
    run=prepare.prepare(kind,version)
    for p in (prepare.ROOT/'runs'/previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
    p=run/'code/runner.py';s=p.read_text()
    old="replies=answer if isinstance(answer,list) else [answer]\n        if not replies or not all(r.get('updated') is True for r in replies):"
    new="if not isinstance(answer,list) or not answer or not all(r is True for r in answer):"
    assert old in s;s=s.replace(old,new)
    p.write_text(s)
    p=run/'code/hicache_audit.py';s=p.read_text().replace('64 GiB KV plus 8 GiB GDN','64 decimal GB KV plus 8 decimal GB GDN');p.write_text(s)
    cfg=json.loads((run/'config.json').read_text())
    cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000)
    cfg['protocol']+=' Equal native per-GPU CPU pools: 64 decimal GB full-Attention KV + 8 decimal GB GDN. Source-verified management reply is List[bool]; every acknowledgement must be exactly true.'
    prepare.save(run/'config.json',cfg)
    shutil.copy2(prepare.ROOT/'management_api_validation.json',run/'management_api_validation.json')
    spec=json.loads((run/'dashboard_spec.json').read_text())
    spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
    for p in [*(run/'code').glob('*.py'),run/'management_api_validation.json']:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
    for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
    prepare.save(run/'dashboard_spec.json',spec)
    print('Validated native List[bool] API and frozen',run)
