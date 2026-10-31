from pathlib import Path
import ast,hashlib,json,shutil,sys
import prepare
root=prepare.ROOT
for kind,version,previous in [('ep','r3','ep_178_r2'),('pp','r2','pp_171_r1')]:
 run=prepare.prepare(kind,version)
 if kind=='pp':
  for p in (root/'runs'/previous/'code').glob('*.py'):shutil.copy2(p,run/'code'/p.name)
 p=run/'code/runner.py';s=p.read_text()
 s=s.replace("if answer.get('updated') is not True:raise RuntimeError('Idle cache fence refused: '+str(answer))", "replies=answer if isinstance(answer,list) else [answer]\n        if not replies or not all(r.get('updated') is True for r in replies):raise RuntimeError('Idle cache fence refused: '+str(answer))")
 s=s.replace("if not answer.get('updated'):raise RuntimeError('PP microbatch update refused '+str(answer))", "replies=answer if isinstance(answer,list) else [answer]\n        if not replies or not all(r.get('updated') is True for r in replies):raise RuntimeError('PP microbatch update refused '+str(answer))")
 s=s.replace("attention_instances=4,aggregate_conservative_tokens=local_cap*4", "attention_instances=row['replicas']*row['dp'],aggregate_conservative_tokens=local_cap*row['replicas']*row['dp']")
 # A real controller/SSH-worker heartbeat keeps long model loading visible.
 s=s.replace("check(run,cfg,os.environ);time.sleep(2)", "check(run,cfg,os.environ)\n                progress_path=run/'task/progress.json'\n                if progress_path.exists():\n                    value=json.loads(progress_path.read_text())\n                    if time.time()-value.get('timestamp',0)>5:\n                        value.update(timestamp=time.time(),controller_pid=os.getpid(),heartbeat_source='live Guard-authenticated controller with SSH worker alive')\n                        h.save(progress_path,value)\n                time.sleep(2)")
 p.write_text(s)
 cfg=json.loads((run/'config.json').read_text());cfg.update(host_pool_unit='decimal GB, native convention',host_kv_bytes_per_rank=64_000_000_000,host_gdn_bytes_per_rank=8_000_000_000)
 cfg['protocol']+=' Equal native per-GPU CPU pools: 64 decimal GB full-Attention KV + 8 decimal GB GDN; 304 GiB node admission includes transient reserve. API management responses are per-rank lists; all ranks must acknowledge.'
 prepare.save(run/'config.json',cfg)
 spec=json.loads((run/'dashboard_spec.json').read_text());spec['model_cache']=cfg['model_cache']
 spec['frozen_inputs']=[r for r in spec['frozen_inputs'] if not str(r['path']).startswith(str(run/'code'))]
 for p in (run/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p));spec['frozen_inputs'].append(dict(path=str(p),sha256=prepare.sha(p)))
 for r in spec['frozen_inputs']:r['sha256']=prepare.sha(r['path'])
 prepare.save(run/'dashboard_spec.json',spec)
 print('Corrected management reply contract frozen',run)
