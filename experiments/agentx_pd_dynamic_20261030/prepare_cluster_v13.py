"""Durable replay results and node-local diagnostic notification; optional audited reuse."""
import argparse,ast,hashlib,importlib.util,json,runpy,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--reuse-run',type=Path)
opts,rest=ap.parse_known_args(sys.argv[1:]);sys.argv=[sys.argv[0]]+rest
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v12.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('durable_prep',ROOT/'prepare_cluster.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
for source,target in [('cluster_runner_v8.py','cluster_runner.py'),('forward_observer_v2.py','forward_observer.py'),('snapshot_transport.py','snapshot_transport.py')]:
 shutil.copyfile(ROOT/source,run/'code'/target);ast.parse((run/'code'/target).read_text())
cfg=prep.read(run/'config.json');extra=[];cfg['imported_episodes']={}
if opts.reuse_run:
 oldrun=opts.reuse_run.resolve();old=prep.read(oldrun/'config.json');clean=prep.read(oldrun/'task/cleanup.json')
 assert clean['passed'] and clean['gpu_idle']
 for key in ('arms','cases','repeats','nodes','profile_override','warmup_cases'):assert cfg[key]==old[key],key
 assert prep.read(run/'launch_manifest.json')['engine_commit']==prep.read(oldrun/'launch_manifest.json')['engine_commit']
 assert prep.desc(Path(cfg['graphs_file']))['sha256']==prep.desc(Path(old['graphs_file']))['sha256']
 assert cfg['arms']==['col_ep4'] and old['clone_lane_policy']=='cyclic_copy_offset_v1'
 assert prep.desc(oldrun/'code/agentx_replay.py')['sha256']==prep.desc(ROOT/'agentx_replay_v2.py')['sha256']
 for p in sorted(oldrun.glob('results/*/*/*/formal.r*.json')):
  rep=p.stem.split('.')[-1];marker=p.with_name('completed.'+rep+'.json');assert prep.read(marker)['passed']
  artifacts=[]
  for original in (p,marker,p.with_name('forwards.'+rep+'.json')):
   relative=original.relative_to(oldrun);target=run/relative;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(original,target)
   artifacts.append(dict(relative_path=str(relative),sha256=prep.desc(target)['sha256'],source=str(original)));extra += [target,original]
  cfg['imported_episodes'][str(p.relative_to(oldrun))]=dict(origin_run=str(oldrun),artifacts=artifacts)
 extra += [oldrun/'config.json',oldrun/'task/cleanup.json',oldrun/'task/progress.json']
 prep.save(run/'imported_episodes.json',cfg['imported_episodes']);extra += [run/'imported_episodes.json']
cfg['forward_snapshot_protocol']='node-local tmpfs epoch notification; completed replay checkpointed before post-replay diagnostics'
cfg['protocol']+=(' Forward-counter notifications now use per-node tmpfs, with remote writes acknowledged outside formal timing; no shared-file mmap notification dependency. Completed client responses are durably checkpointed before post-replay counter collection. '+str(len(cfg['imported_episodes']))+' complete episodes are explicitly imported by hash from the previous same-E5 run; only missing repeats execute. Routing in two/four/eight-owner layouts is identical to V11.')
prep.save(run/'config.json',cfg)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(x['path']) for x in ds['frozen_inputs']]+[ROOT/x for x in ('cluster_runner_v8.py','forward_observer_v2.py','snapshot_transport.py','test_snapshot_recovery.py','snapshot_recovery_tests.log')]+[run/'code/snapshot_transport.py',Path(__file__).resolve()]+extra
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values());prep.save(run/'dashboard_spec.json',ds)
m=prep.read(run/'launch_manifest.json');m.update(protocol=cfg['protocol'],forward_snapshot_protocol=cfg['forward_snapshot_protocol'],imported_episodes=cfg['imported_episodes'],frozen_inputs=ds['frozen_inputs']);prep.save(run/'launch_manifest.json',m)
progress=prep.read(run/'task/progress.json');progress['valid_points']=len(cfg['imported_episodes']);prep.save(run/'task/progress.json',progress)
print('Durable snapshot protocol frozen; imported complete episodes:',len(cfg['imported_episodes']))
