#!/usr/bin/env python3
"""Cluster-wide reconciliation. Never books, preempts normal work, or evicts banks.

Qualified recipes are frozen explicit serving profiles, not inferred launch flags.
Each model manager owns ordinary background Guard leases and its own cleanup.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
from manager import atomic_json, read_json, discover_bookings, registry_blocks, same_process, process_start_ticks
import cache_model_discovery


LEGACY_EPHEMERAL_WORKER=Path(__file__).resolve().parents[1]/'idle_gpu_benchmark/prefill_worker_ephemeral.py'
EPHEMERAL_WORKER=LEGACY_EPHEMERAL_WORKER.with_name('prefill_worker_ephemeral_lease.py')


def artifact_retention(cfg):
    """Background load creates no result archive unless explicitly requested."""
    mode=cfg.get('artifact_retention','none')
    if mode not in ('none','retain'):
        raise ValueError('artifact_retention must be none or retain')
    return mode


def benchmark_worker(cfg):
    mode=artifact_retention(cfg)
    worker=cfg.get('benchmark_worker_script')
    if mode=='none':
        worker=worker or str(EPHEMERAL_WORKER)
        # The old append worker ignores this policy. Refuse that combination
        # rather than claiming to discard results while continuing to save them.
        if Path(worker).resolve() not in (EPHEMERAL_WORKER, LEGACY_EPHEMERAL_WORKER):
            raise ValueError('artifact_retention none requires the ephemeral prefill worker')
    return worker


def read(path,default=None):
    try:return read_json(Path(path))
    except (OSError,ValueError):return {} if default is None else default


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready_keys(bank):
    items=bank.get('status',{}).get('entries',[])
    if not items or any(e.get('state')!='ready' or e.get('evict_pending') for e in items):return set()
    return {e['key'] for e in items}


def select_recipe(booking,bank,recipes):
    if booking.end_at-time.time()<=90:return None,'booking_expiring'
    if not bank:
        for recipe in recipes:
            # Only explicit whole-node fallback recipes may build; hardware is checked before admission.
            if recipe.get('fallback') and 'gpu' in booking.gpu_tres.lower():
                return recipe,'build_qualified_full_model'
        return None,'waiting_for_compatible_recipe'
    if str(bank.get('expected_job_id'))!=booking.job_id:
        return None,'old_booking_cache_requires_verified_retirement'
    if bank.get('state')!='active':return None,'cache_not_active'
    if bank.get('cache_scheme')!='canonical_full_weight':
        return None,'protected_partial_cache_requires_complete_qualified_group'
    keys=ready_keys(bank)
    for recipe in recipes:
        if (keys and set(recipe.get('node_cache_keys',{}).get(booking.node,[])) and set(recipe['node_cache_keys'][booking.node])<=keys and bank.get('source_commit')==recipe['engine_commit']):
            return recipe,'reuse_current_model'
    return None,'waiting_for_compatible_recipe'


def old_booking_recovery(booking, bank, recipes):
    """Advisory death check; the leased Guard helper repeats proof before mutation.

    A different job ID alone never authorizes retirement. In particular SSH
    failure, unreadable proc entries, a live identity or an occupied socket wait.
    No process, socket, weights or ledger is changed by this discovery step.
    """
    manifest=bank.get('manifest',{})
    daemon_id=bank.get('daemon_id')
    socket_path=bank.get('socket_path')
    if (not isinstance(manifest,dict) or not isinstance(daemon_id,str)
            or len(daemon_id)!=32 or any(x not in '0123456789abcdef' for x in daemon_id)
            or not isinstance(socket_path,str) or not Path(socket_path).is_absolute()):
        return None,'old_booking_cache_requires_verified_retirement'
    if (bank.get('state')!='active' or bank.get('uid')!=os.getuid()
            or bank.get('cache_scheme')!='canonical_full_weight'
            or manifest.get('node')!=booking.node
            or manifest.get('cache_id')!=bank.get('daemon_id')
            or any(type(manifest.get(k)) is not int or manifest[k]<=0
                   for k in ('pid','process_start_ticks'))
            or not str(booking.job_id).isdigit()):
        return None,'old_booking_cache_requires_verified_retirement'
    fallback,why=select_recipe(booking,None,recipes)
    if not fallback:return None,why
    # Only a runner with the existing frozen dead-daemon recovery contract may
    # bootstrap. Unknown implementations must not inherit this authorization.
    if Path(fallback.get('node_runner','')).resolve()!=Path(__file__).resolve().parents[2]/'experiments/cluster_backfill_20261020/node_runner_qwen_ephemeral.py':
        return None,'old_booking_cache_requires_verified_retirement'
    payload=dict(node=booking.node,uid=os.getuid(),job=booking.job_id,
                 manifest=manifest,socket=bank.get('socket_path'))
    program=r"""import json,os,socket,sys
from pathlib import Path
p=json.loads(sys.argv[1]);m=p['manifest']
if socket.gethostname().split('.')[0]!=p['node'] or os.getuid()!=p['uid']:
    raise RuntimeError('physical node or UID differs')
cgroup=Path('/proc/self/cgroup').read_text()
if '/job_'+p['job']+'/' not in cgroup:
    raise RuntimeError('probe outside current booked job')
try:
    raw=Path('/proc/'+str(m['pid'])+'/stat').read_text()
except FileNotFoundError:
    try: os.kill(m['pid'],0)
    except ProcessLookupError: dead=True
    else: dead=False
else:
    fields=raw[raw.rfind(')')+2:].split()
    dead=fields[0]=='Z' or int(fields[19])!=m['process_start_ticks']
print(json.dumps(dict(node=p['node'],job=p['job'],daemon_id=m['cache_id'],
                     dead=dead,socket_absent=not os.path.lexists(p['socket']))))
"""
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=5',booking.node,
                           shlex.join(['python3','-c',program,json.dumps(payload)])],
                          capture_output=True,text=True,timeout=20)
    if result.returncode:raise RuntimeError('old booking process inventory unavailable')
    proof=json.loads(result.stdout)
    if (proof.get('node')!=booking.node or proof.get('job')!=booking.job_id
            or proof.get('daemon_id')!=bank['daemon_id']):
        raise ValueError('old booking process proof identity differs')
    if proof.get('dead') is not True or proof.get('socket_absent') is not True:
        return None,'old_booking_cache_requires_verified_retirement'
    recipe=dict(fallback)
    recipe['id']=fallback['id']+'-recover-'+booking.job_id+'-'+bank['daemon_id'][:12]
    recipe['dead_daemon_recovery']=dict(node=booking.node,daemon_id=bank['daemon_id'],manifest=manifest)
    return recipe,'recover_verified_dead_booking_cache'


def live_manager(row):
    if not row:return False
    s=read(Path(row['root'])/'state.json')
    return (s.get('manager_host')==socket.gethostname().split('.')[0]
            and s.get('state')in('running','stopping')
            and same_process(s.get('manager_pid',0),s.get('manager_start_ticks')))


def stop_manager(row):
    if not live_manager(row):return
    s=read(Path(row['root'])/'state.json')
    if s['state']!='stopping':os.kill(s['manager_pid'],signal.SIGTERM)


def matching_service(node,lease,services):
    """Bind recovery to this node's owned manager and exact leased deployment."""
    row=services.get(node)
    if not row:return None,None,None
    state=read(Path(row['root'])/'state.json')
    run=state.get('attempts',{}).get(node,{}).get('run')
    if not run:return None,None,None
    if not Path(run).resolve().is_relative_to(Path(row['root']).resolve()):
        return None,None,None
    deployment=read(Path(run)/'deployment.json')
    if (deployment.get('deployment_id')!=lease.get('experiment_id')
            or deployment.get('node')!=node):
        return None,None,None
    return row,run,deployment


def lease_state(node,registry,services,now):
    # Match Guard's atomic admission expiry rules. Reading registry.json does
    # not prune expired records; treating those records as occupancy prevents
    # the next Guard claim from ever performing its normal expiry transaction.
    # A free registry view still requires the GPU-process audit below, followed
    # by Guard's locked claim. Expiry is never proof of process cleanup.
    def occupies(lease):
        if lease.get('state')=='active':
            return now-float(lease['heartbeat_at'])<=float(lease['ttl_s'])
        if lease.get('state')=='quarantined':
            return now<float(lease.get('cleanup_grace_expires_at',float(lease.get('heartbeat_at',now))+90))
        return False
    matches=[l for l in registry.get('leases',[]) if node in l.get('nodes',[]) and occupies(l)]
    if len(matches)>1:return 'unknown','overlapping_leases',None
    if not matches:return None,None,None
    lease=matches[0]
    if lease['state']=='quarantined':return 'draining','cleanup_unverified',lease
    if lease.get('preemption_requested'):return 'draining','foreground_requested',lease
    if lease.get('resource_priority','normal')!='background':return 'foreground','normal_guard_lease',lease
    # A heartbeat alone is not a successful benchmark: require the exact worker.
    row,run,deployment=matching_service(node,lease,services)
    if row:
        control=read(Path(run)/'benchmark_worker.json')
        if control.get('node',node)==node:
            code=control.get('exit_code')
            if control.get('state') in ('failed','cleanup_unverified') or (type(code)is int and code!=0):
                reason='benchmark_worker_'+str(control.get('state')or'failed')
                if type(code)is int:reason+='_exit_'+str(code)
                return 'failed',reason,lease
        w=read(Path(row['root'])/'prefill/nodes'/f'{node}.json')
        current=(w.get('run')==run and w.get('node',node)==node
                 and -5<=now-w.get('timestamp',0)<20)
        if current and w.get('status')=='running':
            return 'benchmark','verified_live_worker',lease
        if deployment.get('status')=='healthy':
            return 'background_unverified','benchmark_worker_not_current',lease
        return 'loading','benchmark_not_ready',lease
    return 'background_unverified','no_matching_live_worker',lease



def discover_recipe_keys(node,bank,recipes,root):
    candidates=[r for r in recipes if r['engine_commit']==bank.get('source_commit')]
    if not candidates or bank.get('cache_scheme')!='canonical_full_weight':return
    signature=hashlib.sha256(json.dumps([bank.get('daemon_id'),sorted(ready_keys(bank)),[(r['id'],r['profile_sha256'])for r in candidates]],sort_keys=True).encode()).hexdigest()
    path=root/'observations'/(node+'.json');observed=read(path)
    if observed.get('signature')!=signature:
        # CPU-only status and safetensors manifest inspection, never a CUDA
        # import or request. Group keys must be derived on the physical node.
        program=r"""import json,sys,os,importlib.util
from pathlib import Path
p=json.loads(sys.stdin.readline());b=p['bank'];m=b['manifest']
f=Path('/proc/'+str(m['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
assert int(f[19])==m['process_start_ticks'] and f[0]!='Z'
out={}
for r in p['recipes']:
 d=Path(r['node_template']['engine_root'])/'python/sglang/srt/weight_cache';sys.path.insert(0,str(d))
 spec=importlib.util.spec_from_file_location('cpu_canonical_inventory',d/'cpu_canonical.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 from cpu_protocol import CpuWeightCacheClient
 with CpuWeightCacheClient(b['socket_path'],timeout=2)as c:status=c.status()
 assert status['pid']==m['pid'] and status['source_commit']==b['source_commit'] and status['uid']==os.getuid()
 desc=mod.describe_model(r['node_template']['model_path']);out[r['id']]=[i['key']for i in desc['items']]
print(json.dumps(out))
"""
        result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=4',node,'/usr/bin/python3 -c '+shlex.quote(program)],input=json.dumps(dict(bank=bank,recipes=candidates))+'\n',capture_output=True,text=True,timeout=25,check=True)
        observed=dict(signature=signature,keys=json.loads(result.stdout),sampled_at=time.time());atomic_json(path,observed)
    for r in candidates:r.setdefault('node_cache_keys',{}).update({node:observed['keys'][r['id']]})


def recipe_hardware(recipe, slurm_info):
    """Match typed GPU inventory to an explicit whole-node loader qualification.

    Legacy profiles remain A100-only. Generic gpu:4 bookings and hostname
    patterns never establish hardware compatibility.
    """
    allowed=recipe.get('qualified_gpu_types',['A100'])
    if (not isinstance(allowed,list) or not allowed
            or any(kind not in ('A100','H100') for kind in allowed)):
        raise ValueError('Unknown recipe GPU qualification')
    matches=re.findall(r'(?<![\w:])gpu:([A-Za-z0-9_-]+):(\d+)(?!\d)',slurm_info,re.I)
    if len(matches)!=1:return None
    kind,count=matches[0]
    kind=kind.upper()
    if kind not in allowed or int(count)!=4:return None
    return dict(gpu_type=kind,gpu_count=4,source='slurm_typed_node_inventory')


def select_hardware_recipe(booking, bank, candidates, selected, slurm_info):
    """Rank only complete cached models whose loaders support this hardware.

    Discovery already restricts candidates to an authenticated preferred model
    when available. Never expand that set to historical weights on rejection.
    Recovery bootstrap snapshots must remain attached to the selected plan.
    """
    hardware=recipe_hardware(selected,slurm_info)
    if hardware:return dict(selected,hardware_qualification=hardware)
    if not bank:return None
    compatible=[r for r in candidates if recipe_hardware(r,slurm_info)]
    recipe,_=select_recipe(booking,bank,compatible)
    if recipe is None:return None
    return dict(recipe,hardware_qualification=recipe_hardware(recipe,slurm_info))


def make_profile(cfg,recipe,booking,bank,root):
    retention=artifact_retention(cfg)
    worker=benchmark_worker(cfg)
    template=Path(recipe['profile'])
    if digest(template)!=recipe['profile_sha256']:raise RuntimeError('qualified recipe profile changed')
    profile=read_json(template)
    if recipe.get('discovered_model'):
        model=recipe['discovered_model']
        rule=recipe['checkpoint_qualification']
        runner=Path(rule['node_runner'])
        if digest(runner)!=rule['node_runner_sha256']:
            raise ValueError('qualified current-model runner changed')
        profile['model_path']=model['model_root']
        profile['model_revision']='config-and-index-sha256:'+cache_model_discovery.json_sha(model['metadata_sha256'])
        profile['discovered_model']=model
        profile['cache_support_profile']=recipe['node_template']['support_profile']
        profile['loader_template_id']=recipe['template_id']
        profile['loader_template_sha256']=recipe['profile_sha256']
        if rule.get('server_entry'):
            if digest(rule['server_entry'])!=rule['server_entry_sha256']:
                raise ValueError('qualified server entry changed')
            profile.update(server_entry=rule['server_entry'],server_entry_sha256=rule['server_entry_sha256'])
    override=cfg.get('guard_override')
    if override:
        guard=Path(override['script'])
        if digest(guard)!=override['sha256']:
            raise ValueError('Pinned Guard override changed')
        for name,expected in override['dependency_sha256'].items():
            if Path(name).name!=name or digest(guard.parent/name)!=expected:
                raise ValueError('Pinned Guard lock dependency changed')
        profile['guard_script']=str(guard)
    profile['runtime_root']=str(root)
    if recipe.get('hardware_qualification'):
        profile['hardware_qualification']=recipe['hardware_qualification']
    profile['node_runner']=recipe.get('node_runner',cfg['node_runner'])
    profile['artifact_retention']=retention
    profile['benchmark']['artifact_retention']=retention
    if worker:
        profile['benchmark']['worker_script']=worker
    profile['cached_weights_only']=bool(bank)
    profile['model_cache_on_miss']='fail' if bank else 'build'
    ns=dict(recipe['node_template'],node=booking.node,expected_job_id=booking.job_id)
    if bank:
        for field in ('max_bytes','headroom_bytes','socket_path'):ns[field]=bank[field]
    profile['known_node_specs']={booking.node:dict(cache_spec_template=ns)}
    if recipe.get('dead_daemon_recovery'):
        if bank:raise ValueError('Dead-bank bootstrap cannot reuse old cache')
        profile['known_node_specs'][booking.node]['dead_daemon_recovery']=recipe['dead_daemon_recovery']
    profile.pop('default_node_spec',None)
    profile['allowed_gpu_types']=[]
    b=read_json(Path(profile['benchmark']['config_path']));b.update(runtime_root=str(root/'prefill'),fleet_runtime=str(root),nodes=[booking.node])
    b['artifact_retention']=retention
    if retention=='none':
        b['storage_layout']='ephemeral_live_v1'
    elif worker:
        b['storage_layout']='append_per_run_v1'
    atomic_json(root/'benchmark.json',b);profile['benchmark']['config_path']=str(root/'benchmark.json')
    atomic_json(root/'profile.json',profile)
    return profile


def reconcile(cfg,root,services,recipes):
    # Reject unsupported retention policies before any manager may be started.
    retention=artifact_retention(cfg)
    benchmark_worker(cfg)
    now=time.time();bookings=discover_bookings(cfg,now)
    reg=read_json(Path(cfg['gpu_registry_dir'])/'registry.json')
    blocked=registry_blocks(Path(cfg['gpu_registry_dir']),now)
    ledger=read_json(Path(cfg['gpu_registry_dir']).parent/'cpu_weight_daemons/registry.json')
    banks={r['node']:r for r in ledger['entries']};rows={}
    for node,row in list(services.items()):
        if node not in bookings:
            stop_manager(row)
            if not live_manager(row):services.pop(node)
    for node,book in bookings.items():
        state,reason,lease=lease_state(node,reg,services,now)
        if state:
            if (state=='failed' and lease.get('nodes')==[node]
                    and lease.get('resource_priority')=='background'
                    and lease.get('preemptible')is True):
                # Ask only this manager to finish its normal owned Guard cleanup.
                # Keep registry occupancy until that cleanup/release completes;
                # the next reconciliation then uses ordinary recipe admission.
                owned,_,_=matching_service(node,lease,services)
                if owned:stop_manager(owned)
            if state=='benchmark' and node in services and live_manager(services[node]):
                # Seal the mutable control profile after the first successful
                # cold build. Active frozen_profile.json and all model/trace
                # inputs remain untouched. Future rounds may only reuse cache.
                control=Path(services[node]['root'])/'profile.json'
                profile=read(control)
                if profile and not profile.get('cached_weights_only'):
                    profile['cached_weights_only']=True
                    profile['model_cache_on_miss']='fail'
                    atomic_json(control,profile)
                    services[node]['bootstrap_sealed_at']=time.time()
            rows[node]=dict(state=state,reason=reason,experiment=lease.get('experiment_id'))
            continue
        if node in blocked:
            rows[node]=dict(state='reserved',reason=blocked[node]);continue
        # No GPU probing is added to foreground launches. Slurm's actual node
        # inventory is consulted only for fallback qualification.
        bank=banks.get(node)
        candidates=recipes;discovery_reason=None
        if bank and str(bank.get('expected_job_id'))==book.job_id and bank.get('state')=='active' and bank.get('cache_scheme')=='canonical_full_weight':
            try:
                if cfg.get('model_discovery'):
                    observation=cache_model_discovery.discover(node,bank,recipes,cfg['model_discovery'],root)
                    # One overwritten inventory, never per-poll history files.
                    atomic_json(root/'observations'/(node+'.json'),observation)
                    candidates,discovery_reason=cache_model_discovery.current_recipes(node,bank,observation,recipes)
                    if observation.get('status'):
                        bank=dict(bank,status=observation['status'])
                else:discover_recipe_keys(node,bank,recipes,root)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                rows[node]=dict(state='unknown',reason='cache_inventory_unavailable',error=type(exc).__name__)
                continue
        recipe,why=select_recipe(book,bank,candidates)
        if why=='old_booking_cache_requires_verified_retirement':
            try:
                recipe,why=old_booking_recovery(book,bank,recipes)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                rows[node]=dict(state='waiting',reason='old_booking_cache_requires_verified_retirement',error=type(exc).__name__)
                continue
            # The old snapshot is solely a recovery target. It is never passed
            # to cache admission as a usable bank belonging to the new job.
            if recipe:bank=None
        if not recipe and discovery_reason:why=discovery_reason
        current=services.get(node)
        if not recipe:
            if current:stop_manager(current)
            rows[node]=dict(state='waiting',reason=why);continue
        if not bank or recipe.get('discovered_model'):
            # Hardware support belongs to the selected loader, never to a hostname.
            try:
                info=subprocess.check_output([cfg.get('scontrol','/apps/slurm/latest/bin/scontrol'),
                    'show','node',node,'-o'],text=True,timeout=10)
            except (OSError,subprocess.SubprocessError) as exc:
                rows[node]=dict(state='waiting',reason='hardware_inventory_unavailable',error=type(exc).__name__)
                continue
            recipe=select_hardware_recipe(book,bank,candidates,recipe,info)
            if recipe is None:
                if current:stop_manager(current)
                rows[node]=dict(state='waiting',reason='hardware_outside_qualified_recipe');continue
        if current and current['recipe_id']!=recipe['id']:
            stop_manager(current)
            if live_manager(current):
                rows[node]=dict(state='draining',reason='cache_recipe_changed');continue
            services.pop(node,None);current=None
        if current and live_manager(current):
            status=read(Path(current['root'])/'state.json')
            detail=status.get('nodes',{}).get(node,{}).get('reason')
            rows[node]=dict(state='recovering',reason=detail or why);continue
        # A missing Guard lease is not proof that arbitrary shell GPU work is
        # absent. Background audits protect such work without signalling it.
        try:
            probe=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=4',node,
                'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        except (OSError,subprocess.SubprocessError) as exc:
            rows[node]=dict(state='waiting',reason='unregistered_gpu_work_or_inventory_unavailable',error=type(exc).__name__)
            continue
        if probe.returncode or probe.stdout.strip():
            rows[node]=dict(state='waiting',reason='unregistered_gpu_work_or_inventory_unavailable');continue
        target=root/'services'/(node+'-'+uuid.uuid4().hex[:12]);target.mkdir(parents=True,mode=0o700)
        profile=make_profile(cfg,recipe,book,bank,target)
        env=dict(os.environ,EXPERIMENT_GUARD_BACKFILL_CHILD='1')
        # Result-free background load needs live control/Guard evidence only.
        # Keep process output out of project files; foreground and retained
        # profiles retain their existing independent logging contracts.
        log_path=Path(os.devnull) if retention=='none' else target/'manager.log'
        with log_path.open('ab',buffering=0) as log:
            proc=subprocess.Popen([profile['python'],cfg['manager_script'],'run','--config',str(target/'profile.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        services[node]=dict(root=str(target),recipe_id=recipe['id'],pid=proc.pid,created_at=time.time())
        rows[node]=dict(state='starting',reason=why)
    atomic_json(root/'services.json',dict(services=services))
    return dict(timestamp=time.time(),nodes=rows,manager_host=socket.gethostname().split('.')[0],manager_pid=os.getpid(),manager_start_ticks=process_start_ticks(os.getpid()))


def persist_reconciliation(cfg,root,services):
    # Do not admit fresh work when its audit trail cannot be persisted. This
    # probe uses a separate timestamp so it never makes an old fleet snapshot
    # appear fresh. EDQUOT/ENOSPC retries keep the singleton owner alive.
    available=os.statvfs(root).f_favail
    reserve=int(cfg.get('min_free_inodes',1024))
    atomic_json(root/'storage_health.json',dict(timestamp=time.time(),free_inodes=available,reserved_inodes=reserve))
    if 0<=available<reserve:
        # Preserve metadata headroom for foreground leases and cleanup. Signal
        # only identity-verified managers that this coordinator already owns.
        for row in services.values():stop_manager(row)
        state=dict(timestamp=time.time(),state='storage_paused',nodes={},
                   error='Project inode headroom below reserve; background admission paused')
        atomic_json(root/'state.json',state)
        return state
    state=reconcile(cfg,root,services,cfg['recipes'])
    atomic_json(root/'state.json',state)
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    cfg=read_json(args.config);root=Path(cfg['runtime_root']);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(root/'watchdog.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    services=read(root/'services.json').get('services',dict(cfg.get('adopt_services',{})))
    while True:
        try:persist_reconciliation(cfg,root,services)
        except Exception as exc:
            # Logging and failure-report writes may themselves fail when the
            # project exhausts bytes or inodes. Never exit the retry loop merely
            # because those telemetry operations fail too.
            message=type(exc).__name__+': '+str(exc)
            try:print('backfill audit paused: '+message,file=sys.stderr,flush=True)
            except OSError:pass
            try:atomic_json(root/'state.json',dict(timestamp=time.time(),state='audit_failed',error=message))
            except OSError:pass
        if args.once:break
        time.sleep(cfg.get('poll_s',10))

if __name__=='__main__':main()
