from pathlib import Path
import ast,copy,hashlib,json,os,secrets,shutil,subprocess,sys,time
WS=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
ROOT=WS/'experiments/h100_five_layouts_hicache_20261028'
OLD=WS/'experiments/h100_ep4_vs_four_ep1_capacity_20261028/runs/h100178_r3'
REUSE=OLD.parents[1]/'reusable_code'
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,a,b):
 if a not in s:raise RuntimeError('Missing replacement '+a[:100])
 return s.replace(a,b)
def prepare(kind='ep',version='r1'):
 node='spartan-gpgpu178' if kind=='ep' else 'spartan-gpgpu171'
 job='33646188' if kind=='ep' else '33646189'
 run=ROOT/'runs'/f'{kind}_{node[-3:]}_{version}'
 run.mkdir(parents=True,exist_ok=False);shutil.copytree(REUSE,run/'code',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ['task','cache','evidence','results']:(run/name).mkdir()
 cfg=json.loads((OLD/'config.json').read_text())
 cfg.update(id=run.name,node=node,job_id=job,code_root=str(run/'code'),admin_key_file=str(run/'admin.key'),port=25181,dist_port=30211,
  expected_points=30,common_context=8256,large_context=130304,hicache_gib_per_rank=72,
  workload_contexts=[8256,130304],profile_enabled=False,repeats=3,
  protocol='Four H100 per layout, BF16 Qwen3.6-35B-A3B, GPU+CPU HiCache enabled (write_through/kernel/page_first). Same global B64, cached P8256 or P130304; new Prefill Q512->1, Decode Q1->128. Three profiler-off formal cohorts after separate warmup. Both natural repeated-prefix replay and controlled all-GPU-prefix-demoted CPU replay, including H2D in client latency/makespan. All cache tiers and misses recorded; CPU-hit claims require actual native cached_tokens_details.host. No CPU checkpoint replay claimed. Native full-model numerical policy, small T<=16 Triton Expert, same per-GPU request/state bytes. PP candidates selected on pilot cohorts separately from formal samples.')
 layouts=[('ep4',4,1,[[0,1,2,3]]),('two_ep2',2,1,[[0,1],[2,3]]),('four_ep1',1,1,[[0],[1],[2],[3]])] if kind=='ep' else [('two_pp2',1,2,[[0,1],[2,3]]),('pp4',1,4,[[0,1,2,3]]),('ep4_control',4,1,[[0,1,2,3]])]
 cfg['layouts']=[]
 for arm,ep,pp,groups in layouts:
  logs=[dict(id=cfg['id']+'.'+arm+f'.r{r}',path=str(run/'cache'/f'{arm}.r{r}.log'),expected_ranks=[f'PP{i} TP0' for i in range(pp)] if pp>1 else [f'TP{i}' for i in range(ep)]) for r in range(len(groups))]
  cfg['layouts'].append(dict(id=arm,ep=ep,tp=ep,dp=ep,pp=pp,replicas=len(groups),gpu_groups=groups,usage_logs=logs))
 nodecfg=json.loads((OLD/'cache/node.json').read_text());nodecfg.update(node=node,expected_job_id=job)
 save(run/'cache/node.json',nodecfg)
 mc=cfg['model_cache'];mc.update(on_miss='fail' if kind=='ep' else 'build',cpu_memory_budget=dict(schema_version=1,kv_banks_per_node={node:[]}),
  hicache_reserve_bytes_per_node={node:(4*cfg['hicache_gib_per_rank']+16)*(1<<30)},node_specs={node:dict(path=str(run/'cache/node.json'),sha256=sha(run/'cache/node.json'))},usage_logs={node:[u for r in cfg['layouts'] for u in r['usage_logs']]})
 reg=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json')
 save(run/'kv_checkpoint_discovery.json',dict(registry=str(reg),registry_sha256=sha(reg),queried_at=time.time(),model=cfg['model_path'],qwen_matches=str(json.loads(reg.read_text())).lower().count('qwen'),mode='same-process native L2 HiCache; no reusable Qwen checkpoint entry; no checkpoint replay claim'))
 for name in ['pp_ready_policy.bin','eager_policy.bin','fragment_policy.bin','gap_policy.bin']:
  p=run/name;p.write_bytes(b'0');p.chmod(0o600)
 (run/'active_workload.txt').write_text('startup');(run/'active_workload.txt').chmod(0o600)
 (run/'admin.key').write_text(secrets.token_urlsafe(32));(run/'admin.key').chmod(0o600)
 s=(run/'code/runner.py').read_text()
 s=replace(s,'expected=11','expected=cfg[\'expected_points\']')
 s=replace(s,"controller_pid=os.getpid()","controller_pid=json.loads((run/'task/controller_identity.json').read_text())['pid'] if (run/'task/controller_identity.json').exists() else os.getpid()")
 s=replace(s,"ep=layout(cfg,arm)['ep'];d=ep","row=layout(cfg,arm);ep=row['ep'];pp=row['pp'];d=ep;running=cfg['local_requests']*d if pp==1 else cfg['requests']//row['replicas']")
 s=replace(s,"'--pp-size','1'","'--pp-size',str(pp)")
 s=replace(s,"str(cfg['local_requests']*d)","str(running)")
 s=replace(s,"str(cfg['local_mamba_slots']*d)","str(cfg['local_mamba_slots']*d*pp)")
 s=replace(s,"    if ep>1:\n        args+=", "    args+=['--enable-hierarchical-cache','--hicache-size',str(cfg['hicache_gib_per_rank']),'--hicache-host-memory-mode','cache','--hicache-write-policy','write_through','--hicache-io-backend','kernel','--hicache-mem-layout','page_first','--enable-cache-report','--enable-metrics','--enable-metrics-for-all-schedulers']\n    if pp>1:args+=['--pp-max-micro-batch-size','16','--pp-async-batch-depth','0']\n    if ep>1:\n        args+=")
 s=replace(s,"NVSHMEM_DISABLE_P2P='0',DEEPEP_LOW_LATENCY_NVLINK_ONLY='0')","NVSHMEM_DISABLE_P2P='0',DEEPEP_LOW_LATENCY_NVLINK_ONLY='0',PP_GAP_POLICY_PATH=str(run/'gap_policy.bin'),HICACHE_BENCHMARK='1')")
 s=replace(s,"return dict(cfg,port=port(cfg,arm,0 if layout(cfg,arm)['ep']>1 else index%4))","row=layout(cfg,arm)\n    replica=(index%4)//row['ep'] if row['pp']==1 else index%row['replicas']\n    return dict(cfg,port=port(cfg,arm,replica))")
 s=replace(s,"p['data_parallel_rank']=index%4","p['data_parallel_rank']=index%layout(cfg,arm)['dp']")
 a=s.index('\ndef resident_workload(');z=s.index('\ndef worker(',a)
 s=s[:a]+(ROOT/'workloads.py.inc').read_text()+s[z:]
 s=replace(s,"    common_p=None;maximum_p=None", "    common_p=cfg['common_context'];maximum_p=cfg['large_context']")
 s=replace(s,"if not limits or int(limits[-1])!=cfg['local_requests']:","expected_running=cfg['local_requests'] if row['pp']==1 else cfg['requests']//row['replicas']\n                if not limits or int(limits[-1])!=expected_running:")
 s=replace(s,"if len(qs)!=row['ep']", "if len(qs)!=row['ep']*row['pp']")
 s=replace(s,"h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',h.safe_server_info(info))", "h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',h.safe_server_info(info))\n                states=info.get('internal_states',[])\n                if not states or not all(x.get('enable_hierarchical_cache') is True for x in states):raise RuntimeError('HiCache did not activate on every reported scheduler')")
 a=s.index("            if arm=='four_ep1_common':");z=s.index('        finally:',a)
 s=s[:a]+"            for length in cfg['workload_contexts']:\n                replay_workload(run,cfg,arm,key,length,health)\n"+s[z:]
 s=replace(s,"    check(run,cfg,os.environ);packet=", "    h.save(run/'task/controller_identity.json',dict(pid=os.getpid(),started=time.time()))\n    check(run,cfg,os.environ);packet=")
 (run/'code/runner.py').write_text(s)
 # Retain native SSE cache tier metadata instead of inferring CPU hits.
 p=run/'code/base_runner.py';s=p.read_text();s=replace(s,"cached_tokens=final['meta_info'].get('cached_tokens'),","cached_tokens=final['meta_info'].get('cached_tokens'),\n        cached_tokens_details=final['meta_info'].get('cached_tokens_details'),") ;p.write_text(s)
 shutil.copy2(ROOT/'hicache_audit.py',run/'code/hicache_audit.py')
 p=run/'code/runtime_adapter.py';s=p.read_text();s += '\n_base_install=install\ndef install():\n    _base_install()\n    import hicache_audit\n    hicache_audit.install()\n';p.write_text(s)
 save(run/'config.json',cfg)
 spec=json.loads((OLD/'dashboard_spec.json').read_text());spec.update(id=cfg['id'],title='35B HiCache five-layout comparison: '+kind,port=25180 if kind=='ep' else 25280,runtime_dir=str(run/'dashboard_guard'),model_cache=mc)
 spec['tasks'][0].update(nodes=[node],root=str(run/'task'),controller_script=str(run/'code/runner.py'),expected=cfg['expected_points'],label='HiCache Prefill/Decode '+kind)
 frozen=[]
 for r in spec['frozen_inputs']:
  p=Path(r['path'])
  if OLD in p.parents:continue
  frozen.append(r)
 for p in [run/'config.json',run/'cache/node.json',run/'kv_checkpoint_discovery.json',*sorted((run/'code').glob('*.py'))]:frozen.append(dict(path=str(p),sha256=sha(p)))
 for r in frozen:
  if sha(r['path'])!=r['sha256']:raise RuntimeError('Pre-existing frozen dependency changed '+r['path'])
 spec['frozen_inputs']=frozen;save(run/'dashboard_spec.json',spec)
 for p in (run/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 print(json.dumps(dict(run=str(run),id=cfg['id'],node=node,layouts=[x['id'] for x in cfg['layouts']],hicache_gib_peak=304,frozen_files=len(frozen))))
 return run
if __name__=='__main__':prepare(*sys.argv[1:])
