"""Guard-owned EP4 versus four EP1 capacity and phase comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import process_helpers as h
import base_runner as b

class ResidencyError(RuntimeError):
    pass


def check(run,cfg,env,local=False):
    return h.authenticate(run,cfg,env,local=local)


def progress(run,cfg,stage,**extra):
    points=list(run.glob('results/*/metrics.*.json'))
    h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',stage=stage,timestamp=time.time(),controller_pid=json.loads((run/'task/controller_identity.json').read_text())['pid'] if (run/'task/controller_identity.json').exists() else os.getpid(),valid_points=len(points)+(2 if (run/'results/four_ep1_at_ep4max/admission.json').exists() else 0),expected=cfg['expected_points'],**extra))


def layout(cfg,arm):
    return next(x for x in cfg['layouts'] if x['id']==arm)


def port(cfg,arm,replica):
    return cfg['port']+cfg['layouts'].index(layout(cfg,arm))*100+replica*10


def engine_args(cfg,arm,replica):
    row=layout(cfg,arm);ep=row['ep'];pp=row['pp'];d=ep;running=cfg['local_requests']*d if pp==1 else cfg['requests']//row['replicas']
    args=['--model-path',cfg['model_path'],'--host','127.0.0.1','--port',str(port(cfg,arm,replica)),
        '--dist-init-addr',f"127.0.0.1:{cfg['dist_port']+cfg['layouts'].index(layout(cfg,arm))*100+replica*10}",
        '--tp-size',str(ep),'--ep-size',str(ep),'--dp-size',str(ep),'--pp-size',str(pp),
        '--dtype','bfloat16','--kv-cache-dtype','bfloat16','--context-length',str(cfg['max_context']),
        '--mem-fraction-static',str(0.92 if pp>1 else cfg['mem_fraction_static']),'--page-size','64',
        '--max-running-requests',str(running),
        '--max-mamba-cache-size',str(cfg['local_mamba_slots']*d*pp),'--mamba-max-states-per-path','1','--mamba-ssm-dtype','float32',
        '--cuda-graph-max-bs-decode',str(min(16,running) if pp>1 else running),'--cuda-graph-backend-prefill','disabled',
        '--chunked-prefill-size',str(4096*d),'--max-prefill-tokens',str(4096*d),
        '--attention-backend','fa3','--linear-attn-backend','triton',
        '--moe-runner-backend','deep_gemm','--moe-a2a-backend','deepep' if ep>1 else 'none',
        '--disable-custom-all-reduce','--skip-server-warmup','--json-model-override-args','{"language_model_only":true}']
    args+=['--enable-hierarchical-cache','--hicache-size',str(cfg['hicache_gib_per_rank']),'--hicache-host-memory-mode','cache','--hicache-write-policy','write_through','--hicache-io-backend','kernel','--hicache-mem-layout','page_first','--enable-cache-report','--enable-metrics','--enable-metrics-for-all-schedulers']
    if pp>1:args+=['--pp-max-micro-batch-size','16','--pp-async-batch-depth','0']
    if ep>1:
        args+=['--enable-dp-attention','--deepep-mode','auto','--deepep-dispatcher-output-dtype','bf16']
    return args


def engine_entry(run,cfg,arm,replica):
    check(run,cfg,os.environ,True)
    import runtime_adapter
    runtime_adapter.install()
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    load_plugins();args=prepare_server_args(engine_args(cfg,arm,replica))
    args.enable_multimodal=False;args.api_key=args.admin_api_key=h.read_key(cfg)
    try:run_server(args)
    finally:kill_process_tree(os.getpid(),include_parent=False)


def launch(run,cfg,arm,replica,inherited,key):
    row=layout(cfg,arm);alias='ep2' if row['ep']>1 else 'pp2'
    env=b.environment(cfg,alias,inherited)
    evidence=run/'evidence'/f'{arm}.r{replica}';evidence.mkdir(parents=True,exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,row['gpu_groups'][replica])),EP2_PP2_RUN=str(evidence),
        EP_STEADY_HARDWARE='H100',PP_READY_POLICY_PATH=str(run/'pp_ready_policy.bin'),
        STEADY_EAGER_POLICY_PATH=str(run/'eager_policy.bin'),FRAGMENT_POLICY_PATH=str(run/'fragment_policy.bin'),
        FRAGMENT_EXPERT_MODE='triton_small',EXPERT_WORKLOAD_PATH=str(run/'active_workload.txt'),
        NVSHMEM_DISABLE_P2P='0',DEEPEP_LOW_LATENCY_NVLINK_ONLY='0',PP_GAP_POLICY_PATH=str(run/'gap_policy.bin'),HICACHE_BENCHMARK='1',HICACHE_PP_SIZE=str(row['pp']),PP_GAP_SEPARATE_CHANNELS='0')
    command=[cfg['python'],str(Path(cfg['guard_scripts'])/'model_cache_adapter.py'),'exec-local','--bindings',env[h.GUARD_KEYS[3]],'--',
        cfg['python'],str(Path(__file__).resolve()),'--run',str(run),'--engine-entry','--arm',arm,'--replica',str(replica)]
    proc=subprocess.Popen(command,env=env,cwd=cfg['engine_root'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    identity=h.process_record(proc.pid)
    if not identity or identity['pgid']!=proc.pid:
        proc.terminate();raise RuntimeError('Owned model process group absent')
    thread=threading.Thread(target=h.stream_log,args=(proc.stdout,row['usage_logs'][replica]['path'],key),daemon=True);thread.start()
    return dict(process=proc,identity=identity,log_thread=thread,arm=arm,replica=replica)


def endpoint(cfg,arm,index):
    row=layout(cfg,arm)
    replica=(index%4)//row['ep'] if row['pp']==1 else index%row['replicas']
    return dict(cfg,port=port(cfg,arm,replica))


def payload(cfg,arm,tokens,output,index):
    p=dict(input_ids=tokens,sampling_params=dict(temperature=0,max_new_tokens=output,ignore_eos=True),stream=True)
    if layout(cfg,arm)['ep']>1:p['data_parallel_rank']=index%layout(cfg,arm)['dp']
    return p


def cohort(cfg,arm,key,tokens,output,new_tokens=None):
    gate=threading.Barrier(len(tokens))
    ps=[payload(cfg,arm,t,output,i) for i,t in enumerate(tokens)]
    with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
        fs=[pool.submit(b.stream_one,endpoint(cfg,arm,i),key,p,gate) for i,p in enumerate(ps)]
        rows=[f.result() for f in fs]
    duration=max(r['finished'] for r in rows)-min(r['started'] for r in rows)
    return dict(requests=rows,makespan_s=duration,output_tokens_per_s=sum(r['output_tokens'] for r in rows)/duration,
        input_tokens_per_s=(sum(len(t) for t in tokens) if new_tokens is None else len(tokens)*new_tokens)/duration)


def each_server(cfg,arm,fn):
    with ThreadPoolExecutor(max_workers=layout(cfg,arm)['replicas']) as pool:
        return list(pool.map(fn,[dict(cfg,port=port(cfg,arm,r)) for r in range(layout(cfg,arm)['replicas'])]))


def flush(cfg,arm,key):
    return each_server(cfg,arm,lambda c:b.flush_cache(c,key))


def build_tokens(cfg,length):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
    body=tok.encode('Distributed inference sends token activations to experts. Attention reads a private context. Measurements retain cache and topology boundaries.\n',add_special_tokens=False)
    return [[5000+i]+(body*((length+len(body)-1)//len(body)))[:length-1] for i in range(cfg['requests'])]


def capacity_values(info):
    found=[]
    def visit(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in ('max_total_num_tokens','token_capacity') and isinstance(x,(int,float)) and x>0:found.append(int(x))
                if k!='server_args':visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    visit(info)
    if not found:raise RuntimeError('Native server did not export allocator capacity')
    return min(found)


def context_for(cfg,local_capacity):
    reserve=2*cfg['new_prefill_tokens']+cfg['output_tokens']+256
    return min(cfg['max_context']-reserve, int(local_capacity*cfg['capacity_fill']/cfg['local_requests'])-reserve)//64*64


def stable_profiles(out,health,timeout=180):
    deadline=time.monotonic()+timeout;last=None;stable=0
    while time.monotonic()<deadline:
        health();files=sorted(list(out.rglob('*.json.gz'))+list(out.rglob('*.pt.trace.json')))
        signature=[(str(p),p.stat().st_size) for p in files]
        if files and signature==last:
            stable+=1
            if stable>=2:
                try:
                    for p in files:
                        with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:json.load(f)
                    return signature
                except (EOFError,OSError,ValueError):pass
        else:stable=0
        last=signature;time.sleep(1)
    raise TimeoutError('Profiler exports did not become stable and parseable')


def profiles(run,cfg,arm,key,tokens,name,health):
    for phase in ('prefill','decode'):
        out=run/'results'/name/'traces'/phase;out.mkdir(parents=True,exist_ok=True)
        instances=[dict(cfg,port=port(cfg,arm,r)) for r in range(layout(cfg,arm)['replicas'])]
        def start(c):
            target=out/f"port{c['port']}";target.mkdir(exist_ok=True)
            return b.profile_action(c,key,'/start_profile',dict(output_dir=str(target),num_steps=cfg['profile_steps'] if phase=='decode' else None,
                activities=['CPU','GPU'],record_shapes=True,with_stack=False,profile_id=name+'-'+phase,merge_profiles=False,detailed_annotations=True))
        with ThreadPoolExecutor(max_workers=len(instances)) as pool:list(pool.map(start,instances))
        q=cfg['new_prefill_tokens'] if phase=='prefill' else 1
        extended=[t+[7000+int(phase=='decode')]+[198]*(q-1) for t in tokens]
        measured=cohort(cfg,arm,key,extended,1 if phase=='prefill' else 32,new_tokens=q)
        h.save(out/'cohort.json',dict(performance_sample=False,result=measured))
        if phase=='prefill':
            with ThreadPoolExecutor(max_workers=len(instances)) as pool:list(pool.map(lambda c:b.profile_action(c,key,'/stop_profile'),instances))
        files=stable_profiles(out,health)
        h.save(out/'manifest.json',dict(performance_sample=False,files=files))

def cache_command(cfg,arm,key,command):
    def one(c):
        body=json.dumps(dict(server_args=dict(benchmark_cache_cmd=command))).encode()
        req=urllib.request.Request(f"http://127.0.0.1:{c['port']}/set_internal_state",data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as response:answer=json.load(response)
        if not isinstance(answer,list) or not answer or not all(r is True for r in answer):raise RuntimeError('Idle cache fence refused: '+str(answer))
        return h.safe_server_info(h.http_json(c['port'],'/get_server_info',key,timeout=30))
    return each_server(cfg,arm,one)


def tier_summary(m,length):
    tiers={k:sum((r.get('cached_tokens_details') or {}).get(k,0) for r in m['requests']) for k in ('device','host','storage')}
    expected=len(m['requests'])*length
    return dict(tokens=tiers,target_prefix_tokens=expected,prefix_hit_fraction=sum(min(r.get('cached_tokens') or 0,length) for r in m['requests'])/expected,
        cpu_prefix_token_fraction=tiers['host']/expected,gpu_prefix_token_fraction=tiers['device']/expected,
        cpu_hit_requests=sum((r.get('cached_tokens_details') or {}).get('host',0)>0 for r in m['requests']),
        full_prefix_hit_requests=sum((r.get('cached_tokens') or 0)>=length-256 for r in m['requests']),
        native_tier_details_present=all(isinstance(r.get('cached_tokens_details'),dict) for r in m['requests']))


def set_pp_options(run,cfg,arm,key,policy,microbatch):
    import os
    fd=os.open(run/'gap_policy.bin',os.O_RDWR|os.O_NOFOLLOW)
    try:os.pwrite(fd,b'a' if policy==10 else bytes([ord('0')+policy]),0)
    finally:os.close(fd)
    def one(c):
        body=json.dumps(dict(server_args=dict(pp_max_micro_batch_size=microbatch))).encode()
        req=urllib.request.Request(f"http://127.0.0.1:{c['port']}/set_internal_state",data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as response:answer=json.load(response)
        if not isinstance(answer,list) or not answer or not all(r is True for r in answer):raise RuntimeError('PP microbatch update refused '+str(answer))
    each_server(cfg,arm,one)


def select_pp(run,cfg,arm,key,tokens,length,cache_mode,phase,health,out):
    pp=layout(cfg,arm)['pp']
    candidates=[(policy,microbatch) for microbatch in (16,8,4) for policy in (0,1)]
    if pp==4:candidates=[(0,16),(1,16),(1,8),(1,4)]
    pilots=[];reference={};q=cfg['new_prefill_tokens'] if phase=='prefill' else 1
    for n,(policy,microbatch) in enumerate(candidates):
        health();progress(run,cfg,f'select_pp_{arm}_{length}_{cache_mode}_{phase}',candidate=n,policy=policy,microbatch=microbatch)
        set_pp_options(run,cfg,arm,key,policy,microbatch)
        rates=[]
        for repeat in range(2):
            cache_command(cfg,arm,key,2 if cache_mode=='cpu_replay' else 1)
            branch=14000+(100 if phase=='decode' else n*10)+repeat
            extended=[t+[branch]+[198]*(q-1) for t in tokens]
            m=cohort(cfg,arm,key,extended,1 if phase=='prefill' else cfg['output_tokens'],new_tokens=q)
            signature=[r.get('output_ids') or r.get('text') for r in m['requests']]
            if n==0:reference[repeat]=signature
            equivalent=signature==reference[repeat] if phase=='decode' else None
            score=m['input_tokens_per_s'] if phase=='prefill' else m['output_tokens_per_s']
            tiers=tier_summary(m,length)
            h.save(out/f'pilot.{cache_mode}.{phase}.p{policy}.mb{microbatch}.r{repeat}.json',dict(performance_sample=False,optimization_selection=True,output_equal_to_control=equivalent,policy=policy,microbatch=microbatch,result=m,cache_tiers=tiers))
            if phase=='decode' and not equivalent:score=None
            rates.append(score)
        pilots.append(dict(policy=policy,microbatch=microbatch,pilot_scores=rates,mean_score=statistics.mean([x for x in rates if x is not None]) if all(x is not None for x in rates) else -1,unit='new_input_tokens_per_s' if phase=='prefill' else 'output_tokens_per_s'))
    selected=max(pilots,key=lambda r:r['mean_score'])
    set_pp_options(run,cfg,arm,key,selected['policy'],selected['microbatch'])
    h.save(out/f'selection.{cache_mode}.{phase}.json',dict(candidates=pilots,selected=selected,selection_samples_excluded_from_formal_metrics=True))
    return selected


def replay_workload(run,cfg,arm,key,length,health):
    name=f'{arm}_p{length}';out=run/'results'/name;out.mkdir(parents=True,exist_ok=True)
    tokens=build_tokens(cfg,length)
    if layout(cfg,arm)['pp']>1:set_pp_options(run,cfg,arm,key,0,16)
    manifest=dict(arm=arm,node=cfg['node'],B=cfg['requests'],P=length,Q_prefill=cfg['new_prefill_tokens'],Q_decode=1,output_decode=cfg['output_tokens'],hicache=True,
        full_context_tokens=cfg['requests']*length,prompt_hashes=[hashlib.sha256(bytes(str(t),'ascii')).hexdigest() for t in tokens],prefix_origin='model-produced native HiCache L1 GPU and L2 CPU; no checkpoint restore')
    h.save(out/'workload.json',manifest)
    progress(run,cfg,'cold_prefill_'+name);health();flush(cfg,arm,key)
    cold=cohort(cfg,arm,key,tokens,1)
    if any((r.get('cached_tokens') or 0)>64 for r in cold['requests']):raise RuntimeError('Cold prefix hit cache')
    h.save(out/'metrics.cold_prefill.json',b.summarize([cold],dict(manifest,phase='cold_prefill',cached_P=0,Q=length,repeats=1)))
    h.save(out/'host_write_fence.json',cache_command(cfg,arm,key,1))
    for cache_mode in ('natural_replay','cpu_replay'):
        for phase in ('prefill','decode'):
            selected=select_pp(run,cfg,arm,key,tokens,length,cache_mode,phase,health,out) if layout(cfg,arm)['pp']>1 else None
            rounds=[];audits=[]
            for repeat in range(-1,cfg['repeats']):
                health();progress(run,cfg,cache_mode+'_'+phase+'_'+name,repeat=repeat)
                before=cache_command(cfg,arm,key,2 if cache_mode=='cpu_replay' else 1)
                h.save(out/f'{cache_mode}.{phase}.r{repeat}.before.json',before)
                q=cfg['new_prefill_tokens'] if phase=='prefill' else 1
                branch=10000+(100 if phase=='decode' else 0)+(1000 if cache_mode=='cpu_replay' else 0)+repeat+1
                extended=[t+[branch]+[198]*(q-1) for t in tokens]
                m=cohort(cfg,arm,key,extended,1 if phase=='prefill' else cfg['output_tokens'],new_tokens=q)
                tiers=tier_summary(m,length);m['cache_tiers']=tiers
                if not tiers['native_tier_details_present']:raise RuntimeError('Native HiCache tier details missing')
                h.save(out/f'{cache_mode}.{phase}.r{repeat}.json',dict(performance_sample=repeat>=0,result=m))
                if cache_mode=='cpu_replay' and tiers['tokens']['host']<=0:raise RuntimeError('Controlled CPU replay produced no actual host hits')
                if repeat>=0:rounds.append(m);audits.append(tiers)
            h.save(out/f'metrics.{cache_mode}.{phase}.json',dict(b.summarize(rounds,dict(manifest,phase=phase,cache_mode=cache_mode,Q=q,repeats=cfg['repeats'])),cache_tiers_by_round=audits,selected_optimization=selected))


def worker(run,cfg):
    incoming=json.loads(sys.stdin.readline());env=dict(os.environ,**incoming['environment']);check(run,cfg,env,True)
    def interrupted(signum,frame):raise RuntimeError('Guard/controller requested cleanup')
    signal.signal(signal.SIGTERM,interrupted)
    def watch_parent():
        sys.stdin.read();os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=watch_parent,daemon=True).start();key=h.read_key(cfg)
    topo=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,timeout=20)
    (run/'topology.txt').write_text(topo.stdout+topo.stderr)
    def health():check(run,cfg,env,True)
    common_p=cfg['common_context'];maximum_p=cfg['large_context']
    for row in cfg['layouts']:
        arm=row['id'];children=[]
        try:
            progress(run,cfg,'loading_'+arm)
            for replica in range(row['replicas']):children.append(launch(run,cfg,arm,replica,env,key))
            infos=[];deadline=time.monotonic()+cfg['startup_timeout_s']
            for replica,child in enumerate(children):
                while True:
                    health()
                    if child['process'].poll() is not None:raise RuntimeError(arm+f'.r{replica} exited during startup')
                    try:info=h.http_json(port(cfg,arm,replica),'/get_server_info',key,timeout=3);break
                    except Exception:
                        if time.monotonic()>deadline:raise TimeoutError('Full model startup deadline')
                        time.sleep(2)
                infos.append(info)
                startup=Path(row['usage_logs'][replica]['path']).read_text()
                limits=re.findall(r'max_total_num_tokens=\d+.*?max_running_requests=(\d+)',startup)
                expected_running=cfg['local_requests'] if row['pp']==1 else cfg['requests']//row['replicas']
                if not limits or int(limits[-1])!=expected_running:
                    raise RuntimeError('Native effective request limit differs from local B16: '+str(limits))
                evidence=run/'evidence'/f'{arm}.r{replica}'/('ep2' if row['ep']>1 else 'pp2')/'qualification'
                qs=list(evidence.glob('*.json'))
                if len(qs)!=row['ep']*row['pp'] or not all(json.loads(p.read_text()).get('passed') for p in qs):raise RuntimeError('Actual full-model rank qualification missing')
                h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',h.safe_server_info(info))
                states=info.get('internal_states',[])
                if not states or not all(x.get('enable_hierarchical_cache') is True for x in states):raise RuntimeError('HiCache did not activate on every reported scheduler')
                req=urllib.request.Request(f"http://127.0.0.1:{port(cfg,arm,replica)}/freeze_gc",headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=60) as response:
                    if not response.read().decode().startswith('Garbage collection frozen.'):raise RuntimeError('GC freeze not acknowledged')
            caps=[capacity_values(info) for info in infos]
            local_cap=min(caps);h.save(run/'evidence'/arm/'capacity.json',dict(local_allocator_tokens=caps,per_attention_instance_min_tokens=local_cap,attention_instances=row['replicas']*row['dp'],aggregate_conservative_tokens=local_cap*row['replicas']*row['dp'],local_mamba_slots=cfg['local_mamba_slots'],local_max_running=cfg['local_requests']))
            # Untimed compilation/graph warmup, small and identical on all four GPUs.
            flush(cfg,arm,key);cohort(cfg,arm,key,build_tokens(cfg,4096),32)
            for length in cfg['workload_contexts']:
                replay_workload(run,cfg,arm,key,length,health)
        finally:
            if children:
                evidence=h.cleanup(children);h.save(run/'evidence'/arm/'cleanup.json',evidence)
                if not evidence['cleanup_verified']:raise RuntimeError('Owned process cleanup failed')
    return 0


def controller(run,cfg):
    h.save(run/'task/controller_identity.json',dict(pid=os.getpid(),started=time.time()))
    check(run,cfg,os.environ);packet=dict(environment={k:os.environ[k] for k in h.GUARD_KEYS})
    failure=None
    with (run/'controller.log').open('ab',buffering=0) as log:
        command=[cfg['python'],str(Path(__file__).resolve()),'--worker','--run',str(run)]
        proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',cfg['node'],shlex.join(command)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        proc.stdin.write((json.dumps(packet)+'\n').encode());proc.stdin.flush()
        try:
            while proc.poll() is None:
                check(run,cfg,os.environ)
                progress_path=run/'task/progress.json'
                if progress_path.exists():
                    value=json.loads(progress_path.read_text())
                    if time.time()-value.get('timestamp',0)>5:
                        value.update(timestamp=time.time(),controller_pid=os.getpid(),heartbeat_source='live Guard-authenticated controller with SSH worker alive')
                        h.save(progress_path,value)
                time.sleep(2)
            if proc.returncode:raise RuntimeError('Worker failed; inspect exact rank logs')
        except BaseException as e:failure=type(e).__name__;raise
        finally:
            proc.stdin.close()
            try:proc.wait(timeout=85)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
            evidence=[json.loads(p.read_text()) for p in run.glob('evidence/*/cleanup.json')]
            passed=bool(evidence) and all(e.get('cleanup_verified') for e in evidence)
            progress(run,cfg,'failed' if failure else 'complete',error=failure)
            h.save(run/'task/cleanup.json',dict(experiment_id=cfg['id'],task_id='comparison',timestamp=time.time(),passed=passed,gpu_idle=passed,model_processes_exited=passed,source='actual owned-group worker cleanup',worker_evidence_count=len(evidence)))
    return 0


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--engine-entry',action='store_true');ap.add_argument('--arm');ap.add_argument('--replica',type=int,default=0);args=ap.parse_args()
    cfg=json.loads((args.run/'config.json').read_text())
    try:
        result=engine_entry(args.run,cfg,args.arm,args.replica) if args.engine_entry else worker(args.run,cfg) if args.worker else controller(args.run,cfg)
    except BaseException:
        traceback.print_exc();result=1
    sys.exit(result or 0)
