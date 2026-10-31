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
    h.save(run/'task/progress.json',dict(experiment_id=cfg['id'],task_id='comparison',stage=stage,timestamp=time.time(),controller_pid=os.getpid(),valid_points=len(points)+(2 if (run/'results/four_ep1_at_ep4max/admission.json').exists() else 0),expected=11,**extra))


def layout(cfg,arm):
    return next(x for x in cfg['layouts'] if x['id']==arm)


def port(cfg,arm,replica):
    return cfg['port']+cfg['layouts'].index(layout(cfg,arm))*100+replica*10


def engine_args(cfg,arm,replica):
    ep=layout(cfg,arm)['ep'];d=ep
    args=['--model-path',cfg['model_path'],'--host','127.0.0.1','--port',str(port(cfg,arm,replica)),
        '--dist-init-addr',f"127.0.0.1:{cfg['dist_port']+cfg['layouts'].index(layout(cfg,arm))*100+replica*10}",
        '--tp-size',str(ep),'--ep-size',str(ep),'--dp-size',str(ep),'--pp-size','1',
        '--dtype','bfloat16','--kv-cache-dtype','bfloat16','--context-length',str(cfg['max_context']),
        '--mem-fraction-static',str(cfg['mem_fraction_static']),'--page-size','64',
        '--max-running-requests',str(cfg['local_requests']*d),
        '--max-mamba-cache-size',str(cfg['local_mamba_slots']*d),'--mamba-max-states-per-path','1','--mamba-ssm-dtype','float32',
        '--cuda-graph-max-bs-decode',str(cfg['local_requests']*d),'--cuda-graph-backend-prefill','disabled',
        '--chunked-prefill-size',str(4096*d),'--max-prefill-tokens',str(4096*d),
        '--attention-backend','fa3','--linear-attn-backend','triton',
        '--moe-runner-backend','deep_gemm','--moe-a2a-backend','deepep' if ep>1 else 'none',
        '--disable-custom-all-reduce','--skip-server-warmup','--json-model-override-args','{"language_model_only":true}']
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
        NVSHMEM_DISABLE_P2P='0',DEEPEP_LOW_LATENCY_NVLINK_ONLY='0')
    command=[cfg['python'],str(Path(cfg['guard_scripts'])/'model_cache_adapter.py'),'exec-local','--bindings',env[h.GUARD_KEYS[3]],'--',
        cfg['python'],str(Path(__file__).resolve()),'--run',str(run),'--engine-entry','--arm',arm,'--replica',str(replica)]
    proc=subprocess.Popen(command,env=env,cwd=cfg['engine_root'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    identity=h.process_record(proc.pid)
    if not identity or identity['pgid']!=proc.pid:
        proc.terminate();raise RuntimeError('Owned model process group absent')
    thread=threading.Thread(target=h.stream_log,args=(proc.stdout,row['usage_logs'][replica]['path'],key),daemon=True);thread.start()
    return dict(process=proc,identity=identity,log_thread=thread,arm=arm,replica=replica)


def endpoint(cfg,arm,index):
    return dict(cfg,port=port(cfg,arm,0 if layout(cfg,arm)['ep']>1 else index%4))


def payload(cfg,arm,tokens,output,index):
    p=dict(input_ids=tokens,sampling_params=dict(temperature=0,max_new_tokens=output,ignore_eos=True),stream=True)
    if layout(cfg,arm)['ep']>1:p['data_parallel_rank']=index%4
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


def resident_workload(run,cfg,arm,key,length,name,health,do_profiles=True):
    tokens=build_tokens(cfg,length);resultdir=run/'results'/name;resultdir.mkdir(parents=True,exist_ok=True)
    manifest=dict(arm=arm,B=cfg['requests'],P=length,Q_prefill=cfg['new_prefill_tokens'],Q_decode=1,output_decode=cfg['output_tokens'],
        full_context_tokens=cfg['requests']*length,prompt_hashes=[hashlib.sha256(bytes(str(t),'ascii')).hexdigest() for t in tokens],prefix_origin='real model-produced GPU radix and GDN state; no CPU restore')
    h.save(resultdir/'workload.json',manifest)
    progress(run,cfg,'cold_prefill_'+name,P=length,B=cfg['requests']);health();flush(cfg,arm,key)
    cold=cohort(cfg,arm,key,tokens,1)
    if any((r['cached_tokens'] or 0)>64 for r in cold['requests']):raise RuntimeError('Cold prefix unexpectedly hit cache')
    h.save(resultdir/'metrics.cold_prefill.json',b.summarize([cold],dict(manifest,phase='cold_prefill',Q=length,repeats=1)))
    stable=0
    for attempt in range(cfg['cache_prime_limit']):
        health();progress(run,cfg,'priming_'+name,P=length,attempt=attempt)
        prime=cohort(cfg,arm,key,[t+[198] for t in tokens],32,new_tokens=1)
        hit=all((r['cached_tokens'] or 0)>=length-64 for r in prime['requests'])
        h.save(resultdir/f'prime.{attempt}.json',dict(performance_sample=False,all_prefixes_hit=hit,result=prime))
        stable=stable+1 if hit else 0
        if stable>=2:break
    if stable<2:
        h.save(resultdir/'residency_rejected.json',dict(performance_sample=False,status='unstable_prefix_residency',P=length,B=cfg['requests'],prime_attempts=cfg['cache_prime_limit']))
        raise ResidencyError('No two consecutive fully resident priming cohorts')
    for phase in ('prefill','decode'):
        rounds=[]
        for repeat in range(-1,cfg['repeats']):
            health();progress(run,cfg,phase+'_'+name,repeat=repeat,P=length)
            q=cfg['new_prefill_tokens'] if phase=='prefill' else 1
            branch=10000+(0 if phase=='prefill' else 100)+repeat+1
            extended=[t+[branch]+[198]*(q-1) for t in tokens]
            m=cohort(cfg,arm,key,extended,1 if phase=='prefill' else cfg['output_tokens'],new_tokens=q)
            if any((r['cached_tokens'] or 0)<length-64 for r in m['requests']):
                h.save(resultdir/(phase+'.cache_failure.json'),m)
                raise RuntimeError('Resident cohort did not retain model-produced prefix/GDN state')
            if repeat>=0:
                rounds.append(m);h.save(resultdir/(phase+'.partial.json'),dict(rounds=rounds))
        h.save(resultdir/('metrics.'+phase+'.json'),b.summarize(rounds,dict(manifest,phase=phase,Q=q,repeats=cfg['repeats'])))
    if do_profiles:
        progress(run,cfg,'profiles_'+name);profiles(run,cfg,arm,key,tokens,name,health)


def overloaded_workload(run,cfg,arm,key,length,local_capacity,health):
    name='four_ep1_at_ep4max';out=run/'results'/name;out.mkdir(parents=True,exist_ok=True)
    h.save(out/'workload.json',dict(arm=arm,B=cfg['requests'],P=length,local_allocator_capacity=local_capacity,protocol='capacity overload; queue and GPU prefix recomputation included, separate from hot Decode'))
    if length+cfg['output_tokens']>=local_capacity:
        # The actual native allocator admission limit already proves infeasibility.
        # Submit one request to capture native rejection instead of fabricating a time.
        try:
            cohort(cfg,arm,key,build_tokens(dict(cfg,requests=1),length),1)
        except urllib.error.HTTPError as e:
            detail=e.read().decode()[:2000]
            if e.code!=400:raise
            h.save(out/'admission.json',dict(status='rejected',http_status=e.code,error=type(e).__name__,detail=detail,P=length,local_allocator_capacity=local_capacity));return
        raise RuntimeError('Native admission exceeded the exported per-request pool bound')
    tokens=build_tokens(cfg,length);flush(cfg,arm,key)
    for phase in ('cold_prefill','decode'):
        progress(run,cfg,'capacity_overload_'+phase,P=length);health()
        m=cohort(cfg,arm,key,[t if phase=='cold_prefill' else t+[198] for t in tokens],1 if phase=='cold_prefill' else cfg['output_tokens'])
        h.save(out/('metrics.'+phase+'.json'),b.summarize([m],dict(phase=phase,B=cfg['requests'],P=length,cache_admission='overloaded',repeats=1)))


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
    common_p=None;maximum_p=None
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
                if not limits or int(limits[-1])!=cfg['local_requests']:
                    raise RuntimeError('Native effective request limit differs from local B16: '+str(limits))
                evidence=run/'evidence'/f'{arm}.r{replica}'/('ep2' if row['ep']>1 else 'pp2')/'qualification'
                qs=list(evidence.glob('*.json'))
                if len(qs)!=row['ep'] or not all(json.loads(p.read_text()).get('passed') for p in qs):raise RuntimeError('Actual full-model rank qualification missing')
                h.save(run/'evidence'/f'{arm}.r{replica}'/'server_info.json',h.safe_server_info(info))
                req=urllib.request.Request(f"http://127.0.0.1:{port(cfg,arm,replica)}/freeze_gc",headers={'Authorization':'Bearer '+key})
                with urllib.request.urlopen(req,timeout=60) as response:
                    if not response.read().decode().startswith('Garbage collection frozen.'):raise RuntimeError('GC freeze not acknowledged')
            caps=[capacity_values(info) for info in infos]
            local_cap=min(caps);h.save(run/'evidence'/arm/'capacity.json',dict(local_allocator_tokens=caps,per_attention_instance_min_tokens=local_cap,attention_instances=4,aggregate_conservative_tokens=local_cap*4,local_mamba_slots=cfg['local_mamba_slots'],local_max_running=cfg['local_requests']))
            # Untimed compilation/graph warmup, small and identical on all four GPUs.
            flush(cfg,arm,key);cohort(cfg,arm,key,build_tokens(cfg,4096),32)
            if arm=='four_ep1_common':
                common_p=context_for(cfg,local_cap)
                if common_p<4096:raise RuntimeError('EP1 capacity too small for controlled B64 comparison')
                h.save(run/'common_plan.json',dict(P=common_p,B=64,local_capacity=local_cap))
                resident_workload(run,cfg,arm,key,common_p,'four_ep1_common',health)
            elif arm=='ep4':
                resident_workload(run,cfg,arm,key,common_p,'ep4_common',health)
                candidates=[]
                for fill in cfg['capacity_fill_candidates']:
                    maximum_p=context_for(dict(cfg,capacity_fill=fill),local_cap)
                    name='ep4_maximum_f'+str(round(fill*100))
                    candidate=dict(P=maximum_p,B=64,local_capacity=local_cap,context_tokens=maximum_p*64,allocator_fraction=maximum_p*16/local_cap,candidate_fill=fill,result_case=name)
                    candidates.append(candidate)
                    h.save(run/'ep4_capacity_candidates.json',candidates)
                    try:
                        resident_workload(run,cfg,arm,key,maximum_p,name,health)
                    except ResidencyError:
                        candidate['residency']='rejected';h.save(run/'ep4_capacity_candidates.json',candidates);continue
                    candidate['residency']='passed'
                    h.save(run/'ep4_maximum_plan.json',candidate);h.save(run/'ep4_capacity_candidates.json',candidates)
                    break
                else:raise ResidencyError('No near-capacity candidate retained all prefixes')
            else:overloaded_workload(run,cfg,arm,key,maximum_p,local_cap,health)
        finally:
            if children:
                evidence=h.cleanup(children);h.save(run/'evidence'/arm/'cleanup.json',evidence)
                if not evidence['cleanup_verified']:raise RuntimeError('Owned process cleanup failed')
    return 0


def controller(run,cfg):
    check(run,cfg,os.environ);packet=dict(environment={k:os.environ[k] for k in h.GUARD_KEYS})
    failure=None
    with (run/'controller.log').open('ab',buffering=0) as log:
        command=[cfg['python'],str(Path(__file__).resolve()),'--worker','--run',str(run)]
        proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',cfg['node'],shlex.join(command)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        proc.stdin.write((json.dumps(packet)+'\n').encode());proc.stdin.flush()
        try:
            while proc.poll() is None:
                check(run,cfg,os.environ);time.sleep(2)
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
        sys.exit(result or 0)
    except BaseException:
        traceback.print_exc();sys.exit(1)
