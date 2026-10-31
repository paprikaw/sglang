#!/usr/bin/env python3
"""Read-only booked-GPU overview. Never claims, starts inference or changes cache state."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
REMOTE=r'''import importlib.util,json,os,sys
from pathlib import Path
p=json.loads(sys.stdin.readline());out={"sampled_at":__import__('time').time(),"weights":None,"kv":[]}
try:
 source=Path(p['hardware_probe'])
 spec=importlib.util.spec_from_file_location('overview_hardware',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 out['hardware']=mod.probe()
except Exception as exc:out['hardware']={'status':'unknown','nics':[],'gpus':[],'error':type(exc).__name__}
try:
 row=p['bank'];ident=row['manifest'];pid=ident['pid']
 stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
 if int(stat[19])!=ident['process_start_ticks'] or stat[0]=='Z':raise RuntimeError('daemon identity changed')
 source=Path(p['engine'])/'python/sglang/srt/weight_cache/cpu_protocol.py'
 spec=importlib.util.spec_from_file_location('overview_cache_protocol',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 with mod.CpuWeightCacheClient(row['socket_path'],timeout=2)as client:s=client.status()
 if s.get('pid')!=pid or s.get('source_commit')!=row['source_commit'] or s.get('uid')!=os.getuid():raise RuntimeError('daemon status identity differs')
 out['weights']={k:s.get(k)for k in ['ok','current_bytes','used_bytes','source_commit']};out['weights']['entries']=[{k:e.get(k)for k in ['key','state','size']}for e in s.get('entries',[])]
except Exception as exc:out['weight_error']=type(exc).__name__
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for item in p.get('kv',[]):
 path=Path(item['path']);same=boot==item.get('boot_id');exists=path.is_dir()if same else False
 out['kv'].append(dict(path=str(path),same_boot=same,exists=exists,recorded_bytes=item.get('bytes'),verified_at=item.get('verified_at')))
print(json.dumps(out))
'''

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}if default is None else default


def finite(value):
    try:
        value=float(value)
        return value if value>=0 and value<float('inf')else None
    except (ValueError,TypeError):return None


def gpu_telemetry(hardware,now):
    """Physical readings are independent of Guard ownership and serving state."""
    samples=[]
    for gpu in hardware.get('gpus',[]):
        value=finite(gpu.get('utilization_gpu_pct'))
        samples.append(dict(index=gpu.get('index'),utilization_gpu_pct=value if value is not None and value<=100 else None,
                            memory_used_mib=finite(gpu.get('memory_used_mib')),memory_total_mib=finite(gpu.get('memory_total_mib'))))
    values=[g['utilization_gpu_pct']for g in samples if g['utilization_gpu_pct']is not None]
    sampled=finite(hardware.get('sampled_at'))
    status='unknown'
    if values and sampled and sampled<=now+5:
        status='current'if hardware.get('status')=='current'and now-sampled<=90 else'historical'
    return dict(status=status,sampled_at=sampled,source='nvidia-smi',gpus=samples,
                utilization_pct=round(sum(values)/len(values),1)if values else None)


def node_state(node,leases,worker,now):
    active=[l for l in leases if node in l.get('nodes',[])and l.get('state')in('active','quarantined')]
    if len(active)>1:return 'unknown','租约冲突',None
    if active:
        lease=active[0]
        if lease['state']=='quarantined':return 'draining','清理待确认',lease
        if now-float(lease.get('heartbeat_at',0))>float(lease.get('ttl_s',0)):return 'unknown','租约过期',lease
        if lease.get('preemption_requested'):return 'draining','正在让出 GPU',lease
        if lease.get('resource_priority','normal')!='background':return 'foreground','前台持有租约',lease
        if worker.get('deployment_id')==lease['experiment_id']:
            status=worker.get('worker_status',worker.get('status'))
            if status=='failed':return 'failed','benchmark 故障 · 正在恢复',lease
            if status in ('stale','awaiting_worker'):
                return 'unknown','benchmark 运行状态待确认',lease
            timestamp=finite(worker.get('worker_updated_at',worker.get('timestamp',0)))or 0
            if status=='running'and -5<=now-timestamp<45:
                return 'benchmark','空闲 benchmark',lease
            if status=='loading':return 'loading','benchmark 加载中',lease
        return 'unknown','后台运行状态待确认',lease
    return 'idle','空闲／等待 benchmark',None


def state_label(state,label,worker):
    if state!='idle' or not worker.get('state_reason'):return label
    return {
        'waiting_for_compatible_recipe':'等待匹配模型配置',
        'waiting_for_model_identity':'等待核验当前模型身份',
        'waiting_for_model_loader':'等待当前模型加载配置',
        'protected_partial_cache_requires_complete_qualified_group':'等待完整模型配对',
        'cache_inventory_unavailable':'权重缓存查询失败',
        'hardware_inventory_unavailable':'硬件信息暂不可用',
        'hardware_outside_qualified_recipe':'等待匹配 GPU 型号的模型配置',
        'old_booking_cache_requires_verified_retirement':'等待核验旧预约缓存',
        'unregistered_gpu_work_or_inventory_unavailable':'等待核验 GPU 占用',
    }.get(worker['state_reason'],'等待 benchmark 条件')


def snapshot(cfg):
    now=time.time();errors=[];fleet=read(Path(cfg['fleet_root'])/'all_nodes_state.json')
    result=subprocess.run([cfg.get('squeue','/apps/slurm/latest/bin/squeue'),'--me','-h','-o','%i|%T|%N|%L|%b'],capture_output=True,text=True,timeout=15,check=True)
    bookings=[];pending=[]
    for line in result.stdout.splitlines():
        job,state,expr,left,tres=line.split('|',4)
        if 'gpu'not in tres.lower():continue
        if state!='RUNNING':pending.append(dict(job_id=job,state=state));continue
        if re.fullmatch(r'[A-Za-z0-9_.-]+',expr):nodes=[expr]
        else:nodes=subprocess.check_output([cfg.get('scontrol','/apps/slurm/latest/bin/scontrol'),'show','hostnames',expr],text=True,timeout=10).split()
        count=re.search(r'gpu(?::[^:,()]+)?:(\d+)',tres)
        for node in nodes:bookings.append(dict(node=node,job_id=job,remaining=left,gpu_count=int(count[1])if count else None))
    regpath=Path(cfg['registry_root']);registry=read(regpath/'gpu_registry/registry.json')
    if 'leases'not in registry:errors.append('Guard 状态不可用')
    ledger=read(regpath/'cpu_weight_daemons/registry.json');banks={r['node']:r for r in ledger.get('entries',[])}
    kv=read(cfg['kv_registry']);entries=kv.get('entries',{});entries=entries.values()if isinstance(entries,dict)else entries
    locations={}
    for entry in entries:
        for loc in entry.get('locations',[]):
            node=loc.get('node');path=loc.get('path')
            if node and path:locations.setdefault(node,{})[path]=dict(loc,verified_at=entry.get('last_verified_at'))
    rows=[]
    for booking in sorted(bookings,key=lambda r:r['node']):
        node=booking['node'];worker=fleet.get('nodes',{}).get(node,{})
        state,label,lease=node_state(node,registry.get('leases',[]),worker,now)
        label=state_label(state,label,worker)
        if 'leases'not in registry:state,label='unknown','Guard 状态不可用'
        row=dict(booking,state_reason=worker.get('state_reason'),state=state,state_label=label,experiment=lease.get('experiment_id')if lease else None,
                 hardware=dict(status='unknown',nics=[],gpus=[]),
                 weights=dict(status='unknown',label='无当前可核验记录'),kv=dict(runtime_usage_pct=None,runtime_used_tokens=None,persistent_label='历史登记 · 未核验'if locations.get(node)else'无登记记录',persistent_count=0))
        bank=banks.get(node,{});run=worker.get('run')or read(Path(cfg['fleet_root'])/'state.json').get('attempts',{}).get(node,{}).get('run') or cfg.get('cache_runs',{}).get(node);binding={};node_spec={}
        if run:
            binding=read(Path(run)/'dashboard_guard/model_cache_bindings.json').get('nodes',{}).get(node,{})
            descriptor=binding.get('node_spec',{});path=descriptor.get('path')
            if path:
                try:
                    raw=Path(path).read_bytes()
                    if hashlib.sha256(raw).hexdigest()==descriptor.get('sha256'):node_spec=json.loads(raw)
                except(OSError,ValueError):pass
        qualified_cache=(bank.get('state')=='active'and str(bank.get('expected_job_id'))==booking['job_id']and node_spec.get('source_commit')==bank.get('source_commit')and binding.get('daemon_id')==bank.get('daemon_id'))
        # Hardware does not require a model cache or a GPU lease. Reuse this
        # node's existing read-only SSH sample; never run an extra GPU workload.
        payload=dict(bank={k:bank[k]for k in ('manifest','socket_path','source_commit')}if qualified_cache else {},
                     engine=node_spec.get('engine_root'),kv=list(locations.get(node,{}).values()),
                     hardware_probe=str(ROOT/'hardware_inventory.py'))
        try:
            cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=3',node,'/usr/bin/python3 -B -c '+shlex.quote(REMOTE)]
            proc=subprocess.run(cmd,input=json.dumps(payload)+'\n',capture_output=True,text=True,timeout=20,check=True);live=json.loads(proc.stdout)
            hardware=live.get('hardware',{})
            if isinstance(hardware,dict)and hardware.get('node')==node:
                row['hardware']=dict(hardware,job_id=booking['job_id'])
            elif isinstance(hardware,dict)and hardware.get('status')=='unknown':
                row['hardware']['error']=hardware.get('error','HardwareProbeFailed')
            else:row['hardware']['error']='NodeIdentityMismatch'
            status=live.get('weights')if qualified_cache else None
            if status and status.get('ok'):
                items=status.get('entries',[]);expected={e['key']for e in bank.get('status',{}).get('entries',[])if e.get('state')=='ready'};actual={e['key']for e in items if e['state']=='ready'};ready=bool(expected)and expected<=actual and all(e['state']=='ready'for e in items)
                row['weights']=dict(status='ready'if ready else'partial',label='已就绪'if ready else'部分未就绪',model=Path(node_spec['model_path']).name,
                    kind='完整'if bank.get('cache_scheme')=='canonical_full_weight'else'分片',bytes=status.get('current_bytes')or status.get('used_bytes'),sampled_at=live['sampled_at'],gpu_loaded=state=='benchmark')
            elif qualified_cache:row['weights']['label']='守护进程未确认'
            valid=[v for v in live.get('kv',[])if v['same_boot']and v['exists']]
            row['kv'].update(persistent_count=len(valid),persistent_label=(str(len(valid))+' 处目录存在 · 内容未复核')if valid else('旧登记已失效'if locations.get(node)else'无登记记录'))
        except Exception as exc:
            row['hardware']['error']=type(exc).__name__
            if qualified_cache:row['weights'].update(label='实时查询失败',error=type(exc).__name__)
        # Read-only metrics only for an exact active benchmark; never guess a foreground endpoint.
        if state=='benchmark':
            try:
                if len(lease.get('nodes',[]))==1:
                    text=urlopen('http://'+node+':19170/metrics',timeout=2).read(2**20).decode()
                    def metric(name):
                        vals=[finite(l.rsplit(None,1)[-1])for l in text.splitlines()if l.startswith('sglang:'+name+'{')]
                        return max((v for v in vals if v is not None),default=None)
                    value=metric('token_usage');row['kv']['runtime_usage_pct']=value*100 if value is not None else None
                    row['kv']['runtime_used_tokens']=metric('kv_used_tokens')
                else:
                    # Each physical node logs its own Attention DP's actual scheduler usage.
                    log=Path(run)/'ep8'/node/'server.log'
                    if now-log.stat().st_mtime<20:
                        with log.open('rb')as f:f.seek(max(0,log.stat().st_size-32000));tail=f.read().decode(errors='replace')
                        vals=re.findall(r'Prefill batch,.*?token usage: ([0-9.]+)',tail)
                        if vals:row['kv']['runtime_usage_pct']=float(vals[-1])*100
            except Exception:pass
        row['gpu_telemetry']=gpu_telemetry(row['hardware'],time.time())
        row['gpu_utilization_pct']=row['gpu_telemetry']['utilization_pct']
        rows.append(row)
    return dict(schema_version=1,timestamp=time.time(),read_only=True,nodes=rows,pending=pending,errors=errors)


def refresh_ownership(snapshot,cfg):
    now=time.time();registry=read(Path(cfg['registry_root'])/'gpu_registry/registry.json')
    fleet=read(Path(cfg['fleet_root'])/'all_nodes_state.json');result=dict(snapshot);rows=[]
    for old in snapshot.get('nodes',[]):
        row=dict(old);worker=fleet.get('nodes',{}).get(row['node'],{})
        state,label,lease=node_state(row['node'],registry.get('leases',[]),worker,now)
        if 'leases'not in registry:state,label='unknown','Guard 状态不可用'
        row.update(state=state,state_label=state_label(state,label,worker),state_reason=worker.get('state_reason'),experiment=lease.get('experiment_id')if lease else None)
        row['weights']=dict(row['weights'])
        if now-row['weights'].get('sampled_at',0)>120 and row['weights'].get('status')=='ready':row['weights'].update(status='stale',label='采样过期 · 待复核')
        row['weights']['gpu_loaded']=state=='benchmark'and row['weights'].get('status')=='ready'
        hardware=dict(row.get('hardware',dict(status='unknown',nics=[],gpus=[])))
        if hardware.get('status')=='current'and now-hardware.get('sampled_at',0)>120:hardware['status']='historical'
        row['hardware']=hardware
        telemetry=dict(row.get('gpu_telemetry',dict(status='unknown',sampled_at=None,gpus=[],utilization_pct=None)))
        if telemetry.get('status')=='current'and now-(telemetry.get('sampled_at')or 0)>90:telemetry['status']='historical'
        row['gpu_telemetry']=telemetry
        if state!='benchmark' or now-snapshot.get('timestamp',0)>90:row['kv']=dict(row['kv'],runtime_usage_pct=None,runtime_used_tokens=None)
        rows.append(row)
    coordinator=read(Path(cfg['fleet_root'])/'state.json')
    errors=list(result.get('errors',[]))
    if coordinator.get('state')=='storage_paused':
        errors.append('项目剩余文件数不足：后台 benchmark 暂停，保留实验与清理所需空间')
    elif coordinator.get('state')=='audit_failed':
        message=coordinator.get('error','')
        if '122' in message or 'quota' in message.lower() or 'No space left' in message:
            errors.append('后台恢复暂停：项目存储配额不足，正在重试')
        else:errors.append('后台恢复检查失败，正在重试')
    elif now-coordinator.get('timestamp',0)>120:
        errors.append('后台协调器状态已过期，等待服务恢复')
    result.update(nodes=rows,ownership_sampled_at=now,errors=errors)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args();cfg=read(args.config)
    status_file=args.config.parent/'status.json'
    current=read(status_file,dict(timestamp=0,nodes=[],pending=[],errors=['正在读取资源']));lock=threading.Lock()
    def collect():
        nonlocal current
        while True:
            try:result=snapshot(cfg)
            except Exception:result=dict(current,errors=['资源读取失败 · 保留上次状态'],collection_failed_at=time.time())
            with lock:current=result
            try:
                tmp=status_file.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));tmp.replace(status_file)
            except OSError:pass
            time.sleep(10)
    threading.Thread(target=collect,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_GET(self):
            route=self.path.split('?')[0]
            if route in ('/api/status','/api/health'):
                with lock:value=dict(current)
                if route.endswith('status'):value=refresh_ownership(value,cfg)
                if route.endswith('health'):value=dict(ok=True,read_only=True,service='resource-overview-v1')
                data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();kind='application/json; charset=utf-8'
            else:
                paths={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if route not in paths:self.send_error(404);return
                data=(ASSETS/paths[route]).read_bytes();kind={'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}[route]
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    ThreadingHTTPServer(('127.0.0.1',cfg.get('port',19510)),Handler).serve_forever()
if __name__=='__main__':main()
