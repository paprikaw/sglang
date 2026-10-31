"""One exact-node supervisor per fresh DeepEP diagnostic mode."""
import argparse
import hmac
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from common import ROOT,MODES,environment,save,sha
from guard_authorization import validate_guard,authorization_signature
from process_cleanup import enable_subreaper,process_record,track_descendants,stop_owned,gpu_cleanup_query


def command(cfg,run,mode,node):
    if mode not in MODES or node not in cfg['nodes']:
        raise ValueError('Unknown mode/node')
    return [cfg['python'],'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=4',
        f'--node-rank={cfg["nodes"].index(node)}',f'--master-addr={cfg["nodes"][0]}',
        f'--master-port={cfg["master_port"]+MODES.index(mode)}','--max-restarts=0',
        str(ROOT/'runtime/worker.py'),'--run',str(run),'--mode',mode]


def clean_exit(row):
    return row.get('owned_processes_exited') is True and row.get('gpu_cleanup',{}).get('passed') is True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=MODES)
    args=parser.parse_args();run=args.run.resolve()
    cfg=json.loads((run/'config.json').read_text())
    node=os.uname().nodename.split('.')[0]
    packet=json.loads(sys.stdin.readline(65536));payload=packet['payload']
    receipt=validate_guard(run,cfg,payload['guard_env'],run/'visibility'/node)
    if not hmac.compare_digest(packet['signature'],authorization_signature(payload,receipt['token'])):
        raise RuntimeError('Controller packet signature mismatch')
    if (payload['node']!=node or payload['mode']!=args.mode or payload['lease_id']!=receipt['lease_id']
            or payload['config_sha256']!=sha(run/'config.json')):
        raise RuntimeError('Controller node/mode/config binding differs')
    owner=process_record(os.getpid())
    if owner is None or not enable_subreaper():
        raise RuntimeError('Owned process identity/subreaper unavailable')
    cancelled={'signal':0}
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(signum,lambda number,_:cancelled.update(signal=number))
    link=select.poll();link.register(sys.stdin.fileno(),select.POLLIN|select.POLLHUP|select.POLLERR)
    output=run/args.mode;output.mkdir(parents=True,exist_ok=True)
    child,owned,rc,reason,failure=None,{},1,'launch_error',None
    try:
        cmd=command(cfg,run,args.mode,node)
        save(output/f'{node}_launch.json',dict(node=node,mode=args.mode,command=cmd,lease_id=receipt['lease_id']))
        with (output/f'{node}_worker.log').open('x') as log:
            child=subprocess.Popen(cmd,env=environment(cfg,args.mode,run),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=process_record(child.pid)
        if record:owned[child.pid]=record['ticks']
        deadline=time.monotonic()+cfg['worker_timeout_s']
        while child.poll() is None:
            track_descendants(owner['pid'],owner['ticks'],owned)
            if cancelled['signal'] or (run/'STOP').exists():
                raise InterruptedError('Guard/owner cancelled diagnosis')
            if time.monotonic()>deadline:
                raise TimeoutError('Bounded DeepEP diagnostic deadline')
            if link.poll(100):
                data=os.read(sys.stdin.fileno(),4096)
                if not data or data.strip().upper()==b'STOP':
                    raise InterruptedError('Controller channel closed')
        rc=child.returncode;reason='completed' if rc==0 else 'worker_failed'
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error))
    finally:
        exited=stop_owned(child,owned,owner['pid'],owner['ticks']) if child else True
        cleanup=gpu_cleanup_query(20)
        row=dict(node=node,mode=args.mode,returncode=rc,reason=reason,failure=failure,
            owned_processes_exited=exited,gpu_cleanup=cleanup,lease_id=receipt['lease_id'],
            passed=rc==0 and exited and cleanup['passed'])
        save(output/f'{node}_exit.json',row)
    # A native fault is an explicit diagnostic outcome, not a valid profile.
    return 0 if clean_exit(row) else 1


if __name__=='__main__':
    raise SystemExit(main())
