"""Owned cleanup remains active even if preparation/import/model startup fails."""
import argparse,os,time
from pathlib import Path
from common import ROOT,config,save
from auth import authenticate
from owned_worker import run_owned_worker
import supervision

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--node-rank',type=int,required=True)
    a=p.parse_args();cfg=config(a.run);node=os.uname().nodename.split('.')[0]
    if cfg['nodes'][a.node_rank]!=node:raise RuntimeError('Wrong node')
    base=a.run/cfg['arm']/node
    def prepare():
        authenticate(a.run,cfg,local=True)
        save(base/'owner.json',dict(owner=supervision.process_record(os.getpid()),node=node,timestamp=time.time()))
    last=[0]
    def check():
        if time.monotonic()-last[0]>2:
            authenticate(a.run,cfg,local=True,frozen=False);last[0]=time.monotonic()
    code,_=run_owned_worker(command=[cfg['python'],str(ROOT/'runtime/server_entrypoint.py')],environment=dict(os.environ),
        exit_path=base/'exit.json',log_path=base/'server.log',node=node,layout=cfg['layout'],timeout_s=cfg['timeout_s'],
        prepare=prepare,verify_alive=check,supervision=supervision)
    return code
if __name__=='__main__':raise SystemExit(main())
