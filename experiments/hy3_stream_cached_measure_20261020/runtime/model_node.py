"""Keep cleanup owner outside model import hooks, including failed starts."""
import argparse
import os
from pathlib import Path
import sys
import time
from stream_common import ROOT, OLD, config, save, expected_ranks, verify_streaming_entry
from stream_auth import authenticate, private_json
from staging_directory import create_staging_parent
sys.path.append(str(OLD/'runtime'))
import supervision

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--node-rank',type=int,choices=(0,1),required=True);a=p.parse_args()
    cfg=config(a.run); node=os.uname().nodename.split('.')[0]
    if cfg['nodes'][a.node_rank]!=node: raise RuntimeError('Wrong frozen node')
    sys.path.insert(0,cfg['guard_scripts']);from owned_worker import run_owned_worker
    base=a.run/cfg['layout']/node
    def prepare():
        verify_streaming_entry(cfg,owner=True)
        lease,binding=authenticate(a.run,cfg,local=True)
        spec=private_json(binding['hy3_contract']['node_specs'][node]['path'])
        staging=create_staging_parent(cfg['id'],spec['staging_ledger'])
        save(base/'identity.json',dict(node=node,owner=supervision.process_record(os.getpid()),
             lease_id=lease['lease_id'],staging_parent=staging,timestamp=time.time()))
    status,_=run_owned_worker(command=[cfg['python'],str(ROOT/'runtime/server_entrypoint.py')],
        environment=dict(os.environ,HY3_PROCESS_ROLE='server',HY3_STREAMING_ACTIVE='1'),
        exit_path=base/'exit.json',log_path=base/'server.log',node=node,layout=cfg['layout'],
        timeout_s=cfg['timeout_s'],prepare=prepare,
        verify_alive=lambda:authenticate(a.run,cfg,local=True,frozen=False),supervision=supervision)
    return status
if __name__=='__main__': raise SystemExit(main())
