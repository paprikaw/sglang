"""Uniform per-node workers for four/eight rank worlds."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PYTHON

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True)
    a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());phase=next(x for x in cfg['phases'] if x['id']==a.phase)
    node=os.uname().nodename.split('.')[0];rows=[r for r in phase['placement'] if r['node']==node];children=[]
    for local_rank,row in enumerate(rows):
        env=dict(os.environ,RANK=str(row['rank']),LOCAL_RANK=str(local_rank),WORLD_SIZE=str(phase['world']),
                 MASTER_ADDR=phase['nodes'][0],MASTER_PORT=str(phase['port']))
        children.append(subprocess.Popen([PYTHON,str(ROOT/'worker.py'),'--run',str(a.run),'--phase',a.phase],env=env))
    while any(c.poll() is None for c in children):
        if any(c.poll() not in (None,0) for c in children):
            return 1 # The authenticated parent cleans up every descendant.
        time.sleep(.1)
    return int(any(c.returncode for c in children))

if __name__=='__main__':raise SystemExit(main())
