"""Guard child; launch both node supervisors concurrently and retain all evidence."""
import argparse,json,os,shlex,signal,socket,subprocess,time
from pathlib import Path
from common import ROOT,KEYS,authenticate,save,signature,stable

def main(run):
    receipt,spec,binding=authenticate(run,parent=True)
    cfg=json.loads((run/'config.json').read_text());children=[];cancelled=[];failure=None
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(s,lambda n,f:cancelled.append(n))
    try:
        for node in cfg['nodes']:
            payload=dict(node=node,env={k:os.environ[k] for k in KEYS})
            cmd=['python3',str(ROOT/'node.py'),'--run',str(run)]
            log=(run/'tasks'/node/'remote.log').open('ab',buffering=0)
            proc=subprocess.Popen(['ssh','-o','ConnectTimeout=8',node,shlex.join(cmd)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            proc.stdin.write(json.dumps(dict(payload=payload,signature=signature(payload,receipt['token'])))+'\n');proc.stdin.flush()
            children.append((node,proc,log))
        deadline=time.monotonic()+3750
        while any(p.poll() is None for n,p,l in children):
            if cancelled:raise InterruptedError('Guard/user cancellation')
            if time.monotonic()>deadline:raise TimeoutError('Controller deadline')
            time.sleep(.5)
    except BaseException as e:
        failure=dict(type=type(e).__name__,error=str(e));save(run/'failure.json',failure)
    finally:
        for node,p,log in children:
            if p.stdin and not p.stdin.closed:p.stdin.close()
        for node,p,log in children:
            try:p.wait(timeout=50)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()
    all_results=[];cleanup=[]
    for node in cfg['nodes']:
        try:
            all_results.extend(stable(run/'tasks'/node/'result.json')['workers'])
            cleanup.append(stable(run/'tasks'/node/'cleanup.json'))
        except BaseException as e:
            failure=dict(type=type(e).__name__,error=str(e));save(run/'collection_failure.json',failure)
    passed=(failure is None and len(all_results)==8 and all(r.get('passed') is True for r in all_results)
            and len(cleanup)==2 and all(r.get('passed') is True for r in cleanup))
    rows=[dict(**row,worker=r['worker'],model=r['model'],node=r['node'],physical_gpu=r['physical_gpu']) for r in all_results for row in r.get('rows',[])]
    numerical=[dict(**check,worker=r['worker']) for r in all_results for check in r.get('numerical',[])]
    save(run/'profile/result.json',dict(passed=passed,lease_id=receipt['lease_id'],selection_sha256=spec['model_cache']['selection']['sha256'],
         rows=rows,numerical=numerical,workers=[{k:v for k,v in r.items() if k not in ['rows','numerical']} for r in all_results],cleanup=cleanup,failure=failure))
    print(json.dumps(dict(passed=passed,rows=len(rows),run=str(run))),flush=True)
    return 0 if passed else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);raise SystemExit(main(p.parse_args().run.resolve()))
