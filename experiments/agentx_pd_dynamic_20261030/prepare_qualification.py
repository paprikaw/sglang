"""Freeze matched colocated/PD cache qualification on the repaired engine."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
OLD=ROOT.parent/'agentx_pd_feasibility_20261030'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def desc(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser()
    for k in ('run','node','job','family','port','engine','tag'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();run=Path(a.run).resolve();engine=Path(a.engine).resolve()
    commit=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(engine),'status','--porcelain'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(engine),'rev-parse',a.tag+'^{commit}'],text=True).strip()==commit
    spec=importlib.util.spec_from_file_location('agentx_previous_prepare',OLD/'prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(OLD/'prepare.py'),'--run',str(run),'--node',a.node,'--job',a.job,'--family',a.family,'--port',a.port]
    module.main()
    cfg=read(run/'config.json');old_engine=cfg['engine_root']
    def replace(v):
        if isinstance(v,str):return v.replace(old_engine,str(engine))
        if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
        if isinstance(v,list):return [replace(x) for x in v]
        return v
    cfg=replace(cfg);cfg.update(engine_commit=commit,engine_tag=a.tag,
        mooncake_dependency_path=str(OLD/'deps/mooncake_only'),
        request_timeout_s=240,port=int(a.port),dist_port=int(a.port)+3000)
    cfg['protocol']+=' Repaired GDN Decode radix/HiCache ownership; six aligned/unaligned serial probes, same-engine colocated reference; no dynamic claim.'
    save(run/'config.json',cfg)
    worker=run/'code/pd_workload.py';s=worker.read_text()
    s=s.replace('for n in (512,2048,4096)','for n in (512,513,2048,2049,4096,4097)')
    s=s.replace('assert len(reference)==3','assert len(reference)==6')
    worker.write_text(s)
    ds=replace(read(run/'dashboard_spec.json'));ds['title']='AgentX GDN repaired HiCache qualification '+a.family
    files=[Path(x['path']) for x in ds['frozen_inputs']]
    files += [ROOT/'prepare_qualification.py',ROOT/'OBJECTIVE.md']
    files += [engine/p for p in subprocess.check_output(['git','-C',str(engine),'diff','--name-only','0f2f1990994774472b1089d2b0682f0bf926a6b4','HEAD'],text=True).splitlines()]
    dep=OLD/'deps/mooncake_only'
    files+=list((dep/'mooncake').rglob('*.py'))+list((dep/'mooncake').rglob('*.so'))
    files+=list((dep/'mooncake_transfer_engine.libs').rglob('*.so*'))+list(dep.glob('*dist-info/METADATA'))
    ds['frozen_inputs']=list({str(p):desc(p) for p in files}.values())
    save(run/'dashboard_spec.json',ds)
    manifest=replace(read(run/'launch_manifest.json'))
    manifest.update(engine_commit=commit,engine_tag=a.tag,protocol=cfg['protocol'],frozen_inputs=ds['frozen_inputs'],
        claim_boundary='Correctness and bounded AgentX qualification; full static and dynamic comparisons still required.')
    save(run/'launch_manifest.json',manifest)
    print(json.dumps(dict(run=str(run),node=a.node,engine_commit=commit,source_tag=a.tag)))

if __name__=='__main__':main()
