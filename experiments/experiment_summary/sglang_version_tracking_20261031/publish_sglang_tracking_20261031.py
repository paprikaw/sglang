import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
OUT=ROOT/'experiments/experiment_summary/sglang_version_tracking_20261031'
RESULT=[]
def git(p,*args):
    return subprocess.check_output(['git','-C',str(p),*args],stderr=subprocess.STDOUT,text=True).strip()
def create(p,ref,sha):
    try:old=git(p,'rev-parse','--verify',ref)
    except subprocess.CalledProcessError:old=None
    if old is None:git(p,'update-ref',ref,sha,'0'*40)
    elif old!=sha:raise RuntimeError('Ref collision '+ref)
def publish(p,label,branches,tags=()):
    refs=[('refs/heads/'+x) for x in branches]+['refs/tags/'+x for x in tags]
    for start in range(0,len(refs),40):
        batch=refs[start:start+40]
        print('Pushing',label,start,len(batch),flush=True)
        proc=subprocess.run(['git','-C',str(p),'push','--atomic','origin',*[x+':'+x for x in batch]],capture_output=True,text=True,timeout=240)
        (OUT/(label+'_push_'+str(start)+'.log')).write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError(label+' push failed: '+proc.stderr[-2000:])
    remote_text=git(p,'ls-remote','origin')
    (OUT/(label+'_remote_verified.tsv')).write_text(remote_text+'\n')
    remote={line.split()[1]:line.split()[0] for line in remote_text.splitlines()}
    entries=[]
    for ref in refs:
        sha=git(p,'rev-parse',ref)
        if remote.get(ref)!=sha:raise RuntimeError('Remote mismatch '+ref)
        entries.append({'ref':ref,'sha':sha})
    before_file=OUT/('remote_before_'+{'sglang':'0','replay':'1','guard':'2'}[label]+'.tsv')
    before={line.split()[1]:line.split()[0] for line in before_file.read_text().splitlines()}
    if remote.get('refs/heads/main')!=before.get('refs/heads/main'):
        raise RuntimeError(label+' main unexpectedly moved')
    result={'repository':git(p,'remote','get-url','origin'),'label':label,'branches':len(branches),'tags':len(tags),'main_unchanged':True,'refs':entries}
    RESULT.append(result)
    (OUT/'remote_verification.json').write_text(json.dumps(RESULT,indent=2)+'\n')
    print('Verified',label,len(branches),'branches',len(tags),'tags; main unchanged',flush=True)

deps=json.loads(Path('/tmp/sglang_dependency_inventory_20261031.json').read_text())
deep=Path(next(x['path'] for x in deps if '/.third_party/deepep-' in x['path']))
git(ROOT,'fetch','--no-tags','--no-write-fetch-head',str(deep),'refs/heads/codex/*:refs/heads/codex/dependency-deepep/*')
for row in deps:
    if '/.third_party/deepep-' in row['path'] and not row['branch']:
        name=Path(row['path']).name
        ref='refs/heads/codex/dependency-deepep/'+name
        git(ROOT,'fetch','--no-tags','--no-write-fetch-head',str(deep),row['head'])
        create(ROOT,ref,row['head'])
for path,row in json.loads((OUT/'nested_repositories.json').read_text()).items():
    if row['dirty']:raise RuntimeError('Unarchived nested changes '+path)
    target='refs/heads/'+row['branch']
    git(ROOT,'fetch','--no-tags','--no-write-fetch-head',path,row['head'])
    create(ROOT,target,row['head'])
create(ROOT,'refs/heads/codex/agentx-gdn-pd-hicache-r6-20261031','fc014e3553c4d234f38f3088a3158c0333e64830')
create(ROOT,'refs/heads/codex/agentx-hicache-restore-retry-r7-20261031','797f6488cf5fea2f88e38ccaeb1703443deec537')
branches=git(ROOT,'for-each-ref','--format=%(refname:short)','refs/heads/codex').splitlines()
tags=git(ROOT,'tag','-l','serving/sglang/*').splitlines()
publish(ROOT,'sglang',branches,tags)
runner=Path('/data/gpfs/projects/punim2715/vllm_workbench/sglang-cpu-kv-decode-replay')
publish(runner,'replay',git(runner,'for-each-ref','--format=%(refname:short)','refs/heads/codex').splitlines())
guard=ROOT/'.guard-worktrees/qwen-capacity-20261017'
publish(guard,'guard',['codex/archive/qwen-capacity-adapter-wip-20261031'])
print('ALL_REQUESTED_REFS_VERIFIED',flush=True)
