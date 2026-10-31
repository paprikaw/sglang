import importlib.util
import json
import os
from pathlib import Path
import subprocess

spec=importlib.util.spec_from_file_location('archive_helper','/tmp/track_sglang_optimizations_20261031.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.NEW=json.loads((m.OUT/'new_commits.json').read_text())
selected=set(json.loads(Path('/tmp/sglang_filtered_sources_20261031.json').read_text()))
topics=sorted({Path(p).parts[1] for p in selected})
allowed={'.py','.sh','.cu','.cuh','.cpp','.h','.patch','.diff','.md','.yaml','.yml','.toml'}
for topic in topics:
    for code in (m.ROOT/'experiments'/topic).glob('runs/*/code'):
        for p in code.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in allowed and '__pycache__' not in p.parts:
                selected.add(str(p.relative_to(m.ROOT)))
for p in (m.ROOT/'tools').rglob('*'):
    if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
        selected.add(str(p.relative_to(m.ROOT)))
repositories={}
cache={}
def boundary(p):
    if p==m.ROOT:return None
    if p not in cache:
        cache[p]=p if p.is_symlink() or (p/'.git').exists() else boundary(p.parent)
    return cache[p]
clean=set()
for name in selected:
    b=boundary((m.ROOT/name).parent)
    if b is None:clean.add(name)
    elif (b/'.git').exists():repositories[str(b)]=None
groups={}
for name in sorted(clean):
    parts=Path(name).parts
    if parts[0]=='tools':key='tools/'+parts[1]
    elif parts[1]=='experiment_summary':key='reports'
    elif 'agentx' in parts[1]:key='agentx'
    elif any(k in parts[1] for k in ['ep_pp','expert_surface','token_comm','single_layer','bandwidth','deepep']):key='component-and-communication'
    elif any(k in parts[1] for k in ['pp','prefill']):key='pipeline-and-prefill'
    else:key='benchmark-support'
    groups.setdefault(key,[]).append(name)
for repo in repositories:
    repositories[repo]={'head':m.git(repo,'rev-parse','HEAD'),'branch':m.git(repo,'branch','--show-current'),'dirty':m.git(repo,'status','--porcelain','-uno'),'remote':m.git(repo,'remote','-v')}
(m.OUT/'nested_repositories.json').write_text(json.dumps(repositories,indent=2)+'\n')
(m.OUT/'source_selection.json').write_text(json.dumps({'root':str(m.ROOT),'file_count':len(clean),'groups':{k:len(v) for k,v in groups.items()},'excluded_nested_repositories':len(repositories),'excluded_runtime':'Large logs, generated payloads, datasets, weights, KV bytes, native binaries, upstream audit copies and Guard runtime bundles stay in GPFS.'},indent=2)+'\n')
Path('/tmp/sglang_final_source_paths_20261031.json').write_text(json.dumps(sorted(clean),indent=2))
print('Final source files',len(clean),'Nested repositories',len(repositories),flush=True)
m.snapshot(m.ROOT,'codex/sglang-optimization-records-20261031','fa52c3cd95d2c171c9eac9a8bfdc32296aa8c75f',[('chore(research): version '+key+' optimization sources and evidence',paths) for key,paths in sorted(groups.items())])
print('Snapshots complete',sum(len(x['commits']) for x in m.NEW),flush=True)
