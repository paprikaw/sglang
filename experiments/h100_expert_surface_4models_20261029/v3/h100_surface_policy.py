"""Exact user-requested four-model, eight single-GPU Expert-only qualification.

Extends the existing isolated Expert-only profile. No cache mutation, complete
model loading, Attention, communication, new booking, or arbitrary GPU commands.
"""
import hashlib,json,math,os,re,socket
from pathlib import Path

BENCHMARK='h100_four_model_expert_surface_v1'
GEOMETRY={'glm47':(3,40,5120,1536),'hy3':(1,48,4096,1536),
          'qwen35b':(0,256,2048,512),'qwen122b':(0,256,3072,1024)}
MODEL_PATHS={'glm47':'/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ',
             'hy3':'/data/gpfs/projects/punim2565/models/Hy3-BF16',
             'qwen35b':'/data/gpfs/projects/punim2715/huggingface/Qwen3.6-35B-A3B',
             'qwen122b':'/data/gpfs/projects/punim2715/huggingface/Qwen3.5-122B-A10B-FP8'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frozen(spec,d):
    if set(d)!={'path','sha256'} or d not in spec['frozen_inputs']:raise ValueError('Unfrozen Expert input')
    p=Path(d['path'])
    if not p.is_absolute() or p.is_symlink() or not p.is_file() or digest(p)!=d['sha256']:
        raise ValueError('Changed Expert input')
    return p
def validate(spec):
    c=spec['model_cache']
    fields={'policy','reason','benchmark','selection','measurement_plan','workers','entrypoint','worker','command','worker_command'}
    if set(c)!=fields or c['policy']!='expert_layer_only' or c['benchmark']!=BENCHMARK:
        raise ValueError('Unsupported four-model Expert contract')
    if spec.get('engine')!='sglang' or spec.get('attach_existing') or not c['reason']:
        raise ValueError('Fresh Expert-only SGLang experiment required')
    tasks=spec['tasks'];nodes=[n for t in tasks for n in t.get('nodes',[])]
    if len(tasks)!=2 or len(nodes)!=2 or len(set(nodes))!=2 or spec['max_active_nodes']!=2:
        raise ValueError('Exactly two exclusive booked nodes required')
    if spec['workload']!=dict(kind='h100_four_model_expert_layer_compute',model_loaded=False,
        expert_compute=True,attention_compute=False,communication=False,gpu_count=8):
        raise ValueError('Only eight independent single-GPU Expert computations admitted')
    plan=json.loads(frozen(spec,c['measurement_plan']).read_text())
    if (plan['tokens']!=[2**p for p in range(3,16)] or plan['modes']!=['eager','cuda_graph']
        or plan['samples_per_mode']!=30 or plan['blocks']!=3 or plan['samples_per_block']!=10
        or plan['routing']!='balanced' or plan['token_semantics']!='local_token_expert_assignment_rows'):
        raise ValueError('Unsupported grid or timing protocol')
    selections=json.loads(frozen(spec,c['selection']).read_text())
    if set(selections)!=set(GEOMETRY):raise ValueError('Four exact model selections required')
    for name,s in selections.items():
        if (s['model_path']!=MODEL_PATHS[name] or (s['layer'],s['resident'],s['hidden'],s['intermediate'])!=GEOMETRY[name]
            or s['expert_ids']!=list(range(s['resident']))):raise ValueError('Changed model/layer/expert geometry')
        if not s['tensors'] or any('.mlp.experts.' not in k for k in s['tensors']):
            raise ValueError('Selection must contain only real routed Expert weights')
        for filename,key in [('config.json','config_sha256'),('model.safetensors.index.json','index_sha256')]:
            frozen(spec,dict(path=str(Path(s['model_path'])/filename),sha256=s[key]))
    workers=json.loads(frozen(spec,c['workers']).read_text())
    if len(workers)!=8 or len({w['id'] for w in workers})!=8:raise ValueError('Eight unique workers required')
    for node in nodes:
        ws=[w for w in workers if w['node']==node]
        if {w['gpu'] for w in ws}!={0,1,2,3} or {w['model'] for w in ws}!=set(GEOMETRY):
            raise ValueError('Each node must run one model per GPU, covering four models')
    count=sum(sum(e<=t for t in plan['tokens'] for e in selections[m]['active_experts'])*2 for m in GEOMETRY)
    if any(t['expected']!=count for t in tasks):raise ValueError('Incorrect expected cells')
    for desc,command in [(c['entrypoint'],c['command']),(c['worker'],c['worker_command'])]:
        p=frozen(spec,desc)
        if len(command)!=4 or command[1]!=str(p) or command[2]!='--run' or not Path(command[3]).is_absolute():
            raise ValueError('Changed frozen controller or worker command')
        if not re.fullmatch(r'python(?:[23](?:\.[0-9]+)*)?',Path(command[0]).name):raise ValueError('Python entrypoint required')
    if c['command'][3]!=c['worker_command'][3]:raise ValueError('Worker run path differs')
    for d in spec['frozen_inputs']:frozen(spec,d)
    return dict(c)
def worker_gpu(contract,node,worker_id,visible):
    workers=json.loads(Path(contract['workers']['path']).read_text())
    found=[w for w in workers if w['node']==node and w['id']==worker_id]
    if len(found)!=1 or visible!=str(found[0]['gpu']):raise RuntimeError('GPU/model worker differs from admitted plan')
    return visible
def verify_result(contract,binding):
    p=Path(contract['command'][3])/'profile/result.json'
    if p.stat().st_mtime<binding['admitted_at']:raise RuntimeError('Stale Expert result')
    value=json.loads(p.read_text());plan=json.loads(Path(contract['measurement_plan']['path']).read_text())
    selections=json.loads(Path(contract['selection']['path']).read_text());workers=json.loads(Path(contract['workers']['path']).read_text())
    expected={(w['id'],t,e,mode) for w in workers for t in plan['tokens']
              for e in selections[w['model']]['active_experts'] if e<=t for mode in plan['modes']}
    rows=value.get('rows',[])
    if value.get('passed') is not True or len(rows)!=len(expected) or {(r['worker'],r['T'],r['E'],r['mode']) for r in rows}!=expected:
        raise RuntimeError('Missing, duplicate or failed Expert cells')
    if value.get('lease_id')!=binding['lease_id'] or value.get('selection_sha256')!=contract['selection']['sha256']:
        raise RuntimeError('Expert result authorization differs')
    for r in rows:
        xs=r['raw_ms']
        if len(xs)!=30 or any(not math.isfinite(x) or x<=0 for x in xs):raise RuntimeError('Invalid CUDA timing samples')
    if not value.get('numerical') or not all(x['passed'] for x in value['numerical']):raise RuntimeError('Numerical checks failed')
    return dict(passed=True,policy='expert_layer_only',benchmark=BENCHMARK,result=str(p),cells=len(rows))
