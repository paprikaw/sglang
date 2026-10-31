"""Frozen, single-layer H100 experiment utilities. No Torch/CUDA imports here."""
from pathlib import Path
import hashlib, hmac, json, os, socket, stat, sys, time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
ENGINE = WORKSPACE / '.engine-worktrees/glm47-ep16-canonical-profile-20261004'
PYTHON = '/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
CUDA = '/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
GUARD_SCRIPTS = ROOT / 'guard/skills/llm-experiment-guard/scripts'
KEYS = ('EXPERIMENT_GUARD_LEASE_RECEIPT', 'EXPERIMENT_GUARD_ALLOCATION_FILE',
        'EXPERIMENT_GUARD_ACQUIRED_NODES', 'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')
MODELS = {
 'glm47': dict(path='/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ', layer=3,
               hidden=5120, intermediate=1536, resident=40, total_experts=160,
               backend='Humming AWQ W4A16 grouped masked', experts=[1,2,4,8,16,20,32,40]),
 'hy3': dict(path='/data/gpfs/projects/punim2565/models/Hy3-BF16', layer=1,
             hidden=4096, intermediate=1536, resident=48, total_experts=192,
             backend='DeepGEMM BF16 grouped contiguous', experts=[1,2,4,8,16,24,32,48]),
 'qwen35b': dict(path='/data/gpfs/projects/punim2715/huggingface/Qwen3.6-35B-A3B', layer=0,
                 hidden=2048, intermediate=512, resident=256, total_experts=256,
                 backend='DeepGEMM BF16 grouped contiguous', experts=[1,2,4,8,16,32,64,128,256]),
 'qwen122b': dict(path='/data/gpfs/projects/punim2715/huggingface/Qwen3.5-122B-A10B-FP8', layer=0,
                  hidden=3072, intermediate=1024, resident=256, total_experts=256,
                  backend='DeepGEMM FP8 W8A8 grouped contiguous', experts=[1,2,4,8,16,32,64,128,256]),
}
PLAN = dict(schema_version=1, tokens=[2**p for p in range(3,16)], modes=['eager','cuda_graph'],
            samples_per_mode=30, blocks=3, samples_per_block=10, warmup=5,
            routing='balanced', token_semantics='local_token_expert_assignment_rows',
            measured='two expert GEMMs plus SiLU/multiply and internal FP8 activation quantization',
            excluded=['weights','JIT','routing','input packing','initial input quantization','Attention','communication'])

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n'); tmp.replace(path)
def ticks(pid):
    try: return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,ValueError,IndexError): return None
def stable(path):
    sys.path.insert(0,str(GUARD_SCRIPTS))
    from artifact_visibility import wait_for_stable_json
    return wait_for_stable_json(Path(path), timeout_s=20)[0]
def authenticate(run, env=None, parent=False):
    env=os.environ if env is None else env
    receipt_path=Path(env[KEYS[0]]); st=receipt_path.stat()
    assert st.st_uid==os.getuid() and stat.S_IMODE(st.st_mode)==0o600
    receipt=stable(receipt_path); spec=stable(run/'dashboard_spec.json')
    allocation=stable(env[KEYS[1]]); registry=stable(Path(receipt['registry_dir'])/'registry.json')
    lease=next(x for x in registry['leases'] if x['lease_id']==receipt['lease_id'])
    assert lease['state']=='active' and time.time()-lease['heartbeat_at']<lease['ttl_s']
    assert hmac.compare_digest(hashlib.sha256(receipt['token'].encode()).hexdigest(),lease['token_sha256'])
    assert lease['spec_sha256']==receipt['spec_sha256']==sha(run/'dashboard_spec.json')
    assert lease['experiment_id']==allocation['experiment_id']==spec['id']
    nodes=sorted({n for t in spec['tasks'] for n in t['nodes']})
    assert sorted(lease['nodes'])==sorted(allocation['nodes'])==sorted(env[KEYS[2]].split(','))==nodes
    binding=stable(env[KEYS[3]])
    assert binding['policy']=='expert_layer_only' and binding['lease_id']==lease['lease_id']
    assert binding['expert_layer_contract']==spec['model_cache']
    if parent:
        assert lease['owner_host']==socket.gethostname().split('.')[0]
        assert lease['owner_pid']==os.getppid() and lease['owner_start_ticks']==ticks(os.getppid())
    return receipt,spec,binding
def signature(payload, token):
    return hmac.new(token.encode(),json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()
def worker_environment(run, worker, guard_env):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','VLLM_','NVSHMEM_','DEEPEP_','NCCL_','HUMMING_')) and k not in ('LD_PRELOAD','PYTHONPATH')}
    scratch=Path('/tmp')/f'{os.getuid()}-h100-expert-4models-20261029'/worker['id']
    env.update(CUDA_VISIBLE_DEVICES=str(worker['gpu']), PYTHONPATH=str(ENGINE/'python'),
      CUDA_HOME=CUDA, CUDA_PATH=CUDA, LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
      PATH=f'{Path(PYTHON).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
      PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',MAX_JOBS='2',
      TORCH_CUDA_ARCH_LIST='9.0',SGLANG_ENABLE_JIT_DEEPGEMM='1',SGLANG_JIT_DEEPGEMM_PRECOMPILE='0',
      SGLANG_HUMMING_USE_F16_ACCUM='0',HUMMING_COMPILER='nvcc',HUMMING_DISABLE_PARALLEL_BUILD='1',
      HUMMING_CACHE_DIR=str(scratch/'humming'), HUMMING_TMP_DIR=str(scratch/'humming-tmp'),
      TRITON_CACHE_DIR=str(scratch/'triton'),DEEPGEMM_JIT_CACHE_DIR=str(scratch/'deepgemm'),
      SURFACE_WORKER_ID=worker['id'])
    env.update(guard_env); return env
