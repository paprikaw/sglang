"""Pinned PP microbatch A/B with unchanged canonical full CPU-bank admission."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
ENGINE=Path(os.environ.get('GLM_ENGINE_ROOT',str(WORKSPACE/'.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023')))
COMMIT=os.environ.get('GLM_ENGINE_COMMIT')
PYTHON='/data/gpfs/projects/punim2715/vllm_workbench/sglang_overlap_compare/venv-dd15fb57-cu12/bin/python'
MODEL=Path('/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ')
GUARD=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts')
MEMORY=Path('/data/gpfs/projects/punim2715/vllm_workbench/sglang-memory-bank')
DEEP=WORKSPACE/'.third_party/deepep-token-routing-ep4-20261004'
NATIVE=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/runtime/glm47_ep8_tp8_ep4_tp4_c4_20261004_r1/native4/python/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so')
NATIVE_SHA='9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581'
SHIM=WORKSPACE/'experiments/deepep_ah_diagnosis_h100_20260928/trace_verbs.so'
SEED=WORKSPACE/'experiments/glm47_ep8_p2p_vs_tp8_full_20261002/humming_cache'
CUDA='/apps/easybuild-2022/easybuild/software/Core/CUDA/12.8.0'
LAYOUTS={'ep8':dict(tp=8,dp=2,ep=8,pp=1),'pp2ep4':dict(tp=4,dp=1,ep=4,pp=2)}
ARMS={'producer':dict(layout='pp2ep4',mtp=True,role='producer'),
      'ep8_mtp_on':dict(layout='ep8',mtp=True,role='consumer'),
      'ep8_mtp_off':dict(layout='ep8',mtp=False,role='consumer'),
      'pp2ep4_mtp_on':dict(layout='pp2ep4',mtp=True,role='consumer'),
      'pp2ep4_mtp_off':dict(layout='pp2ep4',mtp=False,role='consumer')}
CONSUMER_ORDER=('ep8_mtp_on','ep8_mtp_off','pp2ep4_mtp_on','pp2ep4_mtp_off')
SCHEMA='glm47-semantic-bank-fourway-v1'
PINNED_COMMIT='9e48f876a8eb2d7460317bc713bfa8680d78aea5'
_REUSE_BANK_VERIFIED=set()

def select_arm(cfg,arm):
    if arm not in ARMS:raise ValueError('Unknown frozen arm')
    return dict(cfg,arm=arm,**ARMS[arm])

def execution_arms(cfg):
    return {arm:ARMS[arm] for arm in (CONSUMER_ORDER if cfg.get('reuse_semantic_bank',False) else ARMS)}

def reuse_bank_descriptor(bank_dir,cfg):
    """Read-only qualification of a previously published producer bank."""
    bank=Path(bank_dir)
    if not bank.is_absolute() or not bank.is_dir():raise ValueError('Reuse bank must be an existing absolute directory')
    from semantic_bank import load_manifest,validate_key_record
    from workload import prompts
    os.environ['GLM_SEMANTIC_INPUTS']=cfg['semantic_inputs']
    inputs=prompts();manifest=load_manifest(bank,inputs,cfg['bank_model_identity'])
    gate=manifest.get('publication_gate',{})
    if (gate.get('producer_exited') is not True or gate.get('cleanup_passed') is not True
        or manifest.get('one_semantic_prefill_producer') is not True or manifest.get('second_canonical_copy') is not False):
        raise ValueError('Reusable bank lacks original semantic producer exit/cleanup proof')
    evidence=gate.get('evidence',[])
    if len(evidence)<2 or len({row.get('path') for row in evidence})<2:raise ValueError('Missing original producer evidence')
    for row in evidence:
        if not Path(row['path']).is_absolute() or sha(row['path'])!=row['sha256']:raise ValueError('Original producer evidence changed')
    reference={}
    for piece in manifest['pieces']:
        for row in piece['requests']:
            index=row['request_index']
            for mode,eagle in (('mtp_key',True),('plain_key',False)):
                validate_key_record(inputs[index],eagle,row[mode])
                key=(index,mode)
                if key in reference and reference[key]!=row[mode]:raise ValueError('Producer native hash metadata differs across ranks')
                reference[key]=row[mode]
    ready=bank/'READY.json'
    return dict(path=str(ready),sha256=sha(ready),manifest_sha256=manifest['manifest_sha256'],
        bank_identity_sha256=manifest['bank_identity_sha256'],identity=manifest['identity'])

def _verify_reuse_bank_once(cfg,run):
    # config() is also read by idle/forward instrumentation. Full source JSON,
    # provenance and native-key checks belong to first startup, not each phase.
    # Guard authenticate() still independently hashes all frozen inputs; every
    # cohort import still performs native keys and actual CPU host readback.
    key=(str(Path(run).absolute()),json.dumps(cfg,sort_keys=True,separators=(',',':')))
    if key not in _REUSE_BANK_VERIFIED:
        if reuse_bank_descriptor(cfg['semantic_bank_dir'],cfg)!=cfg.get('source_semantic_bank_ready'):
            raise RuntimeError('Frozen reusable semantic bank identity changed')
        _REUSE_BANK_VERIFIED.add(key)

def bank_identity(cfg):
    return dict(engine_commit=cfg['engine_commit'],model_config_sha256=sha(MODEL/'config.json'),
        model_index_sha256=sha(MODEL/'model.safetensors.index.json'),
        model_revision=cfg['model_revision'],quantization='awq',
        semantic_manifest_sha256=sha(Path(cfg['semantic_inputs'])/'manifest.json'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path=Path(path)
    if not path.is_absolute():
        run=Path(os.environ['GLM_RUN_DIR']);parts=path.parts
        path=run.joinpath(*parts)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def config(run):
    d=json.loads((Path(run)/'config.json').read_text())
    arms=execution_arms(d)
    if (d.get('schema')!=SCHEMA or d.get('layouts')!=LAYOUTS or d.get('arms')!=arms
        or d.get('arm_order')!=list(arms) or type(d.get('reuse_semantic_bank',False)) is not bool
        or d.get('engine_commit')!=PINNED_COMMIT
        or d.get('prefix_lengths')!=[4096] or d.get('production_output_tokens')!=64
        or d.get('production_repeats')!=2 or d.get('profile_enabled') is not True
        or d.get('deterministic_inference') is not True or d.get('prefill_truncation_alignment')!=1024
        or d.get('bank_page_cache_max_bytes_per_node')!=12*1024**3
        or d.get('bank_temporary_copy_buffer_max_bytes_per_rank')!=16*1024**2
        or not Path(d['engine_root']).is_absolute()):
        raise RuntimeError('Frozen four-way semantic-bank identity mismatch')
    if COMMIT is not None and (d['engine_commit']!=COMMIT or Path(d['engine_root'])!=ENGINE):
        raise RuntimeError('Runtime engine differs from frozen configuration')
    os.environ['GLM_SEMANTIC_INPUTS']=d['semantic_inputs']
    if d.get('reuse_semantic_bank'):
        _verify_reuse_bank_once(d,run)
    elif d.get('source_semantic_bank_ready') is not None:
        raise RuntimeError('Unexpected reusable bank identity without reuse mode')
    if os.environ.get('GLM_ARM'):d=select_arm(d,os.environ['GLM_ARM'])
    return d

def expected_ranks(layout,node_rank):
    return ([f'TP{r}' for r in range(4*node_rank,4*node_rank+4)] if layout=='ep8'
            else [f'PP{node_rank} TP{r}' for r in range(4)])

def server_args(cfg,node_rank):
    c=LAYOUTS[cfg['layout']]
    args=['--model-path',str(MODEL),'--served-model-name','GLM-4.7-AWQ','--trust-remote-code',
        '--dtype','bfloat16','--quantization','humming','--moe-runner-backend','humming',
        '--tp-size',str(c['tp']),'--pp-size',str(c['pp']),'--ep-size',str(c['ep']),
        '--dp-size',str(c['dp']),
        '--nnodes','2','--node-rank',str(node_rank),'--dist-init-addr',cfg['nodes'][0]+':'+str(cfg['dist_port']),
        '--host','0.0.0.0','--port',str(cfg['api_port']),
        '--enable-deterministic-inference','--attention-backend','triton','--context-length','131072','--page-size','64',
        '--mem-fraction-static','0.90','--max-running-requests','8',
        '--chunked-prefill-size','2048' if cfg['layout']=='ep8' else '1024','--max-prefill-tokens','1024',
        '--disable-custom-all-reduce','--disable-shared-experts-fusion','--disable-overlap-schedule',
        '--schedule-policy','fcfs','--random-seed','42','--watchdog-timeout','600',
        '--cuda-graph-config',json.dumps({'decode':{'backend':'full','bs':[4],'max_bs':4},'prefill':{'backend':'disabled'}},separators=(',',':')),
        '--reasoning-parser','glm45','--tool-call-parser','glm47',
        '--moe-a2a-backend','deepep','--deepep-mode','low_latency','--deepep-dispatcher-output-dtype','bf16',
        '--enable-hierarchical-cache','--hicache-size','35','--hicache-mem-layout','page_first',
        '--hicache-io-backend','kernel','--hicache-write-policy','write_through','--skip-server-warmup']
    if cfg['layout']=='ep8':args+=['--enable-dp-attention']
    else:args+=['--pp-max-micro-batch-size','4']
    if cfg['mtp']:args+=['--speculative-algorithm','EAGLE','--speculative-draft-model-path',str(MODEL),
        '--speculative-num-steps','2','--speculative-eagle-topk','1','--speculative-num-draft-tokens','3']
    return args

def environment(cfg,node_rank,guard_env):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','VLLM_','NVSHMEM_','DEEPEP_','NCCL_','HUMMING_','GLM_')) and k not in ('LD_PRELOAD','PYTHONPATH')}
    env.update(guard_env)
    scratch=f'/tmp/{os.getuid()}-{cfg["id"]}'
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3',SGLANG_ENABLE_PP_SPEC='1' if cfg['layout']=='pp2ep4' and cfg['mtp'] else '0',PYTHONPATH=os.pathsep.join(map(str,(ROOT/'runtime',GUARD,MEMORY/'src',ENGINE/'python',DEEP,NATIVE.parent))),
        LD_LIBRARY_PATH=f'/home/bxb1/conda/lib:{CUDA}/lib64:/apps/easybuild-2022/easybuild/software/Core/OpenSSL/1.1/lib',
        CUDA_HOME=CUDA,CUDA_PATH=CUDA,CUDA_MODULE_LOADING='EAGER',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PATH=f'{Path(PYTHON).parent}:{CUDA}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',MAX_JOBS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HUMMING_COMPILER='nvcc',HUMMING_DISABLE_PARALLEL_BUILD='1',
        TORCH_EXTENSIONS_DIR=scratch+'/torch',TRITON_CACHE_DIR=scratch+'/triton',
        SGLANG_ENABLE_JIT_DEEPGEMM='0',SGLANG_HUMMING_USE_F16_ACCUM='0',SGLANG_HUMMING_MOE_GEMM_TYPE='indexed',
        TORCH_CUDA_ARCH_LIST='8.0',NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_HCA='mlx5_bond_0',
        SGLANG_NUMA_BIND_V2='0',SGLANG_AUTO_NUMA_BIND='1',NCCL_SOCKET_IFNAME='bond0.3027',GLOO_SOCKET_IFNAME='bond0.3027',
        LD_PRELOAD=str(SHIM),DEEPEP_AH_ZERO_STATIC_RATE='1',NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME='bond0.3027',
        NVSHMEM_HCA_LIST='mlx5_bond_0:1',NVSHMEM_IB_ADDR_FAMILY='AF_INET',NVSHMEM_IB_ADDR_RANGE='172.26.92.0/22',
        NVSHMEM_IBGDA_NIC_HANDLER='cpu_host_memory',NVSHMEM_QP_DEPTH='4096',NVSHMEM_DISABLE_P2P='0',
        DEEPEP_LL_NVLINK_PAIR_ONLY='1',DEEPEP_LL_PCIE_P2P_MAX_TOKENS='0',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='256',SGLANG_DEEPEP_DISABLE_NVLINK_LOW_LATENCY='0',
        SGLANG_MEMORY_BANK_ROOT=str(MEMORY),GLM_RUN_DIR=cfg['run_dir'],GLM_NODE_RANK=str(node_rank),
        GLM_LAYOUT=cfg['layout'],GLM_ARM=cfg['arm'],ARM=cfg['arm'],NODE_RANK=str(node_rank),BANK_PHASE='pp2ep4_measure' if cfg['layout']=='pp2ep4' else 'ep8_measure',
        GLM_ENGINE_ROOT=cfg['engine_root'],GLM_ENGINE_COMMIT=cfg['engine_commit'],
        GLM_SEMANTIC_INPUTS=cfg['semantic_inputs'],
        SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE='1024',
        SGLANG_PP_SPEC_MICROBATCH='1' if cfg['layout']=='pp2ep4' and cfg['mtp'] else '0',SGLANG_ENABLE_COLOCATED_BATCH_GEN='0')
    env['PYTHONPATH']=os.pathsep.join(map(str,(ROOT/'runtime',GUARD,MEMORY/'src',Path(cfg['engine_root'])/'python',DEEP,NATIVE.parent)))
    return env
