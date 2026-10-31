"""Full Hy3, bounded weight streaming, native HiCache; no persistent bank edits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'hy3_a100_qualification_20261018'
spec = importlib.util.spec_from_file_location('_hy3_qualified_common', OLD/'runtime/common.py')
qualified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualified)
ENGINE, PYTHON, MODEL = qualified.ENGINE, qualified.PYTHON, qualified.MODEL
LAYOUTS = {k: qualified.LAYOUTS[k] for k in ('ep8', 'pp2ep4')}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    qualified.save(path, value)

def config(run):
    cfg = json.loads((Path(run)/'config.json').read_text())
    if (cfg['schema'] != 'hy3-stream-cached-v1' or cfg['effective_num_hidden_layers'] != 80
            or cfg['layouts'] != LAYOUTS or cfg['run_dir'] != str(Path(run))):
        raise RuntimeError('Frozen full80 comparison required')
    return cfg

def expected_ranks(node_rank):
    return list(range(4*node_rank, 4*node_rank+4))

def server_args(cfg, node_rank):
    layout = cfg['layout']
    args = qualified.server_args(cfg, layout, node_rank)
    for flag in ('--disable-radix-cache',):
        args.remove(flag)
    for flag in ('--max-total-tokens',):
        i = args.index(flag); del args[i:i+2]
    values = {'--context-length': '262144', '--max-running-requests': '8',
              '--chunked-prefill-size': '2048' if layout=='ep8' else '1024',
              '--max-prefill-tokens': '1024', '--mem-fraction-static': '0.96'}
    if layout=='pp2ep4': values['--pp-max-micro-batch-size']='4'
    for flag, value in values.items(): args[args.index(flag)+1]=value
    args += ['--enable-hierarchical-cache', '--hicache-size', '16',
             '--hicache-mem-layout', 'page_first', '--hicache-io-backend', 'kernel',
             '--hicache-write-policy', 'write_through', '--page-size', '64']
    return args

def environment(cfg, node_rank, guard_env):
    env=qualified.environment(cfg,cfg['layout'],node_rank,guard_env)
    for key in list(env):
        if key.startswith(('HY3_WEIGHT_BANK_', 'HY3_CANONICAL_', 'SGLANG_CANONICAL_WEIGHT_CACHE_',
                           'SGLANG_NODE_LOCAL_WEIGHT_CACHE_', 'SGLANG_CPU_WEIGHT_CACHE_')):
            env.pop(key)
    env.update(HY3_STREAMING_ACTIVE='0', HY3_PROCESS_ROLE='owner', HY3_RUN_DIR=cfg['run_dir'],
               HY3_NODE_RANK=str(node_rank), HY3_LAYOUT=cfg['layout'],
               SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='256')
    env['PYTHONPATH']=os.pathsep.join(map(str,(ROOT/'runtime',ROOT/'streaming',
        Path(cfg['guard_scripts']), ENGINE/'python',qualified.DEEP,qualified.NATIVE.parent)))
    return env

def verify_streaming_entry(cfg, *, owner=False):
    if (os.environ.get('HY3_STREAMING_ACTIVE') != ('0' if owner else '1')
            or os.environ.get('HY3_PROCESS_ROLE') != ('owner' if owner else 'server')
            or os.environ.get('HY3_LAYOUT') != cfg['layout']
            or os.environ.get('HY3_RUN_DIR') != cfg['run_dir']):
        raise RuntimeError('Streaming owner/server separation differs')
