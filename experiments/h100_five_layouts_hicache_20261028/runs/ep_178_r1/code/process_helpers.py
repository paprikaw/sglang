"""Guard-owned A100 native Qwen load/allocator qualification.

Controller: --run ABSOLUTE --model small|large --node BOOKED_NODE.
The parent dashboard_guard owns the lease and canonical bindings. This module
never claims a lease, builds a cache, downloads weights, or substitutes nodes.
SSH transports the small environment packet on stdin, not in command arguments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from experiment_plan import LAYOUTS, state_slot_budget

GUARD_KEYS = ('EXPERIMENT_GUARD_LEASE_RECEIPT', 'EXPERIMENT_GUARD_ALLOCATION_FILE',
              'EXPERIMENT_GUARD_ACQUIRED_NODES', 'EXPERIMENT_GUARD_MODEL_CACHE_BINDINGS')
YARN = {'language_model_only': True, 'text_config': {'rope_parameters': {'mrope_interleaved': True,
    'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000,
    'partial_rotary_factor': .25, 'factor': 4., 'original_max_position_embeddings': 262144}}}
SAFE_ID = re.compile(r'^[A-Za-z0-9_.-]+$')
INFO_KEYS = {'max_total_num_tokens', 'max_total_num_tokens_swa', 'memory_usage',
    'internal_states', 'version', 'attention_backend', 'prefill_attention_backend',
    'decode_attention_backend', 'linear_attn_backend', 'linear_attn_decode_backend',
    'linear_attn_prefill_backend', 'mamba_backend', 'mamba_ssm_dtype',
    'max_mamba_cache_size', 'mamba_cache_size', 'mamba_cache_chunk_size',
    'mamba_scheduler_strategy', 'disable_cuda_graph', 'cuda_graph_config',
    'cuda_graph_max_bs_decode', 'cuda_graph_bs_decode', 'max_running_requests',
    'effective_max_running_requests_per_dp', 'page_size', 'kv_cache_dtype',
    'mem_fraction_static', 'moe_runner_backend', 'moe_a2a_backend', 'deepep_mode',
    'tp_size', 'ep_size', 'dp_size', 'pp_size', 'context_length', 'quantization',
    'dtype', 'disable_overlap_schedule', 'enable_dp_attention', 'token_capacity',
    'token_capacity_swa', 'weight_gb', 'kv_cache_gb', 'target_graph_memory_usage',
    'draft_graph_memory_usage', 'startup_available_gb', 'startup_time'}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')
    os.replace(temporary, path)


def private_json(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError('Guard files must be private UID-owned regular files')
    return json.loads(path.read_text())


def load_config(run, cohort, node):
    run = Path(run)
    if not run.is_absolute():
        raise ValueError('--run must be absolute')
    cfg = json.loads((run / 'config.json').read_text())
    required = ('id', 'cohort', 'node', 'job_id', 'engine_root', 'engine_commit',
                'python', 'model_path', 'model_revision', 'guard_scripts',
                'layouts', 'model_cache', 'admin_key_file')
    if any(key not in cfg for key in required):
        raise ValueError('Frozen qualification config is incomplete')
    if cfg['cohort'] != cohort or cfg['node'] != node or not SAFE_ID.fullmatch(node):
        raise ValueError('Model/node differs from the frozen configuration')
    for key in ('engine_root', 'python', 'model_path', 'guard_scripts', 'admin_key_file'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError('Frozen paths must be absolute: ' + key)
    for key in ('engine_commit', 'model_revision'):
        if not re.fullmatch(r'[a-f0-9]{40}', cfg[key]):
            raise ValueError('Full source/model revision is required')
    selection = cfg.get('qualification_layout_selection', 'all')
    if selection not in ('all', 'minimum_only'):
        raise ValueError('Unsupported qualification layout selection')
    expected = LAYOUTS[cohort][:1] if selection == 'minimum_only' else LAYOUTS[cohort]
    if [row['id'] for row in cfg['layouts']] != [row.id for row in expected]:
        raise ValueError('Qualification must start with the minimum-size layout')
    ids, paths = set(), set()
    declared = cfg['model_cache'].get('usage_logs', {}).get(node, [])
    for row, layout in zip(cfg['layouts'], expected):
        for key in ('replicas', 'tp', 'ep', 'dp', 'pp'):
            if row.get(key) != getattr(layout, key):
                raise ValueError('Frozen layout has incompatible parallelism')
        if row.get('gpu_groups') != layout.gpu_groups:
            raise ValueError('Frozen layout must partition its cohort GPU budget without overlap')
        backend_choices(cfg, row)
        uses = row.get('usage_logs', [])
        if len(uses) != layout.replicas:
            raise ValueError('Each replica requires its own usage log')
        for use in uses:
            if use.get('single_instance_tp1', False) is not (layout.tp == 1):
                raise ValueError('Rankless loader proof is only allowed for a frozen single-instance TP1 layout')
            if (not SAFE_ID.fullmatch(use['id']) or use['id'] in ids
                    or use['path'] in paths or not Path(use['path']).is_absolute()
                    or use.get('expected_ranks') != [f'TP{i}' for i in range(layout.tp)]):
                raise ValueError('Fresh usage IDs/paths and exact TP rank labels are required')
            if sum(all(item.get(k) == use.get(k) for k in ('id', 'path', 'expected_ranks', 'single_instance_tp1'))
                   for item in declared) != 1:
                raise ValueError('Layout usage log is not exactly declared in the Guard contract')
            ids.add(use['id']); paths.add(use['path'])
    if len(declared) != len(ids):
        raise ValueError('Guard usage logs must cover exactly these launches')
    return cfg


def bound_usage_matches(bound, use):
    # Canonical bindings copy identity/ranks plus a baseline, not the TP1
    # marker. The latter stays mandatory in frozen cfg/required contract.
    return sum(all(item.get(key) == use.get(key)
                   for key in ('id', 'path', 'expected_ranks'))
               for item in bound) == 1


def authenticate(run, cfg, env, *, local=False):
    """Read-only live ownership check; no heartbeat, claim, or GPU probe."""
    if any(not env.get(key) for key in GUARD_KEYS):
        raise RuntimeError('A supervising dashboard_guard and canonical cache bindings are required')
    receipt = private_json(env[GUARD_KEYS[0]])
    binding = private_json(env[GUARD_KEYS[3]])
    allocation = json.loads(Path(env[GUARD_KEYS[1]]).read_text())
    sys.path.insert(0, cfg['guard_scripts'])
    registry_module = importlib.import_module('gpu_registry')
    registry = registry_module.SharedGpuRegistry(receipt['registry_dir'])
    digest = hashlib.sha256((Path(run) / 'dashboard_spec.json').read_bytes()).hexdigest()
    lease = registry.assert_owned(receipt, cfg['id'], digest)
    registry.assert_owned_nodes(receipt, [cfg['node']])
    if time.time() - lease['heartbeat_at'] > lease['ttl_s']:
        raise RuntimeError('Guard heartbeat is expired')
    if (receipt['lease_id'] != allocation.get('lease_id')
            or allocation.get('experiment_id') != cfg['id']
            or cfg['node'] not in allocation.get('nodes', [])
            or cfg['node'] not in env[GUARD_KEYS[2]].split(',')
            or binding.get('lease_id') != receipt['lease_id']
            or binding.get('experiment_id') != cfg['id']
            or binding.get('policy') != 'required'
            or cfg['node'] not in binding.get('acquired_nodes', [])
            or binding.get('lease_receipt_sha256') != hashlib.sha256(
                Path(env[GUARD_KEYS[0]]).read_bytes()).hexdigest()):
        raise RuntimeError('Frozen config/allocation/canonical binding differs from the active lease')
    if local and socket.gethostname().split('.')[0] != cfg['node']:
        raise RuntimeError('SSH worker landed on a different node')
    if local:
        job = str(cfg['job_id'])
        if (not job.isdigit() or not any(re.search(rf'(?:^|/)job_{job}(?:/|$)', line.split(':', 2)[-1])
                for line in Path('/proc/self/cgroup').read_text().splitlines())):
            raise RuntimeError('SSH worker is outside its frozen booked Slurm job cgroup')
    bound = binding.get('nodes', {}).get(cfg['node'], {}).get('usage_logs', [])
    for row in cfg['layouts']:
        for use in row['usage_logs']:
            if not bound_usage_matches(bound, use):
                raise RuntimeError('Selected fresh usage log is absent from canonical bindings')
    return {'lease_id': lease['lease_id'], 'node': cfg['node'],
            'experiment_id': cfg['id'], 'authenticated_at': time.time()}


def backend_choices(cfg, layout):
    # Native FP8 W8A8/DeepGEMM is not an A100 implementation. Do not silently
    # dequantize the large checkpoint, change quantization or relabel a fallback.
    if cfg.get('attention_backend', 'triton') != 'triton':
        raise ValueError('A100 qualification requires the frozen Triton attention backend')
    if cfg['cohort'] == 'large':
        fp8_adapter_config(cfg)
        if (layout['tp'] != layout['ep'] or layout.get('dp', 1) != 1
                or layout.get('pp', 1) != 1 or layout['ep'] not in (2, 4)):
            raise ValueError('A100 large requires Attention TP=EP2/EP4, expert TP1, DP1 and PP1')
        return dict(attention='triton', linear_attention='triton', moe_runner='marlin',
                    moe_a2a='none', deepep_mode=None, bf16_dispatch=False,
                    expert_communication='native_standard_dispatch_local_experts_nccl_sum')
    if cfg['cohort'] != 'small':
        raise ValueError('Unknown A100 model cohort')
    if (layout['tp'] != layout['ep'] or layout.get('dp', 1) != 1
            or layout.get('pp', 1) != 1 or layout['ep'] not in (1, 2)):
        raise ValueError('A100 small cohort requires Attention TP=EP in {1,2}, DP1 and PP1')
    return dict(attention='triton', linear_attention='triton', moe_runner='triton',
                moe_a2a='none', deepep_mode=None, bf16_dispatch=False,
                expert_communication='native_standard_dispatch_local_experts_nccl_sum')


def fp8_adapter_config(cfg):
    """A test opt-in is not a formal GPU qualification receipt."""
    adapter = cfg.get('runtime_adapter')
    if (not isinstance(adapter, dict)
            or adapter.get('name') != 'a100_fp8_moe_marlin_w8a16_v1'
            or adapter.get('mode') not in ('qualification', 'validated')
            or not cfg.get('hybrid_bank_json')):
        raise ValueError('A100 large FP8 cohort is not-qualified: explicit Guard qualification '
                         'mode and native hybrid restore smoke are required')
    source = adapter.get('source', {})
    path = Path(source.get('path', ''))
    if (set(source) != {'path', 'sha256'} or not path.is_absolute()
            or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']
            or path.name != 'fp8_marlin_adapter.py'):
        raise ValueError('FP8 Marlin adapter source descriptor changed')
    if cfg.get('engine_commit') != '870269702285d94bdd926a50cc29ad1a20bcf60c':
        raise ValueError('FP8 Marlin requires the pinned native engine commit')
    if adapter['mode'] == 'validated':
        from fp8_qualification_evidence import validate_receipt
        validate_receipt(adapter.get('receipt'), model_path=cfg.get('model_path'),
            model_revision=cfg.get('model_revision'), adapter_sha=source['sha256'],
            engine_commit=cfg['engine_commit'])
    elif cfg.get('formal_profile'):
        raise ValueError('An unqualified adapter mode can never run a formal benchmark')
    return adapter


def engine_args(cfg, layout, replica):
    local_requests = state_slot_budget(LAYOUTS[cfg['cohort']][
        [r.id for r in LAYOUTS[cfg['cohort']]].index(layout['id'])], cfg.get('request_count', 32))
    backend = backend_choices(cfg, layout)
    offset = [r['id'] for r in cfg['layouts']].index(layout['id']) * 10 + replica
    args = ['--model-path', cfg['model_path'], '--host', '127.0.0.1',
        '--port', str(cfg.get('port_base', 19100) + offset),
        '--dist-init-addr', f"127.0.0.1:{cfg.get('dist_port_base', 19200) + offset}",
        '--tp-size', str(layout['tp']), '--ep-size', str(layout['ep']), '--dp-size', '1', '--pp-size', '1',
        '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16',
        '--context-length', str(cfg.get('context_length', 327680)),
        '--mem-fraction-static', str(cfg.get('mem_fraction_static', .95)),
        '--page-size', str(cfg.get('page_size', 64)),
        '--max-running-requests', str(local_requests['max_running_requests']),
        '--max-mamba-cache-size', str(local_requests['max_mamba_cache_size']),
        '--mamba-ssm-dtype', 'float32',
        '--cuda-graph-max-bs-decode', str(local_requests['max_running_requests']),
        '--cuda-graph-backend-prefill', 'disabled',
        '--chunked-prefill-size', str(2048 * layout['tp']),
        '--max-prefill-tokens', str(2048 * layout['tp']),
        '--attention-backend', backend['attention'],
        '--linear-attn-backend', backend['linear_attention'],
        '--linear-attn-decode-backend', backend['linear_attention'],
        '--linear-attn-prefill-backend', backend['linear_attention'],
        '--moe-runner-backend', backend['moe_runner'], '--moe-a2a-backend', backend['moe_a2a'],
        '--disable-custom-all-reduce',
        '--json-model-override-args', json.dumps(YARN, separators=(',', ':'))]
    if backend['deepep_mode']:
        args += ['--deepep-mode', backend['deepep_mode']]
    if cfg['cohort'] == 'large':
        args += ['--quantization', 'fp8']
    if cfg.get('hybrid_bank_json'):
        storage = dict(backend_name='qwen_hybrid_synthetic', module_path='hybrid_bank',
            class_name='HybridPrefixPoolStorage', bank_path=cfg['hybrid_bank_json'],
            model_id=hybrid_plan(cfg)['model_id'], engine_commit=cfg['engine_commit'],
            request_indices=hybrid_indices(cfg, layout, replica),
            prefetch_threshold=64, hicache_storage_pass_prefix_keys=True)
        args += ['--enable-hierarchical-cache', '--hicache-size',
            str(cfg.get('hicache_pool_gb_per_rank', 4)),
            '--hicache-host-memory-mode', 'cache', '--hicache-mem-layout', 'page_first',
            '--hicache-storage-backend', 'dynamic',
            '--hicache-storage-prefetch-policy', 'wait_complete',
            '--hicache-storage-backend-extra-config', json.dumps(storage),
            '--skip-server-warmup']
    return args


def hybrid_plan(cfg):
    return json.loads(Path(cfg['hybrid_bank_json']).read_text())


def hybrid_indices(cfg, layout, replica):
    count = len(hybrid_plan(cfg)['requests'])
    if count < layout['replicas'] or count % layout['replicas']:
        raise ValueError('Hybrid restore smoke requests must divide equally among replicas')
    return list(range(replica * count // layout['replicas'],
                      (replica + 1) * count // layout['replicas']))


def hybrid_control_dir(cfg, layout, replica):
    return Path(cfg['run_dir']) / 'hybrid-control' / f"{layout['id']}.r{replica}"


def server_environment(cfg, layout, replica, inherited):
    backend = backend_choices(cfg, layout)
    prefixes = ('SGLANG_', 'VLLM_', 'DEEPEP_', 'NVSHMEM_', 'NCCL_', 'GLOO_', 'HUMMING_',
                'QWEN_A100_FP8_')
    env = {k: v for k, v in inherited.items() if not k.startswith(prefixes)
           and k not in ('PYTHONPATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES',
                         'MAX_JOBS', 'TORCH_CUDA_ARCH_LIST')}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, layout['gpu_groups'][replica])),
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parent),
            str(Path(cfg['engine_root']) / 'python')]),
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUDA_MODULE_LOADING='EAGER', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1')
    if layout['tp'] == 1:
        logging_config = cfg.get('single_rank_logging')
        if (not isinstance(logging_config, dict)
                or set(logging_config) != {'path', 'sha256'}
                or not Path(logging_config['path']).is_absolute()):
            raise ValueError('TP1 requires the frozen native single-rank logging configuration')
        path = Path(logging_config['path'])
        if (path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != logging_config['sha256']):
            raise ValueError('Frozen native single-rank logging configuration changed')
        env['SGLANG_LOGGING_CONFIG_PATH'] = str(path)
    # Only these frozen non-secret transport/runtime values may be supplied.
    allowed = {'LD_LIBRARY_PATH', 'PATH', 'CUDA_HOME', 'CUDA_PATH', 'NCCL_CUMEM_HOST_ENABLE',
        'NCCL_IB_HCA', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME',
        'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
        'NCCL_IB_DISABLE', 'NCCL_P2P_LEVEL', 'SGLANG_MEMORY_BANK_ROOT',
        'MAX_JOBS', 'TORCH_CUDA_ARCH_LIST'}
    extra = cfg.get('launch_env', {})
    if not isinstance(extra, dict) or set(extra) - allowed:
        raise ValueError('Unreviewed launch environment key')
    if ('MAX_JOBS' in extra or 'TORCH_CUDA_ARCH_LIST' in extra) and (
            cfg['cohort'] != 'large' or fp8_adapter_config(cfg)['mode'] != 'qualification'
            or extra.get('MAX_JOBS') != '4'
            or extra.get('TORCH_CUDA_ARCH_LIST') != '8.0'):
        raise ValueError('Only the explicit large qualification may request bounded SM80 JIT')
    env.update({k: str(v) for k, v in extra.items()})
    if cfg['cohort'] == 'large':
        adapter = fp8_adapter_config(cfg)
        env.update(QWEN_A100_FP8_MOE_MARLIN='1',
            QWEN_A100_FP8_ADAPTER_SHA256=adapter['source']['sha256'])
        if adapter['mode'] == 'validated':
            env['QWEN_A100_FP8_MOE_MARLIN_RECEIPT'] = adapter['receipt']['path']
        else:
            env.update(QWEN_A100_FP8_MOE_MARLIN_QUALIFICATION='1',
                QWEN_A100_FP8_MODEL_MOE_LAYERS='48',
                QWEN_A100_FP8_QUALIFICATION_ATTESTATION_DIR=str(
                    fp8_attestation_dir(cfg, layout, replica)),
                QWEN_A100_FP8_QUALIFICATION_OWNER_JSON=json.dumps(dict(run_id=cfg['id'],
                    layout=layout['id'], replica=replica)))
    if backend['bf16_dispatch']:
        env['SGLANG_DEEPEP_BF16_DISPATCH'] = '1'
    if cfg.get('hybrid_bank_json'):
        env['QWEN_HYBRID_CONTROL_DIR'] = str(hybrid_control_dir(cfg, layout, replica))
        env['QWEN_HYBRID_OWNER_JSON'] = json.dumps(dict(run_id=cfg['id'],
            instance_id=f"{layout['id']}.r{replica}"))
    return env


def fp8_attestation_dir(cfg, layout, replica):
    return Path(cfg['run_dir']) / 'qualification/adapter' / f"{layout['id']}.r{replica}"


def read_fp8_attestations(cfg, layout, replica):
    source = fp8_adapter_config(cfg)['source']
    rows = []
    for rank in range(layout['tp']):
        proof = private_json(fp8_attestation_dir(cfg, layout, replica) / f'rank{rank}.json')
        fixture = proof.get('fixture', {})
        expected_checks = {(batch, changed) for batch in (1, 16) for changed in (False, True)}
        checks = fixture.get('checks', [])
        if (proof.get('status') != 'passed' or proof.get('tp_rank') != rank
                or proof.get('ep_rank') != rank or proof.get('ep_size') != layout['ep']
                or proof.get('owner') != dict(run_id=cfg['id'], layout=layout['id'], replica=replica)
                or proof.get('adapter_sha256') != source['sha256']
                or proof.get('engine_commit') != cfg['engine_commit']
                or proof.get('expected_model_moe_layers') != 48
                or len(proof.get('model_moe_layers', [])) != 48
                or proof.get('all_model_moe_layers_prepared') is not True
                or proof.get('native_dense_unchanged') is not True
                or fixture.get('status') != 'passed' or fixture.get('ep_rank') != rank
                or fixture.get('ep_size') != layout['ep']
                or fixture.get('adapter_sha256') != source['sha256']
                or fixture.get('engine_commit') != cfg['engine_commit']
                or len(checks) != 4
                or {(row.get('batch'), row.get('changed_input')) for row in checks} != expected_checks
                or any(row.get('status') != 'passed' for row in checks)):
            raise RuntimeError('FP8 all-rank model/fixture attestation is incomplete or stale')
        rows.append(proof)
    return rows


def verify_layout_usage(cfg, layout, env):
    """Run Guard's exact delta verifier where its filesystem baseline arose.

    GPFS st_dev identifiers are mount-local and need not match between login
    and compute hosts. Moving this verifier preserves, rather than removes,
    Guard's original device/inode/offset/freshness checks.
    """
    if (not cfg.get('usage_audit_hostname')
            or socket.gethostname().split('.')[0] != cfg['usage_audit_hostname']):
        raise RuntimeError('Native loader usage audit must run on its original admission controller host')
    binding = private_json(env[GUARD_KEYS[3]])
    node = binding['nodes'][cfg['node']]
    adapter = importlib.import_module('model_cache_adapter')
    if Path(adapter.__file__).resolve().parent != Path(cfg['guard_scripts']).resolve():
        raise RuntimeError('Loader verifier came from a different Guard checkout')
    proofs = []
    for use in layout['usage_logs']:
        matches = [row for row in node['usage_logs'] if bound_usage_matches([row], use)]
        if len(matches) != 1:
            raise RuntimeError('Native loader usage is not uniquely bound to this qualification')
        proofs.append(adapter._verify_usage_log(matches[0], node['group_key'], binding['admitted_at']))
    return proofs


def complete_controller_usage_audit(run, cfg, env):
    """Finalize large qualification only after remote forward AND cleanup.

    Workers cannot issue a passing loader receipt using a foreign st_dev
    namespace. The supervising admission controller verifies the original
    baseline without modification, then publishes the combined evidence.
    """
    if cfg['cohort'] != 'large':
        return None
    authenticate(run, cfg, env)
    output = Path(run) / 'qualification'
    result = private_json(output / 'results.json')
    if (result.get('native_full_model_execution_passed') is not True
            or result.get('loader_usage_audit_pending') is not True
            or result.get('native_load_qualification_passed') is True
            or set(result.get('layouts', {})) != {layout['id'] for layout in cfg['layouts']}):
        raise RuntimeError('Remote qualification has not completed every native execution/cleanup boundary')
    for layout in cfg['layouts']:
        row = result['layouts'][layout['id']]
        cleanup = private_json(output / f"{layout['id']}.cleanup.json")
        if (row.get('valid') is not True or row.get('error') is not None
                or row.get('loader_usage_audit_pending') is not True
                or row.get('cleanup_verified') is not True
                or row.get('hybrid_restore_smoke_passed') is not True
                or cleanup.get('cleanup_verified') is not True
                or cleanup.get('process_groups_exited') is not True
                or cleanup.get('node_gpu_idle_observed') is not True):
            raise RuntimeError('Native execution or exact cleanup is incomplete before controller usage audit')
        try:
            row['loader_usage_attestation'] = verify_layout_usage(cfg, layout, env)
        except Exception as error:
            row.update(valid=False, error=type(error).__name__ + ': ' + str(error))
            save(output / f"{layout['id']}.json", row)
            save(output / 'results.json', result)
            raise
        row['loader_usage_audit_pending'] = False
        row['loader_usage_audit_hostname'] = cfg['usage_audit_hostname']
        save(output / f"{layout['id']}.json", row)
    result.update(loader_usage_audit_pending=False,
        native_load_qualification_passed=True,
        loader_usage_audit_hostname=cfg['usage_audit_hostname'])
    save(output / 'results.json', result)
    receipt = fp8_qualification_receipt(cfg, result)
    if receipt['status'] != 'passed':
        raise RuntimeError('Controller usage audit cannot replace missing model/rank/Graph qualification')
    receipt['loader_usage_audit_hostname'] = cfg['usage_audit_hostname']
    save(output / 'fp8_marlin_gpu_qualification.json', receipt)
    save(output / 'progress.json', dict(status='native_load_qualification_passed',
        completed=len(result['layouts']), hybrid_restore_smoke_passed=True))
    return receipt


def read_fp8_graph_attestations(cfg, layout, replica):
    from fp8_qualification_evidence import validate_graph
    source = fp8_adapter_config(cfg)['source']
    owner = dict(run_id=cfg['id'], layout=layout['id'], replica=replica)
    rows = []
    for rank in range(layout['tp']):
        proof = private_json(fp8_attestation_dir(cfg, layout, replica) / f'rank{rank}.graph.json')
        validate_graph(proof, owner, rank, layout['ep'], source['sha256'])
        rows.append(proof)
    return rows


def fp8_qualification_receipt(cfg, result):
    """No EP2-only, fixture-only or load-only evidence may qualify formal use."""
    source = fp8_adapter_config(cfg)['source']
    complete = {row['id'] for row in cfg['layouts']} == {'2xep2', 'ep4'}
    actual = result.get('layouts', {})
    checks = dict(ep2=False, ep4=False, decode=False, prefill=False, changed_inputs=False,
                  decode_graph_replay=False)
    for layout in cfg['layouts']:
        row = actual.get(layout['id'], {})
        replicas = row.get('replicas', [])
        valid = (row.get('valid') is True and row.get('cleanup_verified') is True
            and row.get('hybrid_restore_smoke_passed') is True
            and len(row.get('loader_usage_attestation', [])) == layout['replicas']
            and len(replicas) == layout['replicas'])
        valid = valid and all(len(item.get('fp8_adapter_attestations', [])) == layout['tp']
                              and len(item.get('fp8_decode_graph_attestations', [])) == layout['tp']
                              and all(proof.get('actual_model_decode_graph_replay_verified') is True
                                  and proof.get('replay_count', 0) >= 1
                                  for proof in item.get('fp8_decode_graph_attestations', []))
                              and item.get('hybrid_restore_smoke_passed') is True
                              and item.get('native_request_count', 0) > 0
                              and len(item.get('native_request_tokens', [])) == item.get('native_request_count')
                              and all(reply.get('completion_tokens') == 4
                                  for reply in item.get('native_request_tokens', []))
                              for item in replicas)
        checks[f"ep{layout['ep']}"] = bool(valid)
    complete = complete and checks['ep2'] and checks['ep4']
    checks.update(prefill=complete, decode=complete, changed_inputs=complete,
                  decode_graph_replay=complete)
    return dict(schema_version=2, status='passed' if complete else 'incomplete', checks=checks,
        adapter_sha256=source['sha256'], engine_commit=cfg['engine_commit'],
        model_revision=cfg['model_revision'], model_path=cfg['model_path'],
        run_id=cfg['id'], node=cfg['node'], fixture_only=False,
        full_model_load_restore_and_native_forward_qualified=complete,
        native_loader_usage_attestation_verified=complete,
        tiny_eager_independent_reference_checks_passed=complete,
        actual_model_decode_graph_replay_verified=complete,
        actual_model_graph_numerical_equivalence_checked=False,
        qualification_results=str(Path(cfg['run_dir']) / 'qualification/results.json'),
        note='Changed-input numerical correctness is checked by independent tiny eager FP32-reference '
             'fixtures; actual-model native Decode Graph execution is separately counted, not numerically '
             'compared. Synthetic full-model KV is for execution/restore, not generated text quality')


def hybrid_transition(cfg, layout, replica, child, check, action, *, prepared_id=None):
    from hybrid_control import write_command
    instance = f"{layout['id']}.r{replica}"
    command_id = action + '-smoke'
    command = dict(version=1, command_id=command_id, action=action,
                   run_id=cfg['id'], instance_id=instance)
    if action == 'prepare':
        plan = hybrid_plan(cfg); indices = hybrid_indices(cfg, layout, replica)
        command.update(bank_path=cfg['hybrid_bank_json'], bank_fingerprint=plan['fingerprint'],
            request_indices=indices, request_count=len(indices),
            context_lengths=[plan['requests'][i]['context_tokens'] for i in indices],
            timeout_seconds=cfg.get('prepare_timeout_s', 180))
    elif action == 'begin_timing':
        command.update(prepared_command_id=prepared_id, workload='prefill')
    root = hybrid_control_dir(cfg, layout, replica)
    write_command(root, command)
    deadline = time.monotonic() + cfg.get('prepare_timeout_s', 180) + 20
    while True:
        check()
        if child['process'].poll() is not None:
            raise RuntimeError('Engine exited during hybrid prefix preparation')
        paths = [root / f'ack.{instance}.{command_id}.rank{r}.json' for r in range(layout['tp'])]
        if all(path.exists() for path in paths):
            acknowledgements = [private_json(path) for path in paths]
            if not all(row.get('passed') is True and row.get('command_id') == command_id
                    and row.get('owner', {}).get('instance_id') == instance for row in acknowledgements):
                raise RuntimeError('A TP rank rejected hybrid preparation; inspect private acknowledgements')
            return command_id, acknowledgements
        if time.monotonic() >= deadline:
            raise TimeoutError('All-rank hybrid prefix acknowledgement timed out')
        time.sleep(.1)


def adapter_command(cfg, run, layout, replica, bindings):
    # The audited adapter currently accepts --bindings only; node and use-ID
    # selection are authenticated by this worker and stdout uses that exact log.
    return [cfg['python'], str(Path(cfg['guard_scripts']) / 'model_cache_adapter.py'),
        'exec-local', '--bindings', bindings, '--', cfg['python'], str(Path(__file__).resolve()),
        '--engine-entry', '--run', str(run), '--model', cfg['cohort'], '--node', cfg['node'],
        '--layout', layout['id'], '--replica', str(replica)]


def safe_server_info(info):
    def clean(value, *, nested=False):
        if isinstance(value, dict):
            return {k: clean(v, nested=True) for k, v in value.items()
                    if k in INFO_KEYS or (nested and not any(x in k.lower()
                        for x in ('key', 'tokenizer', 'password', 'secret', 'path', 'env')))}
        if isinstance(value, list):
            return [clean(item, nested=nested) for item in value]
        return value
    return clean(info)


def allocator_capacity(info):
    values = []
    direct = info.get('max_total_num_tokens')
    if type(direct) is int and direct > 0:
        values.append(direct)
    for row in info.get('internal_states', []):
        for value in (row.get('max_total_num_tokens'), row.get('memory_usage', {}).get('token_capacity')):
            if type(value) is int and value > 0:
                values.append(value)
    if not values or len(set(values)) != 1:
        raise RuntimeError('Server did not report one consistent real allocator token capacity')
    return values[0]


def read_key(cfg):
    path = Path(cfg['admin_key_file']); info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise RuntimeError('API key file must be private and UID-owned')
    key = path.read_text().strip()
    if len(key) < 16 or '\n' in key:
        raise RuntimeError('Invalid private API key file')
    return key


def http_json(port, route, key, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{route}', data=data,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def process_record(pid):
    try:
        path = Path('/proc') / str(pid); info = path.stat()
        text = (path / 'stat').read_text(); fields = text[text.rfind(')') + 2:].split()
        return dict(pid=pid, uid=info.st_uid, state=fields[0], pgid=int(fields[2]),
                    session=int(fields[3]), start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def owned_group(identity):
    records = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            record = process_record(int(path.name))
            if record and record['pgid'] == identity['pgid']:
                if (record['uid'] != os.getuid() or record['session'] != identity['pgid']
                        or record['start_ticks'] < identity['start_ticks']
                        or (record['pid'] == identity['pid'] and
                            record['start_ticks'] != identity['start_ticks'])):
                    raise RuntimeError('Refusing to signal a reused or foreign process group')
                records.append(record)
    return records


def gpu_cleanup_evidence(identities):
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=15, check=True).stdout
    apps = []
    for line in output.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 3 and fields[0].isdigit():
            apps.append(dict(pid=int(fields[0]), gpu_uuid=fields[1], used_gpu_memory_mib=fields[2]))
    remaining = [row for identity in identities for row in owned_group(identity)
                 if row['state'] != 'Z']
    own_ids = {row['pid'] for row in remaining} | {row['pid'] for row in identities}
    return dict(timestamp=time.time(), process_groups_exited=not remaining,
        owned_gpu_processes=[row for row in apps if row['pid'] in own_ids],
        observed_gpu_processes=apps, node_gpu_idle_observed=not apps,
        cleanup_verified=not remaining and not apps,
        scope='leased full node; foreign processes are observed, never signalled')


def cleanup(children):
    """Bound owned-group shutdown and reaping; always obtain fresh GPU evidence."""
    identities = [child['identity'] for child in children]
    started = time.monotonic()
    errors, blocked, reaped = set(), set(), set()

    def group(identity):
        pid = identity['pid']
        try:
            # Check the leader separately: a reused PID in a different PGID
            # would otherwise be invisible to owned_group's group scan.
            leader = process_record(pid)
            if leader is not None and any(leader[key] != value for key, value in (
                ('uid', os.getuid()), ('start_ticks', identity['start_ticks']),
                ('pgid', identity['pgid']), ('session', identity['pgid']))):
                raise RuntimeError('Refusing a reused or foreign process identity')
            return owned_group(identity)
        except Exception as error:
            blocked.add(pid)
            errors.add(f'identity:{pid}:{type(error).__name__}')
            return None

    def pending():
        live = False
        for identity in identities:
            rows = group(identity)
            live |= rows is None or any(row['state'] != 'Z' for row in rows)
        for index, child in enumerate(children):
            if index in reaped:
                continue
            try:
                # Nonblocking wait reaps an exited Popen while an alive/D-state
                # wrapper is retried within the same absolute group deadline.
                child['process'].wait(timeout=0)
                reaped.add(index)
            except subprocess.TimeoutExpired:
                pass
            except Exception as error:
                errors.add(f'reap:{child["identity"]["pid"]}:{type(error).__name__}')
        return live or len(reaped) != len(children)

    # All children share these deadlines; slow wrappers cannot add a separate
    # 3-second failure or a per-child shutdown grace after the groups exit.
    for sig, deadline in ((signal.SIGTERM, started + 12),
                          (signal.SIGKILL, started + 42)):
        for identity in identities:
            rows = group(identity)  # Revalidate immediately before every signal.
            if (identity['pid'] not in blocked and rows is not None
                    and any(row['state'] != 'Z' for row in rows)):
                try:
                    os.killpg(identity['pgid'], sig)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.add(f'signal:{identity["pid"]}:{type(error).__name__}')
        while pending():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(.2, remaining))
        if len(reaped) == len(children) and not pending():
            break

    if len(reaped) != len(children):
        errors.add('owned_child_not_reaped_before_deadline')
    log_deadline = min(started + 47, time.monotonic() + 5)
    for child in children:
        try:
            child['log_thread'].join(timeout=max(0, log_deadline - time.monotonic()))
            if child['log_thread'].is_alive():
                errors.add(f'log_thread:{child["identity"]["pid"]}:still_alive')
        except Exception as error:
            errors.add(f'log_thread:{child["identity"]["pid"]}:{type(error).__name__}')

    # Start the last nvidia-smi query by t+60: its native 15-second timeout
    # keeps total cleanup within 75 seconds, below Guard's 90-second grace.
    while True:
        try:
            evidence = gpu_cleanup_evidence(identities)
        except Exception as error:
            errors.add(f'gpu_evidence:{type(error).__name__}')
            evidence = dict(timestamp=time.time(), cleanup_verified=False,
                process_groups_exited=False, node_gpu_idle_observed=False,
                observed_gpu_processes=None, owned_gpu_processes=None,
                scope='fresh cleanup observation failed; state is unverified')
        live = pending()
        evidence.update(process_identities=identities, children_reaped=len(reaped) == len(children),
            cleanup_errors=sorted(errors))
        evidence['cleanup_verified'] = bool(evidence.get('cleanup_verified') is True
            and not live and not errors)
        if (evidence['cleanup_verified'] or errors
                or time.monotonic() >= started + 60):
            return evidence
        time.sleep(min(1, max(0, started + 60 - time.monotonic())))


def stream_log(pipe, path, secret):
    """Preserve Guard's inode/append boundary while redacting startup reprs."""
    pending = b''; secret = secret.encode()
    with Path(path).open('ab', buffering=0) as handle:
        while True:
            # BufferedReader.read(n) can hide an entire live startup below n.
            # read1 returns available pipe data without waiting for 64 KiB.
            chunk = getattr(pipe, 'read1', pipe.read)(65536)
            if not chunk:
                handle.write(pending.replace(secret, b'[REDACTED]')); break
            pending += chunk
            # Keep an entire line so even a split secret cannot escape.
            end = pending.rfind(b'\n')
            if end >= 0:
                handle.write(pending[:end + 1].replace(secret, b'[REDACTED]'))
                pending = pending[end + 1:]
            if len(pending) > 4 * 1024**2:
                raise RuntimeError('Engine emitted an overlong unterminated log line')


def launch(cfg, run, layout, replica, env, key):
    use = layout['usage_logs'][replica]
    private_json_path = Path(use['path']); metadata = private_json_path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise RuntimeError('Fresh usage file must have been created by Guard')
    command = adapter_command(cfg, run, layout, replica, env[GUARD_KEYS[3]])
    child_env = server_environment(cfg, layout, replica, env)
    proc = subprocess.Popen(command, env=child_env, cwd=cfg['engine_root'], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_record(proc.pid)
    if not identity or identity['pgid'] != proc.pid or identity['session'] != proc.pid:
        proc.terminate(); proc.wait(timeout=5)
        raise RuntimeError('Failed to establish the owned engine process group')
    thread = threading.Thread(target=stream_log, args=(proc.stdout, use['path'], key), daemon=True)
    thread.start()
    return dict(process=proc, identity=identity, log_thread=thread, usage_log_id=use['id'],
                usage_log_path=use['path'], replica=replica, gpu_ids=layout['gpu_groups'][replica])


def worker(run, cfg, env, *, control_stream=None):
    authenticate(run, cfg, env, local=True)
    actual = subprocess.check_output(['git', '-C', cfg['engine_root'], 'rev-parse', 'HEAD'], text=True).strip()
    if actual != cfg['engine_commit']:
        raise RuntimeError('Engine checkout differs from the frozen commit')
    key = read_key(cfg); cancelled = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: cancelled.set())
    def check():
        if cancelled.is_set():
            raise RuntimeError('Qualification cancelled')
        if control_stream is not None and select.select([control_stream], [], [], 0)[0]:
            if control_stream.buffer.peek(1) == b'':
                raise RuntimeError('Parent SSH control channel closed')
        authenticate(run, cfg, env, local=True)
    output = Path(run) / 'qualification'; output.mkdir(exist_ok=True)
    result = dict(cohort=cfg['cohort'], node=cfg['node'], model_revision=cfg['model_revision'],
        engine_commit=cfg['engine_commit'],
        source='actual_allocator', performance_measured=False, hybrid_restore_smoke_passed=False,
        native_load_qualification_only=True, layouts={})
    for layout in cfg['layouts']:
        children, observations, failure = [], [], None
        usage_attestation = []
        save(output / 'progress.json', dict(layout=layout['id'], status='loading', completed=len(result['layouts'])))
        try:
            # Qualify the minimum-size replica before spending time on peers.
            for replica in range(layout['replicas']):
                check(); child = launch(cfg, run, layout, replica, env, key); children.append(child)
                offset = [r['id'] for r in cfg['layouts']].index(layout['id']) * 10 + replica
                port = cfg.get('port_base', 19100) + offset
                deadline = time.monotonic() + cfg.get('startup_timeout_s', 1800)
                while True:
                    check()
                    if child['process'].poll() is not None:
                        raise RuntimeError('Engine exited before native qualification; inspect its usage log')
                    try:
                        info = http_json(port, '/get_server_info', key)
                        capacity = allocator_capacity(info); break
                    except (urllib.error.URLError, TimeoutError, OSError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Native server startup timed out')
                        time.sleep(1)
                started = time.monotonic()
                proofs = None
                if cfg.get('hybrid_bank_json'):
                    from hybrid_bank import workload_requests
                    selected_requests = hybrid_indices(cfg, layout, replica)
                    if any(hybrid_plan(cfg)['requests'][i]['context_tokens'] + 8 > capacity
                           for i in selected_requests):
                        raise RuntimeError('Single synthetic request plus output reserve exceeds native GPU KV capacity')
                    prepared_id, proofs = hybrid_transition(cfg, layout, replica, child, check, 'prepare')
                    hybrid_transition(cfg, layout, replica, child, check, 'begin_timing', prepared_id=prepared_id)
                    wanted = set(hybrid_indices(cfg, layout, replica))
                    requests = [row for row in workload_requests(hybrid_plan(cfg), 4, output_tokens=4)
                                if row['index'] in wanted]
                    payloads = [dict(input_ids=row['input_ids'],
                        sampling_params=dict(temperature=0, max_new_tokens=4, ignore_eos=True),
                        stream=False) for row in requests]
                else:
                    requests = [dict(cached_tokens=0, input_ids=[1] * 64)] * 4
                    payloads = [dict(input_ids=row['input_ids'],
                        sampling_params=dict(temperature=0, max_new_tokens=4, ignore_eos=True),
                        stream=False) for row in requests]
                replies = []
                with ThreadPoolExecutor(max_workers=4) as pool:
                    responses = pool.map(lambda payload: http_json(port, '/generate', key, payload, timeout=180), payloads)
                    for request, reply in zip(requests, responses, strict=True):
                        metadata = reply.get('meta_info', {})
                        if (metadata.get('completion_tokens') != 4
                                or metadata.get('prompt_tokens') != len(request['input_ids'])):
                            raise RuntimeError('Tiny native request did not execute the declared token counts')
                        if request['cached_tokens'] and metadata.get('cached_tokens', 0) < request['cached_tokens']:
                            raise RuntimeError('Native inference recomputed the requested synthetic prefix')
                        replies.append({k: metadata[k] for k in ('prompt_tokens', 'completion_tokens', 'cached_tokens')
                                        if k in metadata})
                if cfg.get('hybrid_bank_json'):
                    hybrid_transition(cfg, layout, replica, child, check, 'finish_timing')
                after = http_json(port, '/get_server_info', key)
                if allocator_capacity(after) != capacity:
                    raise RuntimeError('Allocator capacity changed during native qualification')
                observations.append(dict(replica=replica, gpu_ids=child['gpu_ids'],
                    usage_log_id=child['usage_log_id'], expected_ranks=layout['usage_logs'][replica]['expected_ranks'],
                    max_total_num_tokens=capacity, server_info=safe_server_info(after),
                    single_request_300k_forward_executed=any(
                        row['cached_tokens'] >= 300000 for row in requests),
                    single_request_300k_token_capacity_bound_satisfied=(capacity >= 300000 + 513),
                    native_request_count=len(requests), native_request_tokens=replies,
                    hybrid_CPU_hot_GPU_cold_proofs=proofs,
                    qualification_wall_s=time.monotonic() - started,
                    requested_backend=backend_choices(cfg, layout),
                    fp8_adapter_attestations=(read_fp8_attestations(cfg, layout, replica)
                        if cfg['cohort'] == 'large' else None),
                    fp8_decode_graph_attestations=(read_fp8_graph_attestations(cfg, layout, replica)
                        if cfg['cohort'] == 'large' else None),
                    hybrid_restore_smoke_passed=proofs is not None))
        except BaseException as error:
            failure = type(error).__name__ + ': ' + str(error).replace(key, '[REDACTED]')
        finally:
            try:
                evidence = cleanup(children)
            except Exception as error:
                evidence = dict(cleanup_verified=False, cleanup_error_type=type(error).__name__,
                    process_identities=[child['identity'] for child in children])
            save(output / (layout['id'] + '.cleanup.json'), evidence)
        row = dict(source='actual_allocator', replicas=observations,
            per_replica_tokens=[r['max_total_num_tokens'] for r in observations],
            valid=failure is None and evidence['cleanup_verified'], error=failure,
            cleanup_verified=evidence['cleanup_verified'],
            loader_usage_attestation=usage_attestation,
            loader_usage_audit_pending=cfg['cohort'] == 'large',
            hybrid_restore_smoke_passed=bool(observations) and all(
                row['hybrid_restore_smoke_passed'] for row in observations))
        result['layouts'][layout['id']] = row
        save(output / (layout['id'] + '.json'), row); save(output / 'results.json', result)
        if not row['valid']:
            raise RuntimeError(f"Qualification failed for {layout['id']}; see saved error/cleanup evidence")
    result['native_full_model_execution_passed'] = True
    result['loader_usage_audit_pending'] = cfg['cohort'] == 'large'
    result['native_load_qualification_passed'] = cfg['cohort'] != 'large'
    result['hybrid_restore_smoke_passed'] = all(
        row['hybrid_restore_smoke_passed'] for row in result['layouts'].values())
    save(output / 'results.json', result)
    save(output / 'progress.json', dict(status=('controller_loader_audit_pending'
        if cfg['cohort'] == 'large' else 'native_load_qualification_passed'),
        completed=len(result['layouts']), hybrid_restore_smoke_passed=result['hybrid_restore_smoke_passed']))
    return result


def engine_entry(run, cfg, layout_id, replica):
    layout = next(row for row in cfg['layouts'] if row['id'] == layout_id)
    if not 0 <= replica < layout['replicas']:
        raise ValueError('Invalid replica index')
    authenticate(run, cfg, os.environ, local=True)
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    args = prepare_server_args(engine_args(cfg, layout, replica))
    # Disable MM requests. The separate language_model_only HF override above
    # prevents construction of the vision tower in the pinned native model.
    args.enable_multimodal = False
    # Read in-process; never expose the credential through a process argv.
    args.api_key = args.admin_api_key = read_key(cfg)
    try:
        if cfg.get('hybrid_bank_json'):
            from hybrid_entry import launch_http
            args.resolve_once()
            launch_http(args)
        else:
            run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def task_progress(run, cfg, stage, *, error=None):
    """Publish controller liveness, without treating qualification as throughput."""
    result_path = Path(run) / 'qualification/results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    valid = sum(row.get('valid') is True and not row.get('loader_usage_audit_pending', False)
                for row in results.get('layouts', {}).values())
    record = process_record(os.getpid())
    progress = dict(experiment_id=cfg['id'], task_id='qualification',
        timestamp=time.time(), controller_pid=os.getpid(),
        process_start_ticks=record['start_ticks'], stage=stage,
        valid_points=valid, expected=len(cfg['layouts']),
        performance_measured=False, error=error)
    save(Path(run) / 'task/progress.json', progress)
    return progress


def task_cleanup(run, cfg):
    """Only worker evidence can attest GPU/process cleanup, even after failure."""
    root = Path(run) / 'qualification'
    reports = [json.loads(path.read_text()) for path in root.glob('*.cleanup.json')]
    passed = bool(reports) and all(row.get('cleanup_verified') is True and
        row.get('process_groups_exited') is True and
        row.get('node_gpu_idle_observed') is True for row in reports)
    evidence = dict(experiment_id=cfg['id'], task_id='qualification',
        timestamp=time.time(), passed=passed, gpu_idle=passed,
        model_processes_exited=passed,
        worker_evidence_count=len(reports),
        source='saved exact-owned-process-group and leased-node GPU cleanup')
    save(Path(run) / 'task/cleanup.json', evidence)
    return evidence


def controller(run, cfg):
    authenticate(run, cfg, os.environ)
    packet = {'environment': {key: os.environ[key] for key in GUARD_KEYS}}
    command = [cfg['python'], str(Path(__file__).resolve()), '--worker', '--run', str(run),
               '--model', cfg['cohort'], '--node', cfg['node']]
    log = Path(run) / 'qualification_controller.log'
    task_progress(run, cfg, 'starting')
    failure = None
    with log.open('ab', buffering=0) as handle:
        os.chmod(log, 0o600)
        proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            cfg['node'], shlex.join(command)], stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT)
        proc.stdin.write((json.dumps(packet) + '\n').encode()); proc.stdin.flush()
        try:
            while proc.poll() is None:
                authenticate(run, cfg, os.environ)
                task_progress(run, cfg, 'running')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Remote native qualification failed; inspect saved artifacts')
            if cfg['cohort'] == 'large':
                complete_controller_usage_audit(run, cfg, os.environ)
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=50)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait(timeout=5)
            task_progress(run, cfg, 'failed' if failure else 'complete', error=failure)
            task_cleanup(run, cfg)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run', type=Path, required=True)
    result.add_argument('--model', choices=('small', 'large'), required=True)
    result.add_argument('--node', required=True)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--engine-entry', action='store_true')
    result.add_argument('--layout'); result.add_argument('--replica', type=int)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.run, args.model, args.node)
    if args.engine_entry:
        if args.layout is None or args.replica is None:
            raise ValueError('Engine entry requires frozen layout and replica')
        engine_entry(args.run, cfg, args.layout, args.replica); return 0
    if args.worker:
        packet = json.loads(sys.stdin.readline())
        env = packet.get('environment')
        if not isinstance(env, dict) or set(env) != set(GUARD_KEYS):
            raise RuntimeError('Invalid Guard stdin environment packet')
        os.environ.update(env)
        worker(args.run, cfg, dict(os.environ), control_stream=sys.stdin); return 0
    return controller(args.run, cfg)


if __name__ == '__main__':
    raise SystemExit(main())
