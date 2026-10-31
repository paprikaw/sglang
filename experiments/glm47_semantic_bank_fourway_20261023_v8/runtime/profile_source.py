"""Read-only provenance for a profile-only follow-up, never new performance.

The original run remains the only owner of its production metrics. A new
profile may use only the identical engine, serving arguments, kernel flags,
semantic inputs and immutable bank. Calibration contributes labels, not times.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SCHEMA = 'glm47-profile-followup-source-v1'
_VERIFIED = set()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def execution_identity(cfg, arm, common):
    """Only ports/scratch/runtime locations differ between the two processes."""
    selected = common.select_arm(cfg, arm)
    args = []
    environments = []
    for node in (0, 1):
        values = common.server_args(selected, node)
        for flag in ('--dist-init-addr', '--port'):
            values[values.index(flag) + 1] = '<ephemeral>'
        args.append(values)
        env = common.environment(selected, node, {})
        environments.append({key: value for key, value in env.items()
            if key.startswith(('SGLANG_', 'NVSHMEM_', 'DEEPEP_', 'NCCL_', 'HUMMING_'))
            or key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'TORCH_CUDA_ARCH_LIST',
                       'LD_PRELOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    return dict(engine_root=cfg['engine_root'], engine_commit=cfg['engine_commit'],
        model_revision=cfg['model_revision'], model_identity=cfg['bank_model_identity'],
        semantic_manifest_sha256=sha(Path(cfg['semantic_inputs']) / 'manifest.json'),
        source_bank=cfg['source_semantic_bank_ready'], nodes=cfg['nodes'], jobs=cfg['jobs'],
        layout=selected['layout'], mtp=selected['mtp'], server_args_by_node=args,
        kernel_environment_by_node=environments)


def source_common(source_run):
    path = Path(source_run).parents[1] / 'runtime/common.py'
    spec = importlib.util.spec_from_file_location('qualified_profile_source_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _validate_cohort(row, cfg, arm, cohort):
    expected = dict(arm=arm, layout=cfg['arms'][arm]['layout'], mtp=cfg['arms'][arm]['mtp'],
        cohort=cohort, engine_commit=cfg['engine_commit'], request_count=8,
        prefix_tokens_per_request=4096, actual_cached_tokens_per_request=4032,
        query_tokens_per_request=1, output_tokens_per_request=64,
        performance_valid=True, cuda_profiler_off_during_production=True,
        source_bank_import_timed=False, hicache_h2d_timed=True,
        old_context_constructed_by_model=True, synthetic_kv=False,
        atomic_http_batch=True, profile_only_performance_not_production_measurement=False)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Source production protocol/validity differs: ' + arm + '/' + cohort)
    for key in ('actual_graph_proof', 'actual_scheduler_proof'):
        if row.get(key, {}).get('passed') is not True:
            raise ValueError('Source production lacks native Graph/scheduler proof: ' + key)
    if len(row.get('requests', [])) != 8:
        raise ValueError('Source production does not contain all eight requests')
    for key in ('native_cpu_ready_gpu_empty', 'all_rank_after'):
        values = row.get(key, [])
        if len(values) != 8 or {value.get('rank') for value in values} != set(range(8)) or any(value.get('passed') is not True for value in values):
            raise ValueError('Source production lacks eight valid rank receipts: ' + key)
    if any(value.get('profiler', {}).get('armed') or value.get('profiler', {}).get('active')
           for value in row['native_cpu_ready_gpu_empty']):
        raise ValueError('Source production was profiled')


def calibration_files(arm_dir, mtp, pp):
    """Require actual scoped GPU kernels and independent warm0/warm1 pairs."""
    groups = {}
    paths = sorted(Path(arm_dir).glob('calibration*.json'))
    for path in paths:
        values = read(path).get('traceEvents', [])
        if not any(e.get('cat') in ('kernel', 'cuda_kernel') and e.get('ph') == 'X' for e in values):
            raise ValueError('Calibration contains no actual GPU kernels: ' + str(path))
        if not any(e.get('cat') == 'user_annotation' and e.get('name', '').startswith('stage-cal:') for e in values):
            raise ValueError('Calibration contains no actual CPU classification scope: ' + str(path))
        match = re.fullmatch(r'calibration_rank(0|4)_(target|draft)_graph(\d+)_warm([01])\.json', path.name)
        if match:
            rank, role, graph, warm = match.groups()
            groups.setdefault((int(rank), role, int(graph)), set()).add(int(warm))
        elif not re.fullmatch(r'calibration_pp_p2p_rank(0|4)\.json', path.name):
            raise ValueError('Unexpected calibration identity: ' + path.name)
    if any(warms != {0, 1} for warms in groups.values()):
        raise ValueError('Calibration warm0/warm1 pair incomplete')
    for rank in (0, 4):
        # PP's draft model exists only on the last stage. EP8 has a draft
        # worker on both Attention DP replicas. Never require phantom rank0
        # draft calibration for PP, or classify its target as a draft model.
        for role in (('target', 'draft') if mtp and (not pp or rank == 4) else ('target',)):
            if not any(key[:2] == (rank, role) for key in groups):
                raise ValueError('Required same-rank target/draft calibration pair missing')
        if pp and not (Path(arm_dir) / f'calibration_pp_p2p_rank{rank}.json').is_file():
            raise ValueError('PP boundary calibration missing')
    return paths


def build_source_descriptor(source_run, cfg, common):
    """Preparation fails until all four production arms and calibs are ready."""
    source_run = Path(source_run).resolve()
    source_cfg = read(source_run / 'config.json')
    if source_cfg.get('profile_only_followup'):
        raise ValueError('Source must be an original production run')
    if source_cfg.get('arm_order') != list(common.CONSUMER_ORDER):
        raise ValueError('Source does not contain the four exact production arms')
    old_path = source_run.parents[1] / 'runtime/common.py'
    source_spec = read(source_run / 'dashboard_spec.json')
    frozen = {row['path']: row['sha256'] for row in source_spec['frozen_inputs']}
    for path in (source_run / 'config.json', old_path):
        if frozen.get(str(path)) != sha(path):
            raise ValueError('Original config/runtime was not frozen or changed: ' + str(path))
    evidence = [descriptor(source_run / 'dashboard_spec.json')]
    for item in source_spec['frozen_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source frozen program/model/native dependency changed: ' + item['path'])
        evidence.append(dict(item))
    # Load the old identity builder only after its exact bytes and all frozen
    # dependencies have been checked against the original specification.
    old_common, _ = source_common(source_run)
    arms = {}
    for arm in common.CONSUMER_ORDER:
        if execution_identity(source_cfg, arm, old_common) != execution_identity(cfg, arm, common):
            raise ValueError('Follow-up engine/config/kernel/bank identity differs: ' + arm)
        base = source_run / arm
        result = read(base / 'arm_result.json')
        if (result.get('performance_valid') is not True or result.get('cleanup_verified') is not True
            or result.get('failure') is not None or result.get('arm') != arm):
            raise ValueError('Source arm lacks valid performance and verified cleanup: ' + arm)
        cohorts = []
        for name, cohort in (('p4096_decode_results.json', 'timed'), ('p4096_decode_repeat_results.json', 'repeat')):
            path = base / name
            _validate_cohort(read(path), source_cfg, arm, cohort)
            cohorts.append(descriptor(path))
        parity_path = base / 'p4096_decode_repeatability.json'
        parity = read(parity_path)
        if any(parity.get(key) is not True for key in ('passed', 'both_cohorts_unprofiled',
                'both_start_native_cpu_ready_gpu_empty', 'same_serving_process', 'same_immutable_bank')):
            raise ValueError('Source exact repeatability proof missing: ' + arm)
        calibrations = [descriptor(path) for path in calibration_files(base, source_cfg['arms'][arm]['mtp'], source_cfg['arms'][arm]['layout'] == 'pp2ep4')]
        cleanup = []
        for node in source_cfg['nodes']:
            path = base / node / 'exit.json'
            value = read(path)
            if value.get('owned_children_exited') is not True or value.get('gpu_cleanup', {}).get('passed') is not True:
                raise ValueError('Source actual node cleanup missing: ' + arm + '/' + node)
            cleanup.append(descriptor(path))
        row = dict(arm=arm, performance_valid=True, both_cohorts_validated=True,
            execution_identity=execution_identity(source_cfg, arm, old_common),
            arm_result=descriptor(base / 'arm_result.json'), production_cohorts=cohorts,
            repeatability=descriptor(parity_path), cleanups=cleanup, calibrations=calibrations,
            calibration_times_imported=False, new_production_measurement=False)
        arms[arm] = row
        evidence.extend([row['arm_result'], *cohorts, row['repeatability'], *cleanup, *calibrations])
    return dict(schema=SCHEMA, source_run=str(source_run), source_config_sha256=sha(source_run / 'config.json'),
        arms=arms, evidence=evidence, source_production_cohorts=8, new_production_cohorts=0,
        calibration_use='Exact names/launch geometry/ordered scopes only; no prior timing imported')


def validate_source_once(cfg, arm=None):
    value = cfg.get('profile_source', {})
    if not cfg.get('profile_only_followup') or value.get('schema') != SCHEMA or cfg.get('new_production_cohorts') != 0:
        raise RuntimeError('Missing frozen profile-only source contract')
    selected = [arm] if arm else list(value['arms'])
    for name in selected:
        key = (cfg['run_dir'], name, value['source_config_sha256'])
        if key not in _VERIFIED:
            import common
            if name not in common.execution_arms(cfg) or name not in value['arms']:
                raise RuntimeError('Unknown profile source arm')
            row = value['arms'][name]
            if row.get('performance_valid') is not True or row.get('both_cohorts_validated') is not True:
                raise RuntimeError('Profile source production gates failed')
            if execution_identity(cfg, name, common) != row['execution_identity']:
                raise RuntimeError('Profile engine/config/kernel/bank execution identity changed')
            # Full frozen input hashes (including calibration copies/source) are
            # independently authenticated before launch. Recheck small live
            # production/cleanup receipts at first use, not each forward.
            for item in [row['arm_result'], *row['production_cohorts'], row['repeatability'], *row['cleanups']]:
                if sha(item['path']) != item['sha256']:
                    raise RuntimeError('Source production/cleanup evidence changed: ' + item['path'])
            _VERIFIED.add(key)
    return value['arms'][arm] if arm else value
