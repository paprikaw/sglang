"""Synthetic CPU fixtures for provenance and one-collector protocol, not data."""
import ast
import copy
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import common
import controller
import measure
import profile_source as p


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load('followup_stage_runtime', ROOT / 'runtime/stage_activity.py')
a = load('followup_stage_analysis', ROOT / 'analysis/stage_activity.py')
j = load('followup_join_analysis', ROOT / 'analysis/summarize_profile_followup.py')


def event(name, start, duration=1, cat='kernel', **args):
    return dict(name=name, ts=start, dur=duration, ph='X', cat=cat, pid=1, tid=2, args=args)


def test_same_session_proofs_removed_only_by_individually_attested_identity():
    proof = []
    actual = []
    for i in range(3):
        proof.append(dict(actual_gpu_kernel_count=1, correlation=str(i), device=0, stream='9'))
        actual.append(event('observed-spin', i * 10, correlation=i, device=0, stream=9))
    meta = [dict(index=i, start_stream=123, end_stream=123) for i in range(2)]
    actual += [event('observed-spin', 100 + i * 10, correlation=100+i, device=0, stream=9) for i in range(4)]
    receipt = dict(marker_kernel_name='observed-spin', segments=meta,
        same_session_marker_signature=dict(name='observed-spin', observed_proof_count=3, proofs=proof),
        bootstrap={'same_session_marker_proofs': True})
    result = a.marker_segments(receipt, actual)
    assert len(result) == 2 and result[0]['gpu_start_us'] == 100
    wrong = copy.deepcopy(receipt)
    wrong['same_session_marker_signature']['proofs'][0]['correlation'] = 'missing'
    with pytest.raises(ValueError, match='attested'):
        a.marker_segments(wrong, actual)
    # An unscoped extra same-name kernel cannot be hidden by dropping a prefix.
    with pytest.raises(ValueError, match='Marker mismatch'):
        a.marker_segments(receipt, [event('observed-spin', 99, correlation=99, device=0, stream=9), *actual])
    duplicate = copy.deepcopy(receipt)
    duplicate['same_session_marker_signature']['proofs'][1] = proof[0]
    with pytest.raises(ValueError, match='Duplicate'):
        a.marker_segments(duplicate, actual)
    overlap = copy.deepcopy(actual)
    overlap[0]['ts'] = 105
    with pytest.raises(ValueError, match='overlaps'):
        a.marker_segments(receipt, overlap)


def test_offline_calibration_copies_and_original_config_are_hash_bound(tmp_path):
    source = tmp_path / 'original'
    config_path = write(source / 'config.json', {'identity': 'original'})
    arm = tmp_path / 'followup/ep8_mtp_on'
    rows = [event('stage-cal:expert_routed', 0, 20, 'user_annotation'),
            event('cudaLaunchKernel', 1, cat='cuda_runtime', correlation=1),
            event('observed-gemm', 100, correlation=1)]
    traces = []
    for index in (0, 1):
        name = f'calibration_rank0_target_graph0_warm{index}.json'
        original = write(source / name, {'traceEvents': rows})
        copied = write(arm / name, {'traceEvents': rows})
        traces.append(dict(p.descriptor(original), copied_path=str(copied)))
    write(arm.parent / 'config.json', dict(profile_only_followup=True, profile_source=dict(
        source_run=str(source), source_config_sha256=p.sha(config_path),
        arms={'ep8_mtp_on': {'calibrations': traces}})))
    result = a.build_calibration(arm)
    assert result['offline_reuse']['timing_values_imported'] is False
    assert result['exact_names'] == {'observed-gemm': 'expert_routed'}
    with Path(traces[0]['copied_path']).open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError, match='bytes changed'):
        a.build_calibration(arm)


def test_one_collector_stays_running_from_idle_proof_through_formal_cohort(tmp_path, monkeypatch):
    calls = []
    class Profile:
        def start(self): calls.append('start')
        def stop(self): calls.append('stop')
    def profile(**kwargs):
        assert kwargs['activities'] == ['CPU', 'CUDA']
        calls.append('create')
        return Profile()
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append('sync'),
        _sleep=lambda n: calls.append(('proof', n))),
        profiler=SimpleNamespace(profile=profile, record_function=lambda label: nullcontext(),
            ProfilerActivity=SimpleNamespace(CPU='CPU', CUDA='CUDA')),
        empty=lambda *args, **kwargs: SimpleNamespace(fill_=lambda x: calls.append('fill')), int32='int32')
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(r, '_STATE', dict(armed=False, bootstrapped=False, profiler=None, done=False))
    monkeypatch.setattr(r, '_config', lambda: dict(profile_only_followup=True, arm='ep8_mtp_on'))
    monkeypatch.setattr(r, '_require_idle', lambda scheduler: calls.append('idle'))
    monkeypatch.setattr(p, 'validate_source_once', lambda *args, **kwargs: calls.append('source_valid'))
    monkeypatch.setattr(r.time, 'sleep', lambda n: calls.append(('delay', n)))
    result = r.bootstrap(object())
    assert result['bootstrap_deferred_to_same_session'] and 'create' not in calls
    result = r.arm_profile(object(), 'profile-decode-4096')
    assert result['armed'] and result['active']
    assert calls.count('create') == calls.count('start') == 1 and 'stop' not in calls
    assert calls.count(('proof', 100000000)) == 3
    assert calls.count('sync') == 5
    assert r._STATE['marker_name'] is None  # No guessed marker before export.
    assert r._STATE['bootstrap']['collector_lifecycles'] == 1
    with pytest.raises(RuntimeError, match='Only one'):
        r.arm_profile(object(), 'not-a-second-session')


def test_profile_only_path_does_not_run_production_and_keeps_failure_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'validate_source_once', lambda *args, **kwargs: {'original': 'SHA-bound'})
    monkeypatch.setattr(measure, 'warm_profile_followup', lambda *args: {'untimed': True})
    calls = []
    def fail(*args): calls.append('one_profile'); raise ValueError('actual GPU markers absent')
    monkeypatch.setattr(measure, 'profile_once', fail)
    monkeypatch.setattr(measure, 'save', lambda path, value: calls.append((Path(path).name, value)))
    result = measure.measure_all('url', tmp_path, dict(profile_only_followup=True, arm='ep8_mtp_on', layout='ep8', mtp=True))
    assert result[0]['source_performance_valid'] and result[0]['performance_valid'] is None
    assert result[0]['new_production_cohorts'] == 0 and not result[0]['profile_valid']
    assert calls.count('one_profile') == 1
    assert not any(isinstance(x, tuple) and x[0] in ('p4096_decode_results.json', 'p4096_decode_repeat_results.json', 'TIMED') for x in calls)


def test_short_untimed_warm_precedes_first_collector_and_rechecks_native_invariants(tmp_path, monkeypatch):
    calls = []
    saved = {}
    cfg = dict(arm='ep8_mtp_on',layout='ep8',mtp=True)
    monkeypatch.setattr(measure,'import_cold',lambda *args:calls.append('import') or [{'passed':True}])
    cold=[dict(rank=n,profiler=dict(armed=False,active=False)) for n in range(8)]
    after=[dict(rank=n,humming_validation={'passed':True}) for n in range(8)]
    def command(run,cfg,prefix,action,ident):
        calls.append(action)
        return cold if action=='cold' else after
    monkeypatch.setattr(measure,'command',command)
    def batch(url,cfg,cohort,*,output_tokens):
        calls.append(('batch',cohort,output_tokens))
        return [{'actual_meta_checked_by_batch_request':True}],.2
    monkeypatch.setattr(measure,'batch_request',batch)
    monkeypatch.setattr(measure,'validate_receipts',lambda *args,**kwargs:{'passed':True})
    monkeypatch.setattr(measure,'scheduler_proof',lambda *args,**kwargs:{'passed':True})
    monkeypatch.setattr(measure,'save',lambda path,value:saved.update({Path(path).name:value}))
    result=measure.warm_profile_followup('url',tmp_path,cfg)
    assert calls==['import','cold',('batch','warm-decode-4096',8),'observe']
    assert result['untimed'] and result['cuda_profiler_active'] is False
    assert result['production_metric'] is False and result['formal_profile_requires_fresh_reimport']
    assert set(saved)=={'profile_followup_warm_results.json'}
    after[0]['humming_validation']['passed']=False
    with pytest.raises(RuntimeError,match='Humming'):
        measure.warm_profile_followup('url',tmp_path,cfg)
    cold[0]['profiler']['active']=True
    calls.clear()
    with pytest.raises(RuntimeError,match='first collector'):
        measure.warm_profile_followup('url',tmp_path,cfg)
    assert calls==['import','cold']


def test_formal_profile_resets_reimports_after_short_warm_and_only_then_starts_cupti():
    tree=ast.parse((ROOT/'runtime/measure.py').read_text())
    functions={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    followup=functions['profile_followup']
    assert followup.index('warm_profile_followup(')<followup.index('profile_once(')
    profile=functions['profile_once']
    assert profile.index('import_cold(')<profile.index("'profile_cold'")<profile.index('batch_request(')
    warm=functions['warm_profile_followup']
    assert 'output_tokens=8' in warm and 'profile_once' not in warm and 'TIMED' not in warm


def test_no_startup_cupti_in_followup_and_no_new_native_forward():
    tree = ast.parse((ROOT / 'runtime/stage_activity.py').read_text())
    for name in ('capture', 'pp_warm'):
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        first_if = next(n for n in fn.body if isinstance(n, ast.If))
        assert "_config().get('profile_only_followup')" in ast.unparse(first_if.test)
        assert 'original_' in ast.unparse(first_if.body[0])
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'arm_single_session')
    body = ast.unparse(fn)
    assert body.count('profiler.start()') == 1 and '.stop(' not in body
    assert 'forward(' not in body and 'TIMED' not in body


def test_followup_formal_output_is_compared_to_original_same_arm_after_collector_stop():
    tree = ast.parse((ROOT / 'runtime/measure.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'profile_once')
    text = ast.unparse(fn)
    assert text.index("'profile_finish'") < text.index("output_parity(original['requests'], rows)")
    assert text.index("output_parity(original['requests'], rows)") < text.index('profile_valid=True')
    assert 'profile_source_output_parity.json' in text


def test_followup_controller_counts_sources_separately_and_never_counts_new_performance():
    rows = [dict(arm=name, role='consumer', profile_only_followup=True, source_performance_valid=True,
        performance_valid=None, profile_valid=True, passed=True, failure=None, cleanup_verified=True)
        for name in common.CONSUMER_ORDER]
    assert controller.completed_counts(rows) == dict(valid_points=4, performance_points=0, profile_points=4, source_performance_points=4)
    row = dict(rows[0], profile_valid=False, passed=False, profile_failure={'message': 'actual activity missing'})
    assert controller.may_continue(row)
    assert not controller.may_continue(dict(row, source_performance_valid=False))
    assert not controller.may_continue(dict(row, cleanup_verified=False))


def test_joined_report_keeps_missing_profiles_missing_and_requires_output_parity(tmp_path):
    summary = j.load_summary()
    arm = 'ep8_mtp_on'
    assert j.profile_state(tmp_path, arm, summary)['status'] == 'not_measured'
    write(tmp_path / arm / 'arm_result.json', dict(passed=True, cleanup_verified=True,
        source_performance_valid=True, performance_valid=None, profile_valid=True, failure=None))
    profile = dict(profile_valid=True, all_rank_gpu_activity_audited=True,
                   source_production_output_parity={'passed': True})
    path = write(tmp_path / arm / 'p4096_decode_profile_results.json', profile)
    write(tmp_path / arm / 'stage_activity_summary.json', dict(ranks=[{'rank': n} for n in range(8)]))
    assert j.profile_state(tmp_path, arm, summary)['valid']
    profile['source_production_output_parity']['passed'] = False
    write(path, profile)
    assert not j.profile_state(tmp_path, arm, summary)['valid']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / 'production/runs/source'
    source.mkdir(parents=True)
    inputs = tmp_path / 'inputs'
    write(inputs / 'manifest.json', {'semantic': True})
    cfg = dict(arm_order=list(common.CONSUMER_ORDER), arms={arm: common.ARMS[arm] for arm in common.CONSUMER_ORDER},
        engine_root='/qualified/engine', engine_commit='exact-commit', model_revision='exact-model',
        bank_model_identity={'exact': 'model'}, source_semantic_bank_ready={'sha256': 'immutable-bank'},
        semantic_inputs=str(inputs), nodes=['node0', 'node1'], jobs={'node0': '1', 'node1': '2'},
        run_dir=str(tmp_path / 'followup'), api_port=1, dist_port=2, reuse_semantic_bank=True)
    fake = SimpleNamespace(CONSUMER_ORDER=common.CONSUMER_ORDER,
        select_arm=lambda c, arm: dict(c, arm=arm, **c['arms'][arm]),
        server_args=lambda c, n: ['--layout', c['layout'], '--mtp', str(c['mtp']), '--dist-init-addr', str(c['dist_port']), '--port', str(c['api_port'])],
        environment=lambda c, n, env: {'SGLANG_KERNEL': 'exact', 'CUDA_VISIBLE_DEVICES': '0,1,2,3'})
    common_path = source.parents[1] / 'runtime/common.py'
    common_path.parent.mkdir()
    common_path.write_text('# Synthetic frozen test program; not a measurement\n')
    source_config = write(source / 'config.json', cfg)
    write(source / 'dashboard_spec.json', {'frozen_inputs': [p.descriptor(source_config), p.descriptor(common_path)]})
    trace = {'traceEvents': [event('stage-cal:expert_routed', 0, 20, 'user_annotation'),
        event('cudaLaunchKernel', 1, cat='cuda_runtime', correlation=1), event('actual', 100, correlation=1)]}
    for arm in common.CONSUMER_ORDER:
        base = source / arm
        write(base / 'arm_result.json', dict(arm=arm, performance_valid=True, cleanup_verified=True, failure=None))
        receipts = [dict(rank=rank, passed=True, profiler={'active': False, 'armed': False}) for rank in range(8)]
        for name, cohort in (('p4096_decode_results.json', 'timed'), ('p4096_decode_repeat_results.json', 'repeat')):
            write(base / name, dict(arm=arm, layout=cfg['arms'][arm]['layout'], mtp=cfg['arms'][arm]['mtp'],
                cohort=cohort, engine_commit=cfg['engine_commit'], request_count=8,
                prefix_tokens_per_request=4096, actual_cached_tokens_per_request=4032,
                query_tokens_per_request=1, output_tokens_per_request=64, performance_valid=True,
                cuda_profiler_off_during_production=True, source_bank_import_timed=False, hicache_h2d_timed=True,
                old_context_constructed_by_model=True, synthetic_kv=False, atomic_http_batch=True,
                profile_only_performance_not_production_measurement=False, actual_graph_proof={'passed': True},
                actual_scheduler_proof={'passed': True}, requests=[{} for _ in range(8)],
                native_cpu_ready_gpu_empty=receipts, all_rank_after=receipts))
        write(base / 'p4096_decode_repeatability.json', dict(passed=True, both_cohorts_unprofiled=True,
            both_start_native_cpu_ready_gpu_empty=True, same_serving_process=True, same_immutable_bank=True))
        for node in cfg['nodes']:
            write(base / node / 'exit.json', dict(owned_children_exited=True, gpu_cleanup={'passed': True}))
        for rank in (0, 4):
            for role in (('target', 'draft') if cfg['arms'][arm]['mtp'] else ('target',)):
                for warm in (0, 1):write(base / f'calibration_rank{rank}_{role}_graph0_warm{warm}.json', trace)
            if cfg['arms'][arm]['layout'] == 'pp2ep4':write(base / f'calibration_pp_p2p_rank{rank}.json', trace)
    monkeypatch.setattr(p, 'source_common', lambda path: (fake, common_path))
    return source, cfg, fake


def test_source_descriptor_requires_four_complete_arms_and_preserves_original_metric_paths(source_fixture):
    source, cfg, fake = source_fixture
    result = p.build_source_descriptor(source, cfg, fake)
    assert result['source_production_cohorts'] == 8 and result['new_production_cohorts'] == 0
    assert len(result['arms']) == 4
    assert all(row['calibration_times_imported'] is False for row in result['arms'].values())
    assert all(Path(row['production_cohorts'][0]['path']).parent == source / arm for arm, row in result['arms'].items())
    (source / common.CONSUMER_ORDER[-1] / 'p4096_decode_repeat_results.json').unlink()
    with pytest.raises(FileNotFoundError):p.build_source_descriptor(source, cfg, fake)


def test_pp_draft_calibration_required_only_on_last_stage(source_fixture):
    source, cfg, fake = source_fixture
    base = source / 'pp2ep4_mtp_on'
    for path in base.glob('calibration_rank0_draft_*.json'):path.unlink()
    assert p.build_source_descriptor(source, cfg, fake)['arms']['pp2ep4_mtp_on']['performance_valid']
    for path in base.glob('calibration_rank4_draft_*.json'):path.unlink()
    with pytest.raises(ValueError, match='same-rank target/draft'):
        p.build_source_descriptor(source, cfg, fake)


def test_first_runtime_use_revalidates_source_bytes_and_execution_identity(source_fixture, monkeypatch):
    source, cfg, fake = source_fixture
    bound = p.build_source_descriptor(source, cfg, fake)
    cfg = dict(cfg, profile_only_followup=True, profile_source=bound, new_production_cohorts=0)
    monkeypatch.setattr(common, 'server_args', fake.server_args)
    monkeypatch.setattr(common, 'environment', fake.environment)
    p._VERIFIED.clear()
    result = p.validate_source_once(cfg, common.CONSUMER_ORDER[0])
    assert result['production_cohorts'][0]['path'].startswith(str(source))
    p._VERIFIED.clear()
    with pytest.raises(RuntimeError, match='execution identity'):
        p.validate_source_once(dict(cfg, engine_commit='changed-engine'), common.CONSUMER_ORDER[0])
    path = Path(result['production_cohorts'][1]['path'])
    with path.open('a') as stream:stream.write(' ')
    with pytest.raises(RuntimeError, match='evidence changed'):
        p.validate_source_once(cfg, common.CONSUMER_ORDER[0])


def test_profile_metadata_does_not_reparse_large_source_descriptor_per_forward(monkeypatch):
    count = []
    cfg = dict(profile_only_followup=True, arm='ep8_mtp_on')
    monkeypatch.setattr(common, 'config', lambda run: count.append('read') or cfg)
    monkeypatch.setenv('GLM_RUN_DIR', '/synthetic/frozen/run')
    monkeypatch.setattr(r, '_STATE', {})
    assert r._config() is cfg and r._config() is cfg
    assert count == ['read']


@pytest.mark.parametrize('change', ['graph', 'profiled', 'cleanup', 'parity', 'calibration_pair', 'calibration_gpu', 'source_frozen', 'engine', 'bank'])
def test_source_descriptor_rejects_changed_or_invalid_evidence(source_fixture, change):
    source, cfg, fake = source_fixture
    arm = common.CONSUMER_ORDER[0]
    if change in ('graph', 'profiled'):
        path = source / arm / 'p4096_decode_results.json'; row = p.read(path)
        if change == 'graph':row['actual_graph_proof']['passed'] = False
        else:row['native_cpu_ready_gpu_empty'][0]['profiler']['active'] = True
        write(path, row)
    elif change == 'cleanup':
        write(source / arm / 'node0/exit.json', dict(owned_children_exited=False))
    elif change == 'parity':
        write(source / arm / 'p4096_decode_repeatability.json', dict(passed=False))
    elif change == 'calibration_pair':
        (source / arm / 'calibration_rank0_target_graph0_warm1.json').unlink()
    elif change == 'calibration_gpu':
        write(source / arm / 'calibration_rank0_target_graph0_warm1.json', {'traceEvents': []})
    elif change == 'source_frozen':
        with (source / 'config.json').open('a') as stream:stream.write(' ')
    elif change == 'engine':cfg = dict(cfg, engine_commit='changed')
    elif change == 'bank':cfg = dict(cfg, source_semantic_bank_ready={'sha256': 'different-bank'})
    with pytest.raises((ValueError, FileNotFoundError)):
        p.build_source_descriptor(source, cfg, fake)
