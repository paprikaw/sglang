"""CPU-only synthetic trace/unit fixtures, never experimental measurements."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load('fourway_stage_analysis', ROOT / 'analysis' / 'stage_activity.py')
r = load('fourway_stage_runtime', ROOT / 'runtime' / 'stage_activity.py')


class NoReadback:
    shape = (12,)
    def __getattr__(self, name):
        raise AssertionError('Tensor readback attempted: ' + name)


class Mode:
    def __init__(self, name): self.name = name
    def is_decode(self): return self.name == 'DECODE'
    def is_target_verify(self): return self.name == 'TARGET_VERIFY'


def event(name, start, duration, cat='kernel', **args):
    return dict(name=name, ts=start, dur=duration, ph='X', cat=cat,
                pid=1, tid=2, args=args)


def test_batch_metadata_does_not_materialize_even_cpu_tensor():
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid='a'), SimpleNamespace(rid='b')],
                            input_ids=NoReadback(), seq_lens_cpu=NoReadback(),
                            forward_mode=Mode('TARGET_VERIFY'))
    meta = r.batch_metadata(batch)
    assert meta['input_shape'] == [12]
    assert meta['seq_lens_cpu'] is None
    assert meta['request_ids'] == ['a', 'b']
    assert meta['batch_size'] == 2


def test_worker_batch_uses_existing_scheduler_cpu_identity_only():
    batch = SimpleNamespace(batch_size=4, input_ids=NoReadback(), forward_mode=Mode('DECODE'))
    assert r.batch_metadata(batch, {'request_ids': ['a', 'b', 'c', 'd'], 'pp_microstep': 1})['pp_microstep'] == 1
    assert r.batch_metadata(batch)['request_ids'] == []


@pytest.mark.parametrize('mode,draft,expected', [('EXTEND', False, None), ('DECODE', False, 'target_decode'),
                                               ('TARGET_VERIFY', False, 'target_verify'),
                                               ('DECODE', True, 'draft_model')])
def test_only_post_extend_target_phases(mode, draft, expected):
    assert r.target_phase(SimpleNamespace(forward_mode=Mode(mode)), draft) == expected


def test_runtime_contains_no_inner_phase_fences_or_unsafe_forward_stop():
    tree = ast.parse((ROOT / 'runtime' / 'stage_activity.py').read_text())
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    segment = ast.unparse(methods['segment'])
    assert 'synchronize' not in segment and 'Event(' not in segment
    assert '.stop(' not in segment and 'finish_profile' not in segment
    forward = ast.unparse(methods['forward'])
    assert 'finish_profile' not in forward
    assert 'MAX_TARGET_FORWARDS' in segment


def test_gpu_correlation_uses_innermost_native_launch_scope():
    rows = [event('stage-cal:other_gpu', 0, 100, 'user_annotation'),
            event('stage-cal:attention', 10, 50, 'user_annotation'),
            event('stage-cal:nccl_tp', 20, 10, 'user_annotation'),
            event('cudaLaunchKernel', 22, 1, 'cuda_runtime', correlation=7),
            event('generic', 200, 5, correlation=7),
            event('unlinked', 300, 5, correlation=8)]
    result = a.calibration_rows(rows)
    assert result[0]['phase'] == 'nccl_tp'
    assert result[1]['phase'] == 'unclassified_gpu'
    # GPU activity need not be inside the CPU scope: correlation is essential.
    assert result[0]['unique_launch']


def test_ambiguous_runtime_correlation_stays_unknown():
    rows = [event('stage-cal:attention', 0, 100, 'user_annotation'),
            event('launch', 10, 1, 'cuda_runtime', correlation=1),
            event('launch', 20, 1, 'cuda_runtime', correlation=1),
            event('kernel', 200, 5, correlation=1)]
    assert a.calibration_rows(rows)[0]['phase'] == 'unclassified_gpu'


def calibration_fixture(path, first='attention', second='expert_shared'):
    rows = [event('stage-cal:other_gpu', 0, 100, 'user_annotation'),
            event('stage-cal:' + first, 10, 20, 'user_annotation'),
            event('launch', 15, 1, 'cuda_runtime', correlation=1),
            event('stage-cal:' + second, 40, 20, 'user_annotation'),
            event('launch', 45, 1, 'cuda_runtime', correlation=2),
            event('generic', 200, 5, correlation=1),
            event('generic', 210, 5, correlation=2)]
    path.write_text(json.dumps({'traceEvents': rows}))


def test_conflicting_names_need_independent_exact_graph_sequence(tmp_path):
    for index in (0, 1):
        calibration_fixture(tmp_path / f'calibration_rank0_target_graph0_warm{index}.json')
    cal = a.build_calibration(tmp_path)
    assert 'generic' not in cal['exact_names']
    assert cal['audit']['generic']['phases'] == ['attention', 'expert_shared']
    assert len(cal['sequences']) == 1
    kernels = [event('generic', 0, 1), event('generic', 2, 1)]
    assigned, matches = a.assign_phases(kernels, cal, 0, 'target')
    assert assigned == ['attention', 'expert_shared'] and matches
    assert a.assign_phases(kernels, cal, 1, 'target')[0] == ['unclassified_gpu'] * 2
    assert a.assign_phases(kernels, cal, 0, 'draft')[0] == ['unclassified_gpu'] * 2


def test_sequence_phase_mismatch_is_not_used(tmp_path):
    calibration_fixture(tmp_path / 'calibration_rank0_target_graph0_warm0.json')
    calibration_fixture(tmp_path / 'calibration_rank0_target_graph0_warm1.json', second='attention')
    cal = a.build_calibration(tmp_path)
    assert not cal['sequences'] and cal['rejected_sequences']
    assert not cal['exact_names']


def test_single_warm_not_sufficient_for_name_or_sequence(tmp_path):
    calibration_fixture(tmp_path / 'calibration_rank0_target_graph0_warm0.json', second='attention')
    cal = a.build_calibration(tmp_path)
    assert not cal['exact_names'] and not cal['sequences']


def fingerprint_event(grid, *, name='generic', correlation=1):
    return event(name, 200 + correlation * 10, 5, correlation=correlation,
                 grid=[grid, 1, 1], block=[256, 1, 1],
                 **{'registers per thread': 158, 'shared memory': 98304})


def fingerprint_calibration(path, *, second_phase='expert_shared', first_warm=False):
    rows = [event('stage-cal:other_gpu', 0, 100, 'user_annotation'),
            event('stage-cal:attention', 10, 20, 'user_annotation'),
            event('launch', 15, 1, 'cuda_runtime', correlation=1),
            event('stage-cal:' + second_phase, 40, 20, 'user_annotation'),
            event('launch', 45, 1, 'cuda_runtime', correlation=2),
            fingerprint_event(2, correlation=1), fingerprint_event(24, correlation=2)]
    if first_warm:
        # Deliberately invalidate whole-Graph sequence equality. Geometry
        # attribution must not depend on aligning these extra operations.
        rows.append(event('one-time-init', 230, 1, correlation=99))
    path.write_text(json.dumps({'traceEvents': rows}))


def test_launch_fingerprint_needs_both_scoped_warmups_and_exact_full_geometry(tmp_path):
    for index in (0, 1):
        fingerprint_calibration(tmp_path / f'calibration_rank0_target_graph0_warm{index}.json',
                                first_warm=index == 0)
    cal = a.build_calibration(tmp_path)
    assert 'generic' not in cal['exact_names'] and not cal['sequences']
    kernels = [fingerprint_event(2), fingerprint_event(24)]
    assert a.assign_phases(kernels, cal, 0, 'target')[0] == ['attention', 'expert_shared']
    assert a.assign_phases(kernels, cal, 0, 'target', use_launch_fingerprints=False)[0] == ['unclassified_gpu'] * 2
    assert a.assign_phases(kernels, cal, 1, 'target')[0] == ['unclassified_gpu'] * 2
    assert a.assign_phases(kernels, cal, 0, 'draft')[0] == ['unclassified_gpu'] * 2
    incomplete = copy.deepcopy(kernels)
    del incomplete[0]['args']['shared memory']
    incomplete[1]['args']['registers per thread'] = 157
    assert a.assign_phases(incomplete, cal, 0, 'target')[0] == ['unclassified_gpu'] * 2
    partial = copy.deepcopy(kernels[0])
    partial['args']['block'] = [256, 1]
    assert a.launch_fingerprint(partial) is None
    partial['args']['block'] = [True, 1, 1]
    assert a.launch_fingerprint(partial) is None


def test_launch_fingerprint_phase_conflict_stays_unknown(tmp_path):
    fingerprint_calibration(tmp_path / 'calibration_rank0_target_graph0_warm0.json', first_warm=True)
    fingerprint_calibration(tmp_path / 'calibration_rank0_target_graph0_warm1.json', second_phase='attention')
    cal = a.build_calibration(tmp_path)
    shared = [r for r in cal['launch_fingerprint_audit'] if r['fingerprint']['grid'][0] == 24]
    assert shared[0]['phases'] == ['attention', 'expert_shared']
    assert not shared[0]['eligible']
    assert a.assign_phases([fingerprint_event(24)], cal, 0, 'target')[0] == ['unclassified_gpu']


def test_launch_fingerprint_cannot_pair_different_graph_warm_indices(tmp_path):
    fingerprint_calibration(tmp_path / 'calibration_rank0_target_graph0_warm0.json', first_warm=True)
    fingerprint_calibration(tmp_path / 'calibration_rank0_target_graph1_warm1.json')
    cal = a.build_calibration(tmp_path)
    assert not cal['launch_fingerprints']


def test_pp_needs_two_actual_independent_native_launch_scopes(tmp_path):
    rows = []
    for i in range(2):
        rows += [event('stage-cal:pp_boundary', i*10, 5, 'user_annotation'),
                 event('launch', i*10+1, 1, 'cuda_runtime', correlation=i),
                 event('p2p', 100+i*10, 2, correlation=i)]
    (tmp_path / 'calibration_pp_p2p_rank0.json').write_text(json.dumps({'traceEvents': rows}))
    assert a.build_calibration(tmp_path)['exact_names'] == {'p2p': 'pp_boundary'}


def test_partition_preserves_idle_union_overlap_and_denominator():
    rows = [dict(event('a', 0, 5000), phase='attention'),
            dict(event('b', 1000, 2000), phase='attention'),
            dict(event('c', 4000, 3000), phase='deepep_dispatch')]
    result = a.partition(rows, 0, 10000)
    assert result['ms']['attention'] == 4
    assert result['ms']['deepep_dispatch'] == 2
    assert result['ms']['overlap'] == 1
    assert result['ms']['gpu_idle'] == 3
    assert sum(result['percent'].values()) == 100


def test_kmp_repeated_exact_full_sequence():
    assert a.subsequence_positions(['a','b','a','b','a'], ['a','b','a']) == [0,2]
    assert a.subsequence_positions(['a'], ['b']) == []


def receipt_fixture():
    segments = []
    for index in range(3):
        segments.append(dict(index=index, phase='target_verify', start_stream=1234, end_stream=1234,
                             request_ids=['a','b','c','d'], batch_size=4))
    return dict(marker_kernel_name='sleep', segments=segments,
                accepted=[dict(target_segment_index=0, output_tokens=8,
                               output_tokens_by_request=dict(a=2,b=2,c=2,d=2)),
                          dict(target_segment_index=1, output_tokens=7,
                               output_tokens_by_request=dict(a=1,b=2,c=2,d=2)),
                          dict(target_segment_index=2, output_tokens=100,
                               output_tokens_by_request=dict(a=25,b=25,c=25,d=25))])


def test_marker_pair_requires_all_observed_and_same_cupti_stream():
    receipt = receipt_fixture()
    markers = [event('sleep', i*10, 1, stream=9) for i in range(6)]
    assert len(a.marker_segments(receipt, markers)) == 3
    with pytest.raises(ValueError, match='Marker mismatch'):
        a.marker_segments(receipt, markers[:-1])
    markers[1]['args']['stream'] = 8
    with pytest.raises(ValueError, match='changed GPU streams'):
        a.marker_segments(receipt, markers)


def test_only_closed_cycles_use_actual_accepted_outputs_not_width_or_last_cycle():
    receipt = receipt_fixture()
    actual = a.accepted_normalization(receipt, receipt['segments'])
    assert actual['actual_committed_output_tokens'] == 15
    assert actual['complete_local_microbatch_cycles'] == 2
    assert actual['committed_tokens_per_request_per_cycle']['mean'] == 15/8
    missing = copy.deepcopy(receipt)
    missing['accepted'].pop(0)
    assert not a.accepted_normalization(missing, missing['segments'])['valid']


def test_unknown_names_remain_unknown_without_substring_guess():
    cal = dict(exact_names={}, sequences=[])
    kernels = [event('attention_expert_nccl_deepep', 0, 1)]
    assert a.assign_phases(kernels, cal, 0, 'target')[0] == ['unclassified_gpu']


def test_complete_rank_analysis_uses_device_window_and_preserves_unknown(tmp_path):
    receipt = receipt_fixture()
    receipt.update(passed=True, errors=[], arm='ep8_mtp_on', layout='ep8', mtp=True,
                   topology=dict(world_rank=0, pp_rank=0, tp_rank=0, attn_dp_rank=0))
    rows = []
    for index in range(3):
        rows += [event('sleep', index*10, .1, stream=9, device=0),
                 event('validated_attention', index*10+.2, .5, stream=9, device=0),
                 event('unknown_expert_named_kernel', index*10+.8, .1, stream=9, device=0),
                 event('sleep', index*10+1, .1, stream=9, device=0)]
    (tmp_path / 'activity_receipt_rank0.json').write_text(json.dumps(receipt))
    trace = tmp_path / 'activity_trace_rank0.json'
    trace.write_text(json.dumps(dict(traceEvents=rows)))
    cal = dict(exact_names={'validated_attention': 'attention'}, sequences=[])
    result = a.analyze_rank(tmp_path, 0, cal)
    assert result['steady_window']['duration_ms'] == .02
    assert result['steady_window']['ms']['attention'] == pytest.approx(.001)
    assert result['steady_window']['ms']['unclassified_gpu'] == pytest.approx(.0002)
    assert result['attribution_coverage']['before_launch_fingerprint']['unknown_kernel_count'] == 2
    assert result['attribution_coverage']['after_launch_fingerprint']['exclusive_unknown_wall_ms'] == pytest.approx(.0002)
    assert result['actual_output_normalization']['actual_committed_output_tokens'] == 15
    assert result['observed_same_request_slot_start_cadence_ms']['mean'] == .01
    rows[0]['args']['device'] = 1
    trace.write_text(json.dumps(dict(traceEvents=rows)))
    with pytest.raises(ValueError, match='rank-local GPU'):
        a.analyze_rank(tmp_path, 0, cal)


def marker_proof_events(missing=()):
    rows = [event('unrelated-prime-fill', 95, 1, device=0, correlation=999)]
    for index, label in enumerate(r.MARKER_PROOF_SCOPES):
        rows += [event(label, index*10, 5, 'user_annotation'),
                 event('cudaLaunchKernel', index*10+1, 1, 'cuda_runtime', correlation=index+1)]
        if index not in missing:
            rows.append(event('actually-observed-proof-kernel', 100+index*10, 1,
                              device=0, stream=9, correlation=index+1))
    return rows


def test_bootstrap_identity_ignores_unrelated_fill_and_accepts_two_positive_witnesses():
    signature = r.marker_signature(marker_proof_events(missing=(0,)), 0)
    assert signature['name'] == 'actually-observed-proof-kernel'
    assert signature['observed_proof_count'] == 2
    assert signature['missing_proof_count'] == 1
    assert signature['proofs'][0]['runtime_correlations'] == ['1']
    assert signature['proofs'][0]['actual_gpu_kernel_count'] == 0


def test_bootstrap_cpu_proof_is_not_duplicated_by_same_name_gpu_annotation():
    rows = marker_proof_events()
    for scope in [row for row in rows if row.get('cat') == 'user_annotation']:
        mirror = copy.deepcopy(scope)
        mirror.update(cat='gpu_user_annotation', pid=0, tid=9, ts=scope['ts']+100)
        rows.append(mirror)
    signature = r.marker_signature(rows, 0)
    assert signature['observed_proof_count'] == 3
    assert signature['name'] == 'actually-observed-proof-kernel'
    # Two real CPU scopes still remain ambiguous and must be rejected.
    rows.append(copy.deepcopy(next(row for row in rows if row.get('cat') == 'user_annotation')))
    with pytest.raises(RuntimeError, match='one actual CPU scope'):
        r.marker_signature(rows, 0)


def test_gpu_annotation_alone_never_substitutes_for_cpu_scope():
    rows = marker_proof_events()
    for row in rows:
        if row.get('cat') == 'user_annotation':row['cat'] = 'gpu_user_annotation'
    with pytest.raises(RuntimeError, match='one actual CPU scope'):
        r.marker_signature(rows, 0)


def test_bootstrap_identity_never_guesses_from_runtime_or_single_kernel():
    with pytest.raises(RuntimeError, match='at least two'):
        r.marker_signature(marker_proof_events(missing=(0, 1)), 0)
    with pytest.raises(RuntimeError, match='at least two'):
        r.marker_signature(marker_proof_events(missing=(0, 1, 2)), 0)
    # Scope proof is necessary even if another rank observed a plausible name.
    unscoped = [e for e in marker_proof_events() if e['name'] not in r.MARKER_PROOF_SCOPES]
    with pytest.raises(RuntimeError, match='actual CPU scope'):
        r.marker_signature(unscoped, 0)


def test_bootstrap_identity_rejects_conflicting_names_wrong_device_or_multiple_kernels():
    changed = marker_proof_events()
    changed[-1]['name'] = 'different-actual-kernel'
    with pytest.raises(RuntimeError, match='one exact name'):
        r.marker_signature(changed, 0)
    changed = marker_proof_events()
    changed[-1]['args']['device'] = 1
    with pytest.raises(RuntimeError, match='wrong/unknown device'):
        r.marker_signature(changed, 0)
    changed = marker_proof_events()
    changed.append(copy.deepcopy(changed[-1]))
    with pytest.raises(RuntimeError, match='multiple observed GPU kernels'):
        r.marker_signature(changed, 0)


def test_bootstrap_after_prior_profiles_is_primed_scoped_longer_and_only_at_idle(tmp_path, monkeypatch):
    calls, saved = [], {}
    class Profile:
        def __enter__(self): calls.append('start'); return self
        def __exit__(self, *args): calls.append('stop')
        def export_chrome_trace(self, path):
            Path(path).write_text(json.dumps({'traceEvents': marker_proof_events(missing=(0,))}))
    def profile(**kwargs):
        assert kwargs['activities'] == ['CPU', 'CUDA']
        return Profile()
    def record(label):
        calls.append(('scope', label))
        return nullcontext()
    fake = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: calls.append('synchronize'),
                             _sleep=lambda n: calls.append(('sleep', n))),
        profiler=SimpleNamespace(profile=profile, record_function=record,
                                 ProfilerActivity=SimpleNamespace(CPU='CPU', CUDA='CUDA')),
        empty=lambda *args, **kwargs: SimpleNamespace(fill_=lambda value: calls.append('prime_fill')),
        int32='int32')
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(r, '_STATE', dict(bootstrapped=False, armed=False, profiler=None, done=False))
    monkeypatch.setattr(r, '_config', lambda: dict(role='consumer'))
    monkeypatch.setattr(r, '_rank', lambda: 0)
    monkeypatch.setattr(r, '_output', lambda name: tmp_path / name)
    monkeypatch.setattr(r, '_save', lambda path, value: saved.update({path.name: value}))
    monkeypatch.setattr(r, '_require_idle', lambda scheduler: calls.append('confirmed_idle'))
    monkeypatch.setattr(r.time, 'sleep', lambda seconds: calls.append(('activation_wait', seconds)))
    assert r.bootstrap(object())['bootstrapped']
    assert calls[0] == 'confirmed_idle'
    assert calls.count('synchronize') == 5
    assert calls.count(('sleep', 100000000)) == 3
    assert calls.count(('activation_wait', 0.25)) == 1
    assert calls.index('start') < calls.index(('activation_wait', 0.25)) < calls.index('prime_fill')
    assert calls.index('prime_fill') < calls.index(('scope', r.MARKER_PROOF_SCOPES[0]))
    signature = saved['activity_marker_signature_rank0.json']
    assert signature['actual_model_forward_count'] == 0
    assert signature['added_cuda_synchronizations'] == 5
    assert signature['proof_sleep_cycles'] == 100000000
    assert signature['collector_activation_delay_s'] == 0.25
    assert signature['activation_delay_after_profiler_start'] is True
    assert signature['diagnostic_candidate'] is True
    assert signature['collector_failure_root_cause_proven'] is False
    assert signature['observed_proof_count'] == 2
    assert r._STATE['marker_name'] == 'actually-observed-proof-kernel'
    before = list(calls)
    r.bootstrap(object())
    assert calls == before  # No second lifecycle once actual signature is proven.


def test_delayed_bootstrap_rejects_timed_before_collector_or_sleep(tmp_path, monkeypatch):
    arm='ep8_mtp_on';(tmp_path/arm).mkdir();(tmp_path/arm/'TIMED').write_text('test')
    controller=SimpleNamespace(load_queue=[],write_queue=[],ack_load_queue=[],ack_write_queue=[])
    cache=SimpleNamespace(cache_controller=controller,ongoing_load_back={},ongoing_write_through={},
        ongoing_prefetch={},ongoing_backup={},tree_core=SimpleNamespace(has_ongoing_insert=lambda:False))
    scheduler=SimpleNamespace(is_fully_idle=lambda:True,tree_cache=cache)
    monkeypatch.setattr(r,'_STATE',dict(bootstrapped=False,armed=False,profiler=None,done=False))
    monkeypatch.setattr(r,'_config',lambda:dict(role='consumer',run_dir=str(tmp_path),arm=arm))
    def forbidden(*args,**kwargs):raise AssertionError('Collector/sleep touched inside TIMED')
    monkeypatch.setattr(r.time,'sleep',forbidden)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(profiler=SimpleNamespace(profile=forbidden)))
    with pytest.raises(RuntimeError,match='CUPTI cannot overlap'):
        r.bootstrap(scheduler)
