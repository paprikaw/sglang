"""CPU-only telemetry regression; synthetic schedules are never measured evidence."""
import ast
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import dp_proof
import pp_proof
from spec_proof import validate_receipts


def hooks():
    """Execute only the real CPU predicate/wrapper, without importing SGLang."""
    tree = ast.parse((ROOT / 'runtime/scheduler_hooks.py').read_text())
    predicate = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == 'should_record_batch')
    wrapper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == 'run_batch')
    calls = []
    scope = dict(time=time, original_run=lambda *a, **k: calls.append((a, k)) or 'native')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[predicate, wrapper], type_ignores=[])),
                 'real_cpu_telemetry_wrapper', 'exec'), scope)
    return scope['should_record_batch'], scope['run_batch'], calls


def requests(ids):
    return [SimpleNamespace(rid=rid) for rid in ids]


def submit(wrapper, scheduler, ids, *, extend=None, mode='TARGET_VERIFY'):
    batch = SimpleNamespace(reqs=requests(ids), forward_mode=mode, extend_num_tokens=extend)
    assert wrapper(scheduler, batch) == 'native'


def scheduler(*, pp=False, followup=True, dp=0):
    return SimpleNamespace(_glm_record_warm_telemetry=followup,
        ps=SimpleNamespace(pp_size=2 if pp else 1, attn_dp_rank=dp),
        pp_loop_size=2 if pp else 1, _pp_microstep=0)


def actual_fixture():
    value = json.loads((ROOT / 'tests/fixtures/v6_warm_telemetry_failure.json').read_text())
    assert value['fixture_kind'] == 'unaltered_actual_cpu_receipts'
    return value['receipts']


@pytest.mark.parametrize('followup', [False, True])
def test_legacy_and_formal_telemetry_predicate(followup):
    predicate, _, _ = hooks()
    for prefix in ('timed-', 'repeat-', 'profile-'):
        assert predicate(requests([prefix + 'decode-4096-0']), profile_only_followup=followup)
    assert not predicate(requests(['prepare-prefix-0']), profile_only_followup=followup)
    assert not predicate(requests(['warm-other-0']), profile_only_followup=followup)
    for i in range(8):
        assert predicate(requests([f'warm-decode-4096-{i}']),
                         profile_only_followup=followup) is followup
    for rid in ('warm-decode-4096-8', 'warm-decode-4096--1', 'warm-decode-4096-00'):
        assert not predicate(requests([rid]), profile_only_followup=followup)


def test_actual_v6_failure_is_missing_cpu_rows_not_missing_native_graph():
    receipts = actual_fixture()
    assert all(r['batches'] == [] for r in receipts)
    assert all(not r['profiler']['active'] and not r['profiler']['armed'] for r in receipts)
    assert validate_receipts(receipts, 'ep8', 'warm-', True)['passed']
    with pytest.raises(RuntimeError, match='No actual local model forwards'):
        dp_proof.validate(receipts, 'warm-')


def test_real_admission_ids_collect_in_repaired_hook_and_pass_synthetic_dp_replay():
    """Only RIDs/admission/Graph are actual; the two CPU submissions are replay."""
    _, wrapper, calls = hooks()
    replay = copy.deepcopy(actual_fixture())
    for receipt in replay:
        state = scheduler(dp=receipt['rank'] // 4)
        ids = receipt['atomic_admissions'][0]['request_ids']
        submit(wrapper, state, ids, extend=260, mode='EXTEND')
        submit(wrapper, state, ids)
        assert [row['local_submission_ordinal'] for row in state._glm_timed_batches] == [0, 1]
        receipt['batches'] = state._glm_timed_batches
    assert len(calls) == 16  # Every collection delegates exactly once to native run.
    assert dp_proof.validate(replay, 'warm-')['passed']
    # Test does not alter (or pretend to recover) missing measured submissions.
    assert all(r['batches'] == [] for r in actual_fixture())


@pytest.mark.parametrize('mtp', [False, True])
def test_pp_followup_same_hook_collects_two_stable_b4_slots(mtp):
    _, wrapper, calls = hooks()
    replay = copy.deepcopy(actual_fixture())
    ids = [f'warm-decode-4096-{i}' for i in range(8)]
    for receipt in replay:
        state = scheduler(pp=True)
        for microstep in range(4):
            state._pp_microstep = microstep
            group = ids[(microstep % 2) * 4:(microstep % 2 + 1) * 4]
            submit(wrapper, state, group, extend=260 if microstep < 2 else None,
                   mode='EXTEND' if microstep < 2 else ('TARGET_VERIFY' if mtp else 'DECODE'))
        receipt['batches'] = state._glm_timed_batches
        receipt['microbatch'] = dict(enabled=mtp, loop_size=2,
            feedback_group_name='synthetic-test-group' if mtp else None,
            feedback_group_ranks=list(range(8)) if mtp else None)
        receipt['atomic_admissions'] = [dict(cohort='warm-decode-4096', request_ids=ids,
            request_indices=list(range(8)), request_count=8)]
    assert pp_proof.validate(replay, mtp, 'warm-')['passed']
    assert len(calls) == 32
    for receipt in replay:
        assert [b['pp_slot_id'] for b in receipt['batches']] == [0, 1, 0, 1]


def test_legacy_warm_does_not_shift_timed_and_repeat_ordinals():
    _, wrapper, calls = hooks()
    state = scheduler(followup=False)
    for _ in range(3):
        submit(wrapper, state, [f'warm-decode-4096-{i}' for i in range(4)])
    assert not hasattr(state, '_glm_timed_batches')
    submit(wrapper, state, [f'timed-decode-4096-{i}' for i in range(4)])
    assert state._glm_timed_batches[0]['local_submission_ordinal'] == 0
    state._glm_timed_batches = []  # Existing observe cleanup, unchanged.
    submit(wrapper, state, [f'repeat-decode-4096-{i}' for i in range(4)])
    assert state._glm_timed_batches[0]['local_submission_ordinal'] == 0
    assert len(calls) == 5


def test_followup_observe_reset_keeps_formal_profile_ordinal_zero():
    _, wrapper, _ = hooks()
    state = scheduler(followup=True)
    submit(wrapper, state, [f'warm-decode-4096-{i}' for i in range(4)])
    state._glm_timed_batches = []  # Existing followup warm observe cleanup.
    submit(wrapper, state, [f'profile-decode-4096-{i}' for i in range(4)])
    assert state._glm_timed_batches[0]['local_submission_ordinal'] == 0


def test_followup_mode_cached_at_idle_not_read_during_forward():
    tree = ast.parse((ROOT / 'runtime/scheduler_hooks.py').read_text())
    idle = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'idle')
    forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'run_batch')
    assert "self._glm_record_warm_telemetry = cfg.get('profile_only_followup') is True" in ast.unparse(idle)
    text = ast.unparse(forward)
    assert '_glm_record_warm_telemetry' in text
    assert 'config(' not in text and 'read_text(' not in text and 'synchronize' not in text
