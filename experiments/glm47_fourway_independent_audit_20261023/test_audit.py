"""CPU-only evidence fixtures; none of these values are experiment results."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('independent_fourway_audit', Path(__file__).with_name('audit.py'))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def binding(tmp_path):
    production = tmp_path/'production/runs/run'
    profile = tmp_path/'profile/runs/run'
    arm = 'pp2ep4_mtp_on'
    identity = dict(engine='exact-source', arm=arm, serving=['tp4', 'ep4', 'pp2'], bank='semantic')
    original = write(production/'config.json', dict(identity=identity))
    for root in (production, profile):
        runtime = root.parents[1]/'runtime'
        runtime.mkdir(parents=True)
        (runtime/'common.py').write_text('# No model imports in this CPU fixture.\n')
    (profile.parents[1]/'runtime/profile_source.py').write_text(
        'def execution_identity(cfg, arm, common):\n    return cfg["identity"]\n')
    base = production/arm
    cohorts = [a.source(write(base/name, {'cohort': name})) for name in
               ('p4096_decode_results.json', 'p4096_decode_repeat_results.json')]
    calibration = write(base/'calibration_rank4_draft_graph1_warm0.json', {'scopes': 'actual-fixture'})
    copied = profile/arm/calibration.name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(calibration.read_bytes())
    row = dict(arm=arm, performance_valid=True, both_cohorts_validated=True,
        calibration_times_imported=False, new_production_measurement=False,
        execution_identity=identity, production_cohorts=cohorts,
        arm_result=a.source(write(base/'arm_result.json', {'passed': True})),
        repeatability=a.source(write(base/'repeat.json', {'passed': True})),
        cleanups=[a.source(write(base/'exit.json', {'clean': True}))],
        calibrations=[dict(a.source(calibration), copied_path=str(copied))])
    cfg = dict(identity=identity, profile_only_followup=True, new_production_cohorts=0,
        profile_source=dict(schema='glm47-profile-followup-source-v1', source_run=str(production),
            source_config_sha256=a.source(original)['sha256'], arms={arm: row}))
    write(profile/'config.json', cfg)
    return production, profile, arm, cfg


def test_binding_accepts_only_exact_source_and_copied_bytes(binding):
    production, profile, arm, cfg = binding
    result, row = a.bind_profile_source(production, profile, arm)
    assert result == cfg and row['execution_identity']['arm'] == arm


@pytest.mark.parametrize('change', ['production_bytes', 'copy_bytes', 'copy_arm', 'source_run', 'source_arm', 'new_production', 'execution'])
def test_binding_rejects_changed_or_misbound_evidence(binding, change):
    production, profile, arm, cfg = binding
    row = cfg['profile_source']['arms'][arm]
    if change == 'production_bytes':
        Path(row['production_cohorts'][0]['path']).write_text('{"different":"bytes"}')
    elif change == 'copy_bytes':
        Path(row['calibrations'][0]['copied_path']).write_text('{"different":"bytes"}')
    elif change == 'copy_arm':
        row['calibrations'][0]['copied_path'] = str(profile/'ep8_mtp_on/calibration.json')
    elif change == 'source_run':
        cfg['profile_source']['source_run'] = str(production.parent/'another-run')
    elif change == 'source_arm':
        row['arm'] = 'ep8_mtp_on'
    elif change == 'new_production':
        cfg['new_production_cohorts'] = 2
    elif change == 'execution':
        cfg['identity'] = dict(cfg['identity'], engine='another-engine')
    write(profile/'config.json', cfg)
    with pytest.raises(ValueError):
        a.bind_profile_source(production, profile, arm)


def test_missing_profile_remains_missing_not_an_estimate(binding):
    production, profile, arm, cfg = binding
    result = a.followup_profile_rows(production, profile, arm)
    assert result['valid'] is False and result['rows'] == []
    assert 'not_estimated' in result['status']


def test_preprofile_failure_is_explicit_and_not_reinterpreted_as_gpu_timings(binding):
    production, profile, arm, cfg = binding
    write(profile/arm/'arm_result.json', dict(failure={'message': 'No actual warm forwards in CPU telemetry'},
        cleanup_verified=True, profile_valid=False))
    result = a.followup_profile_rows(production, profile, arm)
    assert result['valid'] is False and result['rows'] == []
    assert result['profile_cleanup_verified'] is True
    assert result['failure']['message'] == 'No actual warm forwards in CPU telemetry'
