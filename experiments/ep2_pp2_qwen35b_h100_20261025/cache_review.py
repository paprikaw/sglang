"""Read-only evidence review; never changes a Guard decision or frozen artifact."""
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


def rankless_single_loader(payload, group_key):
    lines = [s for s in payload.decode('utf-8', errors='strict').splitlines()
             if 'CPU weight cache load:' in s]
    assert len(lines) == 1, 'Single process must have exactly one loader record'
    match = re.fullmatch(
        r'\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] '
        r'CPU weight cache load: source=canonical_cpu_daemon '
        r'key=([0-9a-f]{64}) elapsed=([0-9]+(?:\.[0-9]+)?)s', lines[0])
    assert match and match[1] == group_key, 'Invalid single-process canonical loader evidence'
    return dict(source='canonical_cpu_daemon', group_key=group_key,
                elapsed_s=float(match[2]), loader_records=1)


def review(run, post_path):
    """Only review the exact rankless single-process parser failure, not other failures."""
    post = json.loads(post_path.read_text())
    assert post['passed'] is False and post['exit_code'] == 0
    assert post['cleanup_verified'] and post['state'] == 'complete'
    assert post['valid'] == post['expected'] == 12
    assert post['model_cache'] == dict(passed=False,
        error=run.name+'.single: malformed or absent canonical loader logs')
    spec_path = run/'dashboard_spec.json'
    spec = json.loads(spec_path.read_text())
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert spec_hash == post['dashboard']['spec_sha256']
    for item in spec['frozen_inputs']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    binding_path = run/'dashboard_guard/model_cache_bindings.json'
    binding = json.loads(binding_path.read_text())
    assert binding['lease_spec_sha256'] == spec_hash
    assert binding['experiment_id'] == run.name
    cfg = json.loads((run/'config.json').read_text())
    layout = next(x for x in cfg['layouts'] if x['id'] == 'single')
    assert all(layout[k] == 1 for k in ('tp','ep','dp','pp','replicas'))
    assert layout['gpu_groups'] == [[0]]
    preparation = json.loads((run/'preparation.json').read_text())
    guard = next(Path(s) for s in preparation['command'] if s.endswith('/dashboard_guard.py'))
    sys.path.insert(0, str(guard.parent))
    adapter = importlib.import_module('model_cache_adapter')
    logs = []
    for node, item in binding['nodes'].items():
        assert item['passed'] and item['cache_action'] == 'hit'
        assert item['group_key'] == item['model_identity']['group_key']
        declared = {x['id']:x for x in spec['model_cache']['usage_logs'][node]}
        for use in item['usage_logs']:
            original = declared[use['id']]
            assert original['path'] == use['path']
            assert set(original['expected_ranks']) == set(use['expected_ranks'])
            if use['id'] != run.name+'.single':
                logs.append(adapter._verify_usage_log(use,item['group_key'],binding['admitted_at']))
                continue
            assert original['single_instance_tp1'] is True
            assert use['expected_ranks'] == ['TP0']
            payload = adapter._usage_log_delta(use['path'],use['baseline'],binding['admitted_at'])
            proof = rankless_single_loader(payload,item['group_key'])
            qualifiers = [json.loads(s.split('H100_FULL_MODEL_QUALIFIED ',1)[1])
                for s in payload.decode().splitlines() if s.startswith('H100_FULL_MODEL_QUALIFIED ')]
            qualification = json.loads((run/'single/qualification/pp0.tp0.json').read_text())
            assert qualifiers == [qualification]
            assert qualification['passed'] and qualification['layers'] == 40
            assert qualification['ep_size'] == 1 and qualification['tp_rank'] == 0
            logs.append(dict(id=use['id'],path=use['path'],**proof,
                delta_sha256=hashlib.sha256(payload).hexdigest(),delta_bytes=len(payload),
                rank_source='Unranked TP1 log plus frozen single layout and native qualification'))
    assert len(logs) == 3
    result = dict(passed=True,review_mode='supplemental_read_only_rankless_single_loader_review',
        original_guard_passed=False,original_guard_postcheck=str(post_path),
        original_guard_postcheck_sha256=hashlib.sha256(post_path.read_bytes()).hexdigest(),
        original_guard_unchanged=True,admission_or_policy_changed=False,
        frozen_inputs_verified=True,cache_bindings_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        exact_multi_rank_loader_checks_passed=True,logs=logs,
        limitation='This is an independent evidence review, not a successful Guard postcheck.')
    out=run/'analysis';out.mkdir(exist_ok=True)
    (out/'cache_usage_supplemental_review.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
