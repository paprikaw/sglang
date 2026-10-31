"""Join original production metrics with new profiles without relabeling data."""
import argparse
import importlib.util
import json
from pathlib import Path


def load_summary():
    path = Path(__file__).with_name('summarize.py')
    spec = importlib.util.spec_from_file_location('original_production_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_file(item, summary):
    actual = summary.provenance(item['path'])
    if actual['sha256'] != item['sha256']:
        raise ValueError('SHA-bound original evidence changed: ' + item['path'])
    return actual


def profile_state(run, arm, summary):
    base = Path(run) / arm
    result = dict(valid=False, status='not_measured', missing=[], evidence=[])
    values = {}
    for name in ('arm_result.json', 'p4096_decode_profile_results.json', 'stage_activity_summary.json'):
        path = base / name
        if not path.exists():
            result['missing'].append(str(path))
        else:
            values[name] = summary.read(path)
            result['evidence'].append(summary.provenance(path))
    if result['missing']:
        return result
    owned = values['arm_result.json']
    profile = values['p4096_decode_profile_results.json']
    activity = values['stage_activity_summary.json']
    result['valid'] = (owned.get('passed') is True and owned.get('cleanup_verified') is True
        and owned.get('source_performance_valid') is True and owned.get('profile_valid') is True
        and owned.get('performance_valid') is None and owned.get('failure') is None
        and profile.get('profile_valid') is True and profile.get('all_rank_gpu_activity_audited') is True
        and profile.get('source_production_output_parity', {}).get('passed') is True
        and {r.get('rank') for r in activity.get('ranks', [])} == set(range(8)))
    result.update(status='valid_profile' if result['valid'] else 'failed_profile',
        source_output_parity=profile.get('source_production_output_parity'),
        failure=owned.get('profile_failure') or owned.get('failure'), activity=activity)
    return result


def summarize(run):
    run = Path(run)
    summary = load_summary()
    cfg = summary.read(run / 'config.json')
    source = cfg.get('profile_source', {})
    if not cfg.get('profile_only_followup') or cfg.get('new_production_cohorts') != 0:
        raise ValueError('This report only joins a declared profile-only follow-up')
    source_config = summary.provenance(Path(source['source_run']) / 'config.json')
    if source_config['sha256'] != source['source_config_sha256']:
        raise ValueError('Original production configuration changed')
    evidence = [source_config]
    for arm in summary.ARMS:
        row = source['arms'][arm]
        for item in [row['arm_result'], *row['production_cohorts'], row['repeatability'], *row['cleanups']]:
            evidence.append(bound_file(item, summary))
    original = summary.summarize(source['source_run'])
    profiles = {arm: profile_state(run, arm, summary) for arm in summary.ARMS}
    valid_sources = sum(data['status'] == 'measured_repeatable' and source['arms'][arm]['performance_valid'] is True
                        for arm, data in original['arms'].items())
    valid_profiles = sum(row['valid'] for row in profiles.values())
    return dict(schema_version=1, passed=valid_sources == valid_profiles == 4,
        original_production_run=source['source_run'], profile_only_run=str(run.resolve()),
        valid_source_performance_points=valid_sources, valid_profile_points=valid_profiles,
        new_production_points=0, original_production=original, profiles=profiles,
        original_configuration=source_config, evidence=evidence,
        policy='Original production metrics retain their source paths and are not new measurements. New CUDA component times are diagnostic only; absent profiles remain absent. No stage/rank maxima are summed into TPOT.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    output = args.output or args.run / 'fourway_profile_joined.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(str(output))


if __name__ == '__main__':
    main()
