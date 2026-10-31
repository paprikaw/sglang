"""Audit prepared V4 inputs and matching V3 launch settings before scheduling."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from analyze_e7_v4_controls import COMMIT, RUNS

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value, run, base_port, key=None):
    if isinstance(value, dict):
        return {k: normalized(v, run, base_port, k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v, run, base_port) for v in value]
    if isinstance(value, str):
        return value.replace(str(run), '<RUN>').replace(run.name, '<RUN_ID>')
    if key in {'port', 'dist_port', 'bootstrap_port'}:
        return value-base_port
    return value


def main():
    validation_run = ROOT/'runs'/RUNS[0]
    proof = read(validation_run/'v4_cpu_validation.json')
    assert proof['passed'] and proof['checks'] == 16
    prior_names = ['eight_gpu_ep4_queue32_e7_t300_b7200_r1',
                   'eight_gpu_pp2_pd_e7_t300_b7200_r1',
                   'eight_gpu_ep2_split62_e7_t300_b7200_r1']
    common = None
    audit = []
    for i, name in enumerate(RUNS):
        run = ROOT/'runs'/name
        cfg, manifest = read(run/'config.json'), read(run/'launch_manifest.json')
        assert manifest['engine_commit'] == COMMIT
        assert cfg['branch_timing_policy'] == 'official_branch_issue_and_return_v1'
        assert not cfg['imported_episodes']
        frozen = read(run/'dashboard_spec.json')['frozen_inputs']
        for item in frozen:
            assert sha(Path(item['path'])) == item['sha256'], item['path']
        invariant = dict(cases=cfg['cases'], repeats=cfg['repeats'],
            graph_sha256=sha(Path(cfg['graphs_file'])),
            warmup_cases=cfg['warmup_cases'], clone_lane_policy=cfg['clone_lane_policy'],
            episode_timeout_s=cfg['episode_timeout_s'], waiting_timeout_s=cfg['waiting_timeout_s'],
            replay_sha256=sha(run/'code/agentx_replay.py'),
            branch_helper_sha256=sha(run/'code/agentx_branch_timing.py'),
            failure_capture_sha256=sha(run/'code/episode_capture.py'),
            cluster_runner_sha256=sha(run/'code/cluster_runner.py'))
        if common is None:
            common = invariant
        assert common == invariant
        assert invariant['replay_sha256'] == proof['replayer_sha256']
        assert invariant['branch_helper_sha256'] == proof['branch_helper_sha256']
        assert invariant['failure_capture_sha256'] == proof['failure_capture_sha256']
        prior_comparison = None
        if i < len(prior_names):
            oldrun = ROOT/'runs'/prior_names[i]
            old = read(oldrun/'config.json')
            new_port = next(iter(cfg['node_configs'].values()))['port']
            old_port = next(iter(old['node_configs'].values()))['port']
            for key in ['node_configs', 'endpoints', 'model_cache', 'cases', 'arms',
                        'profile_override', 'role_gpu_counts', 'warmup_cases',
                        'decode_extra_slots', 'episode_timeout_s', 'waiting_timeout_s']:
                assert normalized(cfg[key], run, new_port) == normalized(old[key], oldrun, old_port), key
            assert sha(run/'code/cluster_runner.py') == sha(oldrun/'code/cluster_runner.py')
            prior_comparison = dict(run=str(oldrun), same_launch_configuration_except_run_ids_and_ports=True,
                                    same_cluster_runner=True, prior_never_launched=not (oldrun/'supervisor.log').exists())
            assert prior_comparison['prior_never_launched']
        assert all(sum(c.values()) == 8 for c in cfg['role_gpu_counts'].values())
        assert cfg['repeats'] == 3 and len(cfg['cases']) == 1 and cfg['cases'][0]['concurrency'] == 128
        for node in cfg['node_configs'].values():
            assert node['mem_fraction_static'] == .95
        audit.append(dict(run=name, arms=cfg['arms'], planned_episodes=cfg['expected_points'],
            frozen_inputs=len(frozen), never_launched=not (run/'supervisor.log').exists(),
            profile=cfg['profile_override'], role_gpu_counts=cfg['role_gpu_counts'],
            decode_extra_slots=cfg['decode_extra_slots'], prior_comparison=prior_comparison))
    engine = Path(manifest['engine_root'])
    assert subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(engine), 'status', '--porcelain'], text=True).strip()
    result = dict(updated_at=time.time(), passed=True, engine_commit=COMMIT,
        cpu_proof=dict(path=str(validation_run/'v4_cpu_validation.json'), sha256=sha(validation_run/'v4_cpu_validation.json')),
        common=common, runs=audit, planned_episodes=sum(x['planned_episodes'] for x in audit),
        v3_main_pressure_run_excluded='eight_gpu_pd_ep4_e7_t300_b7200_r1',
        note='Preparation and CPU audit only; no V4 GPU result yet. All V4 arms use identical replay/helper/capture/runner bytes.')
    (ROOT/'e7_v4_cohort_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, runs=len(audit), planned_episodes=result['planned_episodes'],
                         frozen_inputs=[x['frozen_inputs'] for x in audit], cpu_checks=16)))


if __name__ == '__main__':
    main()
