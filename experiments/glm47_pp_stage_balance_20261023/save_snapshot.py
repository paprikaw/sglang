"""Archive small, sanitized results; never copy banks, private leases or prompts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / 'glm47_pp_local_feedback_20261023'
ID = 'glm47_pp_mtp_feedback_balance_20261023'
REDACT_KEYS = {'first_output_ids', 'second_output_ids', 'first_text', 'second_text'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k not in REDACT_KEYS}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.name != ID or dest.exists():
        raise ValueError('Require a new, uniquely named canonical snapshot')
    data = read(ROOT/'summary.json')
    first_run = FIRST/'runs/paired_r1'
    balance_run = ROOT/'runs/balance_r1'
    cfg = read(balance_run/'config.json')
    original_cfg = read(first_run/'config.json')
    assert cfg['engine_commit'] == original_cfg['engine_commit']
    assert cfg['nodes'] == original_cfg['nodes'] and cfg['jobs'] == original_cfg['jobs']
    assert data['cleanup_passed'] and data['balanced']['candidate_baseline_output_parity']['passed']
    engine = Path(cfg['engine_root'])
    assert git(engine, 'rev-parse', 'HEAD') == cfg['engine_commit']
    assert not git(engine, 'status', '--porcelain', '--untracked-files=no')
    assert git(engine, 'rev-parse', 'experiment/glm47-pp-mtp-microbatch-b4-qualified-20261023^{commit}') == cfg['engine_commit']
    guard = {}
    registry_path = Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json')
    registry = read(registry_path)
    for label, run in [('feedback', first_run), ('balance', balance_run)]:
        lease = read(run/'dashboard_guard/gpu_registry_lease.json')['lease_id']
        records = [r for r in registry['history'] if r.get('lease_id') == lease]
        assert records and records[-1]['state'] == 'released' and records[-1]['cleanup_verified']
        postfile, = (run/'dashboard_guard/checks').glob('*-post.json')
        post = read(postfile)
        assert post['passed'] and post['scientific_success'] and post['cleanup_verified']
        cleanup = read(run/'task/cleanup.json')
        task = read(run/'task/results.json')
        tests = read(run/'cpu_protocol_tests.json')
        assert cleanup['passed'] and task['passed'] and tests['passed']
        guard[label] = dict(postcheck_passed=True, scientific_success=True,
                            cleanup=cleanup, cpu_protocol_tests_passed=True,
                            registry_state='released', registry_cleanup_verified=True,
                            original_postcheck=str(postfile), postcheck_sha256=sha(postfile))
    rows = []
    originals = []
    for layout, run, arm in [
        ('PP46/46-original', first_run, 'pp2ep4_baseline'),
        ('PP46/46-local-feedback', first_run, 'pp2ep4_local_tail'),
        ('PP50/42-local-feedback', balance_run, 'pp2ep4_balanced')]:
        for cohort, filename in enumerate(('p4096_decode_results.json', 'p4096_decode_repeat_results.json'), 1):
            path = run/arm/filename
            perf = read(path)
            assert perf['performance_valid'] and not perf['weights_loading_timed']
            assert not perf['source_bank_import_timed'] and perf['hicache_h2d_timed']
            row = dict(configuration=layout, cohort=cohort, throughput_tok_s=perf['throughput_tok_s'],
                       wall_s=perf['wall_s'], requests=perf['request_count'],
                       logical_prefix_tokens=perf['prefix_tokens_per_request'],
                       cached_tokens=perf['actual_cached_tokens_per_request'],
                       output_tokens_per_request=perf['output_tokens_per_request'])
            for metric in ('ttft_s', 'tpot_s', 'end_to_end_s'):
                for stat in ('mean', 'median', 'p99'):
                    row[f'{metric}_{stat}'] = perf[metric][stat]
            rows.append(row)
            originals.append(dict(path=str(path), sha256=sha(path)))
    dest.mkdir(parents=True)
    write_json(dest/'summary.json', sanitize(data))
    write_json(dest/'guard_validation.json', guard)
    shutil.copy2(ROOT/'RESULTS.md', dest/'summary.md')
    with (dest/'full_metrics.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Preserve exact opt-in runtime adapters and CPU tests, not the run directory.
    for label, directory in [('feedback', FIRST), ('balance', ROOT)]:
        for source in sorted((directory/'runtime').rglob('*.py')):
            target = dest/'source'/label/'runtime'/source.relative_to(directory/'runtime')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for pattern in ('test*.py', 'analyze.py', 'save_snapshot.py', 'PLAN.md'):
            for source in directory.glob(pattern):
                target = dest/'source'/label/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    manifest = dict(
        id=ID, title='GLM-4.7 PP2xEP4 MTP Decode feedback and stage-balance optimization',
        created_at=datetime.now(timezone.utc).isoformat(), status='complete',
        implementation=dict(engine='SGLang', worktree=str(engine), commit=cfg['engine_commit'],
                            branch=git(engine, 'branch', '--show-current'),
                            source_tag='experiment/glm47-pp-mtp-microbatch-b4-qualified-20261023',
                            tracked_engine_dirty=False,
                            overlays='Frozen opt-in Python runtime adapters, preserved under source/. Not a globally enabled engine change.',
                            overlay_git_publication=False),
        deployment=dict(nodes=cfg['nodes'], slurm_jobs=cfg['jobs'],
                        immutable_run_ids=[original_cfg['id'], cfg['id']],
                        serving_endpoints_retired=True, gateway_route=None,
                        guard_validation='guard_validation.json'),
        workload=dict(requests=8, semantic_inputs=cfg['semantic_inputs'],
                      source_bank=cfg['semantic_bank_dir'], bank_identity_sha256=cfg['source_semantic_bank_ready']['bank_identity_sha256'],
                      logical_P=4096, actual_native_CPU_cached_P=4032, Q=1, recomputed_tail=64,
                      output_tokens_per_request=64, two_B4_PP_slots=True,
                      mtp=dict(topk=1, steps=2, verify_Q=3),
                      production_cohorts_per_configuration=2, weights_and_source_import_timed=False,
                      native_HiCache_H2D_timed=True, synthetic_KV=False),
        model=dict(path='/data/gpfs/projects/punim2715/huggingface/GLM-4.7-AWQ',
                   revision=cfg['model_revision'], weights='Humming INT4', kv_dtype='BF16'),
        testbed=dict(gpu='8 x NVIDIA A100 80GB', NIC='same-model ConnectX-6 Dx',
                     attention='TP4 per stage', experts='EP4 per stage', PP=2,
                     layouts=['46,46 baseline', '46,46 local feedback', '50,42 local feedback']),
        engine_parameters=dict(full_CUDA_Graph=True, deterministic_inference=True, overlap=False,
                               unchanged_expert_kernels=True, communication_policy=cfg['communication_policy'],
                               pcie_threshold=0, CE_decode=False,
                               DeepEP_native_sha256='9dc348878f7a3ad13fa8766a0d2738ca0d19ab116a3a5cc8e6e4e21b12988581'),
        artifacts=dict(summary='summary.json', report='summary.md', metrics='full_metrics.csv',
                       original_runs=[str(first_run), str(balance_run)],
                       raw_measurement_sources=originals, sanitized=True,
                       private_leases_prompts_weights_and_KV_not_copied=True),
        key_results=data['ratios'],
        caveats=['Two cohorts only; no statistical significance or universal optimum claim.',
                 'Historical EP8 reference is not a freshly rerun control; strict cross-layout output equality is unqualified.',
                 'Pre-existing MTP-versus-MTP-off numerical mismatch remains unresolved.',
                 'Profile Graph components are not production TPOT; current-stream gaps are not GPU idle.'])
    write_json(dest/'manifest.json', manifest)
    files = sorted(f for f in dest.rglob('*') if f.is_file())
    (dest/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(dest)}\n' for f in files))
    for entry in (dest/'SHA256SUMS').read_text().splitlines():
        digest, relative = entry.split('  ', 1)
        assert sha(dest/relative) == digest
    print(json.dumps(dict(snapshot=str(dest), files=len(files)+1, metric_rows=len(rows),
                          bytes=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()),
                          guard_and_cleanup_passed=True), indent=2))


if __name__ == '__main__':
    main()
