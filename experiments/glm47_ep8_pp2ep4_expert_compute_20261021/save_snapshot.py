"""Archive validated, non-secret small artifacts; never copy model tensors."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/r2'
EXPERIMENT_ID = 'glm47_ep8_pp2ep4_expert_compute_20261021_r2'
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / EXPERIMENT_ID


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    result = read(RUN / 'profile/result.json')
    spec = read(RUN / 'dashboard_spec.json')
    post = read(next((RUN / 'dashboard_guard/checks').glob('*-post.json')))
    cleanup = read(RUN / 'profile/task/cleanup.json')
    assert result['passed'] and post['scientific_success'] and post['passed']
    assert cleanup['passed'] and cleanup['gpu_idle'] and cleanup['model_processes_exited']
    assert len(result['rows']) == 36 and all(n['passed'] for n in result['numerical'])
    for source in spec['frozen_inputs']:
        assert digest(source['path']) == source['sha256'], source['path']
        assert not source['path'].endswith('.safetensors')
    with (RUN / 'analysis/comparison.csv').open() as f:
        comparisons = list(csv.DictReader(f))
    assert len(comparisons) == 12
    for row in comparisons:
        b = int(row['global_logical_tokens'])
        points = [r for r in result['rows'] if r['mode'] == row['mode'] and r['global_logical_tokens'] == b]
        ep = next(r for r in points if r['layout'] == 'ep8')
        pp = [r for r in points if r['layout'] == 'pp2_ep4']
        for point in points:
            assert point['finite'] and len(point['cuda_event_ms']) == 30
            assert point['local_assignments'] == sum(point['counts']) == 4*b
            assert point['gemm_flops'] == 6*5120*1536*4*b
            assert abs(statistics.median(point['cuda_event_ms']) - point['timing']['median_ms']) < 1e-10
        ratio = statistics.mean(p['timing']['median_ms'] for p in pp) / ep['timing']['median_ms']
        assert abs(float(row['pp_over_ep_latency_ratio']) - ratio) < 1e-12

    SNAPSHOT.mkdir(exist_ok=False)
    files = {p.name: p for p in (RUN / 'analysis').iterdir() if p.is_file()}
    files.update({'raw_result.json': RUN / 'profile/result.json',
                  'frozen_spec.json': RUN / 'dashboard_spec.json',
                  'weight_selection.json': RUN / 'selection.json'})
    for name, source in files.items():
        shutil.copy2(source, SNAPSHOT / name)
        assert digest(SNAPSHOT / name) == digest(source)

    with tarfile.open(SNAPSHOT / 'frozen_sources.tar.gz', 'w:gz') as tar:
        for source in spec['frozen_inputs']:
            p = Path(source['path'])
            tar.add(p, arcname='original_paths/' + str(p).lstrip('/'), recursive=False)
        for p in (ROOT / 'analyze.py', ROOT / 'save_snapshot.py', ROOT / 'README.md'):
            tar.add(p, arcname='postprocessing/' + p.name, recursive=False)
    with tarfile.open(SNAPSHOT / 'frozen_sources.tar.gz', 'r:gz') as tar:
        assert len(tar.getmembers()) == len(spec['frozen_inputs']) + 3
        for source, member in zip(spec['frozen_inputs'], tar.getmembers()):
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == source['sha256']

    validation = {'scientific_postcheck': post,
        'cleanup': {k: cleanup[k] for k in ('timestamp', 'passed', 'model_processes_exited', 'gpu_idle')},
        'independent_recompute': {'rows': 36, 'comparisons': 12, 'samples_per_row': 30,
                                  'frozen_sources_unchanged': len(spec['frozen_inputs']),
                                  'all_passed': True},
        'guard_log': str(RUN / 'guard.log'),
        'failure_preserved': str(ROOT / 'runs/r1')}
    save(SNAPSHOT / 'validation.json', validation)
    manifest = {
        'id': EXPERIMENT_ID, 'title': 'GLM4.7-AWQ local expert compute: EP8 versus PP2 x EP4',
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
        'implementation': {'engine': 'sglang', 'sources': spec['sources'],
            'dirty': False, 'old_implementation': True,
            'benchmark_harness': str(ROOT), 'backend': result['backend'],
            'scope': result['scope']},
        'deployment': {'model_server': False, 'node': 'spartan-gpgpu065',
            'slurm_job_at_launch': '33463294', 'guard_registered_gpus': 4,
            'actual_compute_gpus': 1, 'cuda_visible_devices': '0',
            'lease_id': result['authorization']['lease_id'],
            'cleanup_verified': True, 'lease_released': True},
        'model': {'path': spec['model_cache']['model_path'], 'name': 'GLM-4.7-AWQ',
            'layer': 3, 'weight_precision': 'AWQ INT4', 'activation_precision': 'BF16',
            'hidden_size': 5120, 'intermediate_size': 1536,
            'routed_experts': 160, 'top_k': 8, 'selected_real_experts': list(range(40)),
            'selected_tensor_bytes': 490291200, 'full_model_loaded': False},
        'testbed': result['gpu'],
        'workload': {'kind': 'synthetic balanced routed expert activations with real layer weights',
            'global_logical_new_token_rows': [8, 32, 128, 512, 2048, 4096],
            'physical_source_copy_convention': 4,
            'local_assignments_each_invocation': '4 * B',
            'ep8': {'resident_experts': 20, 'microbatch_rows': 'B'},
            'pp2_ep4': {'resident_experts': 40, 'microbatch_rows': 'B/2', 'slots': 2},
            'routing': 'one expert in each EP8 owner per logical token; balanced counts',
            'request_count': None, 'context_length': None, 'kv_cache': None,
            'modes': ['eager', 'cuda_graph'], 'warmups_per_shape': 5,
            'paired_alternating_blocks': 3, 'cuda_event_samples_per_block': 10,
            'independent_sessions': 1, 'random_seed': 20261021},
        'engine_parameters': {'gemm_type': 'GROUPED_MASKED', 'fixed_receive_capacity_per_expert': 1024,
            'use_f16_accum': False, 'attention_compute': False, 'router_compute': False,
            'shared_expert_compute': False, 'communication': False, 'pipeline_scheduling': False},
        'key_results': comparisons,
        'caveats': ['Single destination-rank emulation, not an eight-GPU end-to-end test.',
            'D=4 matches the saved probe convention; it is not 4B unique user tokens.',
            'Both PP slots timed; compare L*tEP with (L/2)*(tPP0+tPP1).',
            'Block coverage is source-derived, not measured occupancy, HBM traffic or CTA counts.',
            'Small gaps at larger B are within observed block drift; no stable sub-percent winner.',
            'One isolated layer can retain weights in cache differently from a complete model.'],
        'artifacts': {'original_run': str(RUN), 'snapshot': str(SNAPSHOT),
            'source_archive': 'frozen_sources.tar.gz', 'raw_samples': 'raw_result.json',
            'metrics': ['comparison.csv', 'full_metrics.csv', 'kernel_geometry.csv'],
            'plots': ['expert_compute.png', 'expert_compute.pdf'], 'validation': 'validation.json'},
    }
    save(SNAPSHOT / 'manifest.json', manifest)
    checksums = {p.name: digest(p) for p in SNAPSHOT.iterdir() if p.is_file()}
    save(SNAPSHOT / 'checksums.json', checksums)
    for name, expected in checksums.items():
        assert digest(SNAPSHOT / name) == expected
    print(json.dumps({'snapshot': str(SNAPSHOT), 'files': len(checksums), 'verified': True}))


if __name__ == '__main__':
    main()
