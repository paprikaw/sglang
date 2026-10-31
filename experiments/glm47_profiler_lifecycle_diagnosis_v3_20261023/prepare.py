"""CPU-only freeze; root reviews and invokes Guard separately."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from common import GUARD, MODES, PYTHON, save, sha


def prepare(run, node):
    run = Path(run).resolve()
    if run.exists():
        raise FileExistsError('Immutable run already exists')
    if not node.startswith('spartan-gpgpu') or not node.removeprefix('spartan-gpgpu').isdigit():
        raise ValueError('Exact existing booked Spartan node required')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
        capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    run.mkdir(parents=True)
    cfg = dict(id=f'{ROOT.name}-{run.name}', nodes=[node], groups={'lifecycle': {'nodes': [node]}},
        python=PYTHON, master_port=30893, worker_timeout_s=60, modes=list(MODES),
        actual_gpus=[0, 1], registered_gpus=4, payload_elements=20480,
        calibration_sessions_per_graph=0, graph_count=1, profiled_graph_replays=3, no_model=True,
        attention_compute=False, expert_compute=False,
        purpose='First-collector CPU+CUDA versus CUDA-only after manual Graph capture; real NCCL P2P, no startup calibration')
    save(run / 'config.json', cfg)
    save(run / 'cpu_tests.json', dict(passed=True, gpu_execution=False, stdout=tests.stdout, stderr=tests.stderr))
    controller = ROOT / 'runtime/controller.py'
    command = ['python3', str(controller), '--run', str(run)]
    frozen = list((ROOT / 'runtime').glob('*.py')) + list((ROOT / 'tests').glob('*.py')) + [Path(__file__), run / 'config.json', run / 'cpu_tests.json']
    spec = dict(schema_version=1, id=cfg['id'], title='NCCL P2P: first collector activity mode after Graph capture',
        engine='nccl', protocol='task-events-v2', template='parallel-tasks-v5',
        features=['parallel-tasks', 'task-progress', 'task-details', 'gpu-tab', 'gpu-telemetry', 'guard'],
        port=54123, max_active_nodes=1, gpu_lease_ttl_s=90, cleanup_wait_s=60,
        runtime_dir=str(run / 'dashboard_guard'),
        gpu_telemetry=dict(enabled=True, poll_interval_s=10, timeout_s=10),
        workload=dict(kind='synthetic_pp_boundary_communication', model_loaded=False, expert_compute=False, attention_compute=False),
        model_cache=dict(policy='communication_only', benchmark='nccl_pp_boundary',
            reason='No-model actual NCCL P2P lifecycle diagnosis; markers and validation only; no Attention or Expert workload',
            entrypoint=dict(path=str(controller), sha256=sha(controller)), command=command),
        frozen_inputs=[dict(path=str(path.resolve()), sha256=sha(path)) for path in sorted(frozen)],
        tasks=[dict(id='lifecycle', label='First CPU+CUDA versus CUDA-only collectors: real P2P Graph replay',
            root=str(run / 'task'), expected=2, nodes=[node], controller_script=str(controller))])
    sys.path.insert(0, str(GUARD.parent))
    import model_cache_adapter
    model_cache_adapter.validate_contract(spec, launch=True)
    model_cache_adapter.validate_command(spec, command)
    save(run / 'dashboard_spec.json', spec)
    print(json.dumps(dict(run=str(run), spec=str(run / 'dashboard_spec.json'), command=command,
        cpu_tests='passed', gpu_launched=False, nodes=[node], modes=list(MODES))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    prepare(args.run, args.node)
