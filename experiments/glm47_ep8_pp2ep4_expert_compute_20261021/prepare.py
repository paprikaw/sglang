"""Freeze one real expert layer and the one-node compute benchmark on CPU."""
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

from common import ROOT, SG, SG_COMMIT, SG_TAG, MODEL, PYTHON, save, sha

RUN = ROOT / 'runs/r1'
NODE = 'spartan-gpgpu065'
GUARD = ROOT / 'guard'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def selection():
    names = [f'model.layers.3.mlp.experts.{e}.{p}.{f}' for e in range(40)
             for p in ('gate_proj', 'up_proj', 'down_proj') for f in ('qweight', 'qzeros', 'scales')]
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    result = {'model_path': str(MODEL), 'layer': 3, 'expert_ids': list(range(40)),
              'config_sha256': sha(MODEL / 'config.json'),
              'index_sha256': sha(MODEL / 'model.safetensors.index.json'),
              'tensors': {}, 'source_files': {}}
    for filename in sorted({index[n] for n in names}):
        path = MODEL / filename
        before = path.stat()
        with path.open('rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes)
            base = 8 + header_len
            result['source_files'][filename] = {'size': before.st_size,
                    'mtime_ns': before.st_mtime_ns, 'header_sha256': hashlib.sha256(header_bytes).hexdigest()}
            for name in sorted(n for n in names if index[n] == filename):
                item = header[name]
                lo, hi = item['data_offsets']; left = hi - lo
                f.seek(base + lo); digest = hashlib.sha256()
                while left:
                    chunk = f.read(min(left, 4 * 1024**2))
                    if not chunk: raise RuntimeError('Truncated real weight tensor: ' + name)
                    digest.update(chunk); left -= len(chunk)
                result['tensors'][name] = {'file': filename, 'dtype': item['dtype'],
                    'shape': item['shape'], 'bytes': hi - lo, 'sha256': digest.hexdigest()}
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise RuntimeError('Source checkpoint changed during selection')
    result['total_tensor_bytes'] = sum(x['bytes'] for x in result['tensors'].values())
    if result['total_tensor_bytes'] != 490291200 or len(result['tensors']) != 360:
        raise RuntimeError('Unexpected GLM layer selection')
    return result


def main():
    if RUN.exists(): raise RuntimeError('Immutable run already exists')
    RUN.mkdir(parents=True)
    (RUN / 'profile/task').mkdir(parents=True)
    save(RUN / 'selection.json', selection())
    sources = [{'worktree': str(SG), 'commit': SG_COMMIT, 'tag': SG_TAG},
               {'worktree': str(GUARD), 'commit': git(GUARD, 'rev-parse', 'HEAD'),
                'tag': 'experiment/glm47-expert-layer-contract-20261021-r2'}]
    for source in sources:
        root = source['worktree']
        if (git(root, 'rev-parse', 'HEAD') != source['commit']
                or git(root, 'rev-parse', source['tag'] + '^{commit}') != source['commit']
                or git(root, 'cat-file', '-t', source['tag']) != 'tag'
                or git(root, 'status', '--porcelain', '--untracked-files=no')):
            raise RuntimeError('Pinned source identity is not clean and immutable')
    legacy = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/harness')
    paths = list(ROOT.glob('*.py')) + [RUN / 'selection.json', MODEL / 'config.json',
        MODEL / 'model.safetensors.index.json'] + [legacy / x for x in ('profile_entry.py', 'profile_utils.py', 'common.py')]
    paths += [p for p in (Path(PYTHON).parent.parent / 'lib/python3.12/site-packages/humming').rglob('*.py')]
    paths += [SG / 'python/sglang/srt/layers/moe/moe_runner/humming.py',
              SG / 'python/sglang/srt/layers/quantization/humming.py',
              SG / 'python/sglang/kernels/ops/moe/humming_masked_activation.py']
    frozen = [{'path': str(p), 'sha256': sha(p)} for p in sorted(set(paths))]
    def desc(path): return next(x for x in frozen if x['path'] == str(path))
    contract = {'policy': 'expert_layer_only', 'reason': 'User-approved one-layer equal-work compute benchmark; no full model or CPU/KV bank',
        'benchmark': 'glm47_awq_local_experts', 'model_path': str(MODEL), 'layer': 3,
        'expert_ids': list(range(40)), 'selection': desc(RUN / 'selection.json'),
        'entrypoint': desc(ROOT / 'controller.py'), 'worker': desc(ROOT / 'worker.py'),
        'command': ['python3', str(ROOT / 'controller.py'), '--run', str(RUN)],
        'worker_command': [PYTHON, str(ROOT / 'worker.py'), '--run', str(RUN)]}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    spec = {'schema_version': 1, 'id': 'glm47_ep8_pp2ep4_expert_compute_20261021_r1',
        'title': 'GLM expert compute: EP8 vs PP2xEP4 equal local FLOPs',
        'engine': 'sglang', 'protocol': 'task-events-v2', 'template': 'parallel-tasks-v5',
        'features': ['parallel-tasks', 'task-progress', 'task-details', 'gpu-tab', 'gpu-telemetry', 'guard'],
        'workload': {'kind': 'glm47_awq_expert_layer_compute', 'model_loaded': False,
                     'expert_compute': True, 'attention_compute': False, 'communication': False, 'gpu_count': 1},
        'gpu_telemetry': {'enabled': True, 'poll_interval_s': 10, 'timeout_s': 10},
        'port': port, 'max_active_nodes': 1, 'gpu_lease_ttl_s': 90, 'cleanup_wait_s': 60,
        'runtime_dir': str(RUN / 'dashboard_guard'),
        'gpu_registry_dir': '/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry',
        'sources': sources, 'model_cache': contract, 'frozen_inputs': frozen,
        'tasks': [{'id': 'profile', 'label': 'One-layer expert compute, 18 shapes x 2 timing modes',
                   'root': str(RUN / 'profile/task'), 'expected': 36, 'nodes': [NODE],
                   'controller_script': str(ROOT / 'controller.py')}]}
    config = {'id': spec['id'], 'nodes': [NODE], 'groups': {'profile': {'nodes': [NODE]}},
              'cases': {'profile': {'nodes': [NODE], 'ep_capacity': 1024}},
              'model_cache': contract, 'expected_points': 36}
    save(RUN / 'config.json', config)
    # Bind config as well: it does not recursively contain the frozen inputs.
    spec['frozen_inputs'].append({'path': str(RUN / 'config.json'), 'sha256': sha(RUN / 'config.json')})
    save(RUN / 'dashboard_spec.json', spec)
    save(RUN / 'profile/task/progress.json', {'timestamp': time.time(), 'stage': 'waiting',
             'valid_points': 0, 'expected_points': 36, 'note': 'Prepared; GPU run has not started'})
    print(json.dumps({'run': str(RUN), 'port': port, 'selected_weight_MiB': 490291200 / 2**20,
                      'frozen_inputs': len(frozen), 'sources': sources}, indent=2))


if __name__ == '__main__': main()
