"""Freeze one additional native EP2 CUDA Graph diagnostic, without launching."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def save(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')
    p.chmod(0o600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    subprocess.run([sys.executable, str(ROOT/'prepare.py'), '--run', str(run)], check=True)
    cfg = read(run/'config.json')
    cfg.update(arm_order=['native'], expected_points=1, profile_enabled=False, port=24551, dist_port=24552,
               workloads=[c for c in cfg['workloads'] if c['requests']==32],
               protocol=cfg['protocol'] + ' Follow-up: native NCCL only, Decode CUDA Graph bs16 ON; no profile.')
    cfg['layouts'] = [r for r in cfg['layouts'] if r['id']=='native']
    cfg['model_cache']['usage_logs'][cfg['node']] = [cfg['layouts'][0]['usage_logs'][0]]
    save(run/'config.json', cfg)
    p = run/'code/diagnostic.py'
    text = p.read_text().replace("        return result\n", "        result.remove('--disable-cuda-graph')\n"
                               "        result += ['--cuda-graph-bs-decode', '16']\n        return result\n", 1)
    text = text.replace("if case['requests'] == 32:", "if case['requests'] == 32 and cfg.get('profile_enabled', True):")
    p.write_text(text)
    p.chmod(0o600)
    target = run/'code/prepare_graph.py'
    target.write_text(Path(__file__).read_text())
    target.chmod(0o600)
    spec = read(run/'dashboard_spec.json')
    spec.update(title='A100 EP2 native NCCL: one B32 Decode CUDA Graph follow-up',model_cache=cfg['model_cache'],port=24550)
    spec['tasks'][0].update(label='Native EP2 B32 GPU-hot Decode, CUDA Graph ON', expected=1)
    pinned = {d['path']:d for d in spec['frozen_inputs']}
    pinned[str(target)] = dict(path=str(target))
    for path,d in pinned.items():
        d['sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    spec['frozen_inputs'] = list(pinned.values())
    save(run/'dashboard_spec.json', spec)
    m = read(run/'launch_manifest.json')
    m.update(protocol=cfg['protocol'],hypothesis='Native NCCL EP2 permits CUDA Graph, reducing eager host/launch gaps',
             comparison='Follow-up to matched native-eager measurement; same batch/prompts/cache; Graph can change BF16 trajectories',
             reference_eager_run=str(ROOT/'runs/ep2_backend_aba_r1'),
             private_adapters=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                               for p in (run/'code').glob('*.py')])
    save(run/'launch_manifest.json', m)
    save(run/'task/progress.json', dict(experiment_id=run.name,task_id='comparison',stage='queued',valid_points=0,expected=1))
    print(json.dumps(dict(run=str(run),points=1,decode_graph_bs=[16],profile=False)))


if __name__ == '__main__':
    main()
