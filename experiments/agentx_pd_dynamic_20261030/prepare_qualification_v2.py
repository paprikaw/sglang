"""Resume PD qualification after the independent-repeat flush race.

An already completed same-engine colocated reference is immutable input.
The workload, engine and within-episode cache policy are unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    for key in ('run', 'node', 'job', 'family', 'port', 'engine', 'tag', 'reference'):
        ap.add_argument('--' + key, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('qualification_v1', ROOT / 'prepare_qualification.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    reference = Path(args.reference).resolve()
    result = old.read(reference)
    assert result['passed'] and len(result['responses']) == 6
    sys.argv = [str(spec.origin)]
    for key in ('run', 'node', 'job', 'family', 'port', 'engine', 'tag'):
        sys.argv += ['--' + key, getattr(args, key)]
    old.main()
    run = Path(args.run).resolve()
    cfg = old.read(run / 'config.json')
    cfg['colocated_reference'] = str(reference)
    cfg['layouts'] = [row for row in cfg['layouts'] if row['id'] == 'pd']
    cfg['expected_points'] = 2
    cfg['model_cache']['usage_logs'] = {args.node: cfg['layouts'][0]['usage_logs']}
    cfg['protocol'] += ' Independent-repeat flush waits for server request retirement (timeout=30); completed same-engine colocated reference reused.'
    old.save(run / 'config.json', cfg)
    runner = run / 'code/base_runner.py'
    text = runner.read_text()
    needle = '''{cfg['port']}/flush_cache"'''
    assert text.count(needle) == 1
    runner.write_text(text.replace(needle, '''{cfg['port']}/flush_cache?timeout=30"'''))
    ds = old.read(run / 'dashboard_spec.json')
    ds['model_cache'] = cfg['model_cache']
    ds['tasks'][0]['expected'] = 2
    files = [Path(item['path']) for item in ds['frozen_inputs']]
    files += [Path(__file__).resolve(), reference]
    ds['frozen_inputs'] = list({str(p): old.desc(p) for p in files}.values())
    old.save(run / 'dashboard_spec.json', ds)
    manifest = old.read(run / 'launch_manifest.json')
    manifest.update(layouts=cfg['layouts'], protocol=cfg['protocol'],
                    frozen_inputs=ds['frozen_inputs'], colocated_reference=str(reference))
    old.save(run / 'launch_manifest.json', manifest)
    print(json.dumps({'prepared': str(run), 'reference': str(reference)}))


if __name__ == '__main__':
    main()
