"""Add actual host-restore correctness qualification and corrected source metadata."""
import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def main():
    host_only = '--host-only' in sys.argv
    if host_only:
        sys.argv.remove('--host-only')
    spec = importlib.util.spec_from_file_location('qualification_v2', ROOT / 'prepare_qualification_v2.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
    previous.main()
    spec = importlib.util.spec_from_file_location('qualification_v1', ROOT / 'prepare_qualification.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    cfg = old.read(run / 'config.json')
    if host_only:
        cfg['expected_points'] = 1
    cfg['protocol'] += ' Pre-performance qualification includes native GPU-prefix eviction with host copies retained, exact output check after real CPU restore, and Decode load-back evidence.'
    old.save(run / 'config.json', cfg)
    shutil.copyfile(ROOT / 'cache_probe.py', run / 'code/cache_probe.py')
    adapter = run / 'code/runtime_adapter.py'
    adapter.write_text(adapter.read_text() + '\n    import cache_probe\n    cache_probe.install()\n')
    worker = run / 'code/pd_workload.py'
    text = worker.read_text()
    marker = "                for concurrency in (1,8):"
    assert text.count(marker) == 1
    text = text.replace(marker, "                import cache_probe\n                cache_probe.qualify(base,run,cfg,arm,key,one,smoke,reference,health)\n" + marker)
    if host_only:
        text = text.replace(marker, "                base.h.save(run/'results'/arm/'host_restore'/'completed.json',dict(passed=True))\n                continue\n" + marker)
    worker.write_text(text)
    ds = old.read(run / 'dashboard_spec.json')
    ds['tasks'][0]['expected'] = cfg['expected_points']
    for source in ds['sources']:
        if source['role'] == 'inference_engine':
            source.update(worktree=cfg['engine_root'], commit=cfg['engine_commit'], tag=cfg['engine_tag'])
    files = [Path(item['path']) for item in ds['frozen_inputs']]
    files += [Path(__file__).resolve(), ROOT / 'cache_probe.py', run / 'code/cache_probe.py']
    ds['frozen_inputs'] = list({str(p): old.desc(p) for p in files}.values())
    old.save(run / 'dashboard_spec.json', ds)
    manifest = old.read(run / 'launch_manifest.json')
    manifest.update(frozen_inputs=ds['frozen_inputs'], protocol=cfg['protocol'])
    old.save(run / 'launch_manifest.json', manifest)


if __name__ == '__main__':
    main()
