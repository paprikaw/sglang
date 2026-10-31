"""Freeze E7 bounded HiCache restore retry and a real serial reference."""
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'
PARENT = 'fc014e3553c4d234f38f3088a3158c0333e64830'
assert '--reference' in sys.argv
source = Path(sys.argv[sys.argv.index('--reference') + 1]).resolve()
spec = importlib.util.spec_from_file_location('retry_prep', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
reference = prep.read(source)
assert str(reference.get('source_arm', '')).startswith('col_')
assert len(reference['output_ids']) == 6
assert all(len(ids) == 32 for ids in reference['output_ids'])
assert '22 passed' in (ROOT/'unit_r7_restore_pressure_green.log').read_text()
run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v14.py'), run_name='__main__')
cfg = prep.read(run/'config.json')
manifest = prep.read(run/'launch_manifest.json')
engine = Path(manifest['engine_root'])
assert manifest['engine_commit'] == COMMIT
changed = subprocess.check_output(['git', '-C', str(engine), 'diff', '--name-only',
    PARENT, COMMIT], text=True).splitlines()
assert set(changed) == {
    'python/sglang/srt/disaggregation/decode.py',
    'python/sglang/srt/disaggregation/decode_hicache_mixin.py',
    'test/registered/unit/disaggregation/test_decode_hicache_restore_pressure.py'}
assert not cfg.get('imported_episodes')
assert prep.read(run/'evidence/reference.json') == reference
assert prep.read(run/'reference.input.json') == reference
assert prep.desc(run/'code/agentx_replay.py')['sha256'] == prep.desc(ROOT/'agentx_replay_v3.py')['sha256']
cfg['restore_pressure_fix'] = dict(engine_commit=COMMIT, parent_commit=PARENT,
    changed_files=changed, cpu_checks=22, gpu_qualification='pending',
    markers=['MAMBA_CACHE_CHECKPOINT_DEFERRED', 'HICACHE_LOAD_BACK_DEFERRED',
             'HICACHE_LOAD_BACK_RETRY_SUCCEEDED'])
cfg['protocol'] = cfg['protocol'].replace('previous same-E5 run', 'previous identical-source run')
cfg['protocol'] += (' E7 retains the E6 optional non-INT8 checkpoint deferral. '
    'When the promised host prefix is still complete but native restore returns '
    'no device indices, Decode waits and retries up to the existing disaggregation '
    'waiting timeout. Missing prefix coverage and partial restoration still fail. '
    'Successful local restoration remains gated by its H2D completion event and '
    'the native cross-rank transfer gate; the current transferred GDN slot is retained. '
    'HiCache, pool sizes, native kernels, overlap, V3 replay timing and request '
    'lengths remain as declared. No E5/E6 performance samples are imported. '
    'GPU qualification of this source is pending.')
prep.save(run/'config.json', cfg)
preflight = prep.read(run/'preflight.json')
preflight.update(restore_pressure_fix=cfg['restore_pressure_fix'],
    serial_reference=dict(source=str(source), sha256=prep.desc(source)['sha256'],
        source_arm=reference['source_arm'], sequences=6, tokens_per_sequence=32))
prep.save(run/'preflight.json', preflight)
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(x['path']) for x in ds['frozen_inputs']]
paths += [Path(__file__).resolve(), source, ROOT/'unit_r7_restore_pressure_green.log']
paths += [engine/p for p in changed]
paths += [engine/'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py',
          engine/'test/registered/unit/mem_cache/test_mamba_optional_checkpoint_pressure.py']
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest.update(protocol=cfg['protocol'], restore_pressure_fix=cfg['restore_pressure_fix'],
    serial_reference=preflight['serial_reference'], frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json', manifest)
print('E7 restore-pressure candidate frozen; serial reference and 22 CPU checks verified; GPU validation pending')
