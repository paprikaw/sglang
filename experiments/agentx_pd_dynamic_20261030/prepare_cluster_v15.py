"""Freeze the E6 optional-checkpoint pressure repair; retain V3 replay timing."""
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
run=Path(sys.argv[sys.argv.index('--run')+1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v14.py'),run_name='__main__')
spec=importlib.util.spec_from_file_location('pressure_prep',ROOT/'prepare_cluster.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
cfg=prep.read(run/'config.json')
manifest=prep.read(run/'launch_manifest.json')
engine=Path(manifest['engine_root'])
assert manifest['engine_commit']=='fc014e3553c4d234f38f3088a3158c0333e64830'
changed=subprocess.check_output(['git','-C',str(engine),'diff','--name-only',
    '37e0828c642ce4f3795c4c4394f4ce913bcd1a35',manifest['engine_commit']],text=True).splitlines()
assert set(changed)=={
    'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py',
    'test/registered/unit/mem_cache/test_mamba_optional_checkpoint_pressure.py'}
assert '16 passed' in (ROOT/'unit_r6_pressure_green_v2.log').read_text()
assert prep.desc(run/'code/agentx_replay.py')['sha256']==prep.desc(ROOT/'agentx_replay_v3.py')['sha256']
cfg['state_pressure_fix']=dict(engine_commit=manifest['engine_commit'],
    changed_files=changed,cpu_checks=16,gpu_qualification='pending',
    native_pressure_log_marker='MAMBA_CACHE_CHECKPOINT_DEFERRED')
cfg['protocol']=cfg['protocol'].replace('previous same-E5 run','previous identical-source run')
cfg['protocol']+=(' E6 defers an optional non-INT8 unfinished recurrent checkpoint '
    'when all state slots are protected. Existing GPU KV, prefix locks, active '
    'request states and valid tracked checkpoint depth are retained. Completion '
    'may donate the retained checkpoint. HiCache, state pool sizes, transfer queue, '
    'native kernels and V3 replay timing remain configured as declared. '
    'No completed E5 episodes are imported into this E6 run; GPU qualification is pending.')
prep.save(run/'config.json',cfg)
preflight=prep.read(run/'preflight.json')
preflight['state_pressure_fix']=cfg['state_pressure_fix']
prep.save(run/'preflight.json',preflight)
ds=prep.read(run/'dashboard_spec.json')
paths=[Path(x['path']) for x in ds['frozen_inputs']]+[
    Path(__file__).resolve(),ROOT/'unit_r6_pressure_green_v2.log',
    engine/'test/registered/unit/mem_cache/test_mamba_optional_checkpoint_pressure.py']
ds['frozen_inputs']=list({str(p):prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json',ds)
manifest.update(protocol=cfg['protocol'],state_pressure_fix=cfg['state_pressure_fix'],
                frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json',manifest)
print('E6 checkpoint-pressure repair frozen; V3 replay retained; GPU qualification pending')
