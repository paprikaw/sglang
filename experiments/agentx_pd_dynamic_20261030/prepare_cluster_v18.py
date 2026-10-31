"""Freeze the E7 experiment with the engine's native 300-second wait bound.

The inherited 120-second pilot bound expired while Prefill still made progress.
This changes an explicit launch deadline, never engine code or reported latency.
"""
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v17.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('timeout_prep', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
cfg = prep.read(run/'config.json')
for node, local in cfg['node_configs'].items():
    assert local['launch_env']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] == '120'
    local['launch_env']['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '300'
cfg['waiting_timeout_s'] = 300
cfg['protocol'] += (
    ' The inherited pilot SGLANG_DISAGGREGATION_WAITING_TIMEOUT=120 is replaced '
    'by the pinned engine default 300 seconds on both roles. The clock starts '
    'when Decode sends destination metadata and includes waiting for Prefill, '
    'not only network transfer. Request arrival, token lengths, cache behavior, '
    'request timeout 600 seconds, all measured waiting/latency and existing '
    'finite failure gates remain unchanged. Failed 120-second attempts are '
    'retained separately and contribute no performance samples.')
prep.save(run/'config.json', cfg)
preflight = prep.read(run/'preflight.json')
preflight['waiting_timeout'] = dict(previous_seconds=120, seconds=300,
    source_default='python/sglang/srt/environ.py:SGLANG_DISAGGREGATION_WAITING_TIMEOUT',
    reason='120-second pilot deadline expired during ongoing Prefill progress',
    timing_discount_seconds=0)
prep.save(run/'preflight.json', preflight)
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(x['path']) for x in ds['frozen_inputs']] + [Path(__file__).resolve()]
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'], waiting_timeout_s=300,
                frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json', manifest)
print('Native 300-second waiting deadline frozen; no waiting time excluded')
