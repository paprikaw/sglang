"""Extend the finite episode bound and preserve failure-only request diagnostics.

The E7 engine, native 300-second handoff bound, arrivals and timing stay fixed.
This version must be used for every arm in the new replay-bound cohort.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--episode-timeout-s', type=int, default=7200)
opts, rest = parser.parse_known_args(sys.argv[1:])
assert 2400 < opts.episode_timeout_s <= 14400
sys.argv = [sys.argv[0]] + rest
run = Path(sys.argv[sys.argv.index('--run') + 1]).resolve()
runpy.run_path(str(ROOT/'prepare_cluster_v18.py'), run_name='__main__')
spec = importlib.util.spec_from_file_location('episode_prep', ROOT/'prepare_cluster.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


path = run/'code/agentx_replay.py'
text = path.read_text()
text = change(text, 'namespace, idle_gap_cap_s=10, copies=1):',
    'namespace, idle_gap_cap_s=10, copies=1, episode_timeout_s=7200, partial_rows=None):')
text = change(text, '    completed_rows = []',
    '    completed_rows = partial_rows if partial_rows is not None else []\n'
    '    assert not completed_rows and episode_timeout_s > 0')
text = change(text, '    deadline = start + 2400', '    deadline = start + episode_timeout_s')
text = change(text, "raise TimeoutError('Finite AgentX episode exceeded 2400 seconds')",
    "raise TimeoutError(f'Finite AgentX episode exceeded {episode_timeout_s} seconds')")
text = change(text, '                idle_compressed_s=skipped_idle, idle_gap_cap_s=idle_gap_cap_s, slo=slo,',
    '                idle_compressed_s=skipped_idle, idle_gap_cap_s=idle_gap_cap_s, slo=slo,\n'
    '                episode_timeout_s=episode_timeout_s,')
ast.parse(text)
path.write_text(text)

path = run/'code/cluster_runner.py'
text = path.read_text()
text = change(text, 'import agentx_replay as replay',
    'import agentx_replay as replay\nfrom episode_capture import replay_with_failure_capture')
text = change(text, '                result = replay.replay(graphs, endpoints, policy,',
    "                result = replay_with_failure_capture(replay.replay, base.h.save, out/f'partial.r{repeat+1}.json',\n"
    '                    graphs, endpoints, policy,')
text = change(text, "idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1))",
    "idle_gap_cap_s=root['idle_gap_cap_s'], copies=case.get('copies', 1), episode_timeout_s=root['episode_timeout_s'])")
ast.parse(text)
path.write_text(text)
shutil.copyfile(ROOT/'episode_capture.py', run/'code/episode_capture.py')

cfg = prep.read(run/'config.json')
cfg['episode_timeout_s'] = opts.episode_timeout_s
cfg['partial_episode_policy'] = 'Save collected responses on exception only; never count partial files as complete samples'
cfg['protocol'] += (
    f' The inherited 2400-second whole-episode limit is extended to {opts.episode_timeout_s} seconds. '
    'This is separate from the unchanged 300-second handoff bound and 600-second HTTP timeout. '
    'No arrivals, source lengths, cache policy, scheduling or measured waiting time are changed. '
    'On failure only, collected completed responses are saved to a partial diagnostic; '
    'successful episodes perform no additional timed file write. Partial files never enter performance means.')
prep.save(run/'config.json', cfg)
preflight = prep.read(run/'preflight.json')
preflight['episode_timeout'] = dict(previous_seconds=2400, seconds=opts.episode_timeout_s,
    reason='The previous episode hit its global bound while tail requests were still progressing',
    timing_discount_seconds=0, failure_only_diagnostics=True)
prep.save(run/'preflight.json', preflight)
ds = prep.read(run/'dashboard_spec.json')
paths = [Path(x['path']) for x in ds['frozen_inputs']]
paths += [Path(__file__).resolve(), ROOT/'episode_capture.py', run/'code/episode_capture.py']
ds['frozen_inputs'] = list({str(p): prep.desc(p) for p in paths}.values())
prep.save(run/'dashboard_spec.json', ds)
manifest = prep.read(run/'launch_manifest.json')
manifest.update(protocol=cfg['protocol'], episode_timeout_s=opts.episode_timeout_s,
    partial_episode_policy=cfg['partial_episode_policy'], frozen_inputs=ds['frozen_inputs'])
prep.save(run/'launch_manifest.json', manifest)
print('Finite episode bound and failure-only diagnostics frozen:', opts.episode_timeout_s)
