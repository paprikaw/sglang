"""Resolve only the reviewed 30775 backport conflicts; no live source changes."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / '.engine-worktrees/glm47-pp-spec-20261023'
PATTERN = re.compile(r'^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n', re.M | re.S)

def resolve(name, choices):
    path = ROOT / name
    text = path.read_text()
    matches = list(PATTERN.finditer(text))
    if len(matches) != len(choices):
        raise RuntimeError((name, len(matches), len(choices)))
    for match, choose in reversed(list(zip(matches, choices))):
        left, right = match.group(1, 2)
        replacement = choose(left, right)
        text = text[:match.start()] + replacement + text[match.end():]
    path.write_text(text)

left = lambda a, b: a
right = lambda a, b: b
resolve('python/sglang/srt/environ.py', [lambda a,b: '    # Opt-in upstream PP EAGLE/MTP relay; non-overlap, no DP Attention.\n    SGLANG_ENABLE_PP_SPEC = EnvBool(False)\n'])
resolve('python/sglang/srt/managers/scheduler.py', [right])

def early(a, b):
    return b.replace('                exchange_outputs_before_forward = (',
        '                early_proxy = self._pp_requires_immediate_output_relay()\n                exchange_outputs_before_forward = not early_proxy and (')
def late(a, b):
    return a[:a.index('                if get_parallel().pp_async_batch_depth == 0 or early_proxy:')] + b
def payload(a,b):
    return b[:b.index('        has_sampling_mask_output = (')] + a
resolve('python/sglang/srt/managers/scheduler_pp_mixin.py', [
    lambda a,b: b+'\n'+a, early, late, payload, right,
])
resolve('python/sglang/srt/model_executor/model_runner.py', [right])
resolve('python/sglang/srt/model_executor/model_runner_components/layer_setup.py', [
    lambda a,b: 'from sglang.srt.environ import envs\n\n'])
resolve('python/sglang/srt/speculative/eagle_worker_v2.py', [
    right, lambda a,b: b[:b.index('def _qsa_index_share_requested')],
])
resolve('python/sglang/srt/speculative/spec_info.py', [
    lambda a,b: b[b.index('    # Carried between rounds'):],
])
resolve('test/registered/unit/managers/test_generation_auxiliary_output.py', [left])
