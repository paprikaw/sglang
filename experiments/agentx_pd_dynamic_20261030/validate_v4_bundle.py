"""Exercise corrected branch timing and failure capture in a frozen run bundle."""
import argparse
import ast
from enum import Enum
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parent


def branch_suite_module():
    """Use unmodified native enum ASTs without package/plugin initialization."""
    harness = ROOT.parents[1]/'datasets/agentx_20261030/harness/SemiAnalysisAI-agentx-harness-e4bb2f8'
    enum_root = harness/'src/aiperf/common/enums'
    module = ModuleType('frozen_branch_timing_suite')
    module.__file__ = str(ROOT/'test_branch_timing.py')
    module.__dict__['Enum'] = Enum
    module.enum_sources = []
    for filename, names in [
        ('base_enums.py', ['_normalize_name', 'CaseInsensitiveStrEnum']),
        ('enums.py', ['ConversationBranchMode', 'CreditPhase'])]:
        path = enum_root/filename
        nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', None) in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
        module.enum_sources.append(path)
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'aiperf.common.enums']
    assert len(imports) == 1 and [x.name for x in imports[0].names] == ['ConversationBranchMode', 'CreditPhase']
    tree.body.remove(imports[0])
    exec(compile(tree, module.__file__, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run/'code'))
    branch = branch_suite_module()
    import test_episode_capture as capture
    import test_agentx_replay as legacy
    spec = importlib.util.spec_from_file_location('validated_v4_replay', run/'code/agentx_replay.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    assert (ROOT/'agentx_branch_timing.py').read_bytes() == (run/'code/agentx_branch_timing.py').read_bytes()
    assert (ROOT/'episode_capture.py').read_bytes() == (run/'code/episode_capture.py').read_bytes()
    branch.replay = capture.replay = legacy.replay = replay
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(branch.BranchTimingTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(capture.EpisodeCaptureTests)])
    # The old idle-compression fixture assumes V3's artificial post-branch gap.
    # Its branch/once/join invariants are covered by the native-rule V4 tests.
    for name in ['test_rejects_cross_stream_cycle',
                 'test_block_prefix_is_reused_but_session_clones_do_not_alias',
                 'test_replica_and_attention_dp_assignment_use_all_ranks']:
        suite.addTest(legacy.ReplayTests(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    proof = dict(run=str(run), passed=result.wasSuccessful(), checks=result.testsRun,
        replayer_sha256=sha(run/'code/agentx_replay.py'),
        branch_helper_sha256=sha(run/'code/agentx_branch_timing.py'),
        failure_capture_sha256=sha(run/'code/episode_capture.py'),
        native_branch_source_sha256=sha(branch.native_path),
        native_enum_sources=[dict(path=str(p), sha256=sha(p)) for p in branch.enum_sources],
        scope='Three pinned native method ASTs, two helpers and unmodified enum/base ASTs; 24 exported graphs; real-thread overlap/join; failure capture; not full official harness equivalence')
    (run/'v4_cpu_validation.json').write_text(json.dumps(proof, indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
