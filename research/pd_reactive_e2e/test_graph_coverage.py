"""Exercise the real launch adapter without starting serving processes."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize('microbatch_cap', [1, 4])
@pytest.mark.parametrize('mechanism', ['inplace', 'replica'])
@pytest.mark.parametrize('capture_sizes', [[1, 2, 4], [4]])
def test_pd_merged_batch_frontier_has_graph_coverage(monkeypatch, tmp_path,
                                                   microbatch_cap, mechanism, capture_sizes):
    import sglang.srt.server_args as server_args

    source = Path(__file__).with_name('run.py')
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'engine')
    results = []
    monkeypatch.setattr(server_args, 'prepare_server_args', lambda argv: list(argv))
    checks = ModuleType('launch_checks')
    checks.validate_prefill_alignment = lambda args: None
    monkeypatch.setitem(sys.modules, 'launch_checks', checks)
    argv = ['--max-running-requests', '1', '--pp-max-micro-batch-size', '1',
            '--cuda-graph-bs-decode', '1']
    def entry(run, cfg):
        results.append(server_args.prepare_server_args(argv))
    namespace = dict(q=SimpleNamespace(engine_entry=entry),
                     save=lambda *args: None)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'),
         namespace)
    root = dict(servers={'d0': dict(optimized_elastic_pp=True, hicache_size_gib=64)},
                concurrent_kv=True, mechanism=mechanism, kvcached_capacity=False,
                context_length=65536, max_total_tokens=98304,
                max_running_requests=4, pp_micro_batch_size=microbatch_cap,
                decode_graph_batch_sizes=capture_sizes,
                engine_commit='test')
    namespace['engine'](tmp_path, root, 'd0')
    flags = results[0]
    i = flags.index('--cuda-graph-bs-decode') + 1
    captures = []
    while i < len(flags) and not flags[i].startswith('--'):
        captures.append(int(flags[i])); i += 1
    # Keep capture coverage valid for either declared PP microbatch cap.
    # Default graph padding must support every actual batch through the limit.
    actual_limit = int(flags[flags.index('--max-running-requests') + 1])
    assert captures == capture_sizes
    assert all(any(c >= batch for c in captures)
               for batch in range(1, actual_limit + 1))
    assert actual_limit == 4
