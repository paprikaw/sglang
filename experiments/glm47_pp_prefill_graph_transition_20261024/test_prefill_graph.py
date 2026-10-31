"""CPU tests of exact workload boundaries, native PP seams and AB proof."""
from pathlib import Path
import ast
import sys
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import pp_prefill_graph as graph
import pp_seq_cpu_reuse as reuse


def batch(**kwargs):
    values=dict(batch_size=4,extend_seq_lens_cpu=[65]*4,extend_prefix_lens_cpu=[4032]*4,input_ids=list(range(512)))
    values.update(kwargs)
    return NS(**values)


def test_no_cohort_mixing_or_missing_ancestry():
    assert graph.cohort([NS(rid='control-a-0')])['control']
    assert not graph.cohort([NS(rid='timed-0')])['control']
    with pytest.raises(RuntimeError):graph.cohort([])
    with pytest.raises(RuntimeError):graph.cohort([NS(rid='control-a-0'),NS(rid='timed-1')])


@pytest.mark.parametrize('change',[dict(batch_size=8),dict(extend_seq_lens_cpu=[64]*4),dict(extend_prefix_lens_cpu=[4096]*4),dict(input_ids=[1]*1024)])
def test_rejects_workload_change(change):
    assert graph.eligible(batch()) and graph.eligible(batch(input_ids=[1]*260))
    assert not graph.eligible(batch(**change))


@pytest.mark.parametrize('control,stage',[(a,b) for a in (False,True) for b in (0,1)])
def test_actual_replay_and_pp_ownership_receipt(control,stage):
    before=dict(enabled=True,pp_rank=stage,counters={k:0 for k in graph.KEYS})
    counts=dict(before['counters'])
    if control:counts['control_eager']=2
    else:
        for k in ('candidate_eligible','graph_executions','actual_body_replays'):counts[k]=2
        counts['head_proxy_outputs' if stage==0 else 'tail_proxy_copies']=2
    after=dict(before,counters=counts)
    assert graph.validate_delta(before,after,control=control)==counts
    counts['actual_body_replays']+=1
    with pytest.raises(RuntimeError):graph.validate_delta(before,after,control=control)


def test_seq_reuse_can_be_fixed_in_both_cohorts(monkeypatch):
    monkeypatch.setattr(reuse,'_ALWAYS',True)
    ids,control,warm=reuse.cohort(NS(reqs=[NS(rid='control-warm-0')]))
    assert not control and warm


def test_real_pp_capture_input_and_output_seams_are_present():
    root=Path(__file__).resolve().parents[2]/'.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023/python/sglang/srt'
    src=(root/'model_executor/runner/prefill_cuda_graph_runner.py').read_text()
    tree=ast.parse(src);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrefillCudaGraphRunner')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    assert {'_run_forward','_finalize_execute_output','can_run_graph','execute'}<=set(methods)
    assert 'PPProxyTensors is not supported' in ast.unparse(methods['_finalize_execute_output'])
    assert 'pp_proxy_tensors' not in ast.unparse(methods['_run_forward'])
    assert 'PPProxyTensors' in (root/'model_executor/runner_backend/breakable_cuda_graph_backend.py').read_text()
