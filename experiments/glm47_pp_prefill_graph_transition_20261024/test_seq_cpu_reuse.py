"""CPU-only checks of real pinned AST seams, tensor lifetime and AB activation."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import pp_seq_cpu_reuse as reuse

ENGINE=Path(__file__).resolve().parents[2]/'.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023/python/sglang/srt'

class Tensor:
    def __init__(self,values,*,dtype='int64',device='cpu'):
        self.values=list(values);self.dtype=dtype;self.device=NS(type=device);self.shape=(len(values),);self.readbacks=0
    def clone(self):return Tensor(self.values,dtype=self.dtype,device=self.device.type)
    def to(self,device):
        assert device=='cpu';self.readbacks+=1
        return Tensor(self.values,dtype=self.dtype)


def batch(prefix='timed',values=(9,12)):
    return NS(reqs=[NS(rid=f'{prefix}-{i}') for i in range(len(values))],seq_lens_cpu=Tensor(values))

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    reuse._COUNTS.clear();reuse._COUNTS.update({k:0 for k in reuse.COUNTER_KEYS})
    monkeypatch.setitem(sys.modules,'torch',NS(equal=lambda a,b:a.values==b.values))


@pytest.mark.parametrize('control',[False,True])
def test_scheduler_exact_reuse_preserves_isolation_and_old_snapshot(control):
    b=batch('control-a' if control else 'timed');result=NS(new_seq_lens=Tensor([9,12],device='cuda'))
    reuse.capture(b,result)
    b.seq_lens_cpu.values[:]=[3,3]  # scheduler isolation restores old values
    value=reuse.scheduler_value(NS(),b,result)
    assert value.values==[9,12] and b.seq_lens_cpu.values==[3,3]
    assert result.new_seq_lens.readbacks==int(control)
    assert value is not b.seq_lens_cpu
    assert reuse._COUNTS['scheduler_control' if control else 'scheduler_reused']==1


def test_required_readback_not_removed_or_approximated():
    b=batch();result=NS(new_seq_lens=Tensor([9,12],device='cuda'))
    got=reuse.scheduler_value(NS(),b,result)
    assert got.values==[9,12] and result.new_seq_lens.readbacks==1
    assert reuse._COUNTS['scheduler_fallback']==1


@pytest.mark.parametrize('head,control',[(False,False),(False,True),(True,False),(True,True)])
def test_tail_only_retire_reuse(head,control):
    b=batch('control-a' if control else 'timed');snapshot=NS(reqs=b.reqs[:],seq_lens_cpu=Tensor([9,12]))
    new=Tensor([9,12],device='cuda')
    got=reuse.retire_value(NS(pp_group=NS(is_last_rank=not head)),b,NS(fwd_batch=snapshot),new)
    assert got.values==[9,12] and new.readbacks==int(head or control)


def test_warm_equal_proofs_and_formal_no_extra_comparisons():
    b=batch('control-warm');result=NS(new_seq_lens=Tensor([9,12],device='cuda'))
    reuse.capture(b,result);reuse.scheduler_value(NS(),b,result)
    reuse.retire_value(NS(pp_group=NS(is_last_rank=True)),b,NS(fwd_batch=b),result.new_seq_lens)
    assert reuse._COUNTS['warm_scheduler_equal']==reuse._COUNTS['warm_retire_equal']==1
    result._pp_exact_seq_cpu[1].values[0]=999
    with pytest.raises(RuntimeError,match='readback differs'):reuse.scheduler_value(NS(),b,result)


@pytest.mark.parametrize('bad',['rids','shape','dtype','device'])
def test_stale_or_wrong_cpu_snapshot_rejected(bad):
    b=batch();result=NS(new_seq_lens=Tensor([9,12],device='cuda'))
    reuse.capture(b,result)
    if bad=='rids':b.reqs.reverse()
    elif bad=='shape':result._pp_exact_seq_cpu[1].shape=(3,)
    elif bad=='dtype':result._pp_exact_seq_cpu[1].dtype='int32'
    else:result._pp_exact_seq_cpu[1].device.type='cuda'
    with pytest.raises(RuntimeError,match='ancestry|provenance'):reuse.scheduler_value(NS(),b,result)


def test_mixed_cohort_and_missing_retirement_snapshot_fail():
    b=batch();b.reqs[0].rid='control-a-0'
    with pytest.raises(RuntimeError,match='Mixed'):reuse.cohort(b)
    b=batch()
    with pytest.raises(RuntimeError,match='ancestry'):
        reuse.retire_value(NS(pp_group=NS(is_last_rank=True)),b,NS(fwd_batch=None),Tensor([9,12],device='cuda'))


@pytest.mark.parametrize('relative,cls,fn,kind',[
    ('speculative/eagle_worker_v2.py','EAGLEWorkerV2','forward_batch_generation','worker'),
    ('managers/scheduler.py','Scheduler','run_batch','scheduler'),
    ('managers/scheduler_pp_mixin.py','SchedulerPPMixin','_pp_prep_batch_result','retire')])
def test_exact_real_engine_body_one_unique_seam_and_compile(relative,cls,fn,kind):
    source=(ENGINE/relative).read_text();module,proof=reuse.transform(source,cls,fn,kind)
    assert proof['hits']==1
    compile(module,relative,'exec')
    changed=ast.unparse(module)
    if kind=='worker':
        assert "batch.seq_lens_cpu = batch_output.new_seq_lens.to('cpu')" in changed
        assert changed.index('_pp_capture_seq_cpu(')<changed.index('self.draft_worker.draft(batch, with_topology=True)')
    elif kind=='scheduler':
        assert '_pp_scheduler_seq_cpu(self, batch, batch_result)' in changed
        assert 'self.update_cache_from_scheduler(batch, batch_result)' in changed
    else:
        assert '_pp_retire_seq_cpu(self, batch, mb_metadata, new_seq_lens)' in changed
        assert 'new_seq_lens.tolist()' in changed  # merge/filter fallback unchanged
        assert "pp_outputs['spec_accept_lens'].cpu()" in changed
    # Source drift must fail rather than silently skip a purported optimization.
    with pytest.raises(RuntimeError,match='found 0'):
        reuse.transform(source.replace('.to("cpu")','.to("cuda")'),cls,fn,kind)


def test_formal_activation_gate_checks_each_rank_and_warm_validation():
    base=dict(enabled=True,pp_rank=1,**{key:0 for key in reuse.COUNTER_KEYS})
    after=dict(base,captured=50,scheduler_reused=50,retire_reused=50,warm_scheduler_equal=50,warm_retire_equal=50)
    before=dict(base,warm_scheduler_equal=50,warm_retire_equal=50)
    assert reuse.validate_delta(before,after,control=False)['scheduler_reused']==50
    with pytest.raises(RuntimeError,match='AB cohort'):reuse.validate_delta(before,after,control=True)
    head0=dict(base,pp_rank=0);head1=dict(head0,retire_head=50)
    assert reuse.validate_delta(head0,head1,control=False)['retire_head']==50
