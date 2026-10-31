"""Run the pinned producer/Verify consumer branches on CPU test doubles."""
import ast,copy,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from pp_feedback_payload import attach,REMOVED,receipt
SOURCE=ROOT.parents[1]/'.engine-worktrees/glm47-pp-spec-microbatch-invariant-20261023/python/sglang/srt/managers/scheduler_pp_mixin.py'

class Tensor:
    def __init__(self,values):self.values=values
    def contiguous(self):return self
    def to(self,*args,**kwargs):return self
    def cpu(self):return self
    def sum(self):return sum(self.values)

class Payload:
    def __init__(self,d):self.tensors=d
    def __getitem__(self,k):return self.tensors[k]

class RemoveImports(ast.NodeTransformer):
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None

def method(name,env):
    tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SchedulerPPMixin')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name))
    fn=RemoveImports().visit(fn);fn.decorator_list=[]
    fn.returns=None
    for a in fn.args.args:a.annotation=None
    mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));ns=dict(env)
    exec(compile(mod,str(SOURCE),'exec'),ns);return ns[name]

def build_peer():
    env=dict(get_spec=lambda:NS(speculative_eagle_topk=1,speculative_num_draft_tokens=3),
        add_auxiliary_output_to_pp_tensors=lambda d,a:None,
        pop_auxiliary_output_from_pp_tensors=lambda d,o:None,
        GenerationBatchResult=lambda **kwargs:NS(**kwargs))
    class M:
        init_pp_loop_state=lambda self:None
        _pp_prepare_tensor_dict=method('_pp_prepare_tensor_dict',env)
        _pp_prep_batch_result=method('_pp_prep_batch_result',env)
    attach(M,ab=True);p=M();p.init_pp_loop_state();p._pp_spec_relay=True
    p.pp_group=NS(is_first_rank=True);p.tp_worker=NS(model_runner=NS(sampling_observer=None))
    p._pp_spec_compact_accept_kv=lambda *a:None
    p._pp_spec_adopt_relayed_tree=lambda b,r,o:None
    return p

def data(cohort='timed',verify=True):
    result=NS(next_token_ids=Tensor([11,22]),accept_lens=Tensor([1,2]) if verify else None,
        new_seq_lens=Tensor([4098,4099]),accept_index=None,next_verify_chain=Tensor([11,12,13,22,23,24]),
        next_verify_parent_list=Tensor([-1,0]),next_verify_top_scores_index=Tensor([0,1]),logits_output=None,
        next_draft_input=NS(bonus_tokens=Tensor([11,22]),topk_p=Tensor([.7,.8]),topk_index=Tensor([12,23]),hidden_states=Tensor([.1,.2])))
    batch=NS(reqs=[NS(rid=cohort+'-0'),NS(rid=cohort+'-1')],return_logprob=False,seq_lens_cpu=Tensor([4097,4097]))
    meta=NS(fwd_batch=batch,verify_out_cache_loc=None,can_run_cuda_graph=True)
    return result,batch,meta


def test_exactly_three_unused_fields_removed_and_real_verify_consumer_is_unchanged():
    peer=build_peer();original,result_batch,meta=data('control-a')
    full=peer._pp_prepare_tensor_dict(original,result_batch)
    expected=peer._pp_prep_batch_result(result_batch,meta,Payload(full))
    candidate,batch,meta2=data('timed')
    # Reuse precisely the same result tensors: this tests transport, not math.
    trimmed=peer._pp_prepare_tensor_dict(original,batch)
    assert set(full)-set(trimmed)==set(REMOVED)
    assert all(trimmed[k] is full[k] for k in trimmed)
    actual=peer._pp_prep_batch_result(batch,meta2,Payload(trimmed))
    assert vars(actual).keys()==vars(expected).keys()
    assert all(getattr(actual,k) is getattr(expected,k) or getattr(actual,k)==getattr(expected,k) for k in vars(actual))
    assert original.next_draft_input.hidden_states is full['draft_hidden_states']
    assert receipt(peer)['trimmed_verifies']==1 and receipt(peer)['removed_tensors']==3

@pytest.mark.parametrize('cohort,verify,removed', [('control-a',True,False),('control-b',True,False),('timed',True,True),('repeat',True,True),('flow',True,True),('timed',False,False)])
def test_only_candidate_verify_payload_changes(cohort,verify,removed):
    peer=build_peer();result,batch,_=data(cohort,verify)
    out=peer._pp_prepare_tensor_dict(result,batch)
    assert all((k not in out)==removed for k in REMOVED)


def test_mixed_control_cohort_fails_instead_of_silently_changing_wire():
    peer=build_peer();result,batch,_=data()
    batch.reqs[0].rid='control-a-0'
    with pytest.raises(RuntimeError,match='Mixed AB'):peer._pp_prepare_tensor_dict(result,batch)


def test_lossless_layout_alignment_and_bounds():
    from feedback_codec import layout,EXPECTED
    for bs in (0,1,3,4):
        specs={k:('torch.int32' if k in ('spec_accept_lens','spec_new_seq_lens') else 'torch.int64',[bs,3 if k=='spec_next_chain' else 1]) for k in EXPECTED}
        records,total=layout(specs)
        assert len(records)==7 and total%4==0
        for key,dtype,shape,offset,nbytes in records:
            assert offset%(8 if dtype=='torch.int64' else 4)==0
        wrong=dict(specs);wrong['spec_accept_lens']=('torch.float32',[bs])
        with pytest.raises(RuntimeError,match='dtype'):layout(wrong)
    with pytest.raises(RuntimeError,match='fields'):layout({})
