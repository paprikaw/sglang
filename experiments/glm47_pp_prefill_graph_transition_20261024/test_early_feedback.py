"""Run the real pinned exchange against CPU queues; no GPU imports."""
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS
import pytest
import test_local_feedback as original
from pp_early_feedback import attach, receipt

@pytest.mark.parametrize('lengths',[(3,3),(1,4),(4,1),(1,0),(0,1)])
def test_real_pinned_protocol_with_early_submission(monkeypatch,lengths):
    local_attach=original.attach
    def both(mixin,**kwargs):
        local_attach(mixin,**kwargs)
        attach(mixin,can_skip=lambda batch:batch.skip)
    monkeypatch.setattr(original,'attach',both)
    original.test_real_exchange_two_slots_no_echo_and_independent_drain(lengths)

@pytest.mark.parametrize('early',[False,True])
def test_send_order_preserves_real_retirement_and_work_ownership(early):
    mixin=original.real_exchange_class()
    mixin.init_pp_loop_state=lambda self:None
    mixin._pp_recv_dict_from_prev_stage=lambda self:None
    mixin._pp_requires_immediate_output_relay=lambda self:False
    original.attach(mixin,can_skip=lambda b:b.skip,current_stream=lambda self:NS(wait_event=lambda e:None))
    if early:attach(mixin,can_skip=lambda b:b.skip)
    peer=mixin();peer.ps=NS(pp_rank=1,pp_size=2);peer.pp_loop_size=2
    peer._pp_spec_microbatch=peer._pp_spec_relay=True;peer.pp_group=NS(is_last_rank=True)
    peer.copy_stream_ctx=nullcontext();peer.copy_stream=NS(wait_stream=lambda _:None);peer.schedule_stream=None
    peer.device_module=NS(Event=lambda:NS(record=lambda _:None),current_stream=lambda:None)
    peer.init_pp_loop_state()
    a=NS(reqs=[NS(rid='a')],skip=False,forward_mode=NS(is_prebuilt=lambda:False))
    b=NS(reqs=[NS(rid='b')],skip=False,forward_mode=NS(is_prebuilt=lambda:False))
    peer.mb_metadata=[NS(fwd_batch=a),NS(fwd_batch=b)]
    peer._pp_local_feedback.put(0,'ready-a',NS(tensors={'rid':'a'}),('a',))
    queue=deque([('ready-b',NS(tensors={'rid':'b'}))]);events=[];work=object();works=[work]
    peer._pp_send_dict_to_next_stage=lambda payload,**kwargs:events.append(('send',payload['rid'])) or works
    def prep(batch,metadata,payload):
        assert batch is a and payload.tensors=={'rid':'a'}
        events.append(('retire','a'));return object()
    peer._pp_prep_batch_result=prep
    result=peer._pp_send_recv_and_preprocess_output_tensors(1,0,[a,b],peer.mb_metadata,queue,None)
    assert events==([('send','b'),('retire','a')] if early else [('retire','a'),('send','b')])
    assert result[3] is works and result[3][0] is work
    assert not queue and set(peer._pp_local_feedback.pending)=={1}
    peer._pp_local_feedback.pop(1,('b',))
    if early:assert receipt(peer)['sent']==receipt(peer)['consumed']==1


def test_requires_local_feedback_and_refuses_double_install():
    with pytest.raises(RuntimeError,match='requires qualified local-tail'):
        attach(type('Unsupported',(),{}),can_skip=lambda _:False)
    mixin=original.real_exchange_class();mixin.init_pp_loop_state=lambda self:None
    mixin._pp_recv_dict_from_prev_stage=lambda self:None
    original.attach(mixin,can_skip=lambda _:False,current_stream=lambda _:None)
    attach(mixin,can_skip=lambda _:False)
    with pytest.raises(RuntimeError,match='already installed'):attach(mixin,can_skip=lambda _:False)


def test_all_rank_receipt_gate():
    from pp_early_feedback import validate_receipts
    rows=[dict(rank=rank,early_tail_feedback=dict(enabled=True,tail=rank>=4,
        sent=2 if rank>=4 else 0,consumed=2 if rank>=4 else 0,pending=0,
        no_extra_collectives=True,no_new_gpu_fences=True)) for rank in range(8)]
    assert validate_receipts(rows,require_work=True)['passed']
    rows[-1]['early_tail_feedback']['consumed']=1
    with pytest.raises(RuntimeError,match='activation/drain'):validate_receipts(rows)
