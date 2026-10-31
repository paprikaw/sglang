"""CPU-only protocol checks; no Torch import or model launch."""
from types import SimpleNamespace as NS
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
import pytest
from pp_prefill_seed import attach,receipt,validate_receipts


def make_peer():
    events=[]
    class M:
        def init_pp_loop_state(self):pass
        def _pp_prep_batch_result(self,batch,metadata,outputs):
            batch.spec_info=NS(rids=[r.rid for r in metadata.fwd_batch.reqs],root=outputs.tensors['next_token_ids'])
            return 'unchanged-result'
        def _pp_spec_set_relay(self,batch,relay):batch.spec_info=relay;events.append(relay)
    def seed(rids,root,top1,old):
        assert old.rids==rids and old.root is root
        return NS(rids=rids,chain=[[r,t,0] for r,t in zip(root,top1)])
    attach(M,seed=seed)
    peer=M();peer._pp_spec_microbatch=peer._pp_spec_relay=True
    peer.ps=NS(pp_size=2);peer.pp_loop_size=2;peer.init_pp_loop_state()
    return peer,events,M


@pytest.mark.parametrize('final,verify,proposal',[(True,False,True),(False,False,True),(True,True,True),(True,False,False)])
def test_only_final_prefill_reuses_already_computed_proposal(final,verify,proposal):
    peer,events,_=make_peer();batch=NS(contains_last_prefill_chunk=final,reqs=[NS(rid='wrong-live-order')])
    metadata=NS(fwd_batch=NS(reqs=[NS(rid='b'),NS(rid='a')]))
    payload={'next_token_ids':[11,22]}
    if proposal:payload['draft_topk_index']=[33,44]
    if verify:payload['spec_accept_lens']=[2,2]
    before=dict(payload)
    assert peer._pp_prep_batch_result(batch,metadata,NS(tensors=payload))=='unchanged-result'
    eligible=final and not verify and proposal
    assert len(events)==int(eligible)
    if eligible:
        assert batch.spec_info.rids==['b','a']
        assert batch.spec_info.chain==[[11,33,0],[22,44,0]]
    assert payload==before
    assert receipt(peer)['requests']==(2 if eligible else 0)


def test_double_install_rejected_and_all_rank_activation_gate():
    peer,_,mixin=make_peer()
    with pytest.raises(RuntimeError,match='already installed'):attach(mixin,seed=lambda *a:None)
    row=receipt(peer);row.update(batches=2,requests=8)
    rows=[dict(rank=r,prefill_first_proposal=dict(row)) for r in range(8)]
    assert validate_receipts(rows,require_work=True)['passed']
    rows[-1]['prefill_first_proposal']['requests']=4
    with pytest.raises(RuntimeError,match='disagree'):validate_receipts(rows)
