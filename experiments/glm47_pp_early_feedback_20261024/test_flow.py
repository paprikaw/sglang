"""Coverage and phase tests require no CUDA or model imports."""
from pathlib import Path
import copy,sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from pp_flow import phase_for
from flow_validation import validate_all
class Mode:
    def __init__(self,name): self.name=name
    def is_target_verify(self): return self.name=='TARGET_VERIFY'
    def is_decode(self): return self.name=='DECODE'
@pytest.mark.parametrize('name,expected',[('EXTEND','target_prefill'),('TARGET_VERIFY','target_verify'),('DECODE','target_decode')])
def test_phase_keeps_initial_eager_prefill(name,expected):
    assert phase_for(SimpleNamespace(forward_mode=Mode(name)),False)==expected
def traces(layout):
    rows=[]
    for rank in range(8):
        count=2 if layout=='pp2ep4' else 1
        prefill=[dict(index=i,phase='target_prefill',batch_size=4,input_shape=[260],request_ids=[f'r{j+4*i}' for j in range(4)],
            start_ms=i*200,end_ms=i*200+100,elapsed_ms=100) for i in range(count)]
        verify=[dict(index=count+i,phase='target_verify',start_ms=500+i*110,end_ms=550+i*110,elapsed_ms=50) for i in range(12)]
        rows.append(dict(rank=rank,passed=True,all_model_forwards=True,per_replay_fences=0,per_forward_fences=0,
            per_forward_timestamp_reads=0,forwards=prefill+verify,replays=[],batches=[]))
    return rows
@pytest.mark.parametrize('layout',['ep8','pp2ep4'])
def test_accept_complete_initial_and_steady_coverage(layout):
    assert validate_all(traces(layout),layout)['passed']
@pytest.mark.parametrize('mutation',['missing_prefill','fence','duplicate_rank','slot_overlap'])
def test_reject_weak_evidence(mutation):
    rows=copy.deepcopy(traces('pp2ep4'))
    if mutation=='missing_prefill': rows[0]['forwards']=rows[0]['forwards'][1:]
    elif mutation=='fence': rows[0]['per_forward_fences']=1
    elif mutation=='duplicate_rank': rows[1]['rank']=0
    elif mutation=='slot_overlap': rows[0]['forwards'][1]['request_ids']=rows[0]['forwards'][0]['request_ids']
    with pytest.raises(RuntimeError): validate_all(rows,'pp2ep4')


@pytest.mark.parametrize('dp',[0,1])
def test_flow_requests_use_the_same_cpu_bank_dp_and_ordered_b4(dp):
    from admission_adapter import identity,route_tokenized,release_local,new_state
    objects=[]
    for i in reversed(range(dp*4,dp*4+4)):
        obj=SimpleNamespace(rid='flow-decode-4096-'+str(i),routed_dp_rank=None)
        assert identity(obj.rid)==('flow-decode-4096',i)
        objects.append(route_tokenized(obj))
    state=new_state()
    ready=release_local(objects,dp_rank=dp,state=state,tokenized_type=SimpleNamespace)
    assert [x.rid for x in ready]==['flow-decode-4096-'+str(i) for i in range(dp*4,dp*4+4)]
    assert all(x.routed_dp_rank==dp for x in ready)
    assert not state['pending'] and len(state['proofs'])==1


def test_outside_flow_index_still_fails_closed():
    from admission_adapter import identity
    with pytest.raises(RuntimeError): identity('flow-decode-4096-8')


def test_flow_graph_receipts_see_the_same_cohort():
    from spec_proof import cohort
    batch=SimpleNamespace(reqs=[SimpleNamespace(rid='flow-decode-4096-'+str(i)) for i in range(4)])
    assert cohort(batch)=='flow-decode-4096'
