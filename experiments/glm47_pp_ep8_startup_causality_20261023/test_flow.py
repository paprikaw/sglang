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
