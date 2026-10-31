import pytest
from cluster_runner_v6 import forward_delta


def sample(counts):
    return dict(shape={'pp_rank':0,'tp_rank':0},counts=counts)


def test_delta_excludes_preceding_warmup_counts():
    before={'rank0':sample({'DECODE:b1':500,'EXTEND:b1':50})}
    after={'rank0':sample({'DECODE:b1':700,'EXTEND:b1':53,'DECODE:b4':17})}
    assert forward_delta(before,after)['rank0']['counts']=={'DECODE:b1':200,'EXTEND:b1':3,'DECODE:b4':17}


def test_missing_rank_or_process_counter_reset_is_rejected():
    before={'rank0':sample({'DECODE:b1':500})}
    with pytest.raises(AssertionError):
        forward_delta(before,{})
    with pytest.raises(AssertionError,match='Counter reset'):
        forward_delta(before,{'rank0':sample({'DECODE:b1':499})})
