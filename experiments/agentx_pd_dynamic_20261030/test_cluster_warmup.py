from cluster_runner_v5 import warmup_pairs


def ep(name, role, size):
    return dict(id=name,role=role,ep=size)


def test_colocated_warmup_covers_every_rank_of_both_ep4_instances():
    pairs=warmup_pairs([ep('a','colocated',4),ep('b','colocated',4)])
    assert {(p['id'],p['rank']) for p,d in pairs} == {(name,rank) for name in ('a','b') for rank in range(4)}
    assert len(pairs)==8 and all(d is None for p,d in pairs)


def test_two_six_split_warms_every_decode_rank_and_both_prefill_ranks():
    pairs=warmup_pairs([ep('p','prefill',2)]+[ep('d'+str(i),'decode',2) for i in range(3)])
    assert len(pairs)==6
    assert {(d['id'],d['rank']) for p,d in pairs}=={('d'+str(i),rank) for i in range(3) for rank in range(2)}
    assert [sum(p['rank']==rank for p,d in pairs) for rank in (0,1)]==[3,3]


def test_pp_stages_share_one_attention_rank_per_instance():
    pairs=warmup_pairs([ep('p','prefill',1),ep('d0','decode',1),ep('d1','decode',1),ep('d2','decode',1)])
    assert len(pairs)==3
    assert {d['id'] for p,d in pairs}=={'d0','d1','d2'}
    assert all(p['rank']==d['rank']==0 for p,d in pairs)
