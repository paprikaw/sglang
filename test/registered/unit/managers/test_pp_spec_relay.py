"""CPU contracts for the upstream PP proposal relay backport."""
import torch
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput


def proposal(rids, offset=0):
    n = len(rids)
    return PPSpecRelayInput(
        rids=list(rids), tokens=torch.arange(n * 3).reshape(n, 3) + offset,
        parents=torch.tensor([[-1, 0]]).repeat(n, 1),
        top_scores=torch.tensor([[0, 1]]).repeat(n, 1),
    )


def test_degenerate_proposal_filters_by_request():
    item = PPSpecRelayInput.degenerate(['a', 'b'], torch.tensor([12, 21]), 3)
    item.filter_batch(torch.tensor([1]), [1])
    assert item.rids == ['b']
    assert item.tokens.tolist() == [[21, 0, 0]]
    assert item.parents is None


def test_merge_fresh_prefill_and_drafted_rows_keeps_topology():
    item = proposal(['a'])
    item.merge_batch(PPSpecRelayInput.degenerate(['b'], torch.tensor([21]), 3))
    assert item.rids == ['a', 'b']
    assert item.tokens.tolist() == [[0, 1, 2], [21, 0, 0]]
    assert item.parents.tolist() == [[-1, 0], [-1, 0]]
    assert item.top_scores.tolist() == [[0, 1], [0, 1]]


def test_adopt_matches_rid_after_microbatch_recomposition():
    item = proposal(['a', 'c'], 100)
    item.adopt(proposal(['b', 'a'], 200))
    assert item.tokens.tolist() == [[203, 204, 205], [103, 104, 105]]
    assert item.rids == ['a', 'c']


def test_reindex_changes_rows_not_tree_identity():
    item = proposal(['a', 'b'], 10)
    other = item.reindex(['b', 'a'])
    assert other.tokens.tolist() == [[13, 14, 15], [10, 11, 12]]
    assert item.rids == ['a', 'b']
    assert other.rids == ['b', 'a']
