import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import witness


def sample():
    logits=torch.zeros(6,8)
    for i in range(6):logits[i,i+1]=2
    candidates=torch.tensor([[7,1,2],[7,0,5]])
    seq=torch.tensor([4097,4097])
    return dict(rank=4,slot=0,verify_ordinal=0,cohort='timed',draft_token_num=3,
        requests=[dict(request_index=i,committed_history_sha256='history'+str(i)) for i in range(2)],
        pre=dict(batch=dict(seq_lens=seq,seq_lens_cpu=seq.clone(),input_ids=candidates.flatten(),
                            out_cache_loc=torch.arange(6),req_pool_indices=torch.arange(2)),
                 verify={key:candidates.flatten() if key=='draft_token' else torch.arange(6) for key in witness.VERIFY_FIELDS},
                 target_logits=logits),
        post=dict(predict=logits.argmax(-1).int(),accept_lens=torch.tensor([3,1]),
                  accept_index=torch.tensor([[0,1,2],[3,-1,-1]])))


class WitnessTests(unittest.TestCase):
    def test_cpu_reference_matches_chain_and_rejects_bad_acceptance(self):
        s=sample();self.assertTrue(witness.chain_reference(s)['passed'])
        s['post']['accept_lens'][0]=1
        self.assertFalse(witness.chain_reference(s)['passed'])
        s=sample();s['post']['accept_index'][0,0]=-1
        self.assertFalse(witness.chain_reference(s)['passed'])

    def test_same_inputs_same_outputs_and_physical_indices_are_nonsemantic(self):
        a=sample();b=copy.deepcopy(a);b['pre']['batch']['out_cache_loc']+=1024
        r=witness.compare_snapshot(a,b)
        self.assertEqual(r['classification'],'identical_logit_and_acceptance_witness')
        self.assertEqual(r['physical_allocator_fields_changed'],['out_cache_loc'])
        self.assertFalse(r['benign_numerical_branching_proven'])

    def test_history_and_metadata_mismatches_are_separate(self):
        a=sample();b=copy.deepcopy(a);b['requests'][0]['committed_history_sha256']='other'
        self.assertEqual(witness.compare_snapshot(a,b)['classification'],'different_incoming_histories')
        b=copy.deepcopy(a);b['pre']['verify']['positions'][0]=100
        r=witness.compare_snapshot(a,b)
        self.assertEqual(r['classification'],'different_verify_metadata')
        self.assertEqual(r['fields'],['verify.positions'])
        self.assertFalse(r['comparable_logits'])

    def test_same_argmax_logit_drift_is_reported_without_benign_claim(self):
        a=sample();b=copy.deepcopy(a);b['pre']['target_logits'][0,0]=0.001
        r=witness.compare_snapshot(a,b)
        self.assertEqual(r['classification'],'target_logit_drift_same_argmax')
        self.assertAlmostEqual(r['max_abs_logit_delta'],0.001)
        self.assertFalse(r['benign_numerical_branching_proven'])

    def test_logit_drift_changes_argmax_and_acceptance_reference(self):
        a=sample();b=copy.deepcopy(a)
        b['pre']['target_logits'][0,7]=2.001
        b['post']['predict'][0]=7
        b['post']['accept_lens'][0]=1;b['post']['accept_index'][0]=torch.tensor([0,-1,-1])
        r=witness.compare_snapshot(a,b)
        self.assertEqual(r['classification'],'target_logit_drift_changes_argmax')
        self.assertEqual(r['argmax_changed_rows'],[0])
        self.assertEqual(r['native_acceptance_matches_cpu_reference'],[True,True])

    def test_same_logits_bad_output_is_not_numerical_drift(self):
        a=sample();b=copy.deepcopy(a);b['post']['predict'][0]=7
        self.assertEqual(witness.compare_snapshot(a,b)['classification'],'recorded_greedy_acceptance_inconsistent')

    def test_history_matching_ignores_verify_round_indices(self):
        a=sample();b=copy.deepcopy(a);b['verify_ordinal']=9;b['cohort']='repeat'
        r=witness.compare_cohorts([a],[b])
        self.assertEqual(len(r['matched']),1)
        self.assertEqual(r['matched'][0]['right_ordinal'],9)
        self.assertIsNone(r['first_comparable_nonidentical'])
        with self.assertRaises(ValueError):witness.compare_cohorts([a],[b,b])

    def test_individual_history_match_survives_sibling_divergence(self):
        a=sample();b=copy.deepcopy(a);b['requests'][0]['committed_history_sha256']='branched'
        value=witness.compare_requests([a],[b])
        self.assertEqual(len(value['matched_requests']),1)
        self.assertEqual(value['matched_requests'][0]['request_index'],1)
        self.assertFalse(value['matched_requests'][0]['identical_entire_batch_history'])
        self.assertEqual(value['unmatched_left'][0]['request_index'],0)

    def test_owned_clone_does_not_alias_reusable_source(self):
        a=torch.arange(6);b=witness._clone(a);a.fill_(99)
        self.assertTrue(torch.equal(b,torch.arange(6)))

    def test_cpu_history_identity_preserves_emitted_tokens(self):
        req=SimpleNamespace(rid='timed-decode-4096-7',origin_input_ids=[1,2],output_ids=[3])
        a=witness.request_identity(req);req.output_ids.append(4)
        b=witness.request_identity(req)
        self.assertEqual(a['request_index'],7)
        self.assertEqual(a['committed_output_ids'],[3])
        self.assertNotEqual(a['committed_history_sha256'],b['committed_history_sha256'])
        self.assertEqual(a['origin_sha256'],b['origin_sha256'])

    def test_no_forward_readback_or_sync_and_snapshot_order(self):
        tree=ast.parse(Path(witness.__file__).read_text())
        install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
        for n in ast.walk(install):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,('cpu','numpy','tolist','item','synchronize','save'))
        samplefn=next(n for n in ast.walk(install) if isinstance(n,ast.FunctionDef) and n.name=='sample')
        assigns={n.targets[0].id:n.lineno for n in samplefn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        self.assertLess(assigns['pre'],assigns['result']);self.assertLess(assigns['result'],assigns['post'])
        idle=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flush_idle')
        self.assertIn('is_fully_idle',ast.unparse(idle))
        self.assertEqual(witness.MAX_VERIFIES_PER_COHORT_SLOT,16)


if __name__=='__main__':unittest.main()
