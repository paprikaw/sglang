"""CPU checks of role log separation and strict actual-MTP receipt validation."""
import copy
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import io
import json
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).parent / 'runtime'))
import spec_proof
import load_logging
import workload
import ast
import measure
import common
import finalize_pair
import deterministic_proof

def deterministic_sample():
    return dict(source='initialized_scheduler_rank',resolved_enabled=True,declared_enabled=True,
        resolved_attention_backend='triton',actual_target_backend='TritonAttnBackend',
        actual_target_backend_module='sglang.srt.layers.attention.triton_backend',actual_target_deterministic=True,
        nccl_algo='allreduce:tree',nccl_min_nchannels='8',nccl_max_nchannels='8',
        actual_truncation_align_size=1024,actual_chunked_prefill_size=1024,
        effective_max_prefill_tokens=1024,configured_truncation_alignment='1024',
        disable_custom_all_reduce=True,enable_torch_symm_mem=False)

class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.mock_prompts=[[1000+i]*64+list(range(64,4097)) for i in range(8)]
        self.patches=[patch.object(workload,'manifest',return_value=dict(raw_prefix_tokens=4096)),
                      patch.object(workload,'prompts',return_value=self.mock_prompts)]
        for item in self.patches:item.start();self.addCleanup(item.stop)

    def test_loader_filter_requires_actual_role(self):
        record = logging.LogRecord('loader', logging.INFO, '', 1, 'bank proof', (), None)
        target, draft = load_logging.RoleFilter('target'), load_logging.RoleFilter('draft')
        self.assertFalse(target.filter(record))
        token = load_logging._ROLE.set('draft')
        try:
            self.assertFalse(target.filter(record)); self.assertTrue(draft.filter(record))
        finally: load_logging._ROLE.reset(token)
        self.assertFalse(draft.filter(record))

    def receipts(self, layout):
        rows = []
        for rank in range(8):
            has_draft = layout == 'ep8' or rank >= 4
            row = dict(forwards={'target:TARGET_VERIFY': 42}, replays={'target': 42},
                       phases={'draft': 42} if has_draft else {}, batch_sizes={'4': 42}, widths={'target:12': 42})
            rows.append(dict(rank=rank, graph=dict(installed=True, cohorts={'timed-decode-4096': row})))
        return rows

    def test_proof_requires_real_replay_and_correct_draft_placement(self):
        for layout in ('pp2ep4',):
            receipts = self.receipts(layout)
            self.assertTrue(spec_proof.validate_receipts(receipts, layout)['passed'])
            broken = copy.deepcopy(receipts); broken[0]['graph']['cohorts']['timed-decode-4096']['replays'] = {}
            with self.assertRaises(RuntimeError): spec_proof.validate_receipts(broken, layout)
        broken = self.receipts('pp2ep4')
        broken[0]['graph']['cohorts']['timed-decode-4096']['phases']['draft'] = 1
        with self.assertRaises(RuntimeError): spec_proof.validate_receipts(broken, 'pp2ep4')

    def test_only_whole_cohort_is_recognized(self):
        make = lambda ids: SimpleNamespace(reqs=[SimpleNamespace(rid=x) for x in ids])
        self.assertEqual(spec_proof.cohort(make(['timed-decode-4096-0', 'timed-decode-4096-4'])), 'timed-decode-4096')
        self.assertIsNone(spec_proof.cohort(make(['timed-decode-4096-0', 'warm-decode-4096-4'])))
        self.assertIsNone(spec_proof.cohort(make(['capture_dummy'])))

    def test_semantic_inputs_exact_lengths_no_shared_page_and_same_query(self):
        seqs=workload.prefixes(4096)
        self.assertEqual(len(seqs),8)
        self.assertEqual(len({tuple(seq[:64]) for seq in seqs}),8)
        for index,seq in enumerate(seqs):
            self.assertEqual(len(seq),4096)
            self.assertEqual(seq+workload.query(index,1,False),workload.prompts()[index])
            self.assertEqual(workload.query(index,1,True),workload.query(index,1,False))
        self.assertEqual(workload.cached_prefix_length(4096),4032)
        with self.assertRaises(ValueError):workload.query(0,256,False)

    def test_cache_hook_does_not_construct_or_insert_host_kv(self):
        path=Path(__file__).parent/'runtime/scheduler_hooks.py'
        tree=ast.parse(path.read_text())
        forbidden={'fill_','insert_host'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):self.assertNotIn(node.attr,forbidden)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotEqual(node.func.id,'fill')

    def test_profile_uses_same_semantic_input_but_separate_request_ids_and_length(self):
        payloads=[]
        def response(req,timeout):
            payload=json.loads(req.data);payloads.append(payload)
            n=payload['sampling_params']['max_new_tokens']
            meta=dict(prompt_tokens=4097,cached_tokens=4032,
                      cached_tokens_details=dict(host=4032,device=0))
            lines=[]
            for count in [1,n]:
                lines.append(b'data: '+json.dumps(dict(meta_info=dict(meta,completion_tokens=count),output_ids=list(range(count)),text='answer')).encode()+b'\n')
            return io.BytesIO(b''.join(lines)+b'data: [DONE]\n')
        prefix=workload.prefixes(4096)[0]
        with patch.object(measure.urllib.request,'urlopen',side_effect=response):
            production=measure.request('http://unused',dict(layout='pp2ep4',production_output_tokens=64),prefix,0,'decode',False,threading.Barrier(1))
            profile=measure.request('http://unused',dict(layout='pp2ep4',production_output_tokens=64),prefix,0,'decode',False,threading.Barrier(1),
                                    max_output_tokens=64,cohort_prefix='profile')
        self.assertEqual(payloads[0]['input_ids'],payloads[1]['input_ids'])
        self.assertEqual(payloads[0]['rid'],'timed-decode-4096-0')
        self.assertEqual(payloads[1]['rid'],'profile-decode-4096-0')
        self.assertEqual(production['completion_tokens'],64)
        self.assertEqual(profile['completion_tokens'],64)

    def test_profile_cannot_start_without_saved_production_result(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'cannot precede'):
                measure.profile_once('http://unused',Path(root),dict(layout='pp2ep4',production_output_tokens=64),4096,[],
                                     'p4096_decode')

    def test_profile_receipts_are_not_accepted_as_production_receipts(self):
        rows=self.receipts('ep8')
        for row in rows:
            value=row['graph']['cohorts'].pop('timed-decode-4096')
            row['graph']['cohorts']['profile-decode-4096']=value
        with self.assertRaises(RuntimeError):spec_proof.validate_receipts(rows,'ep8')
        self.assertTrue(spec_proof.validate_receipts(rows,'ep8',cohort_prefix='profile-')['passed'])

    def test_flag_changes_scheduler_mode_without_changing_communication(self):
        cfg=dict(layout='pp2ep4',id='unit-test',run_dir='/tmp/unit',
                 engine_root='/tmp/pinned-engine',engine_commit='a'*40,
                 semantic_inputs='/tmp/semantic',mode='baseline')
        left=common.environment(cfg,0,{})
        right=common.environment(dict(cfg,mode='candidate'),0,{})
        changed={key for key in left.keys()|right.keys() if left.get(key)!=right.get(key)}
        self.assertEqual(changed,{'SGLANG_PP_SPEC_MICROBATCH'})
        self.assertEqual(left['SGLANG_PP_SPEC_MICROBATCH'],'0')
        self.assertEqual(right['SGLANG_PP_SPEC_MICROBATCH'],'1')
        self.assertIn('/tmp/pinned-engine/python',right['PYTHONPATH'])
        self.assertEqual(right['DEEPEP_LL_PCIE_P2P_MAX_TOKENS'],'0')

    def microbatch_receipts(self,mode):
        return [dict(rank=rank,deterministic=deterministic_sample(),microbatch=dict(enabled=mode=='candidate',loop_size=2,
                    feedback_group_name='feedback' if mode=='candidate' else None,
                    feedback_group_ranks=[0,4] if mode=='candidate' else None),
                    atomic_admissions=[dict(cohort='timed-decode-4096',request_indices=list(range(8)))],
                    batches=[dict(batch_size=4,pp_slot_id=slot,
                        request_ids=[f'timed-decode-4096-{i}' for i in range(slot*4,slot*4+4)])
                        for slot in range(2)]) for rank in range(8)]

    def test_actual_runtime_mode_and_two_slot_composition_required(self):
        for mode in ('baseline','candidate'):
            proof=measure.validate_microbatch_receipts(self.microbatch_receipts(mode),mode)
            self.assertTrue(proof['passed'])
            self.assertEqual(len(proof['actual_slot_request_composition']),8)
        broken=self.microbatch_receipts('candidate');broken[0]['microbatch']['enabled']=False
        with self.assertRaisesRegex(RuntimeError,'mode/slot'):
            measure.validate_microbatch_receipts(broken,'candidate')
        broken=self.microbatch_receipts('candidate');broken[0]['microbatch']['feedback_group_name']=None
        with self.assertRaisesRegex(RuntimeError,'identity'):
            measure.validate_microbatch_receipts(broken,'candidate')
        broken=self.microbatch_receipts('candidate');broken[0]['batches'].pop()
        with self.assertRaisesRegex(RuntimeError,'eight requests'):
            measure.validate_microbatch_receipts(broken,'candidate')

    def test_pair_mismatch_preserves_metrics_but_disallows_speedup_claim(self):
        cfg=dict(engine_commit='a'*40,engine_root='/tmp/engine',nodes=['a','b'],
            layouts=common.LAYOUTS,prefix_lengths=[4096],production_output_tokens=64,
            profile_output_tokens=64,semantic_inputs='/tmp/semantic',deterministic_inference=True,prefill_truncation_alignment=1024)
        point=dict(prefix_tokens_per_request=4096,actual_cached_tokens_per_request=4032,
            first_forward_new_tokens_per_request=65,output_tokens_per_request=64,
            semantic_input_manifest=dict(requests=[dict(sha256=str(i)) for i in range(8)]),
            requests=[dict(request_index=i,output_ids=[1,2],output_text='answer') for i in range(8)],
            tpot_s=dict(mean=1),throughput_tok_s=8,actual_microbatch_scheduler_proof=dict(passed=True))
        other=copy.deepcopy(point);other['requests'][0]['output_ids']=[3,2]
        with patch.object(finalize_pair,'load_arm',side_effect=[(cfg,point),(cfg,other)]):
            value=finalize_pair.compare(Path('/tmp/baseline'),Path('/tmp/candidate'))
        self.assertEqual(value['correctness_status'],'correctness_unconfirmed')
        self.assertFalse(value['performance_comparison_claim_valid'])
        self.assertEqual(value['baseline']['throughput_tok_s'],8)
        self.assertEqual(value['candidate']['requests'][0]['output_ids'],[3,2])

    def test_equal_pair_without_repeatability_is_not_a_validated_speedup(self):
        cfg=dict(engine_commit='a'*40,engine_root='/tmp/engine',nodes=['a','b'],
            layouts=common.LAYOUTS,prefix_lengths=[4096],production_output_tokens=64,
            profile_output_tokens=64,semantic_inputs='/tmp/semantic',deterministic_inference=True,prefill_truncation_alignment=1024)
        point=dict(prefix_tokens_per_request=4096,actual_cached_tokens_per_request=4032,
            first_forward_new_tokens_per_request=65,output_tokens_per_request=64,
            semantic_input_manifest=dict(requests=[dict(sha256=str(i)) for i in range(8)]),
            requests=[dict(request_index=i,output_ids=[1,2],output_text='answer') for i in range(8)],
            tpot_s=dict(mean=1),throughput_tok_s=8,actual_microbatch_scheduler_proof=dict(passed=True))
        with patch.object(finalize_pair,'load_arm',side_effect=[(cfg,point),(cfg,point)]):
            value=finalize_pair.compare(Path('/tmp/baseline'),Path('/tmp/candidate'))
        self.assertTrue(value['greedy_pair_outputs_equal'])
        self.assertFalse(value['performance_comparison_claim_valid'])
        point['repeatability']=dict(passed=True)
        with patch.object(finalize_pair,'load_arm',side_effect=[(cfg,point),(cfg,point)]):
            value=finalize_pair.compare(Path('/tmp/baseline'),Path('/tmp/candidate'))
        self.assertTrue(value['performance_comparison_claim_valid'])

    def test_pair_rejects_changed_workload(self):
        cfg=dict(engine_commit='a'*40,engine_root='/tmp/engine',nodes=['a','b'],
            layouts=common.LAYOUTS,prefix_lengths=[4096],production_output_tokens=64,
            profile_output_tokens=64,semantic_inputs='/tmp/semantic',deterministic_inference=True,prefill_truncation_alignment=1024)
        with patch.object(finalize_pair,'load_arm',side_effect=[(cfg,{}),(dict(cfg,prefix_lengths=[16384]),{})]):
            with self.assertRaisesRegex(ValueError,'prefix_lengths'):
                finalize_pair.compare(Path('/tmp/baseline'),Path('/tmp/candidate'))

    def test_native_deterministic_flag_is_identical_on_both_arms(self):
        cfg=dict(layout='pp2ep4',nodes=['a','b'],dist_port=1,api_port=2)
        left=common.server_args(dict(cfg,mode='baseline'),0)
        right=common.server_args(dict(cfg,mode='candidate'),0)
        self.assertEqual(left,right)
        self.assertEqual(left.count('--enable-deterministic-inference'),1)
        self.assertEqual(left[left.index('--attention-backend')+1],'triton')

    def test_deterministic_prefill_alignment_does_not_exceed_chunk(self):
        cfg=dict(layout='pp2ep4',id='alignment-test',run_dir='/tmp/test',nodes=['a','b'],
            engine_root='/tmp/pinned',engine_commit='a'*40,semantic_inputs='/tmp/semantic',
            dist_port=1,api_port=2,mode='baseline')
        args=common.server_args(cfg,0);env=common.environment(cfg,0,{})
        alignment=int(env['SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE'])
        chunk=int(args[args.index('--chunked-prefill-size')+1])
        maximum=int(args[args.index('--max-prefill-tokens')+1])
        self.assertEqual(alignment,1024)
        self.assertLessEqual(alignment,chunk)
        self.assertLessEqual(alignment,maximum)
        self.assertEqual(chunk%alignment,0)
        self.assertEqual(common.environment(dict(cfg,mode='candidate'),0,{})['SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE'],str(alignment))

    def test_actual_deterministic_rank_receipts_fail_closed(self):
        rows=[dict(rank=i,deterministic=deterministic_sample()) for i in range(8)]
        self.assertTrue(deterministic_proof.validate(rows)['passed'])
        for key,value in [('resolved_enabled',False),('declared_enabled',False),
                ('actual_target_deterministic',False),('actual_target_backend','FlashInferAttnBackend'),
                ('nccl_algo','ring'),('nccl_min_nchannels','1'),('nccl_max_nchannels','16'),
                ('actual_truncation_align_size',4096),('actual_chunked_prefill_size',4096),
                ('effective_max_prefill_tokens',4096),('configured_truncation_alignment','4096')]:
            broken=copy.deepcopy(rows);broken[3]['deterministic'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'runtime proof'):
                deterministic_proof.validate(broken)
        with self.assertRaises(RuntimeError):deterministic_proof.validate(rows[:-1])

if __name__ == '__main__': unittest.main()
