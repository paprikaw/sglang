"""Bounded EP8 placement/admission CPU tests; no Torch or GPU imports."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
import admission_adapter as a
import common
import dp_proof


class Token(SimpleNamespace):
    pass


def token(index,cohort='timed-decode-4096'):
    return Token(rid=f'{cohort}-{index}',routed_dp_rank=index//4)


def deterministic():
    return dict(source='initialized_scheduler_rank',resolved_enabled=True,declared_enabled=True,
        resolved_attention_backend='triton',actual_target_backend='TritonAttnBackend',
        actual_target_backend_module='sglang.srt.layers.attention.triton_backend',actual_target_deterministic=True,
        nccl_algo='allreduce:tree',nccl_min_nchannels='8',nccl_max_nchannels='8',
        actual_truncation_align_size=1024,actual_chunked_prefill_size=1024,
        effective_max_prefill_tokens=1024,configured_truncation_alignment='1024',
        disable_custom_all_reduce=True,enable_torch_symm_mem=False)


def humming():
    return dict(schema='humming-loaded-kernel-invariant-v1',
        evidence_kind='registered_cubin_metadata_not_launch_trace',proves_actual_launch=False,
        kernels=[dict(kernel_id=i,gemm_type=k,sm_version=80,use_batch_invariant=True,
            use_stream_k=False,block_shape=[64,128,32],warp_shape=[32,32,32],
            kernel_filename='/owned/kernel.cubin',kernel_name='humming',source_sha256='a'*64)
            for i,k in enumerate(('dense','grouped_masked'))])


def receipts():
    rows=[]
    for rank in range(8):
        dp=rank//4;state=a.new_state()
        items=[token(i) for i in range(dp*4,dp*4+4)]
        a.release_local(items,dp_rank=dp,state=state,tokenized_type=Token)
        first=dict(mode='1',batch_size=4,request_ids=[x.rid for x in items],
                   extend_num_tokens=260,attn_dp_rank=dp,local_submission_ordinal=0)
        second=dict(first,mode='2',local_submission_ordinal=1)
        rows.append(dict(rank=rank,microbatch=dict(enabled=False,loop_size=1),
            admission_pending=0,atomic_admissions=state['proofs'],batches=[first,second],
            deterministic=deterministic(),humming=humming()))
    return rows


class AdmissionTests(unittest.TestCase):
    def test_routes_exact_semantic_groups_all_four_cohorts(self):
        for cohort in a.COHORTS:
            for i in range(8):
                item=Token(rid=f'{cohort}-{i}',routed_dp_rank=None)
                self.assertEqual(a.route_tokenized(item).routed_dp_rank,i//4)

    def test_natural_and_health_passthrough(self):
        item=Token(rid='natural-sanity-0',routed_dp_rank=0)
        self.assertIs(a.route_tokenized(item),item)
        state=a.new_state()
        self.assertEqual(a.release_local([item,object()],dp_rank=0,state=state,tokenized_type=Token)[:1],[item])
        self.assertFalse(state['pending'])

    def test_partial_arrivals_and_empty_polls_do_not_release_early(self):
        for dp in (0,1):
            state=a.new_state();items=[token(i) for i in range(dp*4,dp*4+4)]
            for item in items[:3]:
                self.assertEqual(a.release_local([item],dp_rank=dp,state=state,tokenized_type=Token),[])
                self.assertEqual(a.release_local([],dp_rank=dp,state=state,tokenized_type=Token),[])
            self.assertEqual(a.release_local(items[3:],dp_rank=dp,state=state,tokenized_type=Token),items)
            self.assertFalse(state['pending'])
            self.assertEqual(len(state['proofs']),1)

    def test_reversed_arrivals_release_canonical_order(self):
        state=a.new_state();items=[token(i) for i in range(4)]
        result=a.release_local(items[::-1],dp_rank=0,state=state,tokenized_type=Token)
        self.assertEqual(result,items)

    def test_duplicate_before_and_after_release_fail(self):
        state=a.new_state();item=token(0)
        a.release_local([item],dp_rank=0,state=state,tokenized_type=Token)
        with self.assertRaises(RuntimeError):
            a.release_local([item],dp_rank=0,state=state,tokenized_type=Token)
        state=a.new_state();a.release_local([token(i) for i in range(4)],dp_rank=0,state=state,tokenized_type=Token)
        with self.assertRaises(RuntimeError):
            a.release_local([token(0)],dp_rank=0,state=state,tokenized_type=Token)

    def test_foreign_rid_and_conflicting_dp_rejected(self):
        for rid in ('timed-decode-16384-0','timed-decode-4096-8','witness-decode-4096-0'):
            with self.assertRaises(RuntimeError):a.identity(rid)
        with self.assertRaises(RuntimeError):a.route_tokenized(Token(rid='timed-decode-4096-0',routed_dp_rank=1))
        with self.assertRaises(RuntimeError):
            a.release_local([token(4)],dp_rank=0,state=a.new_state(),tokenized_type=Token)

    def test_tp4_dp2_pp1_model_and_no_stock_blocker(self):
        cfg=dict(layout='ep8',nodes=['a','b'],dist_port=1,api_port=2,id='test',run_dir='/tmp/test',
            engine_root='/tmp/pinned',engine_commit='a'*40,semantic_inputs='/tmp/semantic',mode='ep8')
        args=common.server_args(cfg,0);env=common.environment(cfg,0,{})
        self.assertIn('--enable-dp-attention',args)
        self.assertEqual(args[args.index('--tp-size')+1],'8')
        self.assertEqual(args[args.index('--dp-size')+1],'2')
        self.assertEqual(args[args.index('--pp-size')+1],'1')
        self.assertEqual(args[args.index('--ep-size')+1],'8')
        self.assertEqual(env['SGLANG_ENABLE_COLOCATED_BATCH_GEN'],'0')
        self.assertEqual(env['SGLANG_PP_SPEC_MICROBATCH'],'0')
        self.assertEqual(args[args.index('--chunked-prefill-size')+1],'2048')

    def test_no_sync_or_device_operations_in_adapter(self):
        tree=ast.parse((ROOT/'runtime/admission_adapter.py').read_text())
        forbidden={'synchronize','all_reduce','barrier','cuda','cpu','to','item','fill_'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):self.assertNotIn(node.attr,forbidden)

    def test_actual_dp_proof_requires_b4_first_forward_and_matched_tp(self):
        self.assertTrue(dp_proof.validate(receipts())['passed'])
        for kind in ('first_b1','bad_group','bad_tp','missing_rank','bad_humming'):
            rows=receipts()
            if kind=='first_b1':rows[0]['batches'][0]['batch_size']=1
            elif kind=='bad_group':rows[4]['atomic_admissions'][0]['request_indices']=[0,1,2,3]
            elif kind=='bad_tp':rows[1]['batches'][1]['mode']='7'
            elif kind=='missing_rank':rows.pop()
            else:rows[7]['humming']['kernels'][0]['use_stream_k']=True
            with self.subTest(kind=kind),self.assertRaises((RuntimeError,ValueError)):
                dp_proof.validate(rows)

    def test_retirement_shrinkage_is_allowed_without_rebuffer(self):
        rows=receipts()
        for row in rows:
            row['batches'].append(dict(row['batches'][1],batch_size=3,
                request_ids=row['batches'][1]['request_ids'][:3],local_submission_ordinal=2))
        self.assertTrue(dp_proof.validate(rows)['passed'])


if __name__=='__main__':unittest.main()
