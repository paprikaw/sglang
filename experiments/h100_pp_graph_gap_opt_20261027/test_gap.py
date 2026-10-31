"""CPU correctness checks for PP feedback ancestry and schema negotiation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "h100_decode_pp_opt_20261026"))
from gap_schedule import Slots, describe, install_schema, install_early, install_prepare, can_prepare, install_lookahead


class Tensor:
    def __init__(self, shape=(4,2048), dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.is_cuda = device.startswith("cuda")
        self.contiguous = contiguous
    def is_contiguous(self): return self.contiguous


class GapTests(unittest.TestCase):
    def test_lookahead_preserves_launch_send_fifo_and_request_ownership(self):
        calls=[];proxy=object();a=SimpleNamespace(reqs=[SimpleNamespace(rid='a')]);b=SimpleNamespace(reqs=[SimpleNamespace(rid='b')])
        class Fake:
            ps=SimpleNamespace(pp_size=2)
            _gap_proxy_stream_ctx=__import__('contextlib').nullcontext()
            def __init__(self):
                self._gap_ahead={};self._gap_ahead_produced=self._gap_ahead_consumed=0
                self.running_batch=self.cur_batch_for_debug=a;self.send_proxy_work=[]
            def _pp_topology(self):return None
            def _pp_is_last_rank(self):return False
            def _pp_recv_proxy_tensors(self):calls.append('receive');return proxy
            def _pp_commit_comm_work(self,work):calls.append('commit');work.clear()
            def _pp_launch_or_skip_batch(self,slot,batch,proxy,metadata,queue):
                calls.append(('launch',slot));result,event=object(),object();queue.append(result);return result,event
            def _pp_send_launched_proxy(self,batch,result):calls.append(('send',batch.reqs[0].rid))
        install_early(Fake,lambda:8);install_lookahead(Fake,lambda:8);obj=Fake();queue=[]
        result_a,event_a=obj._pp_launch_or_skip_batch(0,a,proxy,None,queue)
        obj._gap_launch_ahead(1,b,None,queue)
        self.assertIs(obj.cur_batch_for_debug,a)
        obj._pp_send_launched_proxy(a,result_a)
        obj.cur_batch_for_debug=b
        cached_proxy=obj._pp_recv_proxy_tensors()
        result_b,event_b=obj._pp_launch_or_skip_batch(1,b,cached_proxy,None,queue)
        obj._pp_send_launched_proxy(b,result_b)
        self.assertEqual([c for c in calls if isinstance(c,tuple)],[('launch',0),('send','a'),('launch',1),('send','b')])
        self.assertEqual(calls.count('receive'),1)
        self.assertEqual(len(queue),2)
        self.assertEqual((obj._gap_ahead_produced,obj._gap_ahead_consumed),(1,1))
        self.assertFalse(obj._gap_ahead);self.assertFalse(obj._gap_proxy_sent)

    def prep_batch(self):
        class Batch:
            def __init__(self):
                self.forward_mode=SimpleNamespace(is_decode=lambda:True)
                self.spec_algorithm=SimpleNamespace(is_none=lambda:True)
                self.return_logprob=self.has_grammar=False
                self.sampling_info=SimpleNamespace(penalizer_orchestrator=SimpleNamespace(is_required=False))
                self.seqs=[4106]*4
                self.seq_lens_cpu=SimpleNamespace(tolist=lambda:self.seqs)
                self.reqs=[SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=True,max_new_tokens=128),
                    output_ids=list(range(10)),finished=lambda:False,is_retracted=False,grammar=None,to_finish=None) for _ in range(4)]
                self.mamba_track_mask_cpu=[False]*4;self.prepared=0
            def check_decode_mem(self):return True
            def copy(self):
                import copy
                return copy.copy(self)
            def prepare_for_decode(self):
                self.prepared+=1;self.seqs=[n+1 for n in self.seqs];self.mamba_track_mask_cpu=[False]*4
        return Batch

    def test_prepare_snapshot_and_exactly_one_allocation(self):
        Batch=self.prep_batch();batch=Batch()
        class Fake:
            _gap_prepared=_gap_prepared_consumed=0
            def _pp_send_recv_and_preprocess_output_tensors(self,*args):return ('original-output','original-result','original-d2h','original-work')
        install_prepare(Fake,Batch,lambda:7,lambda s:64);obj=Fake()
        batches=[None,batch];obj.running_mbs=[None,batch]
        result=obj._pp_send_recv_and_preprocess_output_tensors(0,1,batches,None,None,None)
        self.assertEqual(result,('original-output','original-result','original-d2h','original-work'))
        self.assertIsNot(batches[1],batch)
        self.assertEqual(batches[1].prepared,0);self.assertEqual(batch.prepared,1)
        self.assertEqual(batches[1].mamba_track_mask_next_cpu,batch.mamba_track_mask_cpu)
        batch.prepare_for_decode();self.assertEqual(batch.prepared,1)
        self.assertEqual((obj._gap_prepared,obj._gap_prepared_consumed),(1,1))
        batch.prepare_for_decode();self.assertEqual(batch.prepared,2)

    def test_prepare_rejects_stopping_and_gdn_boundaries(self):
        batch=self.prep_batch()();self.assertTrue(can_prepare(batch,64))
        batch.reqs[0].sampling_params.ignore_eos=False;self.assertFalse(can_prepare(batch,64))
        batch.reqs[0].sampling_params.ignore_eos=True
        batch.seqs=[4096]*4;self.assertFalse(can_prepare(batch,64))
        batch.seqs=[4095]*4;self.assertFalse(can_prepare(batch,64))
        batch.seqs=[4106]*4;batch.reqs[0].output_ids=list(range(127));self.assertFalse(can_prepare(batch,64))

    def test_wider_window_preserves_two_token_finish_margin(self):
        batch=self.prep_batch()()
        for req in batch.reqs:req.output_ids=[1,2]
        self.assertFalse(can_prepare(batch,64))
        self.assertTrue(can_prepare(batch,64,2,2))
        batch.reqs[0].output_ids=list(range(126));self.assertFalse(can_prepare(batch,64,2,2))
        batch.reqs[0].output_ids=list(range(125));self.assertTrue(can_prepare(batch,64,2,2))

    def test_early_send_uses_producer_event_once_at_launch(self):
        calls=[];policy=[1];event=object();result=object();batch=object()
        class Fake:
            ps=SimpleNamespace(pp_size=2)
            def _pp_topology(self):return None
            def _pp_is_last_rank(self):return False
            def _pp_launch_or_skip_batch(self,*args):calls.append('launch');return result,event
            def _pp_send_launched_proxy(self,b,r):calls.append((self.launch_event,b,r))
        install_early(Fake,lambda:policy[0]);obj=Fake()
        self.assertEqual(obj._pp_launch_or_skip_batch(0,batch,None,None,None),(result,event))
        self.assertEqual(calls,['launch',(event,batch,result)])
        obj._pp_send_launched_proxy(batch,result)
        self.assertEqual(len(calls),2)
        policy[0]=0
        obj._pp_launch_or_skip_batch(0,batch,None,None,None)
        self.assertEqual(calls[-1],'launch')
        obj._pp_send_launched_proxy(batch,result)
        self.assertEqual(calls[-1],(event,batch,result))

    def test_dedicated_send_keeps_event_and_restores_scheduling_stream(self):
        calls=[];event=object();result=object();batch=object();active=['schedule']
        class Context:
            def __enter__(self):active[0]='proxy'
            def __exit__(self,*args):active[0]='schedule'
        class Fake:
            ps=SimpleNamespace(pp_size=2)
            _gap_proxy_stream_ctx=Context()
            def _pp_topology(self):return None
            def _pp_is_last_rank(self):return False
            def _pp_launch_or_skip_batch(self,*args):return result,event
            def _pp_send_launched_proxy(self,b,r):calls.append((active[0],self.launch_event,b,r))
        install_early(Fake,lambda:6);obj=Fake()
        obj._pp_launch_or_skip_batch(0,batch,None,None,None)
        self.assertEqual(calls,[('proxy',event,batch,result)])
        self.assertEqual(active[0],'schedule')
        obj._pp_send_launched_proxy(batch,result)
        self.assertEqual(len(calls),1)

    def test_local_ownership_and_slot_ancestry(self):
        slots = Slots(); event, payload = object(), object()
        slots.put(0, event, payload, ["a","b"])
        pending = slots.pop(0, ["a","b"])
        self.assertIs(pending.payload, payload); self.assertIs(pending.event, event)
        self.assertEqual((slots.produced, slots.retired), (1,1))

    def test_local_overwrite_rejected(self):
        slots = Slots();slots.put(0, None, None, ["a"])
        with self.assertRaises(RuntimeError): slots.put(0, None, None, ["a"])
        with self.assertRaises(RuntimeError): slots.put(2, None, None, ["a"])

    def test_wrong_request_history_rejected(self):
        slots = Slots();slots.put(1, None, None, ["a","b"])
        with self.assertRaises(RuntimeError): slots.pop(1, ["b","a"])

    def test_schema_preserves_type_shape_and_metadata(self):
        torch = SimpleNamespace(is_tensor=lambda value:isinstance(value,Tensor))
        tensors, values = describe(torch, {"hidden":Tensor(), "__msg_type__":"proxy"})
        self.assertEqual(tensors[0], ("hidden",(4,2048),"bf16","cuda:0"))
        self.assertEqual(values, (("__msg_type__","proxy"),))
        for tensor in (Tensor((3,2048)), Tensor(device="cpu"), Tensor(contiguous=False)):
            self.assertIsNone(describe(torch,{"hidden":tensor}))

    def test_fast_sender_rejects_schema_or_direction_change(self):
        calls=[]; policy=[0]
        class Fake:
            pp_group=SimpleNamespace(is_last_rank=False,ranks=[0,1],rank_in_group=0,device_group="same-native-group")
            def _pp_send_dict_to_next_stage(self,payload,**kwargs):
                payload["__msg_type__"]=kwargs["msg_type"];return ["native"]
            def _pp_recv_typed_dict(self,**kwargs):return {}
        dist=SimpleNamespace(isend=lambda tensor,peer,group:calls.append((tensor,peer,group)))
        torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,Tensor),distributed=dist)
        install_schema(Fake,torch,lambda:policy[0],lambda handle,payload:SimpleNamespace(work=handle,payload=payload))
        obj=Fake();tensor=Tensor()
        self.assertEqual(obj._pp_send_dict_to_next_stage({"hidden":tensor},msg_type="proxy"),["native"])
        policy[0]=3
        result=obj._pp_send_dict_to_next_stage({"hidden":tensor},msg_type="proxy")
        self.assertIs(result[0].payload,tensor)
        self.assertEqual(calls[0][1:],(1,"same-native-group"))
        with self.assertRaises(RuntimeError):obj._pp_send_dict_to_next_stage({"hidden":Tensor((4,1024))},msg_type="proxy")
        with self.assertRaises(RuntimeError):obj._pp_send_dict_to_next_stage({"hidden":tensor},msg_type="output")

    def test_typed_channels_match_even_if_output_precedes_proxy(self):
        policy=[0];sends=[];receives=[]
        class Handle:
            def wait(self):return None
        class Fake:
            def __init__(self,last):
                self.pp_group=SimpleNamespace(is_last_rank=last,ranks=[0,1],rank_in_group=int(last),device_group="native")
                self._gap_groups=["activation","tail-to-head","head-echo"]
                self._pp_tensor_dict_inbox={}
            def _pp_send_dict_to_next_stage(self,payload,**kw):payload['__msg_type__']=kw['msg_type'];return []
            def _pp_recv_typed_dict(self,expected_kind,**kw):
                return {'value':Tensor(), '__msg_type__':expected_kind}
        def send(tensor,peer,group):sends.append(group);return Handle()
        def receive(tensor,src,group):receives.append(group);return Handle()
        torch=SimpleNamespace(is_tensor=lambda v:isinstance(v,Tensor),
            distributed=SimpleNamespace(isend=send,irecv=receive),empty=lambda shape,**kw:Tensor(shape,**kw))
        install_schema(Fake,torch,lambda:policy[0],lambda h,t:(h,t))
        head,tail=Fake(False),Fake(True)
        for obj in (head,tail):
            for kind in ('output','proxy'):
                obj._pp_send_dict_to_next_stage({'value':Tensor()},msg_type=kind)
                obj._pp_recv_typed_dict(expected_kind=kind)
        policy[0]=5
        # Native depth>0 can echo an output before launching the next proxy.
        head._pp_send_dict_to_next_stage({'value':Tensor()},msg_type='output')
        head._pp_send_dict_to_next_stage({'value':Tensor()},msg_type='proxy')
        tail._pp_recv_typed_dict(expected_kind='proxy')
        tail._pp_recv_typed_dict(expected_kind='output')
        tail._pp_send_dict_to_next_stage({'value':Tensor()},msg_type='output')
        head._pp_recv_typed_dict(expected_kind='output')
        self.assertEqual(sends,['head-echo','activation','tail-to-head'])
        self.assertEqual(receives,['activation','head-echo','tail-to-head'])


if __name__ == "__main__":unittest.main()
