"""CPU-only owned socket reception, nonce, and post-formal cleanup boundary."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from common import save, sha
from counter_socket import exchange, process_ticks
from ce_receipts import collect, fetch_owned_ranks, validate_final
from ce_serving import register_counter_listener


class CounterReceipts(unittest.TestCase):
    def identities(self,out,node_rank=0):
        out.mkdir(parents=True,exist_ok=True);owned={}
        for rank in range(node_rank*4,node_rank*4+4):
            owned[1000+rank]=2000+rank
            save(out/f'ce_config_rank{rank}.json',dict(rank=rank,pid=1000+rank,
                process_start_ticks=2000+rank,counter_socket=f'/tmp/test-r{rank}.sock'))
        return owned

    def receipt(self,out,rank,request):
        identity=json.loads((out/f'ce_config_rank{rank}.json').read_text())
        row=dict(passed=True,rank=rank,pid=identity['pid'],process_start_ticks=identity['process_start_ticks'],
            request_id=request['request_id'],source='unix_socket_listener',received_at=time.time(),
            completed_at=time.time(),after_formal_requests=True,cpu_only=True,timed=False,
            counters=dict(epoch=4,counters=[4,8,100,1,2,2]))
        save(out/f'ce_final_rank{rank}.json',row)
        return dict(passed=True,receipt=row,receipt_sha256=sha(out/f'ce_final_rank{rank}.json'))

    def test_validate_all_owners_before_any_socket_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);owned=self.identities(out);sent=[];request=dict(request_id='r')
            def send(identity,marker):
                sent.append(identity['rank']);return self.receipt(out,identity['rank'],marker)
            def wrong(pid):return dict(ticks=owned[pid] if pid!=1002 else 999,state='S')
            with self.assertRaisesRegex(RuntimeError,'owner identity'):
                fetch_owned_ranks(out,0,owned,request,wrong,send)
            self.assertEqual(sent,[])
            row=fetch_owned_ranks(out,0,owned,request,lambda pid:dict(ticks=owned[pid],state='S'),send)
            self.assertEqual(sent,list(range(4)));self.assertTrue(row['passed'])
            self.assertEqual(row['transport'],'unix_socket')

    def test_all_socket_acks_precede_stop_with_matching_nonce_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);out=run/'pp2ep4';owned={}
            owned.update(self.identities(out,0));owned.update(self.identities(out,1));order=[]
            def read_visible(path,*_args):
                request=json.loads((out/'CE_SNAPSHOT_REQUEST.json').read_text())
                self.assertTrue(request['no_other_model_requests']);self.assertFalse(request['timed'])
                if path.name.startswith('ce_socket_node'):
                    node=int(path.stem[-1]);order.append(path.stem)
                    row=fetch_owned_ranks(out,node,owned,request,lambda pid:dict(ticks=owned[pid],state='S'),
                        lambda identity,marker:self.receipt(out,identity['rank'],marker))
                    save(path,row);return row
                self.assertEqual(order[:2],['ce_socket_node0','ce_socket_node1'])
                order.append(path.stem);return json.loads(path.read_text())
            final=collect(run,'pp2ep4',dict(nodes=['a','b'],world=8),read_visible)
            self.assertEqual(len(final),8);self.assertFalse((out/'STOP').exists())
            row=json.loads((out/'ce_counter_receipts.json').read_text())
            self.assertTrue(row['all_workers_received']);self.assertEqual(row['ranks'],list(range(8)))

    def test_actual_local_socket_binds_marker_identity_and_one_shot_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);out=run/'pp2ep4';out.mkdir();ticks=process_ticks(os.getpid())
            request=dict(arm='pp2ep4',request_id='cpu-request',created_at=time.time(),
                after_formal_requests=True,no_other_model_requests=True,timed=False)
            save(out/'CE_SNAPSHOT_REQUEST.json',request)
            listener=register_counter_listener(run,'pp2ep4',0,object(),ticks,save,
                lambda _:dict(epoch=2,counters=[2,4,100,2,1,1]))
            identity=dict(rank=0,pid=os.getpid(),process_start_ticks=ticks,counter_socket=str(listener.path))
            save(out/'ce_config_rank0.json',identity)
            try:
                with self.assertRaisesRegex(RuntimeError,'quiescent'):
                    exchange(identity,dict(request,request_id='stale'))
                response=exchange(identity,request)
                row=validate_final(out,0,request)
                self.assertEqual(row,response['receipt']);self.assertNotEqual(row['listener_thread_id'],threading.get_ident())
                listener.thread.join(timeout=2)
                self.assertFalse(listener.thread.is_alive());self.assertFalse(listener.path.exists())
                with self.assertRaisesRegex(RuntimeError,'identity/request'):
                    validate_final(out,0,dict(request,request_id='wrong-request'))
            finally:listener.close()

    def test_nonquiescent_or_changing_native_counters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);out=run/'ep8';out.mkdir();ticks=process_ticks(os.getpid());snapshots=[]
            def changing(_):
                snapshots.append(1);return dict(epoch=len(snapshots),counters=[1]*6)
            request=dict(arm='ep8',request_id='changing',after_formal_requests=True,no_other_model_requests=False)
            save(out/'CE_SNAPSHOT_REQUEST.json',request)
            listener=register_counter_listener(run,'ep8',0,object(),ticks,save,changing)
            identity=dict(rank=0,pid=os.getpid(),process_start_ticks=ticks,counter_socket=str(listener.path))
            try:
                with self.assertRaisesRegex(RuntimeError,'quiescent'):exchange(identity,request)
                self.assertEqual(snapshots,[])
                request['no_other_model_requests']=True;save(out/'CE_SNAPSHOT_REQUEST.json',request)
                with self.assertRaisesRegex(RuntimeError,'counters changed'):exchange(identity,request)
                self.assertFalse((out/'ce_final_rank0.json').exists())
            finally:listener.close();listener.thread.join(timeout=2)

    def test_receipt_failure_propagates_before_stop_can_be_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);out=run/'ep8';out.mkdir()
            def missing(*_args):raise TimeoutError('missing node socket receipt')
            with self.assertRaises(TimeoutError):collect(run,'ep8',dict(nodes=['a','b'],world=8),missing)
            self.assertFalse((out/'ce_counter_receipts.json').exists());self.assertFalse((out/'STOP').exists())


if __name__=='__main__':unittest.main()
