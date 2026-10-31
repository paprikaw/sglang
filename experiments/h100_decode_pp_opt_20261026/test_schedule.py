import unittest
from types import SimpleNamespace
from pp_schedule_opt import wrap_methods


class SchedulingTest(unittest.TestCase):
    def make(self, early=False, last=False, topology=None):
        class Fake:
            ps = SimpleNamespace(pp_size=2)
            def __init__(self): self.events=[]
            def _pp_topology(self): return topology
            def _pp_is_last_rank(self): return last
            def _pp_launch_or_skip_batch(self, *args):
                self.events.append('launch'); return object(), object()
            def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *args):
                self.events.append('feedback'); return 'retired'
            def _pp_send_launched_proxy(self, batch, result):
                self.events.append(('send',batch,result))
        policy=[ord('1' if early else '0')]
        wrap_methods(Fake,lambda:policy[0])
        return Fake(),policy

    def step(self, obj):
        batch=object()
        result,event=obj._pp_launch_or_skip_batch(0,batch,None,None,None)
        retired=obj._pp_commit_send_output_work_and_preprocess_output_tensors(0,1)
        obj._pp_send_launched_proxy(batch,result)
        self.assertEqual(retired,'retired')
        return batch,result,event

    def test_original_order(self):
        obj,_=self.make();self.step(obj)
        self.assertEqual([e if isinstance(e,str) else e[0] for e in obj.events],['launch','feedback','send'])

    def test_early_preserves_exact_send_once(self):
        obj,_=self.make(True);batch,result,_=self.step(obj)
        self.assertEqual([e if isinstance(e,str) else e[0] for e in obj.events],['launch','send','feedback'])
        self.assertIs(obj.events[1][1],batch);self.assertIs(obj.events[1][2],result)
        self.assertIsNone(obj._decode_opt_sent)

    def test_policy_switch_at_idle(self):
        obj,policy=self.make();self.step(obj)
        policy[0]=ord('1');self.step(obj)
        policy[0]=ord('0');self.step(obj)
        self.assertEqual(sum(isinstance(e,tuple) for e in obj.events),3)

    def test_tail_does_not_send_early(self):
        obj,_=self.make(True,last=True);self.step(obj)
        self.assertEqual(obj.events[1],'feedback')

    def test_unknown_policy_fails(self):
        obj,p=self.make();p[0]=ord('9')
        with self.assertRaises(RuntimeError): self.step(obj)

    def test_elastic_topology_rejected(self):
        obj,_=self.make(True,topology=object())
        with self.assertRaises(RuntimeError): self.step(obj)


if __name__=='__main__': unittest.main()
