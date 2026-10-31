from types import SimpleNamespace
import unittest

from pp_fixed_cohort import instrument,order_cohort


def req(marker):return SimpleNamespace(origin_input_ids=[marker])


class FixedCohortTest(unittest.TestCase):
    def test_matches_ep_groups(self):
        self.assertEqual([r.origin_input_ids[0] for r in order_cohort([req(i) for i in reversed(range(5000,5008))])],
                         [5000,5002,5004,5006,5001,5003,5005,5007])

    def test_wait_incomplete(self):self.assertIsNone(order_cohort([req(5000)]))

    def test_reject_unrelated_or_duplicate(self):
        for rows in ([req(6000)],[req(5000),req(5000)]):
            with self.assertRaises(RuntimeError):order_cohort(rows)

    def test_wait_bounded_and_admit_once(self):
        class Fake:
            ps=SimpleNamespace(pp_size=2,tp_size=1)
            def _pp_is_first_rank(self):return True
            def _pp_topology(self):return None
            def get_next_batch_to_run(self,**kwargs):
                return SimpleNamespace(batch_to_run='native' if self.waiting_queue else None,
                                       running_batch=kwargs.get('running_batch'))
            def get_internal_state(self):return SimpleNamespace(internal_state={})
        epoch=[1];now=[0.]
        instrument(Fake,lambda:epoch[0],lambda:now[0])
        obj=Fake();obj.waiting_queue=[req(5000)]
        running=object()
        plan=obj.get_next_batch_to_run(running_batch=running)
        self.assertIsNone(plan.batch_to_run);self.assertIs(plan.running_batch,running)
        self.assertEqual(len(obj.waiting_queue),1)
        now[0]=6.
        with self.assertRaises(TimeoutError):obj.get_next_batch_to_run()
        obj.waiting_queue=[req(i) for i in reversed(range(5000,5008))]
        self.assertEqual(obj.get_next_batch_to_run().batch_to_run,'native')
        obj.waiting_queue=[req(5007)]
        self.assertEqual(obj.get_next_batch_to_run().batch_to_run,'native')
        epoch[0]=2
        self.assertIsNone(obj.get_next_batch_to_run().batch_to_run)
        self.assertEqual(obj.get_internal_state().internal_state['experiment_fixed_group_receipt']['epoch'],1)

    def test_tail_uses_same_gate_and_order(self):
        class Tail:
            ps=SimpleNamespace(pp_size=2,tp_size=1)
            def _pp_is_first_rank(self):return False
            def _pp_topology(self):return None
            def get_next_batch_to_run(self):
                return SimpleNamespace(batch_to_run=list(self.waiting_queue),running_batch=None)
            def get_internal_state(self):return SimpleNamespace(internal_state={})
        instrument(Tail,lambda:1)
        obj=Tail();obj.waiting_queue=[req(5000)]
        self.assertEqual(obj.get_next_batch_to_run().batch_to_run,[])
        obj.waiting_queue=[req(i) for i in reversed(range(5000,5008))]
        self.assertEqual([r.origin_input_ids[0] for r in obj.get_next_batch_to_run().batch_to_run],
                         [5000,5002,5004,5006,5001,5003,5005,5007])


if __name__=='__main__':unittest.main()
