"""CPU invariants that prevent unequal routing and misleading timing totals."""
import unittest
from pathlib import Path
from geometry import chunks,layout,SHAPES,validate_grid,workloads,GLOBAL_CHUNK
from analyze import aligned_critical
from node import phase_for
from controller import step_command


class ProtocolTests(unittest.TestCase):
    def test_full_routing_and_chunks_match_across_EP(self):
        self.assertTrue(validate_grid()['passed'])

    def test_large_batch_preserves_ids_across_chunks(self):
        seen=[]
        for start,count in chunks(32768):
            shape=layout(8,count,192,start)
            seen.extend(i for rank_ids in shape['ids'] for i in rank_ids)
        self.assertEqual(seen,list(range(32768)))
        self.assertEqual(len(chunks(16384)),8)
        self.assertEqual(len(chunks(32768)),16)

    def test_components_use_same_critical_rank(self):
        rank,s=aligned_critical([dict(iteration=0,dispatch_cuda_ms=.6,combine_cuda_ms=.6,pair_cuda_ms=1.2),
            dict(iteration=0,dispatch_cuda_ms=1.,combine_cuda_ms=.1,pair_cuda_ms=1.1)])
        self.assertEqual(rank,0)
        self.assertEqual(s['pair_cuda_ms'],1.2)
        with self.assertRaises(AssertionError):
            aligned_critical([dict(iteration=0,pair_cuda_ms=1),dict(iteration=1,pair_cuda_ms=2)])

    def test_no_outside_nodes_or_duplicate_GPUs(self):
        phase=dict(id='p',nodes=['a','b'],world=8,placement=[dict(rank=r,node='ab'[r//4],gpu=r%4) for r in range(8)])
        cfg=dict(phases=[phase],nodes=['a','b'])
        self.assertEqual(phase_for(cfg,'p')['world'],8)
        phase['placement'][7]['gpu']=2
        with self.assertRaises(ValueError):
            phase_for(cfg,'p')

    def test_only_existing_job_steps_and_memory_budget(self):
        command=step_command(dict(booked_jobs={'a':'123'}),dict(id='p',nodes=['a']),Path('/tmp/run'),'a')
        self.assertEqual(command[:3],['srun','--jobid','123'])
        self.assertNotIn('sbatch',command)
        # LL symmetric storage plus every graph-held receive at the largest batch.
        for shape in SHAPES:
            cap=GLOBAL_CHUNK//2
            symmetric=4*shape['experts']*cap*(shape['hidden']*2+shape['hidden']//128*4)
            graph_receives=(32768//GLOBAL_CHUNK)*shape['experts']*cap*shape['hidden']*2
            self.assertLess(symmetric+graph_receives,64*1024**3)
        self.assertEqual(len(workloads()),26)


if __name__=='__main__':
    unittest.main()
