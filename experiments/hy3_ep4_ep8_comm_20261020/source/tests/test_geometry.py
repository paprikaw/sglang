"""CPU conservation checks for Hy3's production dispatch shapes."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import layout, phase_definitions, validate_placement
from common import environment

class Geometry(unittest.TestCase):
    def test_source_and_receive_work_matched_per_rank(self):
        for source in (1, 128):
            for world in (4, 8):
                shape = layout(world, source * world)
                self.assertEqual(shape['local_tokens'], source)
                self.assertEqual(shape['actual_routed_tokens'], source * world)
                local_experts = 192 // world
                counts = shape['global_expert_counts']
                owners = [sum(counts[r*local_experts:(r+1)*local_experts]) for r in range(world)]
                self.assertEqual(owners, [source*8]*world)
                self.assertTrue(all(all(mask) for mask in shape['real_masks_by_rank']))

    def test_three_topology_phases(self):
        phases = phase_definitions()
        self.assertEqual([p['id'] for p in phases], ['local065', 'local167', 'cross8'])
        for phase in phases:
            validate_placement(phase['placement'])
            self.assertEqual(phase['capacity'], 128)

    def test_prefill_covers_all_experts(self):
        for world in (4, 8):
            shape = layout(world, 128*world)
            self.assertEqual(sum(c > 0 for c in shape['global_expert_counts']), 192)
            self.assertLessEqual(max(shape['global_expert_counts']) - min(shape['global_expert_counts']), 1)

    def test_environment_transport_is_unchanged(self):
        env = environment((0, 1, 2, 3))
        self.assertEqual(env['DEEPEP_LL_NVLINK_PAIR_ONLY'], '1')
        self.assertEqual(env['DEEPEP_LL_PCIE_P2P_MAX_TOKENS'], '0')
        self.assertEqual(env['NVSHMEM_IBGDA_NIC_HANDLER'], 'cpu_host_memory')

    def test_aligned_sample_maxima(self):
        from helpers import aggregate
        samples = [[dict(iteration=i, ms=v) for i,v in enumerate(values)]
                   for values in ([1, 4, 2], [3, 2, 5])]
        self.assertEqual(aggregate(samples, 'ms')['per_iteration_rank_max_ms'], [3, 4, 5])

if __name__ == '__main__':
    unittest.main()
