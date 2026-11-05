"""CPU-only fixtures for EP-group ranks versus physical pipeline node ranks."""
import copy
import unittest
import comm_validation as validation

def topology(world,offset=0):
    return [dict(rank=rank,local_rank=rank%4,node_rank=offset+rank//4,
        host=f'node{offset+rank//4}',uuid=f'gpu{offset+rank//4}-{rank%4}',name='NVIDIA A100-SXM4-80GB',
        nvlink_local_peers=[peer in (rank%4,(rank%4)^1) for peer in range(4)]) for rank in range(world)]

class GroupTopology(unittest.TestCase):
    def test_pp0_ep4(self):
        validation.WORLD=4;rows=topology(4,0)
        self.assertEqual(validation.validate_topology_rows(rows),rows)

    def test_pp1_ep4(self):
        validation.WORLD=4;rows=topology(4,1)
        self.assertEqual(validation.validate_topology_rows(rows),rows)

    def test_ep8(self):
        validation.WORLD=8;rows=topology(8,0)
        self.assertEqual(validation.validate_topology_rows(rows),rows)

    def test_rejects_bad_mapping(self):
        for world,offset in [(4,0),(4,1),(8,0)]:
            validation.WORLD=world
            mutations=[lambda r:r[1].update(node_rank=offset+1),
                       lambda r:r[1].update(rank=0),
                       lambda r:r[1].update(local_rank=0),
                       lambda r:r[1].update(uuid=r[0]['uuid']),
                       lambda r:r[1].update(nvlink_local_peers=[True]*4)]
            for mutate in mutations:
                with self.subTest(world=world,offset=offset,mutation=mutate):
                    rows=copy.deepcopy(topology(world,offset));mutate(rows)
                    with self.assertRaises(AssertionError):validation.validate_topology_rows(rows)

if __name__=='__main__':unittest.main()
