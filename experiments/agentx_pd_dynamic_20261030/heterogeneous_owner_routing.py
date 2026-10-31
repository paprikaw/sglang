"""Static sticky placement over Attention-DP owners of unequal EP instances.

The original router is retained byte-for-byte for uniform instance pools. This
only corrects initial placement weights; it does not predict future requests,
change phase timing, migrate state, or implement a dynamic routing policy.
"""
import math


def owner_lane(sid, lane, owners, family_width, copies):
    if copies <= 1 or not family_width or family_width % owners == 0:
        return lane
    copy_id = int(sid.split(':copy', 1)[1].split(':', 1)[0])
    arrival_index = lane - copy_id
    assert arrival_index >= 0
    stride = next(s for s in range(1, owners + 1)
                  if math.gcd(family_width + s, owners) == 1)
    return arrival_index + (arrival_index // family_width) * stride


def install(router_class):
    original = router_class.pick

    def pick(self, sid, turn, lane):
        if self.policy != 'sticky_hash' or len({e['ep'] for e in self.endpoints}) <= 1:
            return original(self, sid, turn, lane)
        # Split larger instances into equal GPU blocks, then interleave ranks
        # within those blocks. EP2+EP4 therefore keeps the same physical owner
        # order as three EP2 instances on the declared 2P6D topology. Each GPU
        # appears once; same-node/cross-node first placements remain matched.
        unit = math.gcd(*(e['ep'] for e in self.endpoints))
        blocks = [(e, offset) for e in self.endpoints
                  for offset in range(0, e['ep'], unit)]
        owners = [(e, offset + rank) for rank in range(unit) for e, offset in blocks]
        lane = owner_lane(sid, lane, len(owners), self.family_width, self.copies)
        chosen, rank = owners[lane % len(owners)]
        old = self.affinity.get(sid)
        switched = old is not None and old != chosen['id']
        self.affinity[sid] = chosen['id']
        self.active[chosen['id']] += 1
        self.pending_output[chosen['id']] += turn['output_tokens']
        return dict(chosen, rank=rank, clone_pool_lane=lane), switched

    router_class.pick = pick
    return original
