"""CPU-only common unique-token routing and two comparison boundaries."""
import hashlib
import json
import statistics

TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MODELS = {'glm': (5120, 160), 'hy3': (4096, 192)}


def choices(token, experts):
    return [owner * (experts // 8) + token % (experts // 8) for owner in range(8)]


def cases(world):
    for tokens in TOKENS:
        if world == 8:
            yield dict(id=f'whole_{tokens}', cohort='whole', tokens=tokens,
                       global_batch=tokens, offset=0)
        elif world == 4:
            for slot in (0, 1):
                yield dict(id=f'half{slot}_{tokens}', cohort=f'half{slot}',
                           tokens=tokens // 2, global_batch=tokens,
                           offset=slot * tokens // 2)
            yield dict(id=f'equal_call_{tokens}', cohort='equal_call',
                       tokens=tokens, global_batch=tokens, offset=0)
        else:
            raise ValueError('Only EP4 and EP8 are admitted')


def layout(world, total, experts, offset=0, capacity=1024):
    assert world in (4, 8) and experts in (160, 192)
    assert total > 0 and total % world == 0 and total // world <= capacity
    local = total // world
    ids = [list(range(offset + r * local, offset + (r + 1) * local))
           for r in range(world)]
    routes = [[choices(token, experts) for token in row] for row in ids]
    counts = [0] * experts
    for row in routes:
        for route in row:
            for expert in route:
                counts[expert] += 1
    return dict(world=world, tokens=total, offset=offset, local_tokens=local,
                ids=ids, routes=routes, counts=counts,
                route_sha256=hashlib.sha256(json.dumps(routes, separators=(',', ':')).encode()).hexdigest(),
                owner_assignments=[sum(counts[r*experts//world:(r+1)*experts//world])
                                   for r in range(world)],
                active_experts=sum(c > 0 for c in counts))


def aggregate(samples_by_rank, key):
    assert samples_by_rank and all(len(s) == 3 for s in samples_by_rank)
    maxima, critical = [], []
    for iteration in range(3):
        assert all(s[iteration]['iteration'] == iteration for s in samples_by_rank)
        values = [s[iteration][key] for s in samples_by_rank]
        assert all(v >= 0 for v in values)
        maxima.append(max(values))
        critical.append(values.index(max(values)))
    return dict(median_ms=statistics.median(maxima), mean_ms=statistics.mean(maxima),
                samples=maxima, critical_ranks=critical)


def validate_grid():
    checked = 0
    for experts in (160, 192):
        for tokens in TOKENS:
            whole = layout(8, tokens, experts)
            halves = [layout(4, tokens//2, experts, slot*tokens//2) for slot in (0, 1)]
            equal = layout(4, tokens, experts)
            assert whole['owner_assignments'] == [tokens] * 8
            assert equal['owner_assignments'] == [tokens*2] * 4
            assert all(h['owner_assignments'] == [tokens] * 4 for h in halves)
            assert sum(whole['counts']) == sum(sum(h['counts']) for h in halves) == tokens*8
            assert whole['routes'][:4] == halves[0]['routes']
            assert whole['routes'][4:] == halves[1]['routes']
            assert len({t for row in whole['ids'] for t in row}) == tokens
            checked += 1
    return {'passed': True, 'geometry_cases': checked, 'gpu_execution': False}
