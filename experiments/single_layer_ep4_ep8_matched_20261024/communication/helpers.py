"""CPU-only geometry, reproducible routing, and aligned rank aggregation."""
import hashlib
import json
import math
import random
import statistics

HIDDEN, EXPERTS, TOPK, CAPACITY = 4096, 192, 8, 512
DEFAULT_SEED = 20261004


def cases(world):
    if world not in (4, 8):
        raise ValueError('EP4 or EP8 required')
    return [('decode',world,'eager'),('prefill',world*128,'eager')]


def expert_choices(token, seed=DEFAULT_SEED):
    # One choice per global EP8 owner, two choices per EP4 owner; same routes
    # for every physical placement and world. All 192 experts active at T>=24.
    return [slot*(EXPERTS//8)+token%(EXPERTS//8) for slot in range(TOPK)]


def layout(world, total, seed=DEFAULT_SEED, local_ranks=4):
    if world not in (4, 8) or total <= 0 or total % (world // 4):
        raise ValueError('Invalid complete-node layout')
    assert local_ranks in (2,4) and world%local_ranks==0
    dp = world // local_ranks
    real_per_dp = total // dp
    local = math.ceil(real_per_dp / local_ranks)
    if local > CAPACITY:
        raise ValueError('Capacity exceeded')
    ids, masks, routes = [], [], []
    for rank in range(world):
        rank_ids, mask, route = [], [], []
        for i in range(local):
            pos = (rank % local_ranks) * local + i
            real = pos < real_per_dp
            token = (rank // local_ranks) * real_per_dp + pos if real else total + rank * local + i
            rank_ids.append(token)
            mask.append(real)
            route.append(expert_choices(token, seed) if real else [-1] * TOPK)
        ids.append(rank_ids)
        masks.append(mask)
        routes.append(route)
    real_ids = [t for rr, mm in zip(ids, masks) for t, ok in zip(rr, mm) if ok]
    if real_ids != list(range(total)):
        raise ValueError('Lost or duplicated logical token')
    counts = [0] * EXPERTS
    real_routes = []
    for rr, mm in zip(routes, masks):
        for choices, real in zip(rr, mm):
            if real:
                real_routes.append(choices)
            for expert in choices:
                if expert >= 0:
                    counts[expert] += 1
    digest = hashlib.sha256(json.dumps(real_routes, separators=(',', ':')).encode()).hexdigest()
    return dict(world=world, global_tokens=total, local_tokens=local,
                real_tokens_per_dp=real_per_dp, padded_tokens_per_dp=local * local_ranks,
                actual_input_tokens=world * local, ids_by_rank=ids,
                real_masks_by_rank=masks, routes_by_rank=routes,
                global_expert_counts=counts, global_route_sha256=digest,
                actual_routed_tokens=sum(counts) // TOPK, route_padding=False,
                local_experts=EXPERTS // world, routing_seed=seed)

def phase_definitions(nodes=None):
    from common import NODES
    NODES=tuple(nodes or NODES)
    definitions=[('local065',[(NODES[0],g) for g in range(4)]),
        ('local167',[(NODES[1],g) for g in range(4)]),
        ('cross8',[(n,g) for n in NODES for g in range(4)])]
    return [dict(id=name,world=len(rows),nodes=list(dict.fromkeys(n for n,g in rows)),
        placement=[dict(rank=r,node=n,gpu=g) for r,(n,g) in enumerate(rows)],
        local_ranks=len(rows)//len(set(n for n,g in rows)),capacity=128,pair_only=True,port=36771+i)
        for i,(name,rows) in enumerate(definitions)]

def validate_placement(rows):
    assert len(rows) in (4,8) and [r['rank'] for r in rows]==list(range(len(rows)))
    assert len({(r['node'],r['gpu']) for r in rows})==len(rows)
    counts=[sum(r['node']==node for r in rows) for node in dict.fromkeys(r['node'] for r in rows)]
    assert len(set(counts))==1 and counts[0] in (2,4)
    for rank in range(0,len(rows),2):
        assert rows[rank]['node']==rows[rank+1]['node'] and rows[rank]['gpu']^1==rows[rank+1]['gpu']

def validate_config(cfg):
    assert cfg['capacity']==512 and cfg['iterations']==3 and cfg['warmup']==3
    assert cfg['model_loaded'] is False and cfg['expert_compute'] is False
    for phase in cfg['phases']:
        validate_placement(phase['placement'])
        assert phase['experts'] in (160,192) and phase['hidden'] in (4096,5120)
    return cfg


def validate_hosts(hosts, world):
    if world not in (4, 8) or len(hosts) != world:
        raise ValueError('Incorrect host list')
    groups = [hosts[i:i + 4] for i in range(0, world, 4)]
    if any(len(set(group)) != 1 for group in groups):
        raise ValueError('Each four ranks must be on one node')
    if len({g[0] for g in groups}) != len(groups):
        raise ValueError('EP8 must span two distinct nodes')
    return [g[0] for g in groups]


def expected_source_ids(shape, source, expert):
    return [i for i, choices in enumerate(shape['routes_by_rank'][source]) if expert in choices]


def stats(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Nonempty finite timing list required')
    return dict(n=len(values), median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                min_ms=min(values), max_ms=max(values))


def aggregate(samples_by_rank, metric):
    if not samples_by_rank or not samples_by_rank[0]:
        raise ValueError('Missing samples')
    n = len(samples_by_rank[0])
    if any(len(rows) != n for rows in samples_by_rank):
        raise ValueError('Unequal sample counts')
    maxima, ranks = [], []
    for i in range(n):
        if any(row[i]['iteration'] != i for row in samples_by_rank):
            raise ValueError('Unaligned iteration IDs')
        values = [rows[i][metric] for rows in samples_by_rank]
        stats(values)
        maxima.append(max(values))
        ranks.append(values.index(max(values)))
    return dict(**stats(maxima), per_iteration_rank_max_ms=maxima, critical_ranks=ranks,
                aggregation='max across aligned ranks per iteration, then descriptive median/mean')
