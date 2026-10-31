"""Identical global tokens and Expert routes across EP2 and EP4; CPU only."""
import hashlib
import json

TOKENS = (8, 16, 32, 256, 1024, 2048)


def layout(world, total, experts=160, offset=0, capacity=1024):
    assert world in (2, 4) and experts == 160 and offset == 0
    assert total > 0 and total % world == 0 and total // world <= capacity
    local = total // world
    ids = [list(range(r*local, (r+1)*local)) for r in range(world)]
    flat = [[g*(experts//8)+t%(experts//8) for g in range(8)] for t in range(total)]
    routes = [flat[r*local:(r+1)*local] for r in range(world)]
    counts = [0]*experts
    for route in flat:
        for expert in route:
            counts[expert] += 1
    return dict(world=world, tokens=total, offset=offset, local_tokens=local,
        ids=ids, routes=routes, counts=counts,
        global_route_sha256=hashlib.sha256(json.dumps(flat,separators=(',',':')).encode()).hexdigest(),
        global_token_ids_sha256=hashlib.sha256(json.dumps(list(range(total))).encode()).hexdigest(),
        owner_assignments=[sum(counts[r*experts//world:(r+1)*experts//world]) for r in range(world)],
        global_remote_assignments=total*8*(world-1)//world,
        active_experts=sum(c>0 for c in counts))


def workloads():
    return [dict(id=f'{mode}_tokens{n}', tokens=n, offset=0, global_batch=n,
                 cohort='equal_global_call', cuda_graph=mode=='graph')
            for n in TOKENS for mode in (['eager','graph'] if n<=32 else ['eager'])]


def validate_grid():
    for n in TOKENS:
        a,b=layout(2,n),layout(4,n)
        assert a['global_route_sha256']==b['global_route_sha256']
        assert a['global_token_ids_sha256']==b['global_token_ids_sha256']
        assert a['owner_assignments']==[n*4]*2 and b['owner_assignments']==[n*2]*4
        assert a['counts']==b['counts'] and sum(a['counts'])==n*8
        assert b['global_remote_assignments']*2==a['global_remote_assignments']*3
    assert len(workloads())==9
    return {'passed':True,'identical_global_route_cases':6,'workloads_per_layout':9,
            'remote_assignments_EP4_over_EP2':1.5,'gpu_execution':False}
