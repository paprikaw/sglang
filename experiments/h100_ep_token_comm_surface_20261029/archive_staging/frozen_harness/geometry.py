"""Equal global original tokens, routes and global chunk boundaries, CPU only."""
import hashlib
import json

TOKENS = tuple(2**i for i in range(3,16))
GLOBAL_CHUNK = 2048
SHAPES = (
    dict(id='glm47', label='GLM-4.7', hidden=5120, experts=160),
    dict(id='hy3', label='HY3', hidden=4096, experts=192),
    dict(id='qwen35b', label='Qwen 35B', hidden=2048, experts=256),
    dict(id='qwen122b', label='Qwen 122B', hidden=3072, experts=256),
)


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def chunks(total, limit=GLOBAL_CHUNK):
    assert total in TOKENS and limit == GLOBAL_CHUNK
    return [(start,min(limit,total-start)) for start in range(0,total,limit)]


def layout(world, total, experts=160, offset=0, capacity=None):
    assert world in (2,4,8) and experts in (160,192,256)
    assert total > 0 and total % world == 0 and offset % world == 0
    capacity = capacity or GLOBAL_CHUNK//world
    assert total//world <= capacity
    local = total//world
    ids = [list(range(offset+r*local, offset+(r+1)*local)) for r in range(world)]
    flat = [[g*(experts//8)+t%(experts//8) for g in range(8)]
            for t in range(offset,offset+total)]
    routes = [flat[r*local:(r+1)*local] for r in range(world)]
    counts = [0]*experts
    source_rows = [[[] for _ in range(world)] for _ in range(experts)]
    for source in range(world):
        for row,choices in enumerate(routes[source]):
            for expert in choices:
                counts[expert] += 1
                source_rows[expert][source].append(row)
    owners = [sum(counts[r*experts//world:(r+1)*experts//world]) for r in range(world)]
    assert owners == [total*8//world]*world
    return dict(world=world,tokens=total,offset=offset,local_tokens=local,
        ids=ids,routes=routes,counts=counts,source_rows=source_rows,
        global_route_sha256=digest(flat),global_token_ids_sha256=digest(list(range(offset,offset+total))),
        owner_assignments=owners,global_remote_assignments=total*8*(world-1)//world,
        active_experts=sum(c>0 for c in counts))


def workloads():
    return [dict(id=f'{mode}_tokens{n}', tokens=n, global_batch=n,
        cohort='fixed_global_unique_tokens_chunk2048', cuda_graph=mode=='graph',
        chunks=[dict(offset=start,tokens=count) for start,count in chunks(n)])
        for n in TOKENS for mode in ('eager','graph')]


def validate_grid():
    cases=0
    for model in SHAPES:
        for n in TOKENS:
            signatures=[]
            for world in (2,4,8):
                parts=[layout(world,count,model['experts'],start) for start,count in chunks(n)]
                signatures.append([(p['global_route_sha256'],p['global_token_ids_sha256']) for p in parts])
                assert sum(p['tokens'] for p in parts)==n
                assert sum(sum(p['counts']) for p in parts)==8*n
                assert [sum(p['owner_assignments'][r] for p in parts) for r in range(world)]==[8*n//world]*world
            assert signatures[0]==signatures[1]==signatures[2]
            cases+=1
    return dict(passed=True,cases=cases,workloads_per_phase=len(workloads()),
        global_original_tokens=list(TOKENS),global_chunk_limit=GLOBAL_CHUNK,
        routing_and_global_chunks_identical_across_EP=True,gpu_execution=False)
