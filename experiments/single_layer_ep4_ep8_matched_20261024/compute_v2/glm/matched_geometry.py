from communication_geometry import TOKENS, layout as routed_layout

def point(model, total, cohort):
    experts,hidden=(160,5120) if model=='glm' else (192,4096)
    world=8 if cohort=='whole' else 4
    tokens=total//2 if cohort.startswith('half') else total
    slot=int(cohort[-1]) if cohort.startswith('half') else 0
    offset=slot*tokens if cohort.startswith('half') else 0
    shape=routed_layout(world,tokens,experts,offset,1024)
    resident=experts//world
    ids=[[] for _ in range(resident)]
    for source,row in zip(shape['ids'],shape['routes']):
        for token,route in zip(source,row):
            for expert in route:
                if expert<resident:ids[expert].append(token)
    counts=shape['counts'][:resident]
    assert list(map(len,ids))==counts
    assert sum(counts)==total*(2 if cohort=='equal_call' else 1)
    return dict(layout='ep8' if world==8 else 'pp2_ep4' if cohort.startswith('half') else 'ep4_equal',
        cohort=cohort,pp_slot=slot,global_logical_tokens=total,global_unique_tokens=total,
        microbatch_logical_tokens=tokens,ep_size=world,local_experts=resident,
        active_experts=sum(n>0 for n in counts),counts=counts,local_assignments=sum(counts),
        local_token_ids=ids,source_rows_per_ep_rank=tokens//world,
        attention_tp_physical_copies=1,expected_m=(tokens*8+experts)//experts,
        route_sha256=shape['route_sha256'],hidden_size=hidden,
        gemm_flops=6*hidden*1536*sum(counts))
