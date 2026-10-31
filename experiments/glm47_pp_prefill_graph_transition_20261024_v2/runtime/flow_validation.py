"""CPU-only coverage gate; no inference or GPU imports."""
def validate_all(traces, layout):
    if len(traces)!=8 or sorted(t['rank'] for t in traces)!=list(range(8)):
        raise RuntimeError('Expected eight unique GPU ranks')
    details=[]
    for trace in traces:
        if (not trace.get('passed') or not trace.get('all_model_forwards') or trace.get('per_replay_fences')!=0
            or trace.get('per_forward_fences')!=0 or trace.get('per_forward_timestamp_reads')!=0):
            raise RuntimeError('Missing unfenced all-forward coverage')
        prefill=[r for r in trace['forwards'] if r['phase']=='target_prefill']
        verify=[r for r in trace['forwards'] if r['phase']=='target_verify']
        expected=2 if layout=='pp2ep4' else 1
        if len(prefill)!=expected or len(verify)<12:
            raise RuntimeError('Initial Prefill / Verify coverage differs')
        if any(r['batch_size']!=4 or len(r['request_ids'])!=4 for r in prefill):
            raise RuntimeError('Initial local B4 missing')
        if any(r.get('input_shape') not in ([260],None) for r in prefill):
            raise RuntimeError('Initial 65 tokens/request shape differs')
        if len({rid for r in prefill for rid in r['request_ids']})!=4*expected:
            raise RuntimeError('Prefill slot request ownership overlaps')
        for kind in ('replays','forwards','batches'):
            if [r['index'] for r in trace[kind]]!=list(range(len(trace[kind]))):
                raise RuntimeError('Trace event slots overwritten')
            if any(r['end_ms']<r['start_ms'] or r['elapsed_ms']<0 for r in trace[kind]):
                raise RuntimeError('Invalid local GPU interval')
        details.append(dict(rank=trace['rank'],prefill_count=len(prefill),verify_count=len(verify)))
    return dict(passed=True,layout=layout,ranks=details,cupti=False,no_per_forward_fences=True,
                no_gpu_idle_fraction_claim=True,nested_spans_not_additive=True)


def validate_cpu_bank_placement(rows,layout):
    if layout not in ('ep8','pp2ep4'):
        raise ValueError('Unsupported flow layout')
    for row in rows:
        actual=row['meta'].get('dp_rank')
        valid=(actual==row['request_index']//4) if layout=='ep8' else (actual in (None,0))
        if not valid:
            raise RuntimeError('Flow request DP placement differs from CPU Bank ownership')
    return dict(passed=True,layout=layout,requests=len(rows))
