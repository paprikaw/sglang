"""Strict, CPU-only acceptance of the all-rank actual-replay receipts."""
def validate_graph_receipts(receipts, layout, prefix=81920, outputs=128):
    if layout not in ('ep8', 'pp2ep4') or len(receipts) != 8:
        raise RuntimeError('Expected a complete eight-rank graph proof')
    reports=[]
    for expected_rank, receipt in enumerate(receipts):
        if receipt.get('rank') != expected_rank or not receipt.get('passed'):
            raise RuntimeError('Incomplete or misordered rank proof')
        graph=receipt.get('graph', {})
        if not graph.get('installed') or not graph.get('passed'):
            raise RuntimeError('Actual graph telemetry failed at rank '+str(expected_rank))
        captures=graph.get('capture_configurations', [])
        if not captures or any(c.get('capture_bs') != [4] or c.get('max_bs') != 4
                               or c.get('backend_type') != 'FullCudaGraphBackend'
                               or c.get('captured_graph_sizes') != [4]
                               or not c.get('captured_graph_count', 0) for c in captures):
            raise RuntimeError('Different graph capture identity')
        cohort_id=f'timed-decode-{prefix}'
        cohort=graph.get('cohorts', {}).get(cohort_id, {})
        expected=(outputs-1)*(2 if layout=='pp2ep4' else 1)
        for field in ('decode_forwards','model_graph_true','execute_successes',
                      'backend_replay_successes','verified_local_b4_replays'):
            if cohort.get(field) != expected:
                raise RuntimeError(f'Rank{expected_rank} {field}: {cohort.get(field)} != {expected}')
        if (not cohort.get('passed') or cohort.get('model_graph_false') or
            cohort.get('exceptions') or cohort.get('violations') or
            cohort.get('batch_sizes') != {'4':expected} or
            cohort.get('replay_key_sizes') != {'4':expected}):
            raise RuntimeError('Real Decode silently fell back, padded differently, or failed')
        groups=cohort.get('request_groups', [])
        expected_groups=2 if layout=='pp2ep4' else 1
        if len(groups) != expected_groups:
            raise RuntimeError('Incorrect number of actual B4 groups')
        seen=set()
        for group in groups:
            ids=set(group.get('request_ids', []))
            if (len(ids)!=4 or seen & ids or group.get('forwards')!=outputs-1
                or group.get('verified_replays')!=outputs-1):
                raise RuntimeError('PP slots / DP request groups were not stable B4 groups')
            seen.update(ids)
        reports.append(dict(rank=expected_rank,actual_decode_replays=expected,
                            local_batch_size=4,request_groups=groups))
    return dict(passed=True,local_capture_bs=4,rank_proofs=reports,
                graph_measured_from_actual_replay=True,profiler_enabled=False)
