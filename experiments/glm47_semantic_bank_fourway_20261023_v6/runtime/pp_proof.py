"""Actual two-slot PP admission/composition proof; physical slot IDs may swap globally."""
from deterministic_proof import validate as validate_deterministic_receipts
def validate(receipts,mtp,cohort_prefix='timed-'):
    mode='candidate' if mtp else 'baseline'
    if mode not in ('baseline','candidate') or len(receipts)!=8 or sorted(r['rank'] for r in receipts)!=list(range(8)):
        raise RuntimeError('Missing exact eight-rank microbatch proof')
    deterministic=validate_deterministic_receipts(receipts)
    from humming_receipt import validate_humming_kernel_receipt
    humming=[]
    for receipt in receipts:
        value=validate_humming_kernel_receipt(receipt.get('humming',{}))
        humming.append(dict(rank=receipt['rank'],**value))
    composition=[]
    for receipt in receipts:
        proof=receipt.get('microbatch',{})
        if proof.get('enabled')!=(mode=='candidate') or proof.get('loop_size')!=2:
            raise RuntimeError('Microbatch mode/slot proof mismatch')
        if mode=='candidate' and (not proof.get('feedback_group_name') or not proof.get('feedback_group_ranks')):
            raise RuntimeError('Candidate feedback group identity missing')
        batches=[b for b in receipt.get('batches',[]) if b.get('request_ids') and all(rid.startswith(cohort_prefix) for rid in b['request_ids'])]
        if not batches or not any(b.get('batch_size')==4 for b in batches):
            raise RuntimeError('No actual B4 microbatch was observed')
        if any(b.get('batch_size',0)>4 for b in batches):
            raise RuntimeError('B4 slot bound exceeded')
        observed={rid for b in batches for rid in b.get('request_ids',[])}
        if len(observed)!=8:
            raise RuntimeError('Two-slot cohort did not cover eight requests')
        groups=sorted({(b.get('pp_slot_id'),tuple(b.get('request_ids',[]))) for b in batches if b.get('batch_size')==4},key=repr)
        expected_groups={tuple(range(4)),tuple(range(4,8))}
        actual_groups={tuple(int(rid.rsplit('-',1)[1]) for rid in ids) for _,ids in groups}
        if len(groups)!=2 or {slot for slot,_ in groups}!={0,1}:
            raise RuntimeError('Expected exactly one stable B4 group per PP slot')
        for rid in observed:
            if len({b['pp_slot_id'] for b in batches if rid in b['request_ids']})!=1:
                raise RuntimeError('Request migrated between PP slots: '+rid)
        if actual_groups!=expected_groups:
            raise RuntimeError('FCFS ordered 2xB4 membership changed: '+str(actual_groups))
        admissions=[a for a in receipt.get('atomic_admissions',[]) if a['cohort'].startswith(cohort_prefix)]
        if len(admissions)!=1 or admissions[0].get('request_indices')!=list(range(8)):
            raise RuntimeError('Actual scheduler atomic B8 admission proof missing')
        composition.append(dict(rank=receipt['rank'],atomic_admission=admissions[0],
            slot_request_groups=[dict(slot_id=slot,request_ids=list(rids)) for slot,rids in groups]))
    return dict(passed=True,mode=mode,actual_slots=2,max_requests_per_slot=4,
                feedback_group='isolated' if mode=='candidate' else 'legacy',ranks=8,deterministic_runtime=deterministic,
                humming_loaded_kernel_invariants=dict(passed=True,ranks=humming,proves_actual_launch=False),
                actual_slot_request_composition=composition)
