"""Stable, content-validated idle witness receipts; no GPU work in the client."""
import hashlib
import io
import json
from pathlib import Path
import time

TAIL_RANKS=(4,5,6,7)
SCHEMA='pp-eagle-first-divergence-witness-v1'


def read_stable_payload(path,*,expected_sha256=None,expected_size=None):
    import torch
    path=Path(path);before=path.stat();data=path.read_bytes();after=path.stat()
    identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if identity(before)!=identity(after) or len(data)!=after.st_size:
        raise RuntimeError('Witness changed while being read')
    digest=hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise RuntimeError('Witness content hash mismatch')
    if expected_size is not None and len(data)!=expected_size:
        raise RuntimeError('Witness file size mismatch')
    value=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if (value.get('schema')!=SCHEMA or value.get('synchronized_only_at_idle') is not True
            or value.get('performance_measurement') is not False):
        raise RuntimeError('Invalid idle witness payload identity')
    rows=value.get('rows',[]);rank=value.get('rank')
    if rank not in TAIL_RANKS or not rows or any(row.get('rank')!=rank for row in rows):
        raise RuntimeError('Witness rank/rows invalid')
    cohorts=sorted({row['cohort'] for row in rows})
    if len(cohorts)!=1 or not cohorts[0].startswith(('witness-','witness-repeat-')):
        raise RuntimeError('Witness flush mixed or unsupported cohorts')
    indices=sorted({req['request_index'] for row in rows for req in row['requests']})
    slots=sorted({row['slot'] for row in rows})
    first={slot:min(row['verify_ordinal'] for row in rows if row['slot']==slot) for slot in slots}
    if indices!=list(range(8)) or slots!=[0,1] or any(v!=0 for v in first.values()):
        raise RuntimeError('Witness must cover all eight requests and both slot first Verifies')
    if len(rows)>32 or any(row['verify_ordinal']>=16 for row in rows):
        raise RuntimeError('Witness exceeded frozen snapshot bound')
    return dict(path=str(path.resolve()),sha256=digest,size_bytes=len(data),rank=rank,
        cohort=cohorts[0],rows=len(rows),request_indices=indices,slots=slots,
        first_verify_ordinal_per_slot={str(k):v for k,v in first.items()},
        synchronized_only_at_idle=True,parsed_as_weights_only_torch_payload=True,
        performance_measurement=False),value


def install(out_dir):
    import witness
    from common import save
    out=Path(out_dir)
    witness.install(out)
    original=witness.flush_idle
    def flush(scheduler):
        if not scheduler.is_fully_idle():
            raise RuntimeError('Witness publication requires fully drained scheduler')
        count=witness._STATE['flush_id']
        original(scheduler)
        if witness._STATE['flush_id']==count:return
        rank=__import__('sglang.srt.runtime_context',fromlist=['get_parallel']).get_parallel().world_rank
        path=out/f'rank{rank}_flush{count:03d}.pt'
        proof,payload=read_stable_payload(path)
        if proof['rank']!=rank:raise RuntimeError('Idle flush rank changed')
        proof.update(passed=True,flush_id=count,published_after_fsync_and_parse=True,
            engine_forward_readbacks_added=0,engine_forward_synchronizations_added=0)
        save(path.with_suffix('.receipt.json'),proof)
    witness.flush_idle=flush


def validate_receipt(receipt_path,cohort,rank):
    path=Path(receipt_path);before=path.read_bytes();receipt=json.loads(before)
    if path.read_bytes()!=before:raise RuntimeError('Witness receipt changed while being read')
    if (receipt.get('passed') is not True or receipt.get('rank')!=rank
            or receipt.get('cohort')!=cohort or receipt.get('published_after_fsync_and_parse') is not True):
        raise RuntimeError('Witness receipt rank/cohort/completion mismatch')
    payload_path=Path(receipt['path'])
    if payload_path.parent.resolve()!=path.parent.resolve() or payload_path.name!=path.name.replace('.receipt.json','.pt'):
        raise RuntimeError('Witness receipt points outside its owned directory')
    proof,payload=read_stable_payload(payload_path,expected_sha256=receipt['sha256'],expected_size=receipt['size_bytes'])
    if proof['rank']!=rank or proof['cohort']!=cohort:
        raise RuntimeError('Parsed witness payload cohort does not match receipt')
    return dict(receipt_path=str(path.resolve()),receipt_sha256=hashlib.sha256(before).hexdigest(),**proof)


def wait_for_cohort(out_dir,cohort,*,timeout=180,poll_s=.2):
    """Gate the next cold command/shutdown on all four parsed idle flushes."""
    out=Path(out_dir);deadline=time.monotonic()+timeout;validated={};last_errors={}
    while len(validated)<4:
        for rank in TAIL_RANKS:
            if rank in validated:continue
            for path in sorted(out.glob(f'rank{rank}_flush*.receipt.json')):
                try:
                    value=json.loads(path.read_text())
                    if value.get('cohort')!=cohort:continue
                    validated[rank]=validate_receipt(path,cohort,rank);break
                except (OSError,ValueError,RuntimeError,EOFError) as error:
                    last_errors[rank]=str(error)
        if len(validated)==4:break
        if time.monotonic()>=deadline:
            raise TimeoutError('Missing valid idle witness flushes for '+cohort+': '+str(dict(validated_ranks=sorted(validated),errors=last_errors)))
        time.sleep(poll_s)
    return dict(passed=True,cohort=cohort,tail_ranks=list(TAIL_RANKS),
        ranks=[validated[rank] for rank in TAIL_RANKS],all_tail_flushes_stably_parsed_and_hashed=True,
        required_before_next_cold_or_shutdown=True,performance_measurement=False)


def status():
    import witness
    state=witness._STATE
    return dict(installed=state is not None,
        pending_snapshots=len(state['pending']) if state else 0,
        pending_gpu_bytes=state['gpu_bytes'] if state else 0,
        total_selected_snapshots=sum(state['counts'].values()) if state else 0,
        flushed_cohorts=state['flush_id'] if state else 0,
        selected_prefixes=['witness-','witness-repeat-'])


def validate_uninstrumented(receipts):
    if len(receipts)!=8 or sorted(r['rank'] for r in receipts)!=list(range(8)):
        raise RuntimeError('Uninstrumented production proof requires all eight ranks')
    proofs=[]
    for receipt in receipts:
        proof=receipt.get('witness',{})
        if (proof.get('installed') is not True or proof.get('pending_snapshots')!=0
            or proof.get('pending_gpu_bytes')!=0 or proof.get('total_selected_snapshots')!=0
            or proof.get('flushed_cohorts')!=0):
            raise RuntimeError('Witness snapshots contaminated uninstrumented production')
        proofs.append(dict(rank=receipt['rank'],**proof))
    return dict(passed=True,ranks=proofs,witness_clones=0,witness_readbacks=0,witness_flushes=0,
        no_cupti=True,performance_comparison_claim_valid=False)
