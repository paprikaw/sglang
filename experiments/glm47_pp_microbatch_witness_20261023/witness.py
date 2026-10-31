"""Untimed first-divergence witness for PP2/EAGLE topk1 CUDA correctness.

Hooks only: no model/engine edits, launch, sampling override or synthetic KV.
The forward path clones owned GPU snapshots on the producer stream; D2H and
serialization happen only when Scheduler.is_fully_idle() is true. Do not use
these instrumented runs as performance measurements.
"""
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os
from pathlib import Path

_FRAME=ContextVar('pp_eagle_witness_frame',default=None)
_STATE=None
MAX_VERIFIES_PER_COHORT_SLOT=16
MAX_RANK_GPU_BYTES=768*1024**2
TENSOR_FIELDS=('seq_lens','seq_lens_cpu','input_ids','out_cache_loc','req_pool_indices')
VERIFY_FIELDS=('draft_token','positions','custom_mask','retrieve_index','retrieve_next_token','retrieve_next_sibling')


def digest_tokens(tokens):
    return hashlib.sha256(json.dumps([int(x) for x in tokens],separators=(',',':')).encode()).hexdigest()


def request_identity(req):
    prompt=list(req.origin_input_ids);output=list(req.output_ids)
    rid=req.rid
    return dict(rid=rid,request_index=int(rid.rsplit('-',1)[1]),
        origin_tokens=len(prompt),origin_sha256=digest_tokens(prompt),
        committed_output_ids=output,committed_history_sha256=digest_tokens(prompt+output),
        history_boundary='CPU Req state observed at target sampling; verify chain and GPU seq_lens are separately snapshotted')


def _clone(value):
    import torch
    return value.detach().clone() if isinstance(value,torch.Tensor) else value


def _tensor_bytes(value):
    import torch
    return value.numel()*value.element_size() if isinstance(value,torch.Tensor) and value.is_cuda else 0


def _cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.cpu()
    if isinstance(value,dict):return {key:_cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [_cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(item) for item in value)
    return value


def install(out_dir,*,cohort_prefixes=('timed-','repeat-'),limit=MAX_VERIFIES_PER_COHORT_SLOT):
    """Install before Scheduler construction. All four CUDA tail TP ranks record.

    At most 16 verify snapshots per (cohort, PP slot); idle flush frees GPU
    snapshots between cohorts. In the current B4×3×vocab151k FP32 case, 32 full
    logits snapshots occupy approximately 232 MB per tail GPU, plus metadata.
    """
    global _STATE
    if _STATE is not None:raise RuntimeError('Witness already installed')
    if not 1<=limit<=MAX_VERIFIES_PER_COHORT_SLOT:raise ValueError('Witness limit must be 1..16')
    if tuple(cohort_prefixes)!=('timed-','repeat-'):raise ValueError('Only two bounded diagnostic cohorts admitted')
    _STATE=dict(out_dir=Path(out_dir),counts={},pending=[],gpu_bytes=0,flush_id=0)
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative import eagle_worker_common
    from sglang.srt.runtime_context import get_parallel
    original_run=Scheduler.run_batch
    @wraps(original_run)
    def run(self,batch,*args,**kwargs):
        p=get_parallel();ids=[req.rid for req in batch.reqs]
        name=ids[0].rsplit('-',1)[0] if ids else ''
        selected=(p.pp_size==2 and p.pp_rank==1 and ids
                  and all(rid.rsplit('-',1)[0]==name for rid in ids)
                  and name.startswith(cohort_prefixes))
        if not selected:return original_run(self,batch,*args,**kwargs)
        frame=dict(rank=p.world_rank,tp_rank=p.tp_rank,slot=int(self._pp_microstep%self.pp_loop_size),
                   microstep=int(self._pp_microstep),cohort=name,
                   requests=[request_identity(req) for req in batch.reqs])
        token=_FRAME.set(frame)
        try:return original_run(self,batch,*args,**kwargs)
        finally:_FRAME.reset(token)
    Scheduler.run_batch=run
    original_sample=eagle_worker_common.eagle_sample
    @wraps(original_sample)
    def sample(verify_input,batch,logits_output,grammar_mask=None):
        frame=_FRAME.get()
        if frame is None:return original_sample(verify_input,batch,logits_output,grammar_mask)
        key=(frame['cohort'],frame['slot'])
        count=_STATE['counts'].get(key,0)
        if count>=limit:return original_sample(verify_input,batch,logits_output,grammar_mask)
        si=batch.sampling_info
        if (not si.is_all_greedy or grammar_mask is not None
            or any(getattr(si,name,None) is not None for name in
                   ('acc_additive_penalties','acc_scaling_penalties','logit_bias'))
            or verify_input.tree_topk!=1 or verify_input.draft_token_num!=3):
            raise RuntimeError('Witness requires unpenalized greedy topk1 three-node verification')
        pre=dict(batch={name:_clone(getattr(batch,name,None)) for name in TENSOR_FIELDS},
                 verify={name:_clone(getattr(verify_input,name,None)) for name in VERIFY_FIELDS},
                 target_logits=_clone(logits_output.next_token_logits))
        # Native sampling may mutate its input logits. This scoped diagnostic
        # only admits no penalties/masks, so pre-argmax logits have same values.
        result=original_sample(verify_input,batch,logits_output,grammar_mask)
        post={name:_clone(value) for name,value in zip(('predict','accept_lens','accept_index'),result)}
        added=sum(_tensor_bytes(v) for v in pre['batch'].values())+sum(_tensor_bytes(v) for v in pre['verify'].values())
        added+=_tensor_bytes(pre['target_logits'])+sum(_tensor_bytes(v) for v in post.values())
        if _STATE['gpu_bytes']+added>MAX_RANK_GPU_BYTES:raise RuntimeError('Bounded witness GPU budget exceeded')
        _STATE['counts'][key]=count+1;_STATE['gpu_bytes']+=added
        _STATE['pending'].append(dict(**frame,verify_ordinal=count,pre=pre,post=post,
            tree_topk=1,draft_token_num=3,all_greedy=True,
            native_cuda_greedy_acceptance_broadcast=False))
        return result
    eagle_worker_common.eagle_sample=sample
    original_idle=Scheduler.on_idle
    @wraps(original_idle)
    def idle(self,*args,**kwargs):
        result=original_idle(self,*args,**kwargs)
        if _STATE['pending'] and self.is_fully_idle():flush_idle(self)
        return result
    Scheduler.on_idle=idle


def flush_idle(scheduler):
    """Only place that performs device synchronization, D2H and binary writes."""
    import torch
    if not scheduler.is_fully_idle():raise RuntimeError('Witness D2H before full idle forbidden')
    if not _STATE['pending']:return
    torch.cuda.synchronize()
    rows=_cpu_tree(_STATE['pending']);rank=rows[0]['rank']
    out=_STATE['out_dir'];out.mkdir(parents=True,exist_ok=True)
    path=out/f'rank{rank}_flush{_STATE["flush_id"]:03d}.pt'
    payload=dict(schema='pp-eagle-first-divergence-witness-v1',rank=rank,rows=rows,
        synchronized_only_at_idle=True,performance_measurement=False,
        exact_gpu_kv_values_captured=False,source='pre_argmax_and_post_acceptance_owned_snapshots')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as handle:
        torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
    _STATE['pending'].clear();_STATE['gpu_bytes']=0;_STATE['flush_id']+=1


def chain_reference(snapshot):
    """Independent CPU topk1 chain acceptance from recorded target logits."""
    import torch
    logits=snapshot['pre']['target_logits'];n=snapshot['draft_token_num']
    candidates=snapshot['pre']['verify']['draft_token'].reshape(-1,n)
    top=logits.argmax(dim=-1).reshape(-1,n)
    lens=[];outputs=[]
    for candidate,prediction in zip(candidates,top):
        accepted=1
        for depth in range(1,n):
            if int(candidate[depth])!=int(prediction[depth-1]):break
            accepted+=1
        lens.append(accepted);outputs.append([int(x) for x in prediction[:accepted]])
    recorded_lens=snapshot['post']['accept_lens'].tolist()
    predict=snapshot['post']['predict'].flatten();indices=snapshot['post']['accept_index']
    actual=[]
    for row,length in zip(indices,recorded_lens):
        chosen=row[:length]
        if bool(((chosen<0)|(chosen>=predict.numel())).any()):
            return dict(passed=False,reason='accept_index_out_of_bounds')
        actual.append([int(x) for x in predict[chosen]])
    return dict(passed=lens==recorded_lens and outputs==actual,
        expected_lens=lens,recorded_lens=recorded_lens,expected_tokens=outputs,recorded_tokens=actual)


def _exact(a,b):
    import torch
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    return a is b if a is None or b is None else a==b


def compare_snapshot(left,right):
    """Classify observations without asserting that logit drift is harmless."""
    import torch
    lh=[(r['request_index'],r['committed_history_sha256']) for r in left['requests']]
    rh=[(r['request_index'],r['committed_history_sha256']) for r in right['requests']]
    if lh!=rh:return dict(classification='different_incoming_histories',comparable_logits=False)
    # Physical cache addresses may legitimately change between cohorts. Compare
    # logical inputs separately and report allocator differences as nonsemantic.
    logical=('seq_lens','seq_lens_cpu','input_ids')
    differences=['batch.'+k for k in logical if not _exact(left['pre']['batch'][k],right['pre']['batch'][k])]
    differences+=['verify.'+k for k in VERIFY_FIELDS if not _exact(left['pre']['verify'][k],right['pre']['verify'][k])]
    if differences:return dict(classification='different_verify_metadata',fields=differences,comparable_logits=False)
    physical=[k for k in ('out_cache_loc','req_pool_indices')
              if not _exact(left['pre']['batch'][k],right['pre']['batch'][k])]
    a,b=left['pre']['target_logits'],right['pre']['target_logits']
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(classification='logit_shape_or_dtype_mismatch',comparable_logits=False)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(classification='nonfinite_target_logits',comparable_logits=True)
    delta=float((a-b).abs().max());ta=a.topk(2,dim=-1);tb=b.topk(2,dim=-1)
    margin_a=ta.values[:,0]-ta.values[:,1];margin_b=tb.values[:,0]-tb.values[:,1]
    changed=ta.indices[:,0]!=tb.indices[:,0]
    checks=[chain_reference(s) for s in (left,right)]
    post_equal=all(_exact(left['post'][k],right['post'][k]) for k in ('predict','accept_lens','accept_index'))
    classification=('recorded_greedy_acceptance_inconsistent' if not all(c['passed'] for c in checks)
                    else 'same_logits_different_acceptance' if delta==0 and not post_equal
                    else 'target_logit_drift_changes_argmax' if bool(changed.any())
                    else 'target_logit_drift_same_argmax' if delta else 'identical_logit_and_acceptance_witness')
    return dict(classification=classification,comparable_logits=True,max_abs_logit_delta=delta,
        argmax_changed_rows=changed.nonzero().flatten().tolist(),min_top12_margin_left=float(margin_a.min()),
        min_top12_margin_right=float(margin_b.min()),physical_allocator_fields_changed=physical,
        native_acceptance_matches_cpu_reference=[c['passed'] for c in checks],post_acceptance_exact=post_equal,
        exact_gpu_kv_values_compared=False,benign_numerical_branching_proven=False,
        boundary='Same observed logical history/metadata; excludes actual KV-value equality and independent target-model correctness')


def history_key(snapshot):
    """No wall-clock/verify-round alignment: pair ordered RIDs by token history."""
    return tuple((row['request_index'],row['committed_history_sha256']) for row in snapshot['requests'])


def compare_cohorts(left,right):
    """Exact common incoming histories only; report unmatched continuations."""
    index={}
    for row in right:
        key=history_key(row)
        if key in index:raise ValueError('Ambiguous duplicate incoming-history snapshot')
        index[key]=row
    matched=[];missing=[];used=set()
    for row in left:
        key=history_key(row)
        if key not in index:
            missing.append(dict(slot=row['slot'],verify_ordinal=row['verify_ordinal'],history=key));continue
        used.add(key)
        matched.append(dict(left_slot=row['slot'],right_slot=index[key]['slot'],
            left_ordinal=row['verify_ordinal'],right_ordinal=index[key]['verify_ordinal'],
            requests=[dict(request_index=i,committed_history_sha256=digest) for i,digest in key],
            result=compare_snapshot(row,index[key])))
    return dict(matched=matched,unmatched_left=missing,
        unmatched_right=[dict(slot=row['slot'],verify_ordinal=row['verify_ordinal'],history=key)
                         for key,row in index.items() if key not in used],
        first_comparable_nonidentical=next((r for r in matched if r['result']['classification']!='identical_logit_and_acceptance_witness'),None),
        performance_measurement=False,actual_kv_equality_not_proven=True)


def _request_tensor(value,index,batch_size,width):
    import torch
    if value is None:return None
    if not isinstance(value,torch.Tensor):return value
    if value.ndim and value.shape[0]==batch_size:return value[index]
    if value.numel()==batch_size*width:return value.reshape(batch_size,width)[index]
    if value.numel()==batch_size:return value.reshape(batch_size)[index]
    raise ValueError('Tensor has no unambiguous request-local layout')


def compare_requests(left,right):
    """Match each RID independently so one divergent request cannot hide others.

    If a RID's verify boundary shifts, only histories actually observed in both
    cohorts are comparable. No interpolation or forced continuation is claimed.
    """
    import torch
    index={}
    for snapshot in right:
        for row,req in enumerate(snapshot['requests']):
            key=(req['request_index'],req['committed_history_sha256'])
            if key in index:raise ValueError('Ambiguous duplicate per-RID incoming history')
            index[key]=(snapshot,row)
    rows=[];missing=[];used=set()
    for a in left:
        for ai,req in enumerate(a['requests']):
            key=(req['request_index'],req['committed_history_sha256'])
            if key not in index:
                missing.append(dict(request_index=key[0],history=key[1]));continue
            b,bi=index[key];used.add(key)
            abs_=len(a['requests']);bbs=len(b['requests']);width=a['draft_token_num']
            differing=[]
            for section,names in [('batch',('seq_lens','seq_lens_cpu','input_ids')),
                    ('verify',('draft_token','positions','retrieve_index','retrieve_next_token','retrieve_next_sibling'))]:
                for name in names:
                    x=_request_tensor(a['pre'][section][name],ai,abs_,width)
                    y=_request_tensor(b['pre'][section][name],bi,bbs,width)
                    if not _exact(x,y):differing.append(section+'.'+name)
            x=a['pre']['target_logits'].reshape(abs_,width,-1)[ai]
            y=b['pre']['target_logits'].reshape(bbs,width,-1)[bi]
            finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            delta=float((x-y).abs().max()) if finite else None
            changed=(x.argmax(-1)!=y.argmax(-1)).nonzero().flatten().tolist() if finite else None
            ar,br=chain_reference(a),chain_reference(b)
            consistent=ar['passed'] and br['passed']
            output_equal=ar.get('recorded_tokens',[[]]*abs_)[ai]==br.get('recorded_tokens',[[]]*bbs)[bi]
            shared_mask_equal=_exact(a['pre']['verify']['custom_mask'],b['pre']['verify']['custom_mask'])
            kind=('different_request_verify_metadata' if differing else
                  'recorded_greedy_acceptance_inconsistent' if not consistent else
                  'nonfinite_target_logits' if not finite else
                  'same_logits_different_accepted_tokens' if delta==0 and not output_equal else
                  'target_logit_drift_changes_argmax' if changed else
                  'target_logit_drift_same_argmax' if delta else 'identical_request_logit_and_acceptance_witness')
            rows.append(dict(request_index=key[0],committed_history_sha256=key[1],
                left_slot=a['slot'],right_slot=b['slot'],left_ordinal=a['verify_ordinal'],right_ordinal=b['verify_ordinal'],
                classification=kind,differing_request_metadata=differing,whole_batch_custom_mask_equal=shared_mask_equal,
                identical_entire_batch_history=history_key(a)==history_key(b),
                max_abs_logit_delta=delta,argmax_changed_tree_rows=changed,
                native_acceptance_cpu_reference_passed=consistent,accepted_tokens_equal=output_equal,
                actual_kv_values_equal_not_proven=True,benign_numerical_branching_proven=False))
    return dict(matched_requests=rows,unmatched_left=missing,
        unmatched_right=[dict(request_index=k[0],history=k[1]) for k in index if k not in used],
        exact_logit_inputs_not_proven_without_actual_kv_values=True,performance_measurement=False)
