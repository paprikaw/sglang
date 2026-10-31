"""Reuse draft-prefill's existing top1 prediction in PP's first Verify tree."""

def attach(mixin, *, seed, check_spec=None):
    if hasattr(mixin,'_pp_prefill_seed_originals'):
        raise RuntimeError('Prefill proposal seed already installed')
    old_init=mixin.init_pp_loop_state
    old_prep=mixin._pp_prep_batch_result
    def init(self,*args,**kwargs):
        result=old_init(self,*args,**kwargs)
        if check_spec is not None:check_spec()
        if not (self._pp_spec_microbatch and self.ps.pp_size==2 and self.pp_loop_size==2):
            raise RuntimeError('Prefill proposal seed needs qualified PP2/two-slot MTP')
        self._pp_prefill_seed_batches=0
        self._pp_prefill_seed_requests=0
        return result
    def prep(self,batch,metadata,outputs,*args,**kwargs):
        # Derive eligibility before the original function clears transient fields.
        eligible=(self._pp_spec_relay and batch.contains_last_prefill_chunk
                  and 'spec_accept_lens' not in outputs.tensors
                  and 'draft_topk_index' in outputs.tensors)
        result=old_prep(self,batch,metadata,outputs,*args,**kwargs)
        if eligible:
            snapshot=metadata.fwd_batch or batch
            rids=[req.rid for req in snapshot.reqs]
            seeded=seed(rids,outputs.tensors['next_token_ids'],outputs.tensors['draft_topk_index'],batch.spec_info)
            # Existing rid-aware merge/reindex semantics; no change to allocation,
            # token acceptance, seq_lens, or request lifecycle.
            self._pp_spec_set_relay(batch,seeded)
            self._pp_prefill_seed_batches+=1
            self._pp_prefill_seed_requests+=len(rids)
        return result
    mixin.init_pp_loop_state=init
    mixin._pp_prep_batch_result=prep
    mixin._pp_prefill_seed_originals=(old_init,old_prep)


def tensor_seed(rids,root,top1,old):
    import torch
    from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput
    if (not isinstance(old,PPSpecRelayInput) or old.rids!=rids or old.tokens.shape!=(len(rids),3)
        or top1.shape!=(len(rids),1) or root.numel()!=len(rids) or old.parents is not None):
        raise RuntimeError('First-proposal seed outside qualified top1/Q3/ancestry')
    tokens=old.tokens.clone()
    tokens[:,1]=top1[:,0].to(dtype=torch.int64,device=tokens.device)
    return PPSpecRelayInput(rids=list(rids),tokens=tokens)


def receipt(scheduler):
    if not hasattr(scheduler,'_pp_prefill_seed_batches'):return dict(enabled=False)
    return dict(enabled=True,batches=scheduler._pp_prefill_seed_batches,
                requests=scheduler._pp_prefill_seed_requests,
                existing_top1_only=True,no_extra_draft_forward=True,no_extra_communication=True)


def validate_receipts(rows,*,require_work=False):
    if sorted(row['rank'] for row in rows)!=list(range(8)):
        raise RuntimeError('Prefill proposal seed needs every worker receipt')
    counts=set()
    for row in rows:
        item=row.get('prefill_first_proposal',{})
        if (item.get('enabled') is not True or item.get('existing_top1_only') is not True
            or item.get('no_extra_draft_forward') is not True or item.get('no_extra_communication') is not True):
            raise RuntimeError('Prefill proposal activation proof missing')
        counts.add((item['batches'],item['requests']))
        if require_work and (item['batches']<=0 or item['requests']<=0):
            raise RuntimeError('No Prefill proposals were adopted')
    if len(counts)!=1:raise RuntimeError('Stages disagree about adopted Prefill proposals')
    return dict(passed=True,workers=8)


def install():
    import os
    from pathlib import Path
    from common import config
    cfg=config(Path(os.environ['GLM_RUN_DIR']))
    if not cfg.get('prefill_first_proposal'):return
    if not (cfg['layout']=='pp2ep4' and cfg['mtp'] and cfg.get('local_tail_feedback')):
        raise RuntimeError('Unsupported Prefill proposal seed arm')
    from sglang.srt.runtime_context import get_spec
    def check_spec():
        spec=get_spec()
        if spec.speculative_eagle_topk!=1 or spec.speculative_num_draft_tokens!=3 or spec.speculative_num_steps!=2:
            raise RuntimeError('Prefill proposal seed only qualifies steps2/top1/Q3')
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    attach(SchedulerPPMixin,seed=tensor_seed,check_spec=check_spec)
