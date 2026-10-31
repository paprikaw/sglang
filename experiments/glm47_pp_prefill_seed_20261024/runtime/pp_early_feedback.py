"""Opt-in PP2 local-tail feedback: submit output before local retirement.

Preserves the pinned exchange/retirement body, slot ancestry, payload ownership,
and P2PWork lifetime. No new GPU operation or inference math is introduced.
"""
from dataclasses import dataclass

@dataclass
class Submitted:
    signature: tuple
    works: object
    consumed: bool = False


def attach(mixin, *, can_skip):
    if hasattr(mixin, '_pp_early_feedback_originals'):
        raise RuntimeError('Early feedback already installed')
    if not hasattr(mixin, '_pp_local_feedback_originals'):
        raise RuntimeError('Early feedback requires qualified local-tail feedback')
    old_init=mixin.init_pp_loop_state
    old_send=mixin._pp_send_output_to_next_stage
    old_exchange=mixin._pp_send_recv_and_preprocess_output_tensors

    def init(self,*args,**kwargs):
        result=old_init(self,*args,**kwargs)
        if not (self._pp_spec_microbatch and self.ps.pp_size==2 and self.pp_loop_size==2):
            raise RuntimeError('Early feedback requires PP2/MTP two slots')
        self._pp_early_submitted=None
        self._pp_early_sent=0
        self._pp_early_consumed=0
        return result

    def signature(slot,mbs,queue,outputs):
        return (slot,id(mbs),id(queue),id(outputs))

    def submit(self,*args):
        # Exposed as an ordinary method so the unfenced diagnostic can observe
        # actual submit CPU time. The existing sender owns all completion work.
        return old_send(self,*args)

    def send(self,slot,mbs,queue,outputs):
        pending=self._pp_early_submitted
        if pending is None:
            return old_send(self,slot,mbs,queue,outputs)
        if pending.signature!=signature(slot,mbs,queue,outputs) or pending.consumed:
            raise RuntimeError('Early feedback duplicate or wrong slot ancestry')
        pending.consumed=True
        self._pp_early_consumed+=1
        return pending.works

    def exchange(self,slot,next_slot,mbs,metadata,queue,outputs):
        if not self.pp_group.is_last_rank:
            return old_exchange(self,slot,next_slot,mbs,metadata,queue,outputs)
        target=mbs[slot]
        if target is None or target.forward_mode.is_prebuilt() or can_skip(target):
            return old_exchange(self,slot,next_slot,mbs,metadata,queue,outputs)
        if self._pp_requires_immediate_output_relay() or self._pp_early_submitted is not None:
            raise RuntimeError('Unsupported or reentrant early feedback exchange')
        works=self._pp_early_submit_feedback(slot,mbs,queue,outputs)
        pending=Submitted(signature(slot,mbs,queue,outputs),works)
        self._pp_early_submitted=pending
        self._pp_early_sent+=1
        try:
            result=old_exchange(self,slot,next_slot,mbs,metadata,queue,outputs)
            if not pending.consumed:
                raise RuntimeError('Pinned exchange failed to consume submitted feedback')
            return result
        finally:
            self._pp_early_submitted=None

    mixin.init_pp_loop_state=init
    mixin._pp_early_submit_feedback=submit
    mixin._pp_send_output_to_next_stage=send
    mixin._pp_send_recv_and_preprocess_output_tensors=exchange
    mixin._pp_early_feedback_originals=(old_init,old_send,old_exchange)


def receipt(scheduler):
    if not hasattr(scheduler,'_pp_early_sent'):
        return dict(enabled=False)
    if scheduler._pp_early_submitted is not None or scheduler._pp_early_sent!=scheduler._pp_early_consumed:
        raise RuntimeError('Early feedback did not drain')
    return dict(enabled=True,tail=bool(scheduler.pp_group.is_last_rank),
                sent=scheduler._pp_early_sent,consumed=scheduler._pp_early_consumed,
                pending=0,no_extra_collectives=True,no_new_gpu_fences=True)


def install():
    import os
    from pathlib import Path
    from common import config
    cfg=config(Path(os.environ['GLM_RUN_DIR']))
    if not cfg.get('early_tail_feedback'):return
    if not (cfg['layout']=='pp2ep4' and cfg['mtp'] and cfg.get('local_tail_feedback')):
        raise RuntimeError('Early feedback outside qualified local-tail PP2/MTP arm')
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin, _pp_can_skip_output_comm
    attach(SchedulerPPMixin,can_skip=_pp_can_skip_output_comm)


def validate_receipts(rows, *, require_work=False):
    if sorted(row['rank'] for row in rows)!=list(range(8)):
        raise RuntimeError('Early feedback needs every worker receipt')
    for row in rows:
        item=row.get('early_tail_feedback',{})
        tail=row['rank']>=4
        if (item.get('enabled') is not True or item.get('tail') is not tail
            or item.get('pending')!=0 or item.get('sent')!=item.get('consumed')
            or item.get('no_extra_collectives') is not True or item.get('no_new_gpu_fences') is not True):
            raise RuntimeError('Early feedback activation/drain proof mismatch')
        if (not tail and item['sent']!=0) or (tail and require_work and item['sent']<=0):
            raise RuntimeError('Early feedback worker role/activity mismatch')
    return dict(passed=True,all_eight_workers=True,payload_sent_once=True)
