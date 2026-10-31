"""Omit PD-only draft fields from non-PD Verify feedback; no math changes."""
REMOVED=('draft_topk_p','draft_topk_index','draft_hidden_states')

def attach(mixin,*,ab=True,pack_payload=None,unpack_payload=None):
    if hasattr(mixin,'_pp_feedback_payload_originals'):
        raise RuntimeError('Feedback payload override already installed')
    old_init=mixin.init_pp_loop_state;old_prepare=mixin._pp_prepare_tensor_dict
    old_recv=getattr(mixin,'_pp_recv_dict_from_prev_stage',None)
    if (pack_payload is None)!=(unpack_payload is None):raise RuntimeError('Codec must be installed on both paths')
    def init(self,*args,**kwargs):
        result=old_init(self,*args,**kwargs)
        self._pp_feedback_trimmed=0;self._pp_feedback_control=0;self._pp_feedback_removed=0;self._pp_feedback_unpacked=0
        return result
    def prepare(self,result,batch):
        payload=old_prepare(self,result,batch)
        if not self._pp_spec_relay or result.accept_lens is None:return payload
        flags=[req.rid.startswith('control-') for req in batch.reqs]
        if ab and any(flags):
            if not all(flags):raise RuntimeError('Mixed AB feedback cohort')
            self._pp_feedback_control+=1
            return payload
        if not all(k in payload for k in REMOVED):raise RuntimeError('Expected three draft auxiliary feedback fields')
        for key in REMOVED:payload.pop(key)
        self._pp_feedback_trimmed+=1;self._pp_feedback_removed+=len(REMOVED)
        return pack_payload(payload) if pack_payload is not None else payload
    def recv(self):
        payload=old_recv(self)
        if unpack_payload is not None and '__pp_verify_layout_v1__' in payload:
            payload=unpack_payload(payload);self._pp_feedback_unpacked+=1
        return payload
    mixin.init_pp_loop_state=init;mixin._pp_prepare_tensor_dict=prepare
    if unpack_payload is not None:
        if old_recv is None:raise RuntimeError('Missing PP feedback receive path')
        mixin._pp_recv_dict_from_prev_stage=recv
    mixin._pp_feedback_payload_originals=(old_init,old_prepare,old_recv)


def receipt(scheduler):
    if not hasattr(scheduler,'_pp_feedback_trimmed'):return dict(enabled=False)
    return dict(enabled=True,trimmed_verifies=scheduler._pp_feedback_trimmed,
        control_verifies=scheduler._pp_feedback_control,removed_tensors=scheduler._pp_feedback_removed,
        codec_cpu_qualification=getattr(scheduler,'_pp_feedback_codec_cpu',None),require_attn_tp_allgather=bool(getattr(scheduler,'require_attn_tp_allgather',False)),removed_keys=list(REMOVED),unpacked=scheduler._pp_feedback_unpacked,packed_verify_tensors=1,source_verify_tensors=10,prefill_unchanged=True,all_consumed_fields_unchanged=True)


def install():
    import os
    from pathlib import Path
    from common import config
    cfg=config(Path(os.environ['GLM_RUN_DIR']))
    if not cfg.get('feedback_payload_ab'):return
    if not (cfg['layout']=='pp2ep4' and cfg['mtp'] and cfg.get('local_tail_feedback')):
        raise RuntimeError('Feedback payload trim only qualified for PP2/two-slot MTP/no PD')
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from feedback_codec import pack,unpack,cpu_qualification
    SchedulerPPMixin._pp_feedback_codec_cpu=cpu_qualification()
    attach(SchedulerPPMixin,ab=False,pack_payload=pack,unpack_payload=unpack)
