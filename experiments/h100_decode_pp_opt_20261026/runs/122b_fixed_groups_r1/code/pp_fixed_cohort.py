"""Experiment-only admission: give EP and PP the same two request groups.

Wait for the declared eight simultaneously submitted requests, then admit the
even document indices followed by the odd ones. No model/tensor/output changes.
The waiting time remains inside client latency. Never used for arbitrary traffic.
"""
from functools import wraps
import os
import time

from pp_schedule_opt import open_policy

_INSTALLED = False
_POLICY = None


def order_cohort(requests):
    markers = [r.origin_input_ids[0] for r in requests]
    if len(markers) != len(set(markers)) or not set(markers).issubset(set(range(5000,5008))):
        raise RuntimeError('Unexpected or duplicate request in fixed eight-request cohort')
    if len(markers) < 8:
        return None
    return sorted(requests,key=lambda r:(r.origin_input_ids[0] % 2,r.origin_input_ids[0]))


def instrument(cls, read_policy, clock=time.monotonic):
    original=cls.get_next_batch_to_run
    internal=cls.get_internal_state

    @wraps(original)
    def get_next(self,*args,**kwargs):
        epoch=read_policy()
        if epoch and self.ps.pp_size == 2 and self._pp_is_first_rank():
            if self.ps.tp_size != 1 or self._pp_topology() is not None:
                raise RuntimeError('Fixed groups are limited to static PP2 x TP1')
            if getattr(self,'_fixed_admitted_epoch',None) != epoch and self.waiting_queue:
                if getattr(self,'_fixed_waiting_epoch',None) != epoch:
                    self._fixed_waiting_epoch=epoch;self._fixed_waiting_since=clock()
                ordered=order_cohort(self.waiting_queue)
                if ordered is None:
                    if clock()-self._fixed_waiting_since > 5:
                        raise TimeoutError('Declared simultaneous cohort did not arrive within five seconds')
                    return None
                self.waiting_queue[:]=ordered
                self._fixed_admitted_epoch=epoch
                self._fixed_group_receipt=dict(epoch=epoch,
                    ordered_markers=[r.origin_input_ids[0] for r in ordered],
                    waiting_ms=(clock()-self._fixed_waiting_since)*1000)
        return original(self,*args,**kwargs)

    @wraps(internal)
    def get_state(self,*args,**kwargs):
        result=internal(self,*args,**kwargs)
        if hasattr(self,'_fixed_group_receipt'):
            result.internal_state['experiment_fixed_group_receipt']=self._fixed_group_receipt
        return result

    cls.get_next_batch_to_run=get_next
    cls.get_internal_state=get_state


def install():
    global _INSTALLED,_POLICY
    if _INSTALLED:return
    from sglang.srt.managers.scheduler import Scheduler
    _POLICY=open_policy(os.environ['PP_FIXED_COHORT_CONTROL'])
    instrument(Scheduler,lambda:_POLICY[0])
    _INSTALLED=True
