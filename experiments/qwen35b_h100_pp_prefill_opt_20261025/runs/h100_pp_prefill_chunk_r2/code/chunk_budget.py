"""Idle-boundary switch for native PP Prefill token admission, not model math."""
from functools import wraps
import os

_DONE=False
_POLICY=None
BUDGETS={ord('0'):4096,ord('1'):2048,ord('2'):1024}


def wrap(original, read_policy):
    @wraps(original)
    def choose(self, *args, **kwargs):
        if self.ps.pp_size==2:
            budget=BUDGETS.get(read_policy())
            if budget is None:
                raise RuntimeError('Unknown frozen PP chunk budget')
            # Both PP stages read the same private byte. The controller may
            # switch only after the preceding cohort has completely drained.
            self.chunked_prefill_size=budget
            self.max_prefill_tokens=budget
        return original(self,*args,**kwargs)
    return choose


def install():
    global _DONE,_POLICY
    if _DONE:return
    import pp_ready_send
    from sglang.srt.managers.scheduler import Scheduler
    _POLICY=pp_ready_send.open_policy(os.environ['PP_CHUNK_POLICY_PATH'])
    Scheduler.get_new_batch_prefill=wrap(Scheduler.get_new_batch_prefill,lambda:_POLICY[0])
    _DONE=True
