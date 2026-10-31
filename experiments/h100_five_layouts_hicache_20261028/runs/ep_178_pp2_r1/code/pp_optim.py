"""Only the qualified native and early-send PP paths are installed."""
import os
_DONE=False
def install():
 global _DONE
 if _DONE:return
 pp=int(os.environ.get('HICACHE_PP_SIZE','1'))
 if pp in (2,4):
  import gap_schedule
  from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
  from sglang.srt.managers.scheduler import Scheduler
  from pp_schedule_opt import open_policy
  gap_schedule._POLICY=open_policy(os.environ['PP_GAP_POLICY_PATH'])
  gap_schedule.install_early(SchedulerPPMixin,gap_schedule.mode)
  previous=Scheduler.get_internal_state
  def state(self,*args,**kwargs):
   result=previous(self,*args,**kwargs)
   result.internal_state['benchmark_pp_policy']=dict(policy=gap_schedule.mode(),pipeline_size=pp,native_transport=True,microbatch=getattr(self,'pp_max_micro_batch_size',None))
   return result
  Scheduler.get_internal_state=state
 _DONE=True
