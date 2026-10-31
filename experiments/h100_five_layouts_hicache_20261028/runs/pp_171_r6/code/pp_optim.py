"""Latest PP2 candidates and static PP4 early activation, idle policy switches."""
import os
_DONE=False

def install():
 global _DONE
 if _DONE:return
 pp=int(os.environ.get('HICACHE_PP_SIZE','1'))
 if pp==2:
  import gap_schedule
  import dynamic_pp_schema
  gap_schedule.install_schema=dynamic_pp_schema.install_schema
  gap_schedule.install()
 elif pp==4:
  import gap_schedule
  from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
  from pp_schedule_opt import open_policy
  gap_schedule._POLICY=open_policy(os.environ['PP_GAP_POLICY_PATH'])
  gap_schedule.install_early(SchedulerPPMixin,gap_schedule.mode)
  # The native event/send/ownership sequence is unchanged. Only the supported
  # stage count check is widened in our private copied install_early below.
 _DONE=True
