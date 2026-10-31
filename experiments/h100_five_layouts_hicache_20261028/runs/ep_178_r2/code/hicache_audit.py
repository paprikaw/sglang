"""Private idle-only cache demotion and read-only evidence; native HiCache I/O."""
from functools import wraps
import time
_DONE=False

def install():
 global _DONE
 if _DONE:return
 from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
 original_split=assembler._split_hicache_size
 def fixed_split(total,pools):
  if int(total)!=72 or len(pools)!=2:raise RuntimeError('Expected the frozen 64 GiB KV plus 8 GiB GDN host budget')
  return (64,8)
 assembler._split_hicache_size=fixed_split
 from sglang.srt.managers.scheduler import Scheduler
 from sglang.srt.managers.io_struct import SetInternalStateReqOutput
 from sglang.srt.mem_cache.base_prefix_cache import EvictParams
 previous_set=Scheduler.set_internal_state;previous_get=Scheduler.get_internal_state
 @wraps(previous_set)
 def set_state(self,req):
  if set(req.server_args)!={'benchmark_cache_cmd'}:return previous_set(self,req)
  cmd=int(req.server_args['benchmark_cache_cmd'])
  if cmd not in (1,2):raise ValueError('Unknown idle-only cache command')
  if self.waiting_queue or (self.running_batch and any(not r.finished() for r in self.running_batch.reqs)):
   return SetInternalStateReqOutput(updated=False)
  tc=self.tree_cache
  if type(tc).__name__!='UnifiedRadixCache' or tc.cache_controller is None:raise RuntimeError('Actual hybrid HiCache absent')
  tc.writing_check(write_back=True)
  before=dict(full=tc.full_evictable_size(),mamba=tc.mamba_evictable_size())
  evicted=tc.evict(EvictParams(num_tokens=1<<50,mamba_num=1<<30)) if cmd==2 else None
  self._benchmark_cache_audit=dict(command=cmd,time=time.time(),before=before,after=dict(full=tc.full_evictable_size(),mamba=tc.mamba_evictable_size()),
   evicted_full=evicted.num_tokens_evicted if evicted else 0,evicted_mamba=evicted.mamba_num_evicted if evicted else 0,ongoing_writes=len(tc.ongoing_write_through),ongoing_loads=len(tc.ongoing_load_back),
   implementation=type(tc).__name__,write_policy=tc.cache_controller.write_policy)
  if cmd==2 and (tc.full_evictable_size() or tc.mamba_evictable_size()):raise RuntimeError('GPU demotion did not drain all evictable prefixes')
  return SetInternalStateReqOutput(updated=True)
 @wraps(previous_get)
 def get_state(self,*args,**kwargs):
  result=previous_get(self,*args,**kwargs)
  result.internal_state['benchmark_hicache_audit']=getattr(self,'_benchmark_cache_audit',None)
  return result
 Scheduler.set_internal_state=set_state;Scheduler.get_internal_state=get_state;_DONE=True
