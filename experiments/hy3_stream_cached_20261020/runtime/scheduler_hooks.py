"""Capture actual native pool capacity on every serving rank."""
import os
from pathlib import Path
import time
from stream_common import config, save

def install():
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler,'_hy3_cached_hook',False): return
    original=Scheduler.on_idle
    def idle(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from sglang.srt.runtime_context import get_parallel
        p=get_parallel(); rank=p.world_rank
        run=Path(os.environ['HY3_RUN_DIR']); cfg=config(run)
        out=run/cfg['layout']/'capacity'/f'rank{rank}.json'
        if not out.exists():
            host=self.tree_cache.token_to_kv_pool_host
            actual=host.kv_buffer.numel()*host.kv_buffer.element_size()
            if actual>16_100_000_000: raise RuntimeError('Native host pool exceeds frozen 16GB/rank bound')
            save(out,dict(timestamp=time.time(),rank=rank,node_rank=int(os.environ['HY3_NODE_RANK']),
                layout=cfg['layout'],token_capacity=self.max_total_num_tokens,
                host_token_capacity=host.size,host_bytes=actual,host_layout=host.layout,
                layer_count=host.layer_num,local_kv_heads=host.head_num,
                page_size=self.tree_cache.page_size,
                parsed_chunked_prefill_size=self.chunked_prefill_size,
                weights_bf16=True,kv_bf16=str(host.dtype)))
        return result
    Scheduler.on_idle=idle;Scheduler._hy3_cached_hook=True
