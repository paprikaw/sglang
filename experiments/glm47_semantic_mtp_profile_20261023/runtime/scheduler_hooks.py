"""Observe real Prefill KV in native HiCache; never construct or mutate KV data."""
from array import array
import json
import os
from pathlib import Path
import time
from common import config, save
from workload import prefixes,cached_prefix_length

def quiescent(cache):
    c=cache.cache_controller
    counts={k:len(getattr(c,k)) for k in ('load_queue','write_queue','ack_load_queue','ack_write_queue')}
    counts.update({k:len(getattr(cache,k)) for k in ('ongoing_load_back','ongoing_write_through')})
    return not any(counts.values()),counts

def install():
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler,'_glm_cached_hook',False): return
    original=Scheduler.on_idle
    def idle(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams,MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.hicache_storage import PoolName
        p=get_parallel(); rank=p.world_rank
        run=Path(os.environ['GLM_RUN_DIR']); cfg=config(run)
        out=run/cfg['layout']/'capacity'/f'rank{rank}.json'
        host=self.tree_cache.host_pool_group.get_pool(PoolName.KV)
        if not out.exists():
            actual=host.kv_buffer.numel()*host.kv_buffer.element_size()
            if actual>35*1024**3: raise RuntimeError('Native host pool exceeds frozen 35GiB/rank bound')
            save(out,dict(timestamp=time.time(),rank=rank,node_rank=int(os.environ['GLM_NODE_RANK']),
                layout=cfg['layout'],token_capacity=self.max_total_num_tokens,
                host_token_capacity=host.size,host_bytes=actual,host_layout=host.layout,
                layer_count=host.layer_num,local_kv_heads=host.head_num,
                page_size=self.tree_cache.page_size,
                parsed_chunked_prefill_size=self.chunked_prefill_size,
                weights_quantization="AWQ_INT4_Humming",kv_bf16=str(host.dtype)))
        base=run/cfg['layout']
        if not self.is_fully_idle():return result
        try: command=json.loads((base/'command.json').read_text())
        except FileNotFoundError:return result
        receipt=base/'receipts'/f"{command['id']}_rank{rank}.json"
        if receipt.exists():return result
        auto_limit=((int(self.max_total_num_tokens*.90)//(4 if cfg['layout']=='ep8' else 8)-384)//64)*64
        allowed=(command['prefix_length'] in cfg['prefix_lengths'] or
                 cfg['prefix_lengths']==[0] and 0<command['prefix_length']<=auto_limit and command['prefix_length']%64==0)
        if command['run']!=cfg['id'] or not allowed:
            raise RuntimeError('Command outside frozen workload')
        quiet,counts=quiescent(self.tree_cache)
        if not quiet:return result
        cache=self.tree_cache;begun=time.time();node_rank=int(os.environ['GLM_NODE_RANK'])
        assigned=list(range(8)) if cfg['layout']=='pp2ep4' else list(range(node_rank,8,2))
        seqs=prefixes(command['prefix_length'])
        if command['action']=='reset':
            if (base/'TIMED').exists():raise RuntimeError('Timed reset forbidden')
            cache.evict(EvictParams(num_tokens=1<<60));cache.reset()
        elif command['action'] in ('ready','cold'):cache.evict(EvictParams(num_tokens=1<<60))
        elif command['action']!='observe':raise RuntimeError('Unknown command')
        matches=[]
        for i in assigned:
            m=cache.match_prefix(MatchPrefixParams(key=RadixKey(array('q',seqs[i]))))
            matches.append(dict(request_index=i,gpu_tokens=len(m.device_indices),cpu_tokens=m.host_hit_length))
        if command['action'] in ('ready','cold') and any(x['gpu_tokens'] or x['cpu_tokens']!=cached_prefix_length(command['prefix_length']) for x in matches):
            raise RuntimeError('Cold native MTP HiCache residency mismatch: '+str(matches))
        if command['action']=='reset' and any(x['gpu_tokens'] or x['cpu_tokens'] for x in matches):
            raise RuntimeError('Real Prefill preparation requires an empty prefix cache')
        from spec_proof import snapshot
        graph = snapshot(reset=command['action'] in ('cold','observe'))
        from mtp_activity import arm
        if command['action']=='cold':arm()
        save(receipt,dict(passed=True,rank=rank,command=command,matches=matches,graph=graph,
             native_hicache_quiescent=True,pending_counts=counts,source_pool_reads=0,
             synthetic_kv_forbidden=True,manual_host_kv_writes=0,
             batches=getattr(self,'_glm_timed_batches',[]),
             prepared_s=time.time()-begun,timestamp=time.time()))
        if command['action']=='observe':self._glm_timed_batches=[]
        return result
    Scheduler.on_idle=idle;Scheduler._glm_cached_hook=True
    original_run=Scheduler.run_batch
    def run_batch(self,batch,*args,**kwargs):
        requests=getattr(batch,'reqs',[])
        if any(str(getattr(r,'rid','')).startswith('timed-') for r in requests):
            rows=getattr(self,'_glm_timed_batches',None)
            if rows is None:rows=[];self._glm_timed_batches=rows
            if len(rows)>=10000:raise RuntimeError('Telemetry memory bound exceeded')
            rows.append(dict(timestamp=time.time(),mode=str(batch.forward_mode),
                batch_size=len(requests),request_ids=[r.rid for r in requests],
                extend_num_tokens=getattr(batch,'extend_num_tokens',None),
                pp_microstep=getattr(self,'_pp_microstep',None)))
        return original_run(self,batch,*args,**kwargs)
    Scheduler.run_batch=run_batch
