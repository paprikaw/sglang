"""Untimed in-place native host KV construction and cold GPU cache proofs."""
from array import array
import json
import os
from pathlib import Path
import time
from common import config, save
from workload import prefixes,page_hashes

def quiescent(cache):
    c=cache.cache_controller
    counts={k:len(getattr(c,k)) for k in ('load_queue','write_queue','ack_load_queue','ack_write_queue')}
    counts.update({k:len(getattr(cache,k)) for k in ('ongoing_load_back','ongoing_write_through')})
    return not any(counts.values()),counts

def fill(host,ids,layer_start,head_start):
    import torch
    if host.layout!='page_first' or host.dtype!=torch.bfloat16:
        raise RuntimeError('Native page_first BF16 host pool required')
    if len(ids)%64 or int(ids[-1])-int(ids[0])+1!=len(ids):
        raise RuntimeError('Fresh contiguous host allocation required')
    v=host.kv_buffer[:,int(ids[0]):int(ids[-1])+1]
    for kv in range(2):
        for layer in range(host.layer_num):
            for head in range(host.head_num):
                scalar=((layer_start+layer)*8+head_start+head+1)*(1 if kv==0 else -1)/4096
                v[kv,:,layer,head,:].fill_(scalar)

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
        if command['action']=='fill':
            if (base/'TIMED').exists():raise RuntimeError('Timed synthesis forbidden')
            cache.evict(EvictParams(num_tokens=1<<60));cache.reset()
            for i in assigned:
                ids=host.alloc(len(seqs[i]))
                if ids is None:raise RuntimeError('HiCache cannot hold entire cohort')
                fill(host,ids,0 if cfg['layout']=='ep8' else node_rank*46,(rank%4)*2)
                inserted=cache.tree_core.insert_host(cache.tree_core.root_node.id,
                    RadixKey(array('q',seqs[i])),ids,page_hashes(seqs[i]))
                cache._apply_cache_actions(inserted.cache_actions)
                if inserted.host_insert_dropped or inserted.prefix_len:
                    raise RuntimeError('Synthetic native host insertion was not a fresh complete prefix')
        elif command['action']=='cold':cache.evict(EvictParams(num_tokens=1<<60))
        elif command['action']!='observe':raise RuntimeError('Unknown command')
        matches=[]
        for i in assigned:
            m=cache.match_prefix(MatchPrefixParams(key=RadixKey(array('q',seqs[i]))))
            matches.append(dict(request_index=i,gpu_tokens=len(m.device_indices),cpu_tokens=m.host_hit_length))
        if command['action']!='observe' and any(x['gpu_tokens'] or x['cpu_tokens']!=command['prefix_length'] for x in matches):
            raise RuntimeError('Cold native HiCache residency mismatch')
        from graph_proof import snapshot
        graph = snapshot(reset=command['action'] in ('cold','observe'))
        save(receipt,dict(passed=True,rank=rank,command=command,matches=matches,graph=graph,
             native_hicache_quiescent=True,pending_counts=counts,source_pool_reads=0,
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
