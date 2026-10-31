"""Idle-only semantic-bank export/import; no source-bank access during timing."""
from array import array
import json,os,time
from pathlib import Path
from common import config,save
from workload import prefixes,prompts,cached_prefix_length

def quiescent(cache):
    c=cache.cache_controller
    counts={k:len(getattr(c,k)) for k in ('load_queue','write_queue','ack_load_queue','ack_write_queue')}
    counts.update({k:len(getattr(cache,k)) for k in ('ongoing_load_back','ongoing_write_through')})
    # URC adds L3 and resumable-insert state outside the legacy D/H queues.
    # Check it before reset/export/import even when no storage backend is used.
    counts.update({k:len(getattr(cache,k,())) for k in ('ongoing_prefetch','ongoing_backup')})
    core=getattr(cache,'tree_core',None)
    counts['ongoing_tree_insert']=int(core.has_ongoing_insert()) if core is not None else 0
    return not any(counts.values()),counts

def scheduler_mode(self,cfg):
    enabled=bool(getattr(self,'_pp_spec_microbatch',False))
    loop=getattr(self,'pp_loop_size',self.ps.pp_size)
    expected=cfg['layout']=='pp2ep4' and cfg['mtp']
    if enabled!=expected or loop!=(2 if cfg['layout']=='pp2ep4' else 1):
        raise RuntimeError('Actual initialized PP microbatch mode mismatch')
    feedback=getattr(self,'_pp_spec_feedback_group',None)
    if enabled and feedback is None:raise RuntimeError('Isolated PP feedback group missing')
    return dict(enabled=enabled,loop_size=loop,feedback_group_name=getattr(feedback,'unique_name',None),
        feedback_group_ranks=getattr(feedback,'ranks',None))

def install():
    from sglang.srt.managers.scheduler import Scheduler
    if getattr(Scheduler,'_glm_cached_hook',False):return
    original=Scheduler.on_idle
    def idle(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams,MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from stage_activity import bootstrap,arm_profile,finish_profile,status
        p=get_parallel();rank=p.world_rank
        run=Path(os.environ['GLM_RUN_DIR']);cfg=config(run);base=run/cfg['arm']
        host=self.tree_cache.host_pool_group.get_pool(PoolName.KV)
        out=base/'capacity'/f'rank{rank}.json'
        if not out.exists():
            actual=host.kv_buffer.numel()*host.kv_buffer.element_size()
            if actual>35*1024**3:raise RuntimeError('Host pool exceeds35GiB per rank')
            save(out,dict(timestamp=time.time(),rank=rank,node_rank=int(os.environ['GLM_NODE_RANK']),
                layout=cfg['layout'],arm=cfg['arm'],mtp=cfg['mtp'],token_capacity=self.max_total_num_tokens,
                host_token_capacity=host.size,host_bytes=actual,host_layout=host.layout,
                layer_count=host.layer_num,local_kv_heads=host.head_num,page_size=self.tree_cache.page_size,
                parsed_chunked_prefill_size=self.chunked_prefill_size,weights_quantization='AWQ_INT4_Humming',kv_bf16=str(host.dtype)))
        if not self.is_fully_idle():return result
        from humming_receipt import collect_humming_kernel_receipt,validate_humming_kernel_receipt
        humming_start=base/f'humming_kernel_startup_rank{rank}.json'
        if not humming_start.exists():
            h=collect_humming_kernel_receipt();save(humming_start,dict(rank=rank,after_model_graph_warmup=True,
                registered_kernel_receipt=h,validation=validate_humming_kernel_receipt(h)))
        try:command=json.loads((base/'command.json').read_text())
        except FileNotFoundError:return result
        receipt=base/'receipts'/f"{command['id']}_rank{rank}.json"
        if receipt.exists():return result
        if command['run']!=cfg['id'] or command['prefix_length']!=4096:raise RuntimeError('Command outside frozen workload')
        quiet,counts=quiescent(self.tree_cache)
        if not quiet:return result
        cache=self.tree_cache;begun=time.time();action=command['action']
        if (base/'TIMED').exists():raise RuntimeError('Idle command cannot overlap timed client cohort')
        if action=='reset':
            cache.evict(EvictParams(num_tokens=1<<60));cache.reset();self._glm_bank_import=None
        elif action=='bank_import':
            if cfg['role']!='consumer':raise RuntimeError('Only fresh consumer may import bank')
            if getattr(self,'_glm_bank_import',None) is not None:raise RuntimeError('Duplicate semantic-bank import')
            from semantic_bank import import_rank
            self._glm_bank_import=import_rank(self,cfg['semantic_bank_dir'],prompts(),cfg['bank_model_identity'])
        elif action=='bank_export':
            if cfg['role']!='producer' or not cfg['mtp'] or cfg['layout']!='pp2ep4':raise RuntimeError('Invalid bank producer')
            from semantic_bank import export_rank
            self._glm_bank_export=export_rank(self,cfg['semantic_bank_dir'],prompts(),cfg['bank_model_identity'])
        elif action in ('ready','cold','profile_cold'):
            cache.evict(EvictParams(num_tokens=1<<60))
        elif action not in ('observe','profile_finish'):raise RuntimeError('Unknown idle command')
        assigned=list(range(8)) if cfg['layout']=='pp2ep4' else list(range(p.attn_dp_rank*4,p.attn_dp_rank*4+4))
        seqs=prefixes(4096);matches=[]
        for i in assigned:
            match=cache.match_prefix(MatchPrefixParams(key=RadixKey(array('q',seqs[i]))))
            matches.append(dict(request_index=i,gpu_tokens=len(match.device_indices),cpu_tokens=match.host_hit_length))
        if action in ('ready','cold','profile_cold','bank_import','bank_export') and any(x['gpu_tokens'] or x['cpu_tokens']!=4032 for x in matches):
            raise RuntimeError('Native CPU4032/GPU0 cache residency mismatch: '+str(matches))
        if action=='reset' and any(x['gpu_tokens'] or x['cpu_tokens'] for x in matches):raise RuntimeError('Reset did not empty native cache')
        from spec_proof import snapshot
        graph=snapshot(reset=action in ('cold','profile_cold','observe','profile_finish'))
        if action=='profile_cold':
            if not (base/'p4096_decode_results.json').exists() or not (base/'p4096_decode_repeat_results.json').exists():
                raise RuntimeError('Profile may start only after both performance cohorts are saved')
            # Collector qualification must not prevent independent production
            # measurements. It remains strict and outside every TIMED region.
            bootstrap(self)
            arm_profile(self,'profile-decode-4096')
        if action=='profile_finish':finish_profile(self)
        activity=status()
        if action=='cold' and (activity['armed'] or activity['active']):raise RuntimeError('Profiler overlaps production')
        from admission_adapter import new_state
        admission=getattr(self,'_glm_ep8_admission_state',new_state())
        if admission['pending']:raise RuntimeError('Idle command reached incomplete atomic local admission')
        atomic=admission['proofs'] if cfg['layout']=='ep8' else getattr(self,'_glm_atomic_admissions',[])
        h=collect_humming_kernel_receipt()
        from deterministic_proof import capture
        save(receipt,dict(passed=True,rank=rank,arm=cfg['arm'],mtp=cfg['mtp'],command=command,matches=matches,graph=graph,
            humming=h,humming_validation=validate_humming_kernel_receipt(h),deterministic=capture(self),
            native_hicache_quiescent=True,pending_counts=counts,source_pool_reads=0,synthetic_kv_forbidden=True,
            manual_host_kv_writes_during_timing=0,bank_import=getattr(self,'_glm_bank_import',None),bank_export=getattr(self,'_glm_bank_export',None),
            batches=getattr(self,'_glm_timed_batches',[]),atomic_admissions=atomic,admission_pending=0,
            profiler=activity,microbatch=scheduler_mode(self,cfg),prepared_s=time.time()-begun,timestamp=time.time()))
        if action in ('observe','profile_finish'):
            self._glm_timed_batches=[];self._glm_atomic_admissions=[];admission['proofs']=[]
        return result
    Scheduler.on_idle=idle;Scheduler._glm_cached_hook=True
    original_run=Scheduler.run_batch
    def run_batch(self,batch,*args,**kwargs):
        requests=getattr(batch,'reqs',[])
        if any(str(getattr(r,'rid','')).startswith(('timed-','repeat-','profile-')) for r in requests):
            rows=getattr(self,'_glm_timed_batches',None)
            if rows is None:rows=[];self._glm_timed_batches=rows
            if len(rows)>=10000:raise RuntimeError('CPU telemetry bound exceeded')
            rows.append(dict(timestamp=time.time(),mode=str(batch.forward_mode),batch_size=len(requests),
                request_ids=[r.rid for r in requests],extend_num_tokens=getattr(batch,'extend_num_tokens',None),
                pp_slot_id=(getattr(self,'_pp_microstep',0)%self.pp_loop_size) if self.ps.pp_size==2 else None,
                pp_microstep=getattr(self,'_pp_microstep',None),local_submission_ordinal=len(rows),attn_dp_rank=self.ps.attn_dp_rank))
        return original_run(self,batch,*args,**kwargs)
    Scheduler.run_batch=run_batch
    original_admission=Scheduler.handle_batch_generate_request
    def atomic_admission(self,recv_req,*args,**kwargs):
        if os.environ['GLM_LAYOUT']=='pp2ep4':
            from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput
            if type(recv_req) is not BatchTokenizedGenerateReqInput:raise RuntimeError('Stock atomic batch class required')
            ids=[r.rid for r in recv_req]
            if ids and all(isinstance(rid,str) and rid.startswith(('prepare-','warm-','timed-','repeat-','profile-')) for rid in ids):
                bases=[rid.rsplit('-',1)[0] for rid in ids];indices=[int(rid.rsplit('-',1)[1]) for rid in ids]
                if len(ids)!=8 or len(set(bases))!=1 or indices!=list(range(8)):raise RuntimeError('Atomic B8 reordered/split')
                rows=getattr(self,'_glm_atomic_admissions',None)
                if rows is None:rows=[];self._glm_atomic_admissions=rows
                if len(rows)>=32:raise RuntimeError('Atomic proof memory bound exceeded')
                rows.append(dict(cohort=bases[0],request_ids=ids,request_indices=indices,
                    input_class=type(recv_req).__name__,request_count=8,stock_scheduler_handler=True,
                    observed_at_scheduler_not_inferred_from_http=True))
        return original_admission(self,recv_req,*args,**kwargs)
    Scheduler.handle_batch_generate_request=atomic_admission
