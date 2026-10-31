"""External CUDA Event diagnostic spans; no CUPTI and no production reruns.

One outer completion/read follows EVERY actual Graph replay, before timestamps
can be overwritten by the next replay (including consecutive Eagle drafts).
This changes diagnostic cadence. Spans include GPU stream dependency waits and
event nodes: they are neither kernel busy time nor a partition of original TPOT.
"""
from collections import defaultdict,deque
from contextvars import ContextVar
from functools import wraps
import inspect,os,time
from pathlib import Path
from activity_metadata import batch_metadata,target_phase,marker_signature,MARKER_PROOF_SCOPES
MAX_TARGET_FORWARDS=12
MAX_REPLAYS=160
_CAP=ContextVar('event_capture',default=None)
_LAYER=ContextVar('event_layer',default=None)
_SHARED=ContextVar('event_shared',default=False)
_FRAME=ContextVar('event_frame',default=None)
_SEGMENT=ContextVar('event_segment',default=None)
_STATE=None

def _config():
    if _STATE.get('config') is None:
        from common import config
        _STATE['config']=config(Path(os.environ['GLM_RUN_DIR']))
    return _STATE['config']
def _rank():
    if _STATE['rank'] is None:
        from sglang.srt.runtime_context import get_parallel
        p=get_parallel();_STATE['rank']=dict(world_rank=p.world_rank,pp_rank=p.pp_rank,tp_rank=p.tp_rank,attn_dp_rank=p.attn_dp_rank)
    return _STATE['rank']['world_rank']
def _output(name):
    cfg=_config();return Path(cfg['run_dir'])/cfg['arm']/name
def _save(path,value):
    from common import save
    save(path,value)
def status():
    return dict(installed=_STATE is not None,bootstrapped=bool(_STATE and _STATE['bootstrapped']),armed=bool(_STATE and _STATE['armed']),
        active=bool(_STATE and _STATE['armed'] and not _STATE['done']),done=bool(_STATE and _STATE['done']),collector='external_cuda_events',cupti_enabled=False)
def _require_idle(scheduler):
    if not scheduler.is_fully_idle():raise RuntimeError('Event control requires confirmed scheduler idle')
    from scheduler_hooks import quiescent
    quiet,counts=quiescent(scheduler.tree_cache)
    if not quiet:raise RuntimeError('Event control requires quiescent HiCache: '+str(counts))
    cfg=_config()
    if (Path(cfg['run_dir'])/cfg['arm']/'TIMED').exists():raise RuntimeError('Event diagnostic cannot overlap production timing')
def bootstrap(scheduler):
    if _STATE is None:raise RuntimeError('install must precede bootstrap')
    _require_idle(scheduler);_STATE['bootstrapped']=True
    return status()
def arm_profile(scheduler,cohort_id):
    if _STATE is None or _STATE['armed']:raise RuntimeError('Only one bounded event cohort per consumer')
    _require_idle(scheduler)
    from profile_source import validate_source_once
    validate_source_once(_config(),arm=_config()['arm'])
    if not _STATE['variants']:raise RuntimeError('No captured Graph event variants')
    _STATE.update(armed=True,bootstrapped=True,cohort_id=str(cohort_id));return status()
def new_pair(torch,label,*,layer_id=None,name=None):
    return dict(label=label,layer_id=layer_id,name=name,stream=int(torch.cuda.current_stream().cuda_stream),
        start=torch.cuda.Event(enable_timing=True,external=True),end=torch.cuda.Event(enable_timing=True,external=True))
def read_variant(variant):
    """One outer completion; all timestamp reads precede same Graph reuse."""
    anchor=variant['anchor'];anchor['end'].synchronize()
    rows=[dict(label=p['label'],layer_id=p['layer_id'],name=p['name'],stream=p['stream'],
        start_ms=float(anchor['start'].elapsed_time(p['start'])),end_ms=float(anchor['start'].elapsed_time(p['end'])),
        elapsed_ms=float(p['start'].elapsed_time(p['end']))) for p in variant['pairs']]
    return dict(graph_span_ms=float(anchor['start'].elapsed_time(anchor['end'])),anchor_stream=anchor['stream'],intervals=rows,
        completion_fences=1,component_fences=0,read_before_same_graph_reuse=True)
def finish_profile(scheduler):
    if _STATE is None or not _STATE['armed']:raise RuntimeError('No armed event profile')
    if _STATE['done']:return status()
    _require_idle(scheduler)
    import torch
    torch.cuda.synchronize() # Idle only; closes unique non-Graph boundary event pairs.
    s,cfg=_STATE,_config();errors=list(s['errors'])
    target=[r for r in s['segments'] if r['phase'].startswith('target_')]
    if len(target)!=12:errors.append('Exactly12 actual target forwards required')
    if any(r['batch_size']!=4 or len(r['request_ids'])!=4 for r in target):errors.append('Actual localB4 missing')
    for r in target:
        if sum(g['role']=='target' and g['segment_index']==r['index'] for g in s['replays'])!=1:errors.append('One actual target Graph per target forward required')
    if sorted(r['target_segment_index'] for r in s['accepted'])!=sorted(r['index'] for r in target):errors.append('Actual committed CPU output records missing')
    boundaries=[{k:v for k,v in p.items() if k not in ('start','end')}|dict(elapsed_ms=float(p['start'].elapsed_time(p['end']))) for p in s['boundaries']]
    path=_output(f'event_spans_rank{_rank()}.json')
    raw=dict(schema_version=1,measurement='external_cuda_event_stream_spans',rank=_rank(),topology=s['rank'],arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],
        segments=s['segments'],replays=s['replays'],accepted=s['accepted'],pp_boundaries=boundaries,
        variants=[dict(identity=v['identity'],role=v['role'],shape_key=v['shape_key'],capture_generation=v['generation'],layer_ids=v['layer_ids'],component_count=len(v['pairs'])) for v in s['variants'].values()],
        cupti_enabled=False,production_timing_affected=False,diagnostic_cadence_changed=True,
        graph_component_semantics='GPU stream spans include waits, event nodes and nesting; not kernel busy',
        residual_semantics='Uncovered Graph stream span, not GPU idle',
        pp_boundary_semantics='Current-stream API dependency/enqueue span; async send completion NOT measured, not pure NIC time',
        no_cross_rank_clock_addition=True,no_production_tpot_partition=True,
        stream_id_semantics='interval/anchor stream IDs are capture-time current streams; replay_stream is the actual calling stream at replay, no CUPTI stream remapping inferred',
        draft_target_ordinal_semantics='Most recently observed target ordinal, NOT an asserted Verify/draft cycle association; keep actual phases/replays separate')
    _save(path,raw)
    receipt=dict(schema_version=2,passed=not errors,errors=errors,rank=_rank(),topology=s['rank'],arm=cfg['arm'],layout=cfg['layout'],mtp=cfg['mtp'],
        target_forward_count=len(target),max_target_forwards=12,segments=s['segments'],accepted=s['accepted'],trace=str(path),measurement=raw['measurement'],
        profile_only=True,cupti_enabled=False,production_timing_affected=False,diagnostic_cadence_changed=True,inner_component_synchronizations=0,
        graph_completion_fences=len(s['replays']),actual_target_graph_replays=sum(g['role']=='target' for g in s['replays']),
        actual_draft_graph_replays=sum(g['role']=='draft' for g in s['replays']),source_performance=cfg.get('profile_source',{}).get('arms',{}).get(cfg['arm']))
    _save(_output(f'activity_receipt_rank{_rank()}.json'),receipt);s['done']=True
    if errors:raise RuntimeError('Event receipt failed: '+'; '.join(errors))
    return receipt

def install():
    global _STATE
    if _STATE is not None:return
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.models.glm4_moe import Glm4MoeAttention,Glm4MoeMLP,Glm4MoeSparseMoeBlock,Glm4MoeDecoderLayer
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplLowLatency
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler
    s=dict(armed=False,bootstrapped=False,done=False,config=None,rank=None,variants={},generation=0,segments=[],replays=[],boundaries=[],accepted=[],
        target_count=0,pending=defaultdict(deque),iterations=defaultdict(int),errors=[],collection_closed=False)
    _STATE=s
    def scope(cls,name,label,shared=False):
        descriptor=inspect.getattr_static(cls,name);original=descriptor.__func__ if isinstance(descriptor,(staticmethod,classmethod)) else descriptor
        @wraps(original)
        def wrapped(*args,**kwargs):
            cap=_CAP.get()
            if cap is None:return original(*args,**kwargs)
            actual='expert_shared' if label=='dense_ffn' and _SHARED.get() else label
            pair=new_pair(torch,actual,layer_id=_LAYER.get(),name=cls.__name__+'.'+name);cap['pairs'].append(pair)
            token=_SHARED.set(True) if shared else None
            pair['start'].record()
            try:return original(*args,**kwargs)
            finally:
                pair['end'].record()
                if token is not None:_SHARED.reset(token)
        setattr(cls,name,staticmethod(wrapped) if isinstance(descriptor,staticmethod) else classmethod(wrapped) if isinstance(descriptor,classmethod) else wrapped)
    original_layer=Glm4MoeDecoderLayer.forward
    @wraps(original_layer)
    def layer(self,*args,**kwargs):
        token=_LAYER.set(int(self.layer_id));cap=_CAP.get()
        if cap is not None:cap['layer_ids'].add(int(self.layer_id))
        try:return original_layer(self,*args,**kwargs)
        finally:_LAYER.reset(token)
    Glm4MoeDecoderLayer.forward=layer
    scope(Glm4MoeAttention,'forward','attention')
    scope(Glm4MoeMLP,'forward','dense_ffn')
    scope(HummingRunnerCore,'_run_grouped_masked_gemm','expert_routed') # Includes both GEMMs+activation.
    scope(Glm4MoeSparseMoeBlock,'_forward_shared_experts','expert_shared',shared=True)
    for name in ('all_reduce','reduce_scatter_along_dim','reduce_scatter_tensor','reduce_scatter','reduce_scatterv','all_gather_into_tensor','all_gather','all_gatherv'):
        if hasattr(GroupCoordinator,name):scope(GroupCoordinator,name,'nccl_tp')
    def exchange_pair(first,second,label):
        a,b=getattr(_DeepEPDispatcherImplLowLatency,first),getattr(_DeepEPDispatcherImplLowLatency,second)
        @wraps(a)
        def begin(self,*args,**kwargs):
            cap=_CAP.get()
            if cap is not None:
                key=(id(self),label,_LAYER.get())
                if key in cap['pending']:raise RuntimeError('Overlapping dispatcher event pair')
                p=new_pair(torch,label,layer_id=_LAYER.get(),name=first+'->'+second);cap['pairs'].append(p);cap['pending'][key]=p;p['start'].record()
            return a(self,*args,**kwargs)
        @wraps(b)
        def end(self,*args,**kwargs):
            value=b(self,*args,**kwargs);cap=_CAP.get()
            if cap is not None:
                p=cap['pending'].pop((id(self),label,_LAYER.get()))
                if int(torch.cuda.current_stream().cuda_stream)!=p['stream']:raise RuntimeError('DeepEP returned on different stream')
                p['end'].record() # AFTER receive hook / combine overlap-stream rejoin.
            return value
        setattr(_DeepEPDispatcherImplLowLatency,first,begin);setattr(_DeepEPDispatcherImplLowLatency,second,end)
    exchange_pair('dispatch_a','dispatch_b','deepep_dispatch');exchange_pair('combine_a','combine_b','deepep_combine')
    original_capture=FullCudaGraphBackend.capture_one
    @wraps(original_capture)
    def capture(self,shape_key,forward_fn,*args,**kwargs):
        s['generation']+=1;generation=s['generation'];role='draft' if self._cuda_graph_runner.model_runner.is_draft_worker else 'target';captured=[]
        def instrumented():
            if not torch.cuda.is_current_stream_capturing():return forward_fn()
            if captured:raise RuntimeError('Repeated capture forward for one variant')
            v=dict(identity=f'{role}:backend{id(self)}:shape{shape_key!r}:capture{generation}',role=role,shape_key=repr(shape_key),generation=generation,
                pairs=[],layer_ids=set(),pending={},anchor=new_pair(torch,'graph'))
            captured.append(v);token=_CAP.set(v);v['anchor']['start'].record()
            try:return forward_fn()
            finally:v['anchor']['end'].record();_CAP.reset(token)
        value=original_capture(self,shape_key,instrumented,*args,**kwargs)
        if len(captured)!=1 or captured[0]['pending']:raise RuntimeError('Incomplete actual event Graph capture')
        v=captured[0];v['layer_ids']=sorted(v['layer_ids']);s['variants'][(id(self),shape_key,role,generation)]=v
        if not hasattr(self,'_glm_event_variants'):self._glm_event_variants={}
        self._glm_event_variants[shape_key]=v
        return value
    FullCudaGraphBackend.capture_one=capture
    original_replay=FullCudaGraphBackend.replay
    @wraps(original_replay)
    def replay(self,shape_key,static_forward_batch,*args,**kwargs):
        value=original_replay(self,shape_key,static_forward_batch,*args,**kwargs);segment=_SEGMENT.get()
        if s['armed'] and not s['done'] and segment is not None:
            if len(s['replays'])>=MAX_REPLAYS:raise RuntimeError('Bounded replay storage exceeded')
            v=self._glm_event_variants[shape_key];sample=read_variant(v)
            s['replays'].append(dict(index=len(s['replays']),segment_index=segment['index'],target_ordinal=segment['target_ordinal'],phase=segment['phase'],
                role=v['role'],variant=v['identity'],layer_ids=v['layer_ids'],request_ids=segment['request_ids'],batch_size=segment['batch_size'],
                pp_microstep=segment['pp_microstep'],replay_stream=int(torch.cuda.current_stream().cuda_stream),cpu_read_ns=time.perf_counter_ns(),**sample))
        return value
    FullCudaGraphBackend.replay=replay
    def segment(function,phase,batch):
        if not s['armed'] or s['done'] or _SEGMENT.get() is not None or s['collection_closed']:return function()
        target=phase in ('target_decode','target_verify')
        if target:
            expected='target_verify' if _config()['mtp'] else 'target_decode'
            if phase!=expected:s['errors'].append('Unexpected target phase '+phase);return function()
            if s['target_count']>=MAX_TARGET_FORWARDS:s['collection_closed']=True;return function()
            s['target_count']+=1
        elif s['target_count']==0:return function()
        meta=batch_metadata(batch,_FRAME.get());key=tuple(sorted(meta['request_ids']))
        row=dict(index=len(s['segments']),phase=phase,target_ordinal=s['target_count'],request_iteration=s['iterations'][key],**meta,cpu_start_ns=time.perf_counter_ns())
        s['segments'].append(row)
        if target:s['iterations'][key]+=1;s['pending'][key].append(row['index'])
        token=_SEGMENT.set(row)
        try:return function()
        finally:row['cpu_end_ns']=time.perf_counter_ns();_SEGMENT.reset(token)
    original_run=Scheduler.run_batch
    @wraps(original_run)
    def run(self,batch,*args,**kwargs):
        if not s['armed'] or s['done']:return original_run(self,batch,*args,**kwargs)
        frame=batch_metadata(batch);frame['pp_microstep']=getattr(self,'_pp_microstep',None);token=_FRAME.set(frame)
        try:return original_run(self,batch,*args,**kwargs)
        finally:_FRAME.reset(token)
    Scheduler.run_batch=run
    original_forward=ModelRunner.forward
    @wraps(original_forward)
    def forward(self,batch,*args,**kwargs):
        phase=target_phase(batch,self.is_draft_worker)
        if phase is None:return original_forward(self,batch,*args,**kwargs)
        return segment(lambda:original_forward(self,batch,*args,**kwargs),phase,batch)
    ModelRunner.forward=forward
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    def draft_wrapper(method,phase):
        @wraps(method)
        def wrapped(self,batch,*args,**kwargs):return segment(lambda:method(self,batch,*args,**kwargs),phase,batch)
        return wrapped
    for name,phase in [('draft','draft'),('_draft_extend_for_decode','draft_extend')]:setattr(EagleDraftWorker,name,draft_wrapper(getattr(EagleDraftWorker,name),phase))
    def boundary(name):
        original=getattr(GroupCoordinator,name)
        @wraps(original)
        def wrapped(self,*args,**kwargs):
            if not s['armed'] or s['done'] or s['collection_closed'] or s['target_count']==0:return original(self,*args,**kwargs)
            if len(s['boundaries'])>=512:raise RuntimeError('Bounded PP boundary storage exceeded')
            p=new_pair(torch,'pp_boundary',name=name);p.update(index=len(s['boundaries']),target_ordinal=s['target_count'],cpu_start_ns=time.perf_counter_ns(),
                async_send=bool(kwargs.get('async_send',False)),completion_scope='current-stream API dependency/enqueue only; async send may remain in flight')
            s['boundaries'].append(p);p['start'].record()
            try:return original(self,*args,**kwargs)
            finally:p['end'].record();p['cpu_end_ns']=time.perf_counter_ns()
        setattr(GroupCoordinator,name,wrapped)
    for name in ('send_tensor_dict','recv_tensor_dict','recv_tensor_dict_into'):
        if hasattr(GroupCoordinator,name):boundary(name)
    original_process=Scheduler.process_batch_result
    @wraps(original_process)
    def process(self,batch,result,*args,**kwargs):
        key=tuple(sorted(r.rid for r in getattr(batch,'reqs',[])));pending=s['pending'].get(key)
        collect=s['armed'] and not s['done'] and bool(pending) and batch.forward_mode.is_decode()
        before={r.rid:len(r.output_ids) for r in batch.reqs} if collect else {};value=original_process(self,batch,result,*args,**kwargs)
        if collect:
            index=pending.popleft();advances={r.rid:len(r.output_ids)-before[r.rid] for r in batch.reqs}
            s['accepted'].append(dict(target_segment_index=index,output_tokens_by_request=advances,output_tokens=sum(advances.values()),cpu_time_ns=time.perf_counter_ns()))
        return value
    Scheduler.process_batch_result=process
