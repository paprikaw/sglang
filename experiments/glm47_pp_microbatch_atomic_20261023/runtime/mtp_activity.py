"""Dormant calibration hooks, armed only after production results are saved.

Calibration uses existing untimed graph warmups (no extra model calls). During
production Decode CUPTI and GPU markers are off. A separate subsequent cohort
uses CUDA-only activity, two tiny markers per API and no inner phase fences.
"""
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path

_CAL=ContextVar('mtp_calibration_scope',default=False)
_DEPTH=ContextVar('mtp_profile_segment_depth',default=0)
_STATE=None

def save(path,value):
    from common import save as atomic_save
    atomic_save(path,value)

def arm():
    if _STATE is None:raise RuntimeError('Profiler hooks were not installed')
    if _STATE['armed']:raise RuntimeError('One bounded profile per process only')
    import torch
    # This command runs only after Scheduler.is_fully_idle and cache-controller
    # quiescence. Bootstrap before any profile request is released; never fence
    # a model forward that may have outstanding PP NCCL work.
    torch.cuda.synchronize()
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,profile_memory=False,with_stack=False)
    profiler.start()
    bootstrap=torch.empty(1,device='cuda',dtype=torch.int32)
    bootstrap.fill_(0)
    torch.cuda.synchronize()
    _STATE['profiler']=profiler
    _STATE['bootstrap']=dict(at_confirmed_scheduler_idle=True,
        before_profile_client_release=True,non_marker_fill_kernel=True,
        cuda_synchronizations=2,inside_model_forward=False)
    _STATE['armed']=True

def status():
    return dict(installed=_STATE is not None,
        armed=bool(_STATE and _STATE['armed']),
        active=bool(_STATE and _STATE['profiler'] is not None and not _STATE['done']),
        done=bool(_STATE and _STATE['done']))

def install():
    global _STATE
    if _STATE is not None:return
    import torch
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import FullCudaGraphBackend
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.models.glm4_moe import Glm4MoeAttention,Glm4MoeMLP,Glm4MoeSparseMoeBlock
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplLowLatency
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    state=dict(armed=False,profiler=None,verify_count=0,segments=[],calibrations=0,
               done=False,rank=None,cohort_iterations={})
    _STATE=state

    def rank():
        if state['rank'] is None:
            p=get_parallel();state['rank']=p.world_rank
        return state['rank']
    def output(name):
        return Path(os.environ['GLM_RUN_DIR'])/os.environ['GLM_LAYOUT']/name

    original_load=ModelRunner.load_model
    @wraps(original_load)
    def load(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        if not output(f'mtp_marker_signature_rank{rank()}.json').exists():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as marker:
                torch.cuda._sleep(1)
            path=output(f'mtp_marker_startup_rank{rank()}.json')
            marker.export_chrome_trace(str(path))
            kernels=[e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
            if len(kernels)!=1:raise RuntimeError('Cannot uniquely resolve actual startup GPU marker')
            save(output(f'mtp_marker_signature_rank{rank()}.json'),dict(passed=True,rank=rank(),name=kernels[0]))
        return result
    ModelRunner.load_model=load

    def scope(cls,name,label):
        descriptor=inspect.getattr_static(cls,name)
        original=descriptor.__func__ if isinstance(descriptor,(staticmethod,classmethod)) else descriptor
        @wraps(original)
        def wrapped(*args,**kwargs):
            if not _CAL.get():return original(*args,**kwargs)
            with torch.profiler.record_function('mtp-cal:'+label):return original(*args,**kwargs)
        setattr(cls,name,staticmethod(wrapped) if isinstance(descriptor,staticmethod) else
                classmethod(wrapped) if isinstance(descriptor,classmethod) else wrapped)
    for cls,name,label in [
        (Glm4MoeAttention,'forward','attention'),
        (Glm4MoeMLP,'forward','expert_shared_or_dense'),
        (Glm4MoeSparseMoeBlock,'_forward_shared_experts','expert_shared_or_dense'),
        (HummingRunnerCore,'_run_grouped_masked_gemm','expert_routed'),
        (HummingRunnerCore,'apply_masked_activation','expert_routed'),
        (_DeepEPDispatcherImplLowLatency,'dispatch_a','dispatch'),
        (_DeepEPDispatcherImplLowLatency,'dispatch_b','dispatch'),
        (_DeepEPDispatcherImplLowLatency,'combine_a','combine'),
        (_DeepEPDispatcherImplLowLatency,'combine_b','combine')]:scope(cls,name,label)
    for name in ('all_reduce','reduce_scatter_along_dim','reduce_scatter_tensor','reduce_scatter',
                 'reduce_scatterv','all_gather_into_tensor','all_gather','all_gatherv'):
        if hasattr(GroupCoordinator,name):scope(GroupCoordinator,name,'tp_communication')

    original_capture=FullCudaGraphBackend.capture_one
    @wraps(original_capture)
    def capture(self,shape_key,forward_fn,*args,**kwargs):
        if rank() not in (0,4):return original_capture(self,shape_key,forward_fn,*args,**kwargs)
        ordinal=state['calibrations'];state['calibrations']+=1
        calls=[0]
        def calibrated():
            if torch.cuda.is_current_stream_capturing():return forward_fn()
            index=calls[0];calls[0]+=1
            token=_CAL.set(True)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                     record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    result=forward_fn()
                path=output(f'calibration_rank{rank()}_graph{ordinal}_warm{index}.json')
                prof.export_chrome_trace(str(path))
                return result
            finally:_CAL.reset(token)
        return original_capture(self,shape_key,calibrated,*args,**kwargs)
    FullCudaGraphBackend.capture_one=capture

    def finish():
        if state['done'] or state['profiler'] is None:return
        state['profiler'].stop()
        path=output(f'mtp_activity_rank{rank()}.json')
        state['profiler'].export_chrome_trace(str(path))
        save(output(f'mtp_activity_receipt_rank{rank()}.json'),dict(passed=True,rank=rank(),
             segments=state['segments'],verify_count=state['verify_count'],trace=str(path),
             marker_kernel='torch.cuda._sleep(1), exact signature resolved from trace/calibration',
             profile_only=True,inner_phase_synchronizations=0,bootstrap=state.get('bootstrap'),
             stop_synchronization_outside_last_selected_forward=True))
        state['done']=True

    def segment(function,phase,batch):
        active=state['profiler'] is not None and not state['done']
        if not active or _DEPTH.get():return function()
        token=_DEPTH.set(1)
        # Shape/CPU lengths are read without touching a device tensor.
        mode=getattr(getattr(batch,'forward_mode',None),'name',None)
        reqs=getattr(batch,'reqs',None)
        batch_size=len(reqs) if reqs is not None else getattr(batch,'batch_size',None)
        ids=[r.rid for r in reqs] if reqs is not None else getattr(batch,'rids',None)
        ids=list(ids) if isinstance(ids,(list,tuple)) and all(isinstance(x,str) for x in ids) else []
        key=(phase,tuple(ids))
        iteration=state['cohort_iterations'].get(key,0)
        state['cohort_iterations'][key]=iteration+1
        state['segments'].append(dict(phase=phase,batch_size=batch_size,mode=mode,
             request_ids=ids,cohort_iteration=iteration,
             tokens_in_batch=getattr(getattr(batch,'input_ids',None),'numel',lambda:None)()))
        torch.cuda._sleep(1)
        try:return function()
        finally:
            torch.cuda._sleep(1);_DEPTH.reset(token)

    original_forward=ModelRunner.forward
    @wraps(original_forward)
    def forward(self,batch,*args,**kwargs):
        verify=not self.is_draft_worker and batch.forward_mode.is_target_verify()
        if state['armed'] and verify and not state['done']:
            if state['profiler'] is None:
                raise RuntimeError('Profile must bootstrap while idle before profile clients are released')
            if state['verify_count']>=12:finish()
            else:state['verify_count']+=1
        return segment(lambda:original_forward(self,batch,*args,**kwargs),
                       'draft_model' if self.is_draft_worker else 'target_verify' if verify else 'target_prefill',batch)
    ModelRunner.forward=forward
    def draft_wrapper(method,phase):
        @wraps(method)
        def wrapper(self,batch,*args,**kwargs):
            return segment(lambda:method(self,batch,*args,**kwargs),phase,batch)
        return wrapper
    for name,phase in [('draft','draft'),('_draft_extend_for_decode','draft_extend')]:
        setattr(EagleDraftWorker,name,draft_wrapper(getattr(EagleDraftWorker,name),phase))
