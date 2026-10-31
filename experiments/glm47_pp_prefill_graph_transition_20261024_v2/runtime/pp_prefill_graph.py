"""Opt-in residual Prefill Graph, with owned PP activation inputs.

Only B4, cached4032 + new65/request target EXTEND is admitted. Controls use
the original eager path in the SAME process. Native BCG owns all attention
metadata/Graph buffers; this adapter supplies the missing PP body inputs and
PP output forwarding. Draft and Verify kernels/protocol are unchanged.
"""
import collections
import contextvars
from pathlib import Path

_COHORT=contextvars.ContextVar('prefill_graph_cohort',default=None)
_COUNTS=collections.Counter()
_INSTALLED=False


def cohort(reqs):
    ids=[r.rid for r in reqs]
    if not ids:raise RuntimeError('Prefill Graph requires request ancestry')
    flags=[rid.startswith('control-') for rid in ids]
    if any(flags) and not all(flags):raise RuntimeError('Mixed Prefill Graph cohort')
    return dict(control=all(flags),rids=tuple(ids))


def eligible(batch):
    return (batch.batch_size==4 and list(batch.extend_seq_lens_cpu or [])==[65]*4
            and list(batch.extend_prefix_lens_cpu or [])==[4032]*4
            and len(batch.input_ids) in (260,512))


def copy_proxy(storage,live):
    if set(storage.tensors)!=set(live.tensors):raise RuntimeError('PP Graph activation fields changed')
    for key,dst in storage.tensors.items():
        src=live.tensors[key]
        if src.shape[1:]!=dst.shape[1:] or src.shape[0]>dst.shape[0] or src.dtype!=dst.dtype or src.device!=dst.device:
            raise RuntimeError('PP Graph activation geometry/dtype/device changed')
        if src.shape[0]<dst.shape[0]:dst.zero_()
        dst[:src.shape[0]].copy_(src)


def global_layer_map(local,layer_model,total=92):
    """Retain global layer IDs used by RadixAttention/quantized MoE ops.

    PPMissingLayer modules are absent from the collected local lists. Those
    positions must be explicit None slots, never reindexed as local layer0.
    Every owned layer must still have its real Standard GQA module.
    """
    start,end=layer_model.start_layer,layer_model.end_layer
    if (start,end) not in ((0,50),(50,92)):raise RuntimeError('Unqualified PP layer partition')
    if any(len(part)!=end-start for part in local):raise RuntimeError('Incomplete owned PP layer lists')
    if any(attn is None or attn.layer_id!=i for i,attn in enumerate(local.attention_layers,start)):
        raise RuntimeError('Missing or wrongly numbered owned attention layer')
    return type(local)(*([None]*start+list(part)+[None]*(total-end) for part in local))


def install():
    global _INSTALLED
    import os
    from common import config
    cfg=config(Path(os.environ['GLM_RUN_DIR']))
    if not cfg.get('prefill_graph_ab'):return
    if _INSTALLED:raise RuntimeError('Prefill Graph adapter already installed')
    if cfg['layout']!='pp2ep4' or not cfg['mtp']:raise RuntimeError('Unqualified Prefill Graph topology')
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
    from sglang.srt.model_executor import model_runner
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend

    original_run=Scheduler.run_batch
    def run_batch(self,batch,*args,**kwargs):
        token=_COHORT.set(cohort(batch.reqs))
        try:return original_run(self,batch,*args,**kwargs)
        finally:_COHORT.reset(token)
    Scheduler.run_batch=run_batch

    original_capture=cuda_graph_setup.capture_prefill_graph
    original_collect=cuda_graph_setup.compute_attention_and_moe_layers
    def capture(**kwargs):
        if kwargs['model_runner'].is_draft_worker:
            return cuda_graph_setup.GraphCapture(runner=None,memory_phase='draft_prefill',memory_usage_gb=0,capture_time=0)
        def collect(layer_model):return global_layer_map(original_collect(layer_model),layer_model)
        cuda_graph_setup.compute_attention_and_moe_layers=collect
        try:return original_capture(**kwargs)
        finally:cuda_graph_setup.compute_attention_and_moe_layers=original_collect
    cuda_graph_setup.capture_prefill_graph=model_runner.capture_prefill_graph=capture

    original_can=PrefillCudaGraphRunner.can_run_graph
    def can(self,batch):
        ctx=_COHORT.get()
        if ctx is None:raise RuntimeError('Serving Prefill Graph has no scheduler ancestry')
        if self.model_runner.is_draft_worker:raise RuntimeError('Draft Prefill must remain eager')
        # Exact geometry is checked even in controls, to prevent changing work.
        if not eligible(batch):raise RuntimeError('Prefill outside frozen B4/P4032/Q65 workload')
        if ctx['control']:
            _COUNTS['control_eager']+=1
            return False
        if not original_can(self,batch):raise RuntimeError('Native Prefill Graph eligibility failed; no silent fallback')
        _COUNTS['candidate_eligible']+=1
        return True
    PrefillCudaGraphRunner.can_run_graph=can

    original_body=PrefillCudaGraphRunner._run_forward
    def body(self,batch,num_tokens):
        if self.model_runner.pp_group.is_first_rank:return original_body(self,batch,num_tokens)
        proxy=getattr(self,'_pp_graph_proxy',None)
        if proxy is None:
            proxy=PPProxyTensors({key:torch.zeros((self.max_num_tokens,self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,device=self.device) for key in ('hidden_states','residual')})
            self._pp_graph_proxy=proxy
        original_layer=self.layer_model.forward
        def layer(*args,**kwargs):
            if len(args)>=5 or 'pp_proxy_tensors' in kwargs:raise RuntimeError('Native PP Prefill capture seam changed')
            return original_layer(*args,pp_proxy_tensors=proxy,**kwargs)
        self.layer_model.forward=layer
        try:return original_body(self,batch,num_tokens)
        finally:self.layer_model.forward=original_layer
    PrefillCudaGraphRunner._run_forward=body

    original_execute=PrefillCudaGraphRunner.execute
    def execute(self,batch,**kwargs):
        ctx=_COHORT.get()
        if ctx is None or ctx['control'] or not eligible(batch):raise RuntimeError('Unexpected Prefill Graph execute')
        if not self.model_runner.pp_group.is_first_rank:
            live=kwargs.get('pp_proxy_tensors')
            if live is None:raise RuntimeError('PP tail has no live activation input')
            copy_proxy(self._pp_graph_proxy,live)
            _COUNTS['tail_proxy_copies']+=1
        result=original_execute(self,batch,**kwargs)
        _COUNTS['graph_executions']+=1
        return result
    PrefillCudaGraphRunner.execute=execute

    original_finalize=PrefillCudaGraphRunner._finalize_execute_output
    def finalize(self,output):
        if isinstance(output,PPProxyTensors):
            if not self.model_runner.pp_group.is_first_rank:raise RuntimeError('Unexpected PP tail proxy output')
            _COUNTS['head_proxy_outputs']+=1
            return PPProxyTensors({key:value[:self.raw_num_tokens] for key,value in output.tensors.items()})
        return original_finalize(self,output)
    PrefillCudaGraphRunner._finalize_execute_output=finalize

    original_replay=BreakableCudaGraphBackend.replay
    def replay(self,*args,**kwargs):
        ctx=_COHORT.get()
        if ctx is not None:
            if ctx['control']:raise RuntimeError('Control unexpectedly replayed Prefill Graph')
            _COUNTS['actual_body_replays']+=1
        return original_replay(self,*args,**kwargs)
    BreakableCudaGraphBackend.replay=replay
    _INSTALLED=True


KEYS=('control_eager','candidate_eligible','graph_executions','actual_body_replays','tail_proxy_copies','head_proxy_outputs')


def receipt(scheduler):
    return dict(enabled=_INSTALLED,pp_rank=int(scheduler.ps.pp_rank),
        counters={key:_COUNTS[key] for key in KEYS},backend='native_breakable',bucket=512,
        live_batch_size=4,cached_tokens_per_request=4032,new_tokens_per_request=65,
        semantic_bank_unchanged=True,draft_prefill_eager=True,verify_graph_unchanged=True)


def validate_delta(before,after,*,control):
    if not before['enabled'] or not after['enabled']:raise RuntimeError('Missing Prefill Graph proof')
    delta={k:after['counters'][k]-before['counters'][k] for k in KEYS}
    expected={k:0 for k in KEYS}
    if control:expected['control_eager']=2
    else:
        for k in ('candidate_eligible','graph_executions','actual_body_replays'):expected[k]=2
        expected['head_proxy_outputs' if after['pp_rank']==0 else 'tail_proxy_copies']=2
    if delta!=expected:raise RuntimeError(f'Prefill Graph activation differs from frozen cohort: {delta} != {expected}')
    return delta
