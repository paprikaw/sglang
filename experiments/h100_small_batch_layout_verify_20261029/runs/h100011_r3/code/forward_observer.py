"""Count Python-visible real forwards; no CUDA operations, tensor reads or fences."""
from collections import Counter
from functools import wraps

_COUNTS=Counter()
_DONE=False


def install():
    global _DONE
    if _DONE:return
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.managers.scheduler import Scheduler
    original=ModelRunner.forward
    @wraps(original)
    def forward(self, forward_batch, *args, **kwargs):
        _COUNTS[f'{forward_batch.forward_mode.name}:b{forward_batch.batch_size}']+=1
        return original(self,forward_batch,*args,**kwargs)
    ModelRunner.forward=forward
    graph=DecodeCudaGraphRunner.can_run_graph
    @wraps(graph)
    def can_graph(self,batch):
        accepted=graph(self,batch)
        if accepted:_COUNTS[f'graph_eligible:b{batch.batch_size}']+=1
        return accepted
    DecodeCudaGraphRunner.can_run_graph=can_graph
    internal=Scheduler.get_internal_state
    @wraps(internal)
    def state(self,*args,**kwargs):
        result=internal(self,*args,**kwargs)
        result.internal_state['experiment_forward_counts']=dict(_COUNTS)
        result.internal_state['experiment_parallel_shape']=dict(pp_rank=self.ps.pp_rank,tp_rank=self.ps.tp_rank,
            pp_size=self.ps.pp_size,tp_size=self.ps.tp_size,ep_size=self.ps.moe_ep_size)
        return result
    Scheduler.get_internal_state=state
    _DONE=True
