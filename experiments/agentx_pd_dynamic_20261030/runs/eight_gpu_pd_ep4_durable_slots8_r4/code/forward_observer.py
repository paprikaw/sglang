"""Count Python-visible real forwards; no CUDA operations, tensor reads or fences."""
from collections import Counter
from functools import wraps
import json
import mmap
import os
from pathlib import Path
import threading
import time

_COUNTS=Counter()
_DONE=False
_SHAPE=None


def install():
    global _DONE
    if _DONE:return
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.managers.scheduler import Scheduler
    original=ModelRunner.forward
    @wraps(original)
    def forward(self, forward_batch, *args, **kwargs):
        global _SHAPE
        if _SHAPE is None:
            _SHAPE=dict(pp_rank=self.ps.pp_rank,tp_rank=self.ps.tp_rank,
                pp_size=self.ps.pp_size,tp_size=self.ps.tp_size,ep_size=self.ps.moe_ep_size)
        _COUNTS[f'{forward_batch.forward_mode.name}:b{forward_batch.batch_size}']+=1
        return original(self,forward_batch,*args,**kwargs)
    ModelRunner.forward=forward
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
    dispatch=DeepEPDispatcher.dispatch
    @wraps(dispatch)
    def count_dispatch(self,*args,**kwargs):
        result=dispatch(self,*args,**kwargs)
        _COUNTS['dispatch_output:'+type(result).__name__]+=1
        return result
    DeepEPDispatcher.dispatch=count_dispatch
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
    # PP's HTTP introspection reports the head stage only. Each rank publishes
    # a counter snapshot in response to an idle-boundary epoch; file writes are
    # performed by this CPU thread outside all measured HTTP intervals.
    from snapshot_transport import open_control
    fd=open_control(os.environ['BENCH_FORWARD_COUNTER_CONTROL'])
    try:control=mmap.mmap(fd,1,access=mmap.ACCESS_READ)
    finally:os.close(fd)
    root=Path(os.environ['EP2_PP2_RUN'])/'forward_counters'
    root.mkdir(parents=True,exist_ok=True)
    def publish():
        previous=0
        while True:
            epoch=control[0]
            if epoch and epoch!=previous and _SHAPE is not None:
                value=dict(epoch=epoch,shape=_SHAPE,counts=dict(_COUNTS))
                path=root/f'pp{_SHAPE["pp_rank"]}.tp{_SHAPE["tp_rank"]}.json'
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(value))
                os.replace(temporary,path)
                previous=epoch
            time.sleep(.02)
    threading.Thread(target=publish,daemon=True).start()
    _DONE=True
