"""Startup-only topology proof, kernel warmup and native PP boundary setup."""
import os
from pathlib import Path
import shutil
from common import *
_INSTALLED=None

def isolate_humming():
    base=Path('/tmp')/f'{os.getuid()}-{Path(os.environ["GLM_RUN_DIR"]).name}'/os.environ['GLM_ARM']/str(os.getpid())
    target=base/'humming'
    if not target.exists():shutil.copytree(SEED,target,ignore=shutil.ignore_patterns('lock','*.lock'))
    (base/'tmp').mkdir(parents=True,exist_ok=True)
    os.environ['HUMMING_CACHE_DIR']=str(target);os.environ['HUMMING_TMP_DIR']=str(base/'tmp')

def install():
    global _INSTALLED
    if _INSTALLED==os.getpid():return
    isolate_humming()
    from verify_as_decode import install as install_verify_fix
    install_verify_fix()
    from pp_local_feedback import install as install_local_feedback
    install_local_feedback()
    from admission_adapter import install_scheduler
    if os.environ['GLM_LAYOUT']=='ep8':install_scheduler()
    from scheduler_hooks import install as cache_hooks
    cache_hooks()
    from spec_proof import install as install_graph_proof
    install_graph_proof()
    import deep_ep,deep_ep_cpp
    actual=Path(deep_ep_cpp.__file__).resolve()
    if actual!=NATIVE or sha(actual)!=NATIVE_SHA:raise RuntimeError('Different DeepEP native')
    original_init=deep_ep.Buffer.__init__
    def init(self,*args,**kwargs):
        if kwargs.get('low_latency_mode') is not True:raise RuntimeError('LL required')
        original_init(self,*args,**kwargs)
        if not self.runtime.get_low_latency_nvlink_pair_only() or self.runtime.get_low_latency_pcie_p2p_max_tokens()!=0:
            raise RuntimeError('Different communication route')
        from sglang.srt.runtime_context import get_parallel
        p=get_parallel()
        save(f'{os.environ["GLM_ARM"]}/route_rank{p.world_rank}.json',dict(passed=True,rank=p.world_rank,
            low_latency=True,native=str(actual),sha256=sha(actual),nvlink_pair_only=True,pcie_threshold=0,
            route_policy='NVLink within pairs, RDMA outside pairs; policy proof, not packet trace'))
    deep_ep.Buffer.__init__=init
    from sglang.srt.model_executor.model_runner import ModelRunner
    original=ModelRunner.load_model
    def load(self,*args,**kwargs):
        from load_logging import loading
        with loading(self.is_draft_worker):
            result=original(self,*args,**kwargs)
        from sglang.srt.runtime_context import get_parallel
        p=get_parallel();arm=os.environ['GLM_LAYOUT'];outarm=os.environ['GLM_ARM'];rank=p.world_rank
        expected=(8,4,2,8,1) if arm=='ep8' else (4,4,1,4,1)
        actual=(p.tp_size,p.attn_tp_size,p.attn_dp_size,p.moe_ep_size,p.moe_tp_size)
        if actual!=expected:raise RuntimeError('Different Attention/Expert layout: '+str(actual))
        expected_experts=20 if arm=='ep8' else 40
        layers=[]
        for name,layer in self.model.named_modules():
            if name.endswith('.mlp.experts'):
                if type(layer.quant_method).__name__!='HummingMoEMethod' or layer.num_local_experts!=expected_experts:
                    raise RuntimeError('Unexpected Expert kernel/partition')
                layers.append(name)
        if self.is_draft_worker:
            if len(layers)!=1:raise RuntimeError('MTP must load exactly one expert layer')
            save(f'{outarm}/draft_weights_rank{rank}.json',dict(passed=True,rank=rank,layers=layers,local_experts=expected_experts,canonical_key=getattr(self.model,'_canonical_cpu_weight_cache_key',None)))
            return result
        cfg=config(Path(os.environ['GLM_RUN_DIR']))
        partition=cfg.get('target_pp_partition',[46,46])
        expected_moe=partition[p.pp_rank]-(3 if p.pp_rank==0 else 0)
        if len(layers)!=(89 if arm=='ep8' else expected_moe):
            raise RuntimeError('Incomplete model layer coverage')
        if arm=='pp2ep4' and (self.model.start_layer,self.model.end_layer)!=(sum(partition[:p.pp_rank]),sum(partition[:p.pp_rank+1])):
            raise RuntimeError('Live PP layer partition differs from frozen configuration')
        changes=[]
        if arm=='pp2ep4':
            from pp_boundary import apply_boundaries
            from sglang.srt.layers.communicator import ScatterMode
            changes=apply_boundaries(self.model.model,ScatterMode.TP_ATTN_FULL)
        save(f'{outarm}/weights_rank{rank}.json',dict(passed=True,rank=rank,layout=arm,
            topology=list(actual),local_experts=expected_experts,moe_layers=layers,boundary_changes=changes))
        from expert_preload import preload,warm_communication
        preload(self.model,arm,rank);warm_communication(self.model,arm,rank)
        if arm=='pp2ep4':
            from pp_p2p_warm import warm
            warm(self,arm,rank)
        return result
    ModelRunner.load_model=load
    from stage_activity import install as install_activity
    install_activity()
    from pp_flow import install as install_flow
    install_flow()
    _INSTALLED=os.getpid()
