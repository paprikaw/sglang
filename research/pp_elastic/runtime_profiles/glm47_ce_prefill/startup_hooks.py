"""Untimed ownership, A100 mixed-topology and eager CE communication evidence."""
import os
from pathlib import Path
from common import save, native_info, sha, case
_INSTALLED_PID=None

def install():
    global _INSTALLED_PID
    if _INSTALLED_PID==os.getpid():return dict(passed=True,already_installed=True)
    from sglang.srt.model_executor.model_runner import ModelRunner
    arm=os.environ['ARM'];selected=case(arm)
    original=ModelRunner.load_model
    def load(self):
        result=original(self)
        from sglang.srt.runtime_context import get_parallel
        p=get_parallel()
        assert (p.tp_size,p.attn_tp_size,p.attn_dp_size,p.moe_ep_size,p.moe_tp_size,p.pp_size)==(selected['tp'],4,selected['dp'],selected['ep'],1,selected['pp'])
        rank=p.pp_rank*4+p.tp_rank if selected['pp']>1 else p.tp_rank
        start=p.pp_rank*4 if selected['pp']>1 else p.tp_rank//4*4
        assert list(p.attn_tp_group.ranks)==list(range(start,start+4))
        if selected['pp']>1:
            from pp_boundary import apply_boundaries
            from sglang.srt.layers.communicator import ScatterMode
            changes=apply_boundaries(self.model.model,ScatterMode.TP_ATTN_FULL)
            save(f'{arm}/boundary_rank{rank}.json',dict(passed=True,rank=rank,changes=changes,startup_only=True))
        records=[]
        for name,layer in self.model.named_modules():
            if not name.endswith('.mlp.experts'):continue
            assert type(layer.quant_method).__name__=='HummingMoEMethod'
            assert (layer.num_local_experts,layer.intermediate_size_per_partition)==(160//selected['ep'],1536)
            records.append(dict(name=name,experts=layer.num_local_experts,intermediate=layer.intermediate_size_per_partition,
                quant_method=type(layer.quant_method).__name__,dispatcher=type(layer.dispatcher).__name__))
        assert len(records)==selected['moe_layers_by_rank'][rank]
        import pynvml
        pynvml.nvmlInit()
        try:
            handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
            local_rank=p.tp_rank%4;name=pynvml.nvmlDeviceGetName(handles[local_rank]);name=name.decode() if isinstance(name,bytes) else name
            assert 'A100' in name
            links=[i==local_rank or pynvml.nvmlDeviceGetP2PStatus(handles[local_rank],h,pynvml.NVML_P2P_CAPS_INDEX_NVLINK)==pynvml.NVML_P2P_STATUS_OK for i,h in enumerate(handles)]
            assert links==[i in (local_rank,local_rank^1) for i in range(4)],links
        finally:pynvml.nvmlShutdown()
        save(f'{arm}/weights_rank{rank}.json',dict(passed=True,rank=rank,arm=arm,pp_rank=p.pp_rank,
            attn_tp=4,attn_dp=selected['dp'],moe_ep=p.moe_ep_size,moe_tp=p.moe_tp_size,layers=records,gpu=name,nvlink_peers=links,startup_only=True))
        from expert_preload import preload,warm_communication
        preload(self.model,arm,rank);warm_communication(self.model,arm,rank)
        return result
    ModelRunner.load_model=load
    import deep_ep,deep_ep_cpp
    actual=Path(deep_ep_cpp.__file__).resolve();expected=native_info(arm)
    assert str(actual)==expected['path'] and sha(actual)==expected['sha256']
    from ce_serving import install as install_ce
    ce_installation=install_ce(Path(os.environ['RUN_DIR']).resolve(),arm)
    original_init=deep_ep.Buffer.__init__
    def init(self,*args,**kwargs):
        assert kwargs.get('low_latency_mode') is True
        group=args[0] if args else kwargs['group']
        import comm_validation as check
        import torch
        check.torch=torch;check.WORLD=selected['ep'];check.LOCAL_EXPERTS=160//selected['ep']
        check.EP_CAPACITY=check.PREFILL_DISPATCH_ROWS=selected['ep_capacity']
        topology=check.pair_topology(group)
        original_init(self,*args,**kwargs)
        assert self.runtime.get_low_latency_nvlink_pair_only() is True
        threshold=selected['ep_capacity'] if selected['transport']=='ce' else 0
        assert self.runtime.get_low_latency_pcie_p2p_max_tokens()==threshold
        rank=int(os.environ['NODE_RANK'])*4+self.rank if selected['pp']>1 else self.rank
        mapped=check.peer_mapping(self)
        save(f'{arm}/route_rank{rank}.json',dict(passed=True,rank=rank,local_ep_rank=self.rank,
            native_path=str(actual),native_sha256=sha(actual),low_latency=True,topology=topology,peer_mapped=mapped,
            nvlink_pair_only_enabled=True,pcie_p2p_max_tokens=threshold,transport=selected['transport'],ce_installation=ce_installation,capacity_per_dispatch_rank=selected['ep_capacity'],
            nvlink_allowed=kwargs.get('allow_nvlink_for_low_latency_mode'),effective_disable_p2p=os.environ.get('NVSHMEM_DISABLE_P2P'),
            nic_handler_requested=os.environ.get('NVSHMEM_IBGDA_NIC_HANDLER'),startup_only=True,
            interpretation='Policy verification; transport-specific tracing is not collected in formal samples.'))
        save(f'{arm}/comm_check_rank{rank}.json',dict(**check.check_communication(self,group,capture=False),global_rank=rank))
        if selected['transport']=='ce':
            from ce_serving import snapshot
            counters=snapshot(self)
            assert counters['counters'][0]>0 and counters['counters'][1]>0
            save(f'{arm}/ce_after_validation_rank{rank}.json',dict(passed=True,rank=rank,
                counters=counters,startup_only=True,timed=False))
    deep_ep.Buffer.__init__=init
    _INSTALLED_PID=os.getpid()
    return dict(passed=True,source_sha256=sha(__file__),startup_only=True)
