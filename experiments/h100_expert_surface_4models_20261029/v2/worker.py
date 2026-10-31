"""Real single-layer Expert compute. Imports CUDA only after Guard authentication."""
import argparse,gc,hashlib,importlib.util,json,math,os,socket,statistics,struct,sys,time,traceback
from pathlib import Path
from common import ROOT,ENGINE,MODELS,PLAN,authenticate,save,sha

LEGACY=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments/glm47_ep4_local_expert_profile_20261005_r2/harness')
def load_helper():
    sys.path.insert(0,str(LEGACY))
    s=importlib.util.spec_from_file_location('real_awq_expert_loader',LEGACY/'profile_entry.py')
    m=importlib.util.module_from_spec(s)
    import types
    alias=types.ModuleType('common')
    alias.MODEL=Path(MODELS['glm47']['path']);alias.SG=ENGINE
    alias.SG_COMMIT='870269702285d94bdd926a50cc29ad1a20bcf60c';alias.save=save;alias.sha=sha
    previous=sys.modules.get('common');sys.modules['common']=alias
    try:s.loader.exec_module(m)
    finally:
        if previous is None:sys.modules.pop('common',None)
        else:sys.modules['common']=previous
    return m
def summary(xs):
    ys=sorted(xs)
    return dict(n=len(xs),median_ms=statistics.median(xs),mean_ms=statistics.mean(xs),min_ms=min(xs),max_ms=max(xs),p90_ms=ys[math.ceil(.9*len(ys))-1])

def main(run):
    receipt,spec,binding=authenticate(run)
    worker_id=os.environ['SURFACE_WORKER_ID'];cfg=json.loads((run/'config.json').read_text())
    w=next(x for x in cfg['workers'] if x['id']==worker_id);node=socket.gethostname().split('.')[0]
    assert node==w['node'] and os.environ['CUDA_VISIBLE_DEVICES']==str(w['gpu'])
    model=w['model'];m=MODELS[model];selected=json.loads((run/'selection.json').read_text())[model]
    assert selected['model_path']==m['path'] and selected['resident']==m['resident']
    folder=run/'workers'/worker_id;rows=[];numerical=[];provenance={};begun=time.time()
    def progress(stage,**extra):save(folder/'progress.json',dict(timestamp=time.time(),stage=stage,valid_points=len(rows),**extra))
    progress('loading',model=model)
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    torch.cuda.set_device(0);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability(0)==(9,0) and 'H100' in torch.cuda.get_device_name(0)
    set_global_server_args_for_scheduler(ServerArgs(model_path=m['path'],dtype='bfloat16',
        quantization='humming' if model=='glm47' else 'fp8' if model=='qwen122b' else None,
        moe_runner_backend='humming' if model=='glm47' else 'deep_gemm',attention_backend='triton',
        disable_radix_cache=True,disable_custom_all_reduce=True,disable_shared_experts_fusion=True))
    root=Path(m['path'])
    assert sha(root/'config.json')==selected['config_sha256'] and sha(root/'model.safetensors.index.json')==selected['index_sha256']
    for file,info in selected['source_files'].items():
        p=root/file;st=p.stat();assert st.st_size==info['size'] and st.st_mtime_ns==info['mtime_ns']
        with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=f.read(n)
        assert hashlib.sha256(header).hexdigest()==info['header_sha256']
    resident,H,I=m['resident'],m['hidden'],m['intermediate'];raw_awq=None
    if model=='glm47':
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore,HummingMoeQuantInfo,HummingRunnerInput,HummingGemmType
        helper=load_helper();helper.LOCAL_EXPERTS=resident;helper.EXPECTED_CHECKPOINT_TENSOR_BYTES=selected['total_tensor_bytes']
        layer,raw_awq,origin=helper.load_real_experts(root)
        for expert,ps in raw_awq.items():
            for projection,fs in ps.items():
                for field,tensor in fs.items():
                    name=f'model.layers.3.mlp.experts.{expert}.{projection}.{field}'
                    actual=hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert actual==selected['tensors'][name]['sha256'];provenance[name]=actual
        core=HummingRunnerCore(MoeRunnerConfig(num_experts=160,num_local_experts=resident,hidden_size=H,
               intermediate_size_per_partition=I,top_k=8,activation='silu'))
        quant=HummingMoeQuantInfo(layer)
    else:
        fp8=model=='qwen122b';dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
        w13=torch.empty((resident,2*I,H),device='cuda',dtype=dtype);w2=torch.empty((resident,H,I),device='cuda',dtype=dtype)
        if fp8:
            s13=torch.empty((resident,2*I//128,H//128),device='cuda',dtype=torch.float32)
            s2=torch.empty((resident,H//128,I//128),device='cuda',dtype=torch.float32)
        grouped={}
        for name,item in selected['tensors'].items():grouped.setdefault(item['file'],[]).append(name)
        for file,names in sorted(grouped.items()):
            with safe_open(root/file,framework='pt',device='cpu') as handle:
                for name in names:
                    value=handle.get_tensor(name).clone();item=selected['tensors'][name]
                    assert list(value.shape)==item['shape'] and value.nbytes==item['bytes']
                    digest=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    if 'sha256' in item:assert digest==item['sha256']
                    provenance[name]=digest
                    if model=='qwen35b':
                        (w13 if name.endswith('gate_up_proj') else w2).copy_(value)
                    else:
                        suffix=name.split('.mlp.experts.')[1].split('.');e=int(suffix[0]);projection,field=suffix[1:]
                        if field=='weight':
                            if projection=='gate_proj':w13[e,:I].copy_(value)
                            elif projection=='up_proj':w13[e,I:].copy_(value)
                            else:w2[e].copy_(value)
                        else:
                            if projection=='gate_proj':s13[e,:I//128].copy_(value.float())
                            elif projection=='up_proj':s13[e,I//128:].copy_(value.float())
                            else:s2[e].copy_(value.float())
                    del value
        from sglang.srt.layers import deep_gemm_wrapper as dg
        from sglang.kernels.ops.activation.activation import silu_and_mul
        if fp8:
            from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8 as quantize
            from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
            assert not dg.DEEPGEMM_SCALE_UE8M0,'This qualified H100 path uses FP32 block scales'
    torch.cuda.synchronize();torch.manual_seed(20261029)
    save(folder/'weights.json',dict(model=model,tensors=provenance,selected_weight_bytes=selected['total_tensor_bytes']))
    weight_digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    props=torch.cuda.get_device_properties(0)
    gpu=dict(name=props.name,total_bytes=props.total_memory,capability=list(torch.cuda.get_device_capability(0)),uuid=str(getattr(props,'uuid','unknown')))
    references={}
    def reference(expert,x):
        if model=='glm47':return helper.reference_expert(x,raw_awq[expert])
        if expert not in references:
            if model=='qwen122b':
                a=w13[expert].float()*s13[expert].repeat_interleave(128,0).repeat_interleave(128,1)
                b=w2[expert].float()*s2[expert].repeat_interleave(128,0).repeat_interleave(128,1)
            else:a=w13[expert].float();b=w2[expert].float()
            references[expert]=(a,b)
        a,b=references[expert]
        gu=F.linear(x.float(),a).to(torch.bfloat16)
        act=(F.silu(gu[:,:I].float())*gu[:,I:].float()).to(torch.bfloat16)
        return F.linear(act.float(),b).to(torch.bfloat16).float()
    def prepared(total,active,all_x):
        cs=[total//active+(e<total%active) if e<active else 0 for e in range(resident)]
        if model=='glm47':
            # Match the old A100 fixed stride through T8192; extend the stride
            # only at larger T. It stays identical across all E at a given T.
            capacity=max(8192,total)
            recv=torch.zeros((resident,capacity,H),device='cuda',dtype=torch.bfloat16)
            offset=0
            for e,n in enumerate(cs[:active]):recv[e,:n].copy_(all_x[offset:offset+n]);offset+=n
            counts=torch.tensor(cs,device='cuda',dtype=torch.int32)
            inp=HummingRunnerInput(hidden_states=recv,topk_ids=torch.zeros((max(1,total//8),8),device='cuda',dtype=torch.int32),
              topk_weights=torch.full((max(1,total//8),8),1/8,device='cuda',dtype=torch.float32),expert_num_tokens=counts,
              expected_m=max(1,math.ceil(total/resident)),gemm_type=HummingGemmType.GROUPED_MASKED,apply_routed_scaling_factor=False)
            def invoke():return core.run(inp,quant,{}).hidden_states
            def select(out,e,n):return out[e,:n]
            meta=dict(actual_padded_rows=None,capacity_per_expert=capacity,expected_m=inp.expected_m)
        else:
            sizes=[math.ceil(n/128)*128 for n in cs[:active]];padded=sum(sizes)
            x=torch.zeros((padded,H),device='cuda',dtype=torch.bfloat16)
            ids=torch.empty((padded,),device='cuda',dtype=torch.int32);offset=0;original=0;offsets=[]
            for e,(n,pad) in enumerate(zip(cs,sizes)):
                offsets.append(offset);x[offset:offset+n].copy_(all_x[original:original+n]);ids[offset:offset+pad].fill_(e)
                offset+=pad;original+=n
            gu=torch.empty((padded,2*I),device='cuda',dtype=torch.bfloat16)
            down=torch.empty((padded,I),device='cuda',dtype=torch.bfloat16)
            out=torch.empty((padded,H),device='cuda',dtype=torch.bfloat16)
            if model=='qwen122b':
                xf,sx=quantize(x,128,column_major_scales=False,scale_tma_aligned=False,scale_ue8m0=False)
                if dg.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:sx=tma_align_input_scale(sx)
                def invoke():
                    dg.grouped_gemm_nt_f8f8bf16_contig((xf,sx),(w13,s13),gu,ids)
                    silu_and_mul(gu,down)
                    q,s=quantize(down,128,column_major_scales=False,scale_tma_aligned=False,scale_ue8m0=False)
                    if dg.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:s=tma_align_input_scale(s)
                    dg.grouped_gemm_nt_f8f8bf16_contig((q,s),(w2,s2),out,ids)
                    return out
            else:
                def invoke():
                    dg.grouped_gemm_nt_bf16_contig(x,w13,gu,ids)
                    silu_and_mul(gu,down)
                    dg.grouped_gemm_nt_bf16_contig(down,w2,out,ids)
                    return out
            def select(output,e,n):return output[offsets[e]:offsets[e]+n]
            meta=dict(actual_padded_rows=padded,padding_ratio=padded/total,alignment_rows=128)
        return cs,invoke,select,meta
    try:
        progress('warming',model=model)
        for total in PLAN['tokens']:
            all_x=torch.randn((total,H),device='cuda',dtype=torch.bfloat16)*.125
            es=[e for e in m['experts'] if e<=total];raw={(e,mode):[] for e in es for mode in PLAN['modes']};metas={};block_stats={key:[] for key in raw}
            for block in range(3):
                order=es if block==0 else list(reversed(es)) if block==1 else es[len(es)//2:]+es[:len(es)//2]
                for active in order:
                    progress('measuring',model=model,T=total,E=active,block=block)
                    cs,fn,select,meta=prepared(total,active,all_x)
                    for _ in range(5):output=fn()
                    torch.cuda.synchronize()
                    if block==0:
                        for e in sorted({0,active-1}):
                            n=min(2,cs[e]);original=sum(cs[:e]);expected=reference(e,all_x[original:original+n]);actual=select(output,e,n).float()
                            delta=actual-expected;nrms=(delta.norm()/expected.norm().clamp_min(1e-12)).item();err=delta.abs().max().item();peak=expected.abs().max().item()
                            limit=.12 if model=='qwen122b' else .02
                            valid=bool(torch.isfinite(actual).all()) and nrms<=limit and err<=((.30 if model=='qwen122b' else .04)*peak+1e-4)
                            numerical.append(dict(T=total,E=active,expert=e,nrms=nrms,max_abs=err,reference_peak=peak,relative_l2_limit=limit,passed=valid))
                            assert valid,numerical[-1]
                    for mode in PLAN['modes']:
                        graph=None
                        if mode=='cuda_graph':
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):captured=fn()
                            call=graph.replay
                        else:call=fn
                        for _ in range(3):call()
                        starts=[torch.cuda.Event(enable_timing=True) for _ in range(10)];ends=[torch.cuda.Event(enable_timing=True) for _ in range(10)]
                        for a,b in zip(starts,ends):a.record();call();b.record()
                        ends[-1].synchronize();xs=[a.elapsed_time(b) for a,b in zip(starts,ends)]
                        raw[(active,mode)].extend(xs);block_stats[(active,mode)].append(statistics.median(xs))
                        del call,graph
                        if mode=='cuda_graph':del captured
                    metas[active]=meta
                    del fn,select,output;gc.collect();torch.cuda.empty_cache()
            for active in es:
                for mode in PLAN['modes']:
                    xs=raw[(active,mode)];timing=summary(xs)
                    rows.append(dict(T=total,E=active,resident=resident,mode=mode,backend=m['backend'],counts=[total//active+(e<total%active) if e<active else 0 for e in range(resident)],
                         gemm_flops=6*H*I*total,raw_ms=xs,block_median_ms=block_stats[(active,mode)],timing=timing,effective_tflops=6*H*I*total/timing['median_ms']/1e9,**metas[active]))
            save(folder/'partial.json',dict(passed=None,rows=rows,numerical=numerical))
            progress('measuring',model=model,T=total,note='T slice complete')
            print('EXPERT_SURFACE_SLICE',worker_id,total,len(rows),flush=True)
            del all_x;gc.collect();torch.cuda.empty_cache()
        expected_rows=sum(e<=t for t in PLAN['tokens'] for e in m['experts'])*2
        assert len(rows)==expected_rows and all(x['passed'] for x in numerical)
        value=dict(passed=True,worker=worker_id,model=model,node=node,physical_gpu=w['gpu'],replica=w['replica'],gpu=gpu,
            selection_sha256=spec['model_cache']['selection']['sha256'],lease_id=receipt['lease_id'],rows=rows,numerical=numerical,
            selected_weight_bytes=selected['total_tensor_bytes'],weight_digest=weight_digest,elapsed_s=time.time()-begun,
            engine=str(ENGINE),scope=PLAN,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(folder/'result.json',value);progress('complete',model=model);print('EXPERT_SURFACE_COMPLETE',worker_id,flush=True)
        return 0
    except BaseException as e:
        save(folder/'result.json',dict(passed=False,worker=worker_id,model=model,node=node,physical_gpu=w['gpu'],rows=rows,numerical=numerical,
             failure=dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc())))
        progress('failed',model=model,error=str(e));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run.resolve()
    try:code=main(run)
    except BaseException as e:
        worker_id=os.environ.get('SURFACE_WORKER_ID');cfg=json.loads((run/'config.json').read_text())
        w=next(x for x in cfg['workers'] if x['id']==worker_id);folder=run/'workers'/worker_id
        if not (folder/'result.json').exists():
            save(folder/'result.json',dict(passed=False,worker=worker_id,model=w['model'],node=w['node'],physical_gpu=w['gpu'],rows=[],numerical=[],
                 failure=dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc())))
        raise
    raise SystemExit(code)
