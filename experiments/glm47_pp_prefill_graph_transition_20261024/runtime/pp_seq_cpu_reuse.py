"""Reuse exact CPU sequence lengths already read for tail draft planning.

No speculative state, KV placement, GPU stream, proposal or model operation is
changed. Controls keep both redundant D2H copies. Both arms capture the same
small immutable CPU tensor; only warm controls compare it against original D2H.
The adapter compiles three exact pinned function bodies, replacing one seam in
each. Failing to find a unique seam aborts startup.
"""
import ast
import collections
import copy
from pathlib import Path

COUNTER_KEYS=('captured','scheduler_reused','scheduler_control','scheduler_fallback',
              'retire_reused','retire_control','retire_head','warm_scheduler_equal','warm_retire_equal')
_COUNTS=collections.Counter({k:0 for k in COUNTER_KEYS})
_INSTALLED=False
_TRANSFORMS={}
_ALWAYS=False


def cohort(batch):
    rids=[r.rid for r in batch.reqs]
    if not rids:raise RuntimeError('Sequence CPU reuse requires a nonempty batch')
    flags=[rid.startswith('control-') for rid in rids]
    if any(flags) and not all(flags):raise RuntimeError('Mixed sequence CPU reuse AB cohort')
    return rids,all(flags) and not _ALWAYS,all(rid.startswith('control-warm-') for rid in rids)


def capture(batch,result):
    rids,_,_=cohort(batch)
    cpu=batch.seq_lens_cpu
    if cpu is None or cpu.device.type!='cpu':raise RuntimeError('Draft has no exact CPU sequence lengths')
    result._pp_exact_seq_cpu=(tuple(rids),cpu.clone())
    _COUNTS['captured']+=1


def validated(cpu,device):
    if cpu.device.type!='cpu' or cpu.shape!=device.shape or cpu.dtype!=device.dtype:
        raise RuntimeError('CPU sequence length provenance/shape/dtype mismatch')
    return cpu


def scheduler_value(scheduler,batch,result):
    rids,control,warm=cohort(batch)
    saved=getattr(result,'_pp_exact_seq_cpu',None)
    if saved is None:
        # Prefill and any zero-step path do not have a prior exact CPU readback.
        _COUNTS['scheduler_fallback']+=1
        return result.new_seq_lens.to('cpu')
    if saved[0]!=tuple(rids):raise RuntimeError('CPU sequence length request ancestry changed')
    cpu=validated(saved[1],result.new_seq_lens)
    if control:
        actual=result.new_seq_lens.to('cpu')
        _COUNTS['scheduler_control']+=1
        if warm:
            import torch
            if not torch.equal(cpu,actual):raise RuntimeError('Warm scheduler CPU sequence readback differs')
            _COUNTS['warm_scheduler_equal']+=1
        return actual
    if warm:
        import torch
        if not torch.equal(cpu,result.new_seq_lens.to('cpu')):raise RuntimeError('Warm scheduler CPU sequence readback differs')
        _COUNTS['warm_scheduler_equal']+=1
    _COUNTS['scheduler_reused']+=1
    return cpu


def retire_value(scheduler,batch,metadata,new_seq_lens):
    _,control,warm=cohort(batch)
    if not scheduler.pp_group.is_last_rank:
        _COUNTS['retire_head']+=1
        return new_seq_lens.to('cpu')
    snapshot=metadata.fwd_batch
    if snapshot is None or [r.rid for r in snapshot.reqs]!=[r.rid for r in batch.reqs]:
        raise RuntimeError('CPU retirement reuse requires exact saved forward ancestry')
    cpu=validated(snapshot.seq_lens_cpu,new_seq_lens)
    if control:
        actual=new_seq_lens.to('cpu')
        _COUNTS['retire_control']+=1
        if warm:
            import torch
            if not torch.equal(cpu,actual):raise RuntimeError('Warm tail retirement CPU sequence readback differs')
            _COUNTS['warm_retire_equal']+=1
        return actual
    if warm:
        import torch
        if not torch.equal(cpu,new_seq_lens.to('cpu')):raise RuntimeError('Warm tail retirement CPU sequence readback differs')
        _COUNTS['warm_retire_equal']+=1
    _COUNTS['retire_reused']+=1
    return cpu


def transform(source,class_name,function_name,kind):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==function_name))
    class Patch(ast.NodeTransformer):
        hits=0
        def visit_Assign(self,node):
            if len(node.targets)!=1 or ast.unparse(node.targets[0])!='batch.seq_lens_cpu':return self.generic_visit(node)
            expected={'worker':"batch_output.new_seq_lens.to('cpu')",'scheduler':"batch_result.new_seq_lens.to('cpu')",'retire':"new_seq_lens.to('cpu')"}[kind]
            if ast.dump(node.value,include_attributes=False)!=ast.dump(ast.parse(expected,mode='eval').body,include_attributes=False):
                return self.generic_visit(node)
            self.hits+=1
            if kind=='worker':
                # Keep the REQUIRED pre-draft D2H and capture its exact CPU output.
                extra=ast.parse('_pp_capture_seq_cpu(batch, batch_output)').body[0]
                return [node,ast.copy_location(extra,node)]
            expr='_pp_scheduler_seq_cpu(self, batch, batch_result)' if kind=='scheduler' else '_pp_retire_seq_cpu(self, batch, mb_metadata, new_seq_lens)'
            node.value=ast.copy_location(ast.parse(expr,mode='eval').body,node.value)
            return node
    patch=Patch();fn=patch.visit(fn)
    if patch.hits!=1:raise RuntimeError(f'Expected one {kind} CPU-copy seam, found {patch.hits}')
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    return ast.fix_missing_locations(mod),dict(class_name=class_name,function=function_name,kind=kind,hits=patch.hits)


def install():
    global _INSTALLED,_ALWAYS
    import os
    from common import config
    cfg=config(Path(os.environ['GLM_RUN_DIR']))
    if not cfg.get('seq_cpu_reuse_ab'):return
    _ALWAYS=bool(cfg.get('seq_cpu_reuse_always'))
    if _INSTALLED:raise RuntimeError('CPU sequence reuse already installed')
    if not (cfg['layout']=='pp2ep4' and cfg['mtp'] and cfg.get('local_tail_feedback')):
        raise RuntimeError('CPU sequence reuse only qualified for two-slot PP2 MTP')
    from sglang.srt.speculative import eagle_worker_v2
    from sglang.srt.managers import scheduler,scheduler_pp_mixin
    for module,class_name,function,kind in (
        (eagle_worker_v2,'EAGLEWorkerV2','forward_batch_generation','worker'),
        (scheduler,'Scheduler','run_batch','scheduler'),
        (scheduler_pp_mixin,'SchedulerPPMixin','_pp_prep_batch_result','retire')):
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(Path(cfg['engine_root']).resolve()):raise RuntimeError('CPU-copy seam outside frozen engine')
        code,proof=transform(path.read_text(),class_name,function,kind)
        scope=dict(vars(module),_pp_capture_seq_cpu=capture,_pp_scheduler_seq_cpu=scheduler_value,_pp_retire_seq_cpu=retire_value)
        exec(compile(code,str(path),'exec'),scope)
        setattr(getattr(module,class_name),function,scope[function])
        _TRANSFORMS[kind]=dict(proof,path=str(path))
    _INSTALLED=True


def receipt(scheduler):
    return dict(enabled=_INSTALLED,**dict(_COUNTS),source_seams=_TRANSFORMS,
                exact_existing_readback=True,model_math_unchanged=True,required_planning_d2h_retained=True,
                controls_keep_original_copies=not _ALWAYS,always_reuse=_ALWAYS,immutable_cpu_snapshot=True,
                pp_rank=int(scheduler.ps.pp_rank))


def validate_delta(before,after,*,control):
    if not before.get('enabled') or not after.get('enabled'):raise RuntimeError('CPU sequence reuse receipt missing')
    delta={key:after[key]-before[key] for key in COUNTER_KEYS}
    if after['pp_rank']==0:
        if delta['retire_head']<=0 or any(delta[k] for k in ('captured','scheduler_reused','scheduler_control','retire_reused','retire_control')):
            raise RuntimeError('Head CPU-copy path changed unexpectedly')
    else:
        active=('scheduler_control','retire_control') if control else ('scheduler_reused','retire_reused')
        inactive=('scheduler_reused','retire_reused') if control else ('scheduler_control','retire_control')
        if delta['captured']<=0 or any(delta[k]<=0 for k in active) or any(delta[k] for k in inactive):
            raise RuntimeError('Tail CPU-copy activation does not match AB cohort')
        if after['warm_scheduler_equal']<=0 or after['warm_retire_equal']<=0:
            raise RuntimeError('No warm equality proof against both original D2H copies')
    if any(delta[k] for k in ('warm_scheduler_equal','warm_retire_equal')):
        raise RuntimeError('Warm-only CPU comparisons contaminated formal measurement')
    return delta
