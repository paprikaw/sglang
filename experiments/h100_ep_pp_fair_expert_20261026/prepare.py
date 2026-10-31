"""Freeze a new fair EP/PP run without changing previous experiment inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'h100_pp_expert_fragmentation_20261026'


def descriptor(path):
    path=Path(path).absolute()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['native','triton_small'],required=True)
    p.add_argument('--arms',nargs='+',choices=['ep2','pp2'],default=['ep2','pp2'])
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.run.exists():
        raise RuntimeError('Refusing to modify an existing run; choose a new immutable run directory')
    module_spec=importlib.util.spec_from_file_location('fragment_prepare',REFERENCE/'prepare.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    result=module.build(args.run,args.mode,args.arms,args.repeats)
    run=args.run.absolute();code=run/'code'
    for name in ['ep_triton.py','ep_fragmentation.py','fair_validation.py']:
        shutil.copyfile(ROOT/name,code/name)
    adapter=code/'runtime_adapter.py'
    adapter.write_text(adapter.read_text().replace('fragmentation.install()',
        'fragmentation.install()\n    import ep_fragmentation\n    ep_fragmentation.install()'))
    benchmark=code/'benchmark.py'
    benchmark.write_text(benchmark.read_text().replace("if arm == 'pp2':", "if arm in ('ep2','pp2'):")
        .replace('validation.append(base.cohort(cfg, key, packets))',
                 'from fair_validation import atomic_validation\n        validation.append(atomic_validation(base,cfg,arm,key,tokens))'))
    cfg=json.loads((run/'config.json').read_text())
    cfg.update(port=24421,dist_port=24422,fair_expert_comparison=True,
        numerical_layer_relative_l2_limit=.015,
        shared_small_gemm='sglang.kernels.ops.moe.fused_moe_triton_kernels.invoke_fused_moe_kernel',
        ep_layout_adapter='ep_triton.py',
        comparison_scope='Same BF16 small-batch Triton GEMM family on both EP2 and PP2; unchanged DeepEP and Attention; all layout-adapter costs included')
    save(run/'config.json',cfg)
    spec=json.loads((run/'dashboard_spec.json').read_text())
    spec.update(title='H100 fair small-batch Expert comparison: '+args.mode,port=24420)
    spec['tasks'][0]['label']='Fair EP2 / PP2 '+args.mode
    inputs=[d for d in spec['frozen_inputs'] if not Path(d['path']).is_relative_to(code)
            and d['path']!=str(run/'config.json')]
    inputs += [descriptor(path) for path in sorted(code.glob('*.py'))]+[descriptor(run/'config.json')]
    spec['frozen_inputs']=list({d['path']:d for d in inputs}.values())
    save(run/'dashboard_spec.json',spec)
    manifest=json.loads((run/'launch_manifest.json').read_text())
    manifest.update(fair_expert_comparison=True,private_adapters=[descriptor(path) for path in sorted(code.glob('*.py'))],
        comparison_scope=cfg['comparison_scope'],shared_small_gemm=cfg['shared_small_gemm'])
    save(run/'launch_manifest.json',manifest)
    registry=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/kv_checkpoints/registry.json')
    data=json.loads(registry.read_text())
    if 'Qwen3.6-35B-A3B' in str(data):
        raise RuntimeError('A matching model checkpoint candidate now exists; perform live compatibility validation before launch')
    save(run/'kv_checkpoint_discovery.json',dict(registry=str(registry),sha256=descriptor(registry)['sha256'],
        registered_model_candidate=False,reason='No checkpoint for this Qwen hybrid GDN model',
        selected_protocol='Real native model-produced GPU radix prefixes and GDN state; no CPU KV restore'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
