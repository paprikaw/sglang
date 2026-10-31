"""CPU-only archival after verified Guard release. Never touches GPU ownership."""
import argparse,csv,datetime,json,shutil,sys
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ACTIVE=BASE/'v4';sys.path.insert(0,str(ACTIVE))
from common import sha,save,MODELS

def replace_block(path,marker,body):
    begin=f'<!-- begin {marker} -->';end=f'<!-- end {marker} -->'
    original=path.read_text();content=original;block=begin+'\n'+body.strip()+'\n'+end
    if begin in content:
        a=content.index(begin);b=content.index(end,a)+len(end);content=content[:a]+block+content[b:]
    else:content=content.rstrip()+'\n\n'+block+'\n'
    prior=path.read_text()
    assert prior==original,'Concurrent report-index change; retry from latest contents'
    temp=path.with_name(path.name+'.h100-surface.tmp')
    temp.write_text(content);temp.replace(path)
def main(run):
    ident='h100_four_model_expert_surface_20261029'
    report=ACTIVE/'report';result=json.loads((run/'profile/result.json').read_text());spec=json.loads((run/'dashboard_spec.json').read_text())
    assert result['passed'] is True and len(result['rows'])==1584
    registry=Path('/data/gpfs/projects/punim2715/vllm_workbench/.experiment_guard/gpu_registry/registry.json')
    history=json.loads(registry.read_text())['history']
    released=[x for x in history if x['experiment_id']==spec['id'] and x['lease_id']==result['lease_id']]
    assert len(released)==1 and released[0]['state']=='released' and released[0]['cleanup_verified'] is True,'Guard has not verified and released this run'
    public={k:released[0].get(k) for k in ['experiment_id','lease_id','state','cleanup_verified','nodes','acquired_at','ended_at','end_reason']}
    aggregate=list(csv.DictReader((report/'aggregate_metrics.csv').open()))
    data={(r['model'],int(r['T_assignment_rows']),int(r['E_active']),r['mode']):float(r['median_mean_ms']) for r in aggregate}
    ratios={m:{str(t):data[m,t,32,'cuda_graph']/data[m,t,2,'cuda_graph'] for t in [128,8192,32768]} for m in MODELS}
    now=datetime.datetime.now().astimezone().isoformat()
    snapshot=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ident
    snapshot.mkdir(exist_ok=True)
    for p in report.iterdir():
        if p.is_file():shutil.copyfile(p,snapshot/p.name)
    shutil.copyfile(ACTIVE/'analyze.py',snapshot/'analyze.py')
    shutil.copyfile(BASE/'report_tools/enrich_report.py',snapshot/'enrich_report.py')
    save(snapshot/'manifest.json',dict(id=ident,title='H100 four-model real single-layer Expert token-by-active-expert surfaces',created_at=now,status='complete',
      implementation=dict(engine='sglang',sources=spec['sources'],controller=str(ACTIVE/'controller.py'),worker=str(ACTIVE/'worker.py'),
        frozen_inputs_spec=dict(path=str(run/'dashboard_spec.json'),sha256=sha(run/'dashboard_spec.json'))),
      model=MODELS,kernel_options={w['worker']:w.get('kernel_options',{}) for w in result['workers']},deployment=dict(gpu_count=8,per_worker_gpu_count=1,nodes=public['nodes'],independent_processes_per_model=2,communication=False,full_model_loaded=False),
      workload=json.loads((run/'measurement_plan.json').read_text()),gpu_release=public,
      metrics=dict(cells=1584,cuda_events_per_cell=30,E32_over_E2_graph_ratios=ratios),
      artifacts={p.name:dict(path=str(snapshot/p.name),sha256=sha(p)) for p in report.iterdir() if p.is_file()},
      original_run=str(run),report=str(snapshot/'REPORT.md'),
      attempts=dict(first_admission=str(BASE/'runs/r1'),exploratory_failed_wrapper=str(BASE/'runs/r2'),
        corrected_prelaunch_wait=str(BASE/'v2/runs/r3'),
        deepgemm_valid_glm_tma_failed=str(BASE/'v2/runs/r4'),
        glm_adapter_schema_failed=str(BASE/'v3/runs/r5'),accepted=str(run)),
      limitations=['isolated compute core, not complete EP/PP','initial packing/input quantization excluded','warm resident weights','two independent processes, event samples are not independent processes',
        'HY3 A100 baseline used Triton, H100 uses DeepGEMM','H100 DeepGEMM contiguous core does not cover all PP masked adapters','GLM Humming explicitly disables zero-point TMA with use_tma_bzp=False; default grouped AWQ TMA path failed']))
    body=f'''### {ident}

Snapshot: {snapshot}

- GLM-4.7 AWQ/Humming、HY3 BF16/DeepGEMM、Qwen3.6-35B BF16/DeepGEMM、Qwen3.5-122B FP8/DeepGEMM 的真实单层二维组件实验。
- 两个现有 H100 节点、8 张 GPU，每模型两份独立单 GPU 进程并行；T=8..32768 配对行，活跃 E 单独控制，均衡路由；eager/Graph 分开。共 1584 点，每点 30 个事件样本。
- 只测两次 Expert GEMM、SiLU/multiply 及 FP8 内部激活量化；不含路由、初始 packing/量化、Attention、通信和整模型指标。两份进程逐张量权重摘要一致，dense reference 通过，Guard 核实清理并释放。
- GLM 使用原生 Humming 的 use_tma_bzp=False 兼容配置，绕过 grouped AWQ 零点 TMA 描述符错误；AWQ 精度未改变，不能标为默认 TMA 路径。
- 提供绝对延迟、相对 E1 延迟二维图与曲线；各模型 E32/E2 在 T128、8192、32768 的比值：{json.dumps(ratios,ensure_ascii=False)}。

Report: [{ident}]({snapshot/'REPORT.md'})。Source run: {run}。
'''
    registry_index=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    replace_block(registry_index,ident,body)
    project=BASE.parents[0]/'experiment_summary/README.md'
    # BASE sits directly under autoscaling/experiments.
    project=BASE.parent/'experiment_summary/README.md'
    short=f'''### H100 四模型：单层 Expert 的 token × 活跃专家数二维图

真实单层，8 张 H100 并行，每模型两份独立单 GPU 进程。T 是本地 token–expert 配对行，范围 8..32768；均衡路由、resident 专家固定，eager 与 CUDA Graph 分开，共 1584 点。GLM-4.7 为 AWQ/Humming，HY3/35B 为 BF16 DeepGEMM，122B 为 FP8 DeepGEMM。

GLM 使用 Humming 的 use_tma_bzp=False 兼容配置；其余模型使用各自原生 DeepGEMM 路径。固定 T 的 E32/E2 Graph 耗时比：{json.dumps(ratios,ensure_ascii=False)}。这些是计算核心的比值，不能替代整模型 EP/PP 性能；DeepGEMM 的 packing 和初始量化在计时外。数值、两份权重摘要、计时样本和 Guard 清理/释放全部核实。

[报告与二维图]({snapshot/'REPORT.md'}) · [原始逐进程 CSV]({snapshot/'full_metrics.csv'}) · [来源与限制]({snapshot/'manifest.json'})。
'''
    replace_block(project,ident,short)
    # Verify idempotent indexing and every copied artifact checksum.
    before=sha(project);replace_block(project,ident,short);assert sha(project)==before
    before=sha(registry_index);replace_block(registry_index,ident,body);assert sha(registry_index)==before
    for p in report.iterdir():
        if p.is_file():assert sha(p)==sha(snapshot/p.name)
    print(json.dumps(dict(snapshot=str(snapshot),report=str(snapshot/'REPORT.md'),ratios=ratios,gpu_release=public),ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
