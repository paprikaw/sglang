"""Archive a derived analysis without touching either original experiment."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
import shutil

ROOT=Path(__file__).resolve().parent
ID='h100_ep_pp_component_tradeoff_20261029'
DEST=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments')/ID


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    validation=json.loads((ROOT/'validation.json').read_text())
    browser=json.loads((ROOT/'browser_validation.json').read_text())
    sources=json.loads((ROOT/'sources.json').read_text())
    data=json.loads((ROOT/'data.json').read_text())
    assert validation['passed'] and browser['passed']
    assert sources['no_new_gpu_measurements']
    for item in sources['inputs']:assert sha(item['path'])==item['sha256']
    assert sha(ROOT/'analyze.py')==sources['generator']['sha256']
    assert not DEST.exists(),'Preserve an existing snapshot'
    files=['analyze.py','report.py','publish.py','browser-check.mjs','visual-template.html','data.json',
        'REPORT.md','component_estimates_event01.csv','sources.json','validation.json','browser_validation.json',
        'tradeoff_cuda_graph.png','tradeoff_cuda_graph.pdf','tradeoff_eager.png','tradeoff_eager.pdf']
    hashes={name:sha(ROOT/name) for name in files}
    complete=sum(c['status']['cuda_graph']['classification']!='missing' for c in data['cases'])
    manifest=dict(id=ID,title='H100 Expert versus communication layout component trade-off',
        created_at=datetime.now(ZoneInfo('Australia/Melbourne')).isoformat(),status='complete',
        evidence='derived component model, not a new GPU benchmark or PP/EP end-to-end measurement',
        implementation=dict(analysis_generator_sha256=sha(ROOT/'analyze.py'),
            report_generator_sha256=sha(ROOT/'report.py'),inputs=sources['inputs']),
        workload=dict(global_original_tokens=data['tokens'],minimum_ep=sources['minimum_ep'],
            candidates=['GLM/HY3: EP8 versus PP2xEP4 only',
                        '35B/122B: EP2/4/8 versus pure PP2/4/8; EP8 versus PP2xEP4'],
            routing='balanced deterministic top8, matching communication routes',
            gpu_budget='equal within each row; no comparison of different GPU counts as equal resources'),
        testbed=dict(gpu='H100 80GB HBM3',topology='EP2/EP4 intra-node NVLink; EP8 two nodes plus RDMA'),
        metrics=dict(unit='ms',aggregation='none',cases=len(data['cases']),complete_cases_per_mode=complete,
            modes=['cuda_graph','eager'],raw_events_per_point=30,
            default_compute_process=1,default_communication_round=1,default_event_1based=1,
            classification='conservative min/max propagation across all observed samples; not a confidence interval'),
        limitations=['ideal saturated pipeline throughput accounting, not single-request latency',
            'PP transfers, bubbles, Attention, packing, initial FP8 quantization, overlap and contention unmeasured',
            'compute sums across multiple calls are component estimates',
            'large N also changes computation chunking across layouts',
            'PP contiguous/masked implementation may differ from measured core',
            'source protocol is custom SM90 DeepEP low latency, with BF16 communication for all models',
            'full expert row points beyond32768 are missing and never extrapolated'],
        artifacts=hashes)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    DEST.mkdir()
    for name in files+['manifest.json']:shutil.copy2(ROOT/name,DEST/name)
    for name,digest in hashes.items():assert sha(DEST/name)==digest
    old=ROOT.parent/'h100_ep_token_comm_surface_20261029/report_tools/archive_results.py'
    spec=importlib.util.spec_from_file_location('existing_index_helper',old)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.ID=ID
    body=f'''<!-- begin {ID} -->
### H100：Expert 计算与通信的布局权衡组件模型（2026-10-29）

按现有单层 Expert 与 DeepEP LL 通信原始数据做同 GPU 预算比较；没有新跑 GPU。
GLM/HY3 最小 EP4，仅 EP8 对 PP2×EP4；35B/122B 另含纯 PP2/4/8。
130 个条件，每种模式 {complete} 个有完整组件；Graph/Eager、两份计算与两轮通信分开，30次事件无平均。
差额为省通信减多计算的绝对 ms；以全部观测样本 min/max 的保守包络标 PP/EP/边界。
理想饱和流水估算，未测 PP 交接、气泡、Attention 与 overlap；大 N 同时改变分块，缺点不外推。
[报告]({DEST}/REPORT.md) · [区域图]({DEST}/tradeoff_cuda_graph.png) · [组件绝对值 CSV]({DEST}/component_estimates_event01.csv) · [来源]({DEST}/sources.json)
<!-- end {ID} -->'''
    for index in [Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),
            ROOT.parents[1]/'experiments/experiment_summary/README.md']:
        helper.update_index(index,body)
        before=sha(index);helper.update_index(index,body);assert sha(index)==before
    print(json.dumps(dict(snapshot=str(DEST),cases=len(data['cases']),complete_per_mode=complete,
        archived_files=len(files)+1,index_updates_idempotent=True)))


if __name__=='__main__':main()
