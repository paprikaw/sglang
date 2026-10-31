"""Archive validated small-model results and link one general summary row."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def archive(run, destination):
    cfg=json.loads((run/'config.json').read_text())
    data=json.loads((run/'analysis/validated_metrics.json').read_text())
    post=json.loads(Path(data['guard_postcheck']).read_text())
    assert data['passed'] and post['cleanup_verified']
    supplemental=data.get('supplemental_cache_review')
    assert (post['passed'] and post['model_cache']['passed']) or (
        supplemental and supplemental['passed'] and supplemental['original_guard_unchanged']
        and supplemental['original_guard_passed'] is False)
    destination.mkdir(parents=True,exist_ok=False)
    copied=[]
    def copy(src,name):
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,p);copied.append(p)
    for p in (run/'analysis').iterdir():
        if p.is_file():copy(p,'analysis/'+p.name)
    archived_report=destination/'analysis/REPORT.md'
    archived_report.write_text(archived_report.read_text().replace(
        '[配置](../config.json)',f'[原运行私有配置]({run/"config.json"})'))
    copy(Path(data['guard_postcheck']),'original_guard_postcheck.json')
    if data.get('decode_followup'):
        follow=Path(data['decode_followup']['run'])
        copy(Path(data['decode_followup']['guard_postcheck']),'decode_followup/guard_postcheck.json')
        for name in ('completed.json','worker_cleanup.json'):
            copy(follow/name,'decode_followup/'+name)
        for p in follow.glob('pp2/traces/*/sampling_validation.json'):
            copy(p,'decode_followup/'+p.parent.name+'/sampling_validation.json')
    copy(Path(__file__).with_name('README.md'),'protocol.md')
    copy(Path(__file__).with_name('cpu_tests.txt'),'cpu_tests.txt')
    for name in ('devices.csv','topology.txt','capacity_matrix.json','worker_cleanup.json',
                 'single_gpu_fit.json','single_gpu_cleanup.json'):
        copy(run/name,name)
    for arm in ('single','ep2','pp2'):
        copy(run/arm/'gc_freeze.json',arm+'/gc_freeze.json')
    for arm in ('ep2','pp2'):
        for name in ('tuning.json','cache_residency_qualification.json'):
            copy(run/arm/name,arm+'/'+name)
        for p in (run/arm/'qualification').glob('*.json'):
            copy(p,arm+'/qualification/'+p.name)
    z=destination/'code.zip'
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as a:
        for p in (run/'code').glob('*.py'):a.write(p,'frozen_run/'+p.name)
        for p in Path(__file__).parent.glob('*.py'):a.write(p,'analysis_and_preparation/'+p.name)
    copied.append(z)
    manifest=dict(id=destination.name,title='Single-GPU-sized Qwen35B: H100 EP2 versus PP2',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',
        implementation=dict(engine='SGLang',worktree=cfg['engine_root'],commit=cfg['engine_commit'],
            tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated'),
        model=dict(path=cfg['model_path'],revision=cfg['model_revision'],precision=cfg['precision']),
        deployment=dict(node=cfg['node'],job_id=cfg['job_id'],retired=True,
            layouts=[{k:r[k] for k in ('id','tp','ep','dp','pp','gpu_groups')} for r in cfg['layouts']]),
        workload=json.loads((run/'completed.json').read_text()),
        parameters=dict(graph=True,overlap_ep=True,overlap_pp=False,hicache=False,mtp=False,
            context_limit=262144,repeats=cfg['repeats'],mamba_max_states_per_path=1),
        communication=dict(wrapper=cfg['deepep_wrapper'],native=cfg['deepep_native'],
            native_sha256=hashlib.sha256(Path(cfg['deepep_native']).read_bytes()).hexdigest(),topology='NV6'),
        validation=dict(guard_postcheck=data['guard_postcheck'],guard_original_passed=post['passed'],
            supplemental_cache_review=bool(supplemental),cache_evidence_verified=True,cleanup=True,
            decode_followup=data.get('decode_followup'),
            actual_graph=True,request_metrics_recomputed=True),
        artifacts=dict(original_run=str(run),report='analysis/REPORT.md',metrics='analysis/full_metrics.csv',
            phase_profiles='analysis/prefill_components.csv and analysis/decode_components.csv'),
        key_results=data['rows'],limitations=['Three rounds; descriptive percentiles',
            'Separate NVLink pairs share node resources','Different supported EP/PP overlap paths',
            'Cumulative GPU kernel work is not exclusive critical-path share',
            'Original Guard postcheck rejected rankless single-GPU loader log; independent read-only review retained'])
    p=destination/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n');copied.append(p)
    checksums={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in copied}
    (destination/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    assert all(hashlib.sha256((destination/n).read_bytes()).hexdigest()==h for n,h in checksums.items())
    summary=run.parents[2]/'experiment_summary';report=run/'analysis/REPORT.md'
    overview=summary/'component_reports/Q04.md'
    overview.write_text('# Q04：单 GPU 大小模型的 H100 EP2／PP2\n\n'
        '模型为 Qwen3.6-35B-A3B BF16，区别于 Q03 的 122B FP8。各配置使用两张 H100；'
        '中等 8 请求，长缓存固定总量约 288 万 token，分成 16／32 请求。旧前缀在 GPU；'
        '准备、预热与 profile 都不计正式性能。\n\n'
        '同工作量 PP 的 Prefill 吞吐分别高 14.15%、高 11.72%、低 4.48%；'
        'Decode 吞吐低 18%–25%。长缓存 Attention 成为大的执行项，不能仅凭省去专家通信就预测 PP 更快。\n\n'
        f'[完整报告、全部指标及逐阶段诊断]({report})。\n\n'
        f'[校验归档]({destination/"manifest.json"})；原始逐卡 trace 保留在原运行目录。\n')
    readme=summary/'README.md';text=readme.read_text()
    assert '| [Q04：' not in text
    row=(f'| [Q04：单 GPU 大小模型 H100 EP2／PP2]({overview}) | Qwen3.6-35B-A3B BF16；'
         '178 节点两对 NV6 双卡并行；8 请求中等负载，16／32 请求固定总量长缓存。 | '
         '同工作量 PP Prefill 吞吐 +14.15%／+11.72%／−4.48%；Decode 低 18%–25%。'
         '全部逐请求 TTFT／TPOT／端到端均值、中位数及 P99，见完整报告。 | '
         '真实 BF16 DeepGEMM、FA3／Triton、Decode Graph；分开采全部 Prefill chunk 和实际 Decode Graph，'
         '可看 Attention、Expert、Dispatch／Combine 以及 PP 边界驻留。 |')
    anchor=next(x for x in text.splitlines() if x.startswith('| [Q03：'))
    readme.write_text(text.replace(anchor,anchor+'\n'+row))
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    marker='<!-- begin '+destination.name+' -->'
    with registry.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);existing=f.read()
        if marker not in existing:
            f.seek(0,2);f.write('\n'+marker+'\n### '+destination.name+'\n\n'
                f'Snapshot: `{destination}`; Q04 in general experiment summary.\n'
                'Complete Qwen3.6-35B-A3B BF16 on H100178, disjoint NV6 EP2/PP2 pairs. '
                'Moderate and measured near-KV-capacity Prefill/Decode; three formal rounds, separate GPU phase profiles. '
                'Request counts/cache hits/output lengths, native BF16 kernels, actual Graph and cleanup verified. '
                'Original Guard postcheck failed on its rankless single-GPU loader parser; read-only supplemental evidence review retained.\n'
                f'<!-- end {destination.name} -->\n')
    print(json.dumps(dict(archived=True,files=len(checksums),destination=str(destination),index=str(overview))))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    archive(a.run.resolve(),a.destination.resolve())
