"""Archive the offline comparison and link it once in the existing summary."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / HERE.name
INDEX = ROOT / 'experiments/experiment_summary'
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    v = json.loads((HERE / 'validation.json').read_text())
    assert v['passed'] and v['e2e_rows'] == 12 and v['communication_comparisons'] == 6
    assert v['e2e_exact_workload_match'] is False
    report = str(HERE / 'REPORT.md')
    short = f'''# Q10：H100 两卡与八卡的通信域、EP／PP部署取舍

这是历史数据的离线归纳，没有新GPU实验；两卡和八卡使用不同模型、不同请求条件，**不是严格相同workload的网络消融**。

- 两卡：Qwen3.5-122B-A10B FP8，H100 178内NVLink，EP2对两槽PP2。8请求；Prefill无旧缓存、每条新输入4096、输出1；Decode每条GPU已缓存4096、新输入1、生成128。
- 八卡：GLM-4.7-AWQ INT4，H100 174/175各4卡，跨节点EP8对PP2×节点内EP4。4或16请求；Prefill无旧缓存、每条新输入8192、输出1；Decode冷输入128、生成512。
- 主图采用各自cohort的EP吞吐÷PP吞吐；不是跨模型绝对吞吐排名。Prefill计新输入token/s，Decode计输出token/s，均为3轮保存结果。

两卡Decode EP2比PP2快24.87%；八卡Decode EP8比PP慢21.50%／17.12%。Prefill两卡EP比PP慢6.01%，八卡慢约69%–70%。早期八卡native库、Expert布局及本地chunk不同，不能将完整差距全归因于网络。

独立同库纯通信使用BF16、隐藏维度5120、160专家、每token命中8专家，固定全局逻辑token数，无模型计算。节点内EP4分别保留174/175；跨节点EP8的Dispatch＋Combine，小消息约5.7–6.1倍、1024-token消息约9.8–10.1倍。三次预热、三次对齐rank最大跨度采样后取中位数；EP宽度／每rank工作划分也不同，不是固定rank只更换链路。

[完整术语、12行吞吐／TTFT／TPOT／端到端指标与两张图]({report}) · [原始路径与哈希]({HERE / 'source_manifest.json'}) · [离线校验]({HERE / 'validation.json'})。
'''
    index_report = INDEX / 'component_reports/Q10.md'
    index_report.write_text(short)
    (HERE / 'index_Q10.md').write_text(short)
    readme_path = INDEX / 'README.md'
    readme = readme_path.read_text()
    readme = readme.replace('单节点双卡补充见第 8 节 Q01–Q09。',
        '单节点双卡补充见第 8 节 Q01–Q09；H100跨规模通信域对照见第9节Q10。')
    begin = '<!-- BEGIN H100 COMM DOMAIN Q10 -->'
    end = '<!-- END H100 COMM DOMAIN Q10 -->'
    readme = re.sub(re.escape(begin) + r'.*?' + re.escape(end) + r'\n*', '', readme, flags=re.S)
    readme = readme.replace('## 9. 以后怎么追踪', '## 10. 以后怎么追踪')
    section = f'''{begin}
## 9. 跨规模通信域对照：H100两卡与八卡

不同规模与模型的历史cohort不混入同workload排名；端到端现象和独立纯通信证据分开呈现。

| ID／简短报告 | 两组部署与workload | 已有观察 | 数据、图与限制 |
|---|---|---|---|
| [Q10：H100通信域与EP／PP选择]({index_report}) | 2卡NVLink：Qwen122B FP8，8请求；冷Prefill4096→1，GPU缓存4096后Decode1→128。8卡两节点：GLM INT4，4／16请求；冷Prefill8192→1，冷输入128→512。 | 两卡Decode EP比PP快24.87%；跨节点八卡EP比PP慢21.50%／17.12%。同库纯通信的跨节点EP8／节点内EP4：小消息约5.7–6.1倍，大消息约9.8–10.1倍。 | [12行指标、归一化图、绝对通信图及来源]({report})。模型、缓存、batch、native及chunk不同；没有严格同workload网络因果估计；纯通信不包含计算或PP调度。 |

{end}

'''
    marker = '## 10. 以后怎么追踪'
    assert readme.count(marker) == 1
    readme = readme.replace(marker, section + marker)
    if readme != readme_path.read_text():
        readme_path.write_text(readme)
    assert readme.count(begin) == 1

    (HERE / 'summary.md').write_text(
        '# H100 historical EP/PP communication-domain comparison\n\n'
        'Offline analysis only; no new GPU samples. Two-GPU Qwen122B FP8 '
        'NVLink versus eight-GPU GLM INT4 cross-node deployments, each '
        'comparing EP and PP on its own saved workload. Different models, '
        'caches, request counts, backends and chunks prohibit network-only '
        'E2E causal attribution. Same-native EP4/EP8 communication provides '
        'a separate mechanism check.\n\n'
        '[Complete report and both scientific figures](REPORT.md) · '
        '[12 formal metric rows](full_metrics.csv) · '
        '[source hashes](source_manifest.json) · [validation](validation.json).\n')
    entry = f'''### {HERE.name}

Snapshot: {SNAPSHOT}

- Complete offline H100 comparison; no GPU allocation, launch or new measurements. Q10 in the general experiment summary; historical cohorts kept distinct.
- Two-GPU Qwen3.5-122B-A10B FP8 NVLink EP2/PP2: B8 cold Prefill4096->1 and realGPUcached4096/new1->output128. Eight-GPU GLM-4.7-AWQ INT4 on174/175 EP8/PP2xEP4: B4/B16 cold Prefill8192->1 and cold128->512 Decode. Each original point three formal trials.
- EP/PP throughput ratio: 2GPU Prefill0.940, Decode1.249; 8GPU B4 Prefill0.300, Decode0.785, B16 Prefill0.308, Decode0.829. Different models/cache/batch/native/expert layouts/local chunks: not network-only E2E causal evidence.
- Separate same-native pure communication, hidden5120 BF16/top8/160experts, fixed logical tokens: EP4 local174/local175 vs EP8cross. D+C median 0.050848/0.047680 vs0.290080ms at4tokens; 0.043552/0.043712 vs0.257344 at16tokens; 0.187360/0.193152 vs1.884736 at1024tokens. Three warmups/three aligned rank-max samples; no compute. EP width/QP/row partition also changes.
- Twelve full throughput/TTFT/TPOT/E2E rows, six normalized contrasts, six communication contrasts, PNG/SVG/PDF figures, Markdown report, copied small source tables/manifests, hashes and regeneration scripts. No weights, KV payloads, raw traces or keys copied. Optimized followup two-GPU EP advantage14.89% remains separately disclosed, not pooled.
- Report: {HERE / 'REPORT.md'}
- Project entry: {index_report}
'''
    registry = REGISTRY.read_text()
    pattern = r'(?ms)^### ' + re.escape(HERE.name) + r'\n.*?(?=^### |^## |\Z)'
    matches = re.findall(pattern, registry)
    assert len(matches) <= 1
    updated = re.sub(pattern, lambda _: entry + '\n', registry) if matches else registry.rstrip() + '\n\n' + entry + '\n'
    if updated != registry:
        REGISTRY.write_text(updated)
    assert len(re.findall(pattern, updated)) == 1
    assert re.sub(pattern, lambda _: entry + '\n', updated) == updated

    files = ['build_report.py', 'archive_report.py', 'REPORT.md', 'summary.md', 'index_Q10.md',
             'manifest.json', 'full_metrics.csv', 'e2e_comparisons.csv',
             'communication_comparisons.csv', 'source_manifest.json', 'validation.json']
    files += [stem + '.' + ext for stem in ('e2e_communication_domain', 'communication_locality')
              for ext in ('png', 'svg', 'pdf')]
    files += [r['snapshot'] for r in json.loads((HERE / 'source_manifest.json').read_text())]
    for name in files:
        source = HERE / name
        target = SNAPSHOT / name
        assert source.is_file() and source.stat().st_size < 4 * 1024 * 1024
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    checksums = ''.join(f'{sha(HERE / name)}  {name}\n' for name in sorted(files))
    (HERE / 'checksums.sha256').write_text(checksums)
    (SNAPSHOT / 'checksums.sha256').write_text(checksums)
    print(json.dumps(dict(snapshot=str(SNAPSHOT), byte_verified_files=len(files),
        project_index='Q10', registry_idempotent=True, new_gpu_samples=0)))


if __name__ == '__main__':
    main()
