"""Idempotently archive the small offline report and index it as Q11."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INDEX = ROOT / 'experiments/experiment_summary'
REGISTRY = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
SNAPSHOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager/experiments') / HERE.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = json.loads((HERE / 'validation.json').read_text())
    assert validation['passed'] and validation['full_metric_rows'] == 12
    assert validation['independently_recomputed_comm_points'] == 280
    report = HERE / 'REPORT.md'
    short = f'''# Q11：A100 两卡与八卡的 EP／PP 及通信域对照

历史数据的离线复核及作图，没有新 GPU 实验。**两卡／八卡不是相同模型、相同缓存或相同请求协议，不能由端到端差值唯一归因于网络。**

## 负载和术语

EP 是同层专家跨卡；PP 是模型按层分两个阶段。两卡 EP2 为 Attention TP1×DP2；PP2 每阶段 EP1，固定两槽。八卡 EP8 为 Attention TP4×DP2；PP2×EP4 每节点一个阶段。

- 两卡 A100 NV12：Qwen122B FP8 或35B BF16，8请求。Prefill 每条无旧缓存、新输入4096、输出1；Decode 每条GPU已有真实4096-token缓存、新输入1、输出128，无CPU恢复。每点三轮。
- 八卡 A100：GLM4.7 AWQ INT4，168/169，4请求。Prefill每条无旧缓存、新输入8192、输出1；Decode冷输入128、输出512。节点内Prefill chunk均1024，EP全局2048／PP全局1024；每点单cohort。pair内NVLink，pair外及跨节点走旧RDMA。
- 通信毫秒是一次 Dispatch＋Combine 的事件跨度，没有Attention、Expert计算或完整PP调度；不是请求TPOT。

| 部署与阶段 | EP 吞吐 token/s | PP 吞吐 token/s | PP 相对 EP |
|---|---:|---:|---:|
| 两卡122B Prefill | 8670.75 | 8185.11 | −5.60% |
| 两卡122B Decode | 72.08 | 97.26 | +34.92% |
| 两卡35B Prefill | 27644.14 | 31841.16 | +15.18% |
| 两卡35B Decode | 90.80 | 121.76 | +34.09% |
| 八卡GLM Prefill | 1322.77 | 1943.53 | +46.93% |
| 八卡GLM Decode | 100.06 | 101.92 | +1.85% |

本组A100没有H100那种Decode偏好翻转；八卡Decode差很小，不声称稳定优胜。原122B EP r7的Guard包装器失败状态保留，测量独立复算；后续两槽PP通过。旧四槽PP不是两槽PP，不能混用来制造反转。Q02后续Prefill优化没有新EP基线，单列在完整报告。

同库T01纯通信固定单次输入、均衡路由。1024-token时，CE策略节点内EP4为1.3348ms、跨节点EP8为7.0528ms（5.28倍）；RDMA策略则为8.4070／6.3670ms，方向相反。8-token时RDMA为0.1852／0.2846ms。EP4每卡配对行数是EP8两倍，Graph关闭，三次rank-max中位数；CE只替换节点内pair外搬运、跨节点仍RDMA。另一个节点结果和完整8–4096扫描均保存。

[完整报告：两张图、TTFT／TPOT／端到端与所有来源]({report}) · [12行指标]({HERE / 'full_metrics.csv'}) · [来源与哈希]({HERE / 'source_manifest.json'}) · [校验]({HERE / 'validation.json'})。
'''
    (INDEX / 'component_reports/Q11.md').write_text(short)
    (HERE / 'index_Q11.md').write_text(short)
    (HERE / 'summary.md').write_text('# A100 historical EP/PP communication-domain evidence\n\n'
        'Offline analysis only; no GPU measurements. Two-GPU Qwen122B FP8 and Qwen35B BF16 '
        'versus eight-GPU GLM INT4. No H100-like Decode preference reversal in these cohorts. '
        'Same-native communication sweeps show transport-policy- and size-dependent locality costs.\n\n'
        '[Complete report and plots](REPORT.md) · [Metrics](full_metrics.csv) · '
        '[Source hashes](source_manifest.json) · [Validation](validation.json).\n')

    index_path = INDEX / 'README.md'
    text = index_path.read_text()
    text = text.replace('## 9. 跨规模通信域对照：H100两卡与八卡',
                        '## 9. 跨规模通信域对照：H100与A100两卡及八卡')
    text = text.replace('；H100跨规模通信域对照见第9节Q10。',
                        '；H100／A100跨规模通信域对照见第9节Q10–Q11。')
    begin, end = '<!-- BEGIN A100 COMM DOMAIN Q11 -->', '<!-- END A100 COMM DOMAIN Q11 -->'
    text = re.sub(re.escape(begin) + r'.*?' + re.escape(end) + r'\n*', '', text, flags=re.S)
    section = f'''{begin}
### A100 对应证据

| ID／简短报告 | 部署与准确负载 | 已有观察 | 数据、图与限制 |
|---|---|---|---|
| [Q11：A100两卡／八卡通信域对照]({INDEX / 'component_reports/Q11.md'}) | 两卡Qwen122B FP8／35B BF16，8请求：无旧缓存Prefill新输入4096→1输出；GPU缓存4096后新输入1→128输出。八卡GLM INT4，4请求：无旧缓存8192→1及冷输入128→512。 | 两卡Decode PP快34.92%／34.09%；八卡GLM Prefill PP快46.93%，Decode约1.85%。同库1024-token通信CE策略节点内1.3348／跨节点7.0528ms；RDMA策略8.4070／6.3670ms，方向相反。 | [完整图、12行指标、20行同调用通信表与来源]({report})。没有H100式Decode反转；模型、内核、缓存、Graph不同，不能唯一归因于网络。T01没有计算或完整PP调度，旧四槽结果单列。 |

{end}

'''
    marker = '## 10. 以后怎么追踪'
    assert text.count(marker) == 1
    text = text.replace(marker, section + marker)
    if text != index_path.read_text():
        index_path.write_text(text)
    assert text.count(begin) == 1

    entry = f'''### {HERE.name}

Snapshot: {SNAPSHOT}

- Complete offline historical analysis, no GPU launch. Q11 in the general experiment summary, alongside H100 Q10. Original run records and existing metric tables remain unchanged.
- Same-pair A100 NV12 Qwen122B FP8 two-slot EP2/PP2 and supplementary35B BF16, eight requests, cold4096->1 Prefill and GPUcached4096/new1->128 Decode; three rounds. GLM INT4 eight-GPU EP8/PP2xEP4, nodes168/169, four requests, cold8192->1 and cold128->512, one cohort; matched localchunk1024 but different global steps.
- PP versus EP throughput: 122B Prefill-5.60%, Decode+34.92%;35B+15.18%/+34.09%; eight-GPU GLM+46.93%/+1.85%. No H100-like Decode preference reversal; models/kernels/cache/Graph/workload are not a network-only ablation. Original122B EP7 wrapper Guard failure disclosed, raw measurements independently verified; four-slot and later PP-only optimization cohorts not pooled.
- Same-native T01 pure communication, GLM BF16hidden5120/E160/top8, same tokens per call, both local065/local167 retained, no compute or KV, eager, three aligned rank-max samples. At1024 on065, localEP4/crossEP8 CE1.3348/7.0528ms; RDMA8.4070/6.3670ms. Actual local policy changes the direction; EP4 assignments/GPU are twice EP8, not fixed per-rank work.
- Two figures PNG/SVG/PDF, twelve full latency metric rows, six E2E contrasts, twenty equal-call communication rows and old RDMA controls. Recomputed280 communication medians and140 within-call transport-identity groups, verified8 original request-source hashes. Small source snapshots only, no weights/KV/raw traces/credentials.
- Report: {report}
'''
    registry = REGISTRY.read_text()
    pattern = r'(?ms)^### ' + re.escape(HERE.name) + r'\n.*?(?=^### |^## |\Z)'
    assert len(re.findall(pattern, registry)) <= 1
    updated = re.sub(pattern, lambda _: entry + '\n', registry) if re.search(pattern, registry) else registry.rstrip() + '\n\n' + entry + '\n'
    if updated != registry:
        REGISTRY.write_text(updated)
    assert re.sub(pattern, lambda _: entry + '\n', updated) == updated

    files = sorted(p for p in HERE.rglob('*') if p.is_file()
                   and p.name != 'checksums.sha256' and '__pycache__' not in p.parts)
    for p in files:
        assert p.stat().st_size < 4 * 1024 * 1024
        target = SNAPSHOT / p.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        assert sha(p) == sha(target)
    checksums = ''.join(f'{sha(p)}  {p.relative_to(HERE)}\n' for p in files)
    (HERE / 'checksums.sha256').write_text(checksums)
    (SNAPSHOT / 'checksums.sha256').write_text(checksums)
    print(json.dumps(dict(canonical_snapshot=str(SNAPSHOT), verified_files=len(files), index='Q11', no_gpu_launch=True)))


if __name__ == '__main__':
    main()
