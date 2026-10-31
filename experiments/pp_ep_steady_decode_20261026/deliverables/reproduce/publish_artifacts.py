"""Index the completed result and save a small, checksummed archive."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from finalize import ROOT, OUT, ID, ARCHIVE


def checksums(directory, ignored=()):
    return [dict(path=str(p.relative_to(directory)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ignored and p.suffix!='.zip']


def main():
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['passed']
    short=ROOT.parent/'experiment_summary/component_reports/Q08.md'
    short.parent.mkdir(exist_ok=True)
    assert not short.exists(), 'Q08 was claimed by another result; choose a fresh index ID'
    short.write_text(f'''# Q08：35B A100 / H100 稳态 Decode 补测与 PP / EP 决策 PPT

本轮使用 Qwen3.6-35B-A3B BF16，全 40 层；每节点同一对 GPU0/1 顺序运行 EP2 / 两槽 PP2。A100069 为 NV12，H100178 为 NV6。EP Attention TP1×DP2、40层 / 128专家每层每卡；PP Attention TP1、20层 / 256专家每层每卡。两平台保留原生后端、Graph 和 overlap 差异。

8 请求同时到达；每条真实 GPU 缓存 P=4096、新输入 Q=1、输出128，总旧缓存32768、新输入8、每轮输出1024 token。各配置3轮，无 profiler；准备、预热与独立 profile 不计时。

| 平台 | EP2 / PP2 输出 token/s | EP2 / PP2 平均 TPOT ms | 此负载的 TPOT 选择 |
|---|---:|---:|---|
| A100 | 91.40 / 123.10 | 86.258 / 64.271 | PP 降低25.5% |
| H100 | 858.86 / 668.65 | 7.962 / 11.149 | EP 降低28.6% |

每配置3段独立分项 profile，每rank丢弃4个实际Decode forward再选24个完整B4 forward；共36份rank trace / 864个rank forward / 25920次层调用。A100 Attention 含投影变化−0.8%、Routed Expert +7.3%且PP误差较大，逐层专家通信明显；H100原生Graph Attention核心−3.8%、Routed GEMM +52.9%。H100另有已核验的eager模块诊断，独立展示，不套用到Graph原生性能。PP阶段NCCL驻留A100约0.037、H100约1.745 ms/卡/所选forward，含等待与重叠，不是纯传输时延。

结合Q01–Q07历史数据，可展示模型、平台、workload与Attention干扰的作用，支持比较Expert执行变化和通信成本。尚无硬件或模型大小的单变量因果，也没有跨节点切换阈值。分项可重叠，不能直接相加预测TPOT。

- [12页中文PPT]({OUT}/PP_EP_deployment_decision_20261026.pptx) · [PDF]({OUT}/PP_EP_deployment_decision_20261026.pdf)
- [完整报告]({OUT}/REPORT.md) · [逐请求]({OUT}/data/requests.csv) · [原生性能]({OUT}/data/native_metrics.csv) · [逐卡分项]({OUT}/data/components_per_rank.csv)
- [协议与统计边界]({OUT}/data/protocol.json) · [验证]({OUT}/validation.json) · [来源哈希]({OUT}/data/source_hashes.json)
- [原始补测]({ROOT}/runs) · [轻量归档]({ARCHIVE})

最终a100_r2 / h100_r2的Guard、canonical CPU权重缓存、清理和lease释放通过。a100_r1准备失败、h100_r1首轮TTFT异常和未生效eager钩子均保留；H100 r2补齐形状预热后完整重测三轮，未挑选旧轮样本。原始大trace与私有运行配置留在原目录。
''')
    index=ROOT.parent/'experiment_summary/README.md'
    row=f'| [Q08：35B A100／H100 稳态 Decode 与决策 PPT]({short}) | 每平台同一对 NVLink 双卡顺序 EP2／两槽 PP2；8请求，各 GPU 缓存4096、新输入1、输出128，各3轮。 | A100 PP 的 TPOT **86.258→64.271 ms（−25.5%）**；H100 EP **11.149→7.962 ms（−28.6%）**。Expert／通信取舍、Attention对照与例外纳入12页中文PPT。 | 36份rank trace，每份24个稳态B4 forward；原生Graph与eager诊断分开、PP边界单列；96请求计时、来源哈希、Guard／cache／清理及释放验证。 [PPT]({OUT}/PP_EP_deployment_decision_20261026.pptx)／[PDF]({OUT}/PP_EP_deployment_decision_20261026.pdf)。 |\n'
    with index.open('r+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        content=handle.read()
        if '[Q08：35B A100／H100' not in content:
            marker='\n<!-- END Q01 QWEN EP2 PP2 -->'
            assert content.count(marker)==1
            content=content.replace(marker,row+marker)
            handle.seek(0);handle.write(content);handle.truncate()
    shutil.copy2(short,OUT/'Q08.md')
    shutil.copy2(ROOT/'publish_artifacts.py',OUT/'reproduce/publish_artifacts.py')
    # Plain aliases match the experiment-manager convention.
    shutil.copy2(OUT/'REPORT.md',OUT/'summary.md')
    shutil.copy2(OUT/'data/native_metrics.csv',OUT/'full_metrics.csv')
    shutil.copy2(OUT/'data/components_per_rank.csv',OUT/'stage_metrics.csv')
    local_checks=checksums(OUT,ignored=('checksums.json',))
    (OUT/'checksums.json').write_text(json.dumps(local_checks,ensure_ascii=False,indent=2)+'\n')
    bundle=OUT/'PP_EP_decision_bundle_20261026.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as output:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix!='.zip':output.write(path,arcname=str(path.relative_to(OUT)))
    with zipfile.ZipFile(bundle)as output:assert output.testzip()is None
    ARCHIVE.mkdir(parents=True,exist_ok=True)
    assert not (ARCHIVE/'manifest.json').exists(), 'Refusing to overwrite an existing archive'
    for path in sorted(OUT.iterdir()):
        destination=ARCHIVE/path.name
        if path.is_dir():shutil.copytree(path,destination)
        else:shutil.copy2(path,destination)
    for entry in local_checks:
        assert hashlib.sha256((ARCHIVE/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert hashlib.sha256((ARCHIVE/bundle.name).read_bytes()).digest()==hashlib.sha256(bundle.read_bytes()).digest()
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    entry=f'''\n<!-- begin {ID} -->
### {ID}

Snapshot: {ARCHIVE}

- New complete40-layer Qwen3.6-35B-A3B BF16 fixed2GPU EP2/PP2 steady Decode; each platform sequential on GPU0/1, A100069 NV12 / H100178 NV6. Native Triton/eager/no-overlap versus BF16 DeepGEMM/FA3/Graph/EP-overlap; not a hardware-only comparison. Frozen SGLang870269702285d94bdd926a50cc29ad1a20bcf60c and per-run adapter hashes preserved.
- 8concurrent requests, each realGPUcached4096 + new1 -> output128;3 unprofiled rounds/cell,96request rows. Outputtok/s EP/PP A10091.40/123.10, H100858.86/668.65; meanTPOT A10086.258/64.271ms, H1007.962/11.149ms. PP improves A100 TPOT25.5%, EP improves H100 TPOT28.6%.
- 36rank profiles /864 selected steadyB4 rank forwards /25920layer calls. Attention includes projections in nativeA100/eagerH100 diagnostics; nativeH100 Graph uses actual correlated core/GEMM scope. H100 PP routedGEMM+52.9%, Attentioncore-3.8%; A100 Expert+7.3% with widerPP scatter, Attention-0.8%. PPboundaryNCCL residency separately retained; components overlap, not additive TPOT shares.
- Guard/cache/cleanup and shared final lease release verified for both r2 runs; original CPU preparation failure, first H100 TTFT outlier and inactive diagnostic hook preserved. All H100r2 formal rounds repeated after shape warmup; no cherry-picking old rounds. No cross-node threshold, size-only or hardware-only causality claim.
- 12-slide editable Chinese PPTX, PDF preview, scientificPNG/vectorPDF charts, portableCSV/JSON, report/speaker notes, provenance, lightweight source/validation and checksummed share ZIP. General indexQ08; priorQ07 preserved. No raw traces, weight/KV payloads or authentication values copied.

<!-- end {ID} -->
'''
    with registry.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        content=handle.read()
        assert f'<!-- begin {ID} -->'not in content
        handle.seek(0,2);handle.write(entry)
    result=dict(passed=True,archived_at=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE),
        local_files_checked=len(local_checks),bundle=str(bundle),bundle_bytes=bundle.stat().st_size,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),summary_id='Q08',registry_updated=True)
    (ROOT/'publication_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
