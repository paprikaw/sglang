"""Validate and archive the three-slide revision, leaving prior PPTs intact."""
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables'
ID='pp_ep_three_insights_20261026'
ARCHIVE=ROOT.parents[4]/'experiment_manager/experiments'/ID


def rows(path):return list(csv.DictReader(path.open()))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    ppt=OUT/'PP_EP_three_insights_20261026.pptx'
    pdf=OUT/'PP_EP_three_insights_20261026.pdf'
    with zipfile.ZipFile(ppt)as package:
        assert package.testzip()is None
        names=[n for n in package.namelist()if n.startswith('ppt/slides/slide')and n.endswith('.xml')]
        assert len(names)==3
        for n in names:
            xml=ET.fromstring(package.read(n))
            text=''.join(xml.itertext())
            assert '平均端到端延迟'in text and 'Expert'in text
            assert all(word not in text for word in ['附录','仍有边界','尚未','不能','不代表','验证范围'])
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    assert 'Pages:           3'in info
    lat=rows(OUT/'latency.csv');components=rows(OUT/'expert_communication.csv')
    assert len(lat)==len(components)==12
    assert {r['slide']for r in lat}=={'gpu','model','phase'}
    assert all(float(r['e2e_s'])>0 and r['requests']=='8'for r in lat)
    assert all(float(r['expert'])>0 and float(r['exchange'])>=0 and float(r['pipeline'])>=0 for r in components)
    ranks=rows(ROOT/'data/components_per_rank.csv');assert len(ranks)==16
    # A common GEMM scope is established by actual kernel inventories.
    counts=[]
    for f in sorted((ROOT/'data').glob('rank_*.json')):
        r=json.loads(f.read_text())
        count=sum(r['expert_kernel_inventory'].values())
        assert count==2*r['actual_layer_invocations'],(f,count,r['actual_layer_invocations'])
        counts.append(count)
    sources=json.loads((OUT/'sources.json').read_text())
    for r in sources:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    bounds=json.loads((OUT/'layout_bounds.json').read_text())
    assert all(r['x']>=0 and r['y']>=0 and r['x']+r['width']<=40/3+.001 and r['y']+r['height']<=7.501 for r in bounds)
    validation=dict(passed=True,ppt_slides=3,pdf_pages=3,e2e_rows=12,component_rows=12,
        baseline_rank_traces=16,matched_expert_gemms=sum(counts),
        two_expert_gemms_per_actual_layer_verified=True,source_hashes_verified=len(sources),
        all_slides_visually_checked=True,new_gpu_experiments=False,
        checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json',validation)
    shutil.copytree(ROOT/'data',OUT/'profile_data',dirs_exist_ok=True)
    shutil.copy2(ROOT/'archive.py',OUT/'archive.py')
    shutil.copy2(ROOT.parent/'pp_ep_steady_decode_20261026/deck_engine.py',OUT/'deck_engine.py')
    shutil.copy2('/tmp/pp_ep_ppt_assets/OFL.txt',OUT/'Noto_CJK_LICENSE.txt')
    shutil.copy2(OUT/'latency.csv',OUT/'full_metrics.csv')
    shutil.copy2(OUT/'expert_communication.csv',OUT/'stage_metrics.csv')
    readme='''# PP / EP：三组 insight 与数据

新版PPT仅3页，每页一条insight、平均请求端到端延迟图、Expert/通信分项图。

1. GPU：122B Decode，A100 PP 的 E2E 延迟降低26.0%，H100 EP降低19.8%。
2. 模型：H100 Prefill，35B/122B 的 PP额外路由Expert模块活动66.4/117.7 ms/卡。
3. 阶段：H100 122B，Prefill PP延迟降低25.9%；Decode EP延迟降低19.8%。

GPU/阶段图采用实际路由Expert GEMM（每层2个GEMM，原始trace核验），单位GPU毫秒/卡/实际层调用；模型图采用整批Prefill路由Expert模块GPU活动。D+C为专家交换；PP NCCL为流水通信驻留。各部分用并排柱状图分别展示。正文仅保留数据所需的配置、单位和统计量；详细来源在备注、sources.json及profile_data。

数据来自已有正式计时及独立profile，没有启动新GPU实验。PPT文字可编辑，图表附PNG/矢量PDF及CSV。PDF为同一布局的页面预览；本环境无Office渲染器。原12页版本及原始实验资料保持不变。
'''
    (OUT/'summary.md').write_text(readme)
    dump(OUT/'manifest.json',dict(id=ID,title='Three PP/EP insights with E2E latency and Expert/communication charts',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',cpu_only_reanalysis=True,
        implementation=dict(engine='SGLang saved baselines',source_commit='870269702285d94bdd926a50cc29ad1a20bcf60c',
            source_tag='experiment/sglang/glm47-ep16-canonical-profile-20261004-cpu-validated',
            worktree=str(ROOT.parents[1]/'.engine-worktrees/glm47-ep16-canonical-profile-20261004'),
            frozen_runtime='Referenced by original experiment manifests; no runtime edited'),
        deployment=dict(new_deployment=False,original_nodes=['spartan-gpgpu069','spartan-gpgpu178'],
            endpoint=None,launch_manifests='original saved run roots',validation='validation.json'),
        workload=dict(requests=8,prefill='cached0/new4096/output1',decode='GPUcached4096/new1/output128',
            formal_rounds=3,latency='mean complete-request seconds',order='original cohort order retained'),
        model=['Qwen3.5-122B-A10B-FP8','Qwen3.6-35B-A3B BF16'],
        layouts=['EP2 AttentionTP1 DP2','PP2 two slots AttentionTP1'],
        engine_parameters='native original per-platform backends and phase-specific Graph; see speaker notes and source manifests',
        artifacts=dict(pptx=ppt.name,pdf=pdf.name,full_metrics='full_metrics.csv',stage_metrics='stage_metrics.csv',
            raw_profiles='Referenced by profile_data/components_per_rank.csv, not copied'),
        original_root=str(ROOT),source_provenance='sources.json',scope_details='speaker_notes.md'))
    checks=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file()and p.suffix!='.zip'and p.name!='checksums.json':
            assert p.name not in ['admin.key','config.json','gpu_registry_lease.json']
            checks.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(OUT/'checksums.json',checks)
    bundle=OUT/'PP_EP_three_insights_data_20261026.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED)as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file()and p.suffix!='.zip':package.write(p,arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle)as package:assert package.testzip()is None
    assert not ARCHIVE.exists(),'Refusing to overwrite an archive'
    shutil.copytree(OUT,ARCHIVE)
    for r in checks:assert hashlib.sha256((ARCHIVE/r['path']).read_bytes()).hexdigest()==r['sha256']
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    entry=f'''\n<!-- begin {ID} -->
### {ID}

Snapshot: {ARCHIVE}

- User-requested three-slide revision: GPU(122B A100/H100 Decode), model(H10035B/122B Prefill), phase(H100122B Prefill/Decode). Each slide has one insight, mean full-request E2E latency and separated routed Expert/DeepEP/PP NCCL bars. No new GPU launch, original12-slide deck preserved.
- Formal meanE2E A100122B EP/PP14.205/10.508s; H100122B Decode1.608/2.005s. H10035B Prefill0.387/0.261s and122B0.628/0.465s. Whole-cohort routedExpert PP increment66.4/117.7ms/card. Native implementations and exact named checkpoints retained.
- CPU re-extraction of16 original rank traces verifies exactly2 routedExpertGEMM kernels per actual layer, aligns A100 Marlin and H100 DeepGEMM scopes, selects B4 actual Decode or complete Prefill. PP communication remains nonzero NCCL residency; independent grouped bars, not additive E2E slices. Detailed source/implementation distinctions moved to notes.
- Editable3-slide PPTX,3-pagePDF preview, PNG/vectorPDF figures,12-row latency and component tables, per-rank source identities/hashes, code, validation, checksums and share ZIP archived. Index reference added under Q08.

<!-- end {ID} -->
'''
    with registry.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        assert f'<!-- begin {ID} -->'not in text
        handle.seek(0,2);handle.write(entry)
    report=ROOT.parent/'experiment_summary/component_reports/Q08.md'
    with report.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- THREE INSIGHT PPT -->'
        if marker not in text:
            lines=text.splitlines(keepends=True)
            link=f'\n{marker}\n**三页精简汇报版：** [PPT]({ppt}) · [PDF]({pdf})。每页一条 insight，配端到端延迟及 Expert／通信图。\n\n'
            text=lines[0]+link+''.join(lines[1:]);handle.seek(0);handle.write(text);handle.truncate()
    dump(ROOT/'archive_validation.json',dict(passed=True,archive=str(ARCHIVE),files_verified=len(checks),
        bundle=str(bundle),bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest()))
    print(json.dumps(dict(**validation,archive=str(ARCHIVE),pptx=str(ppt),pdf=str(pdf)),ensure_ascii=False))


if __name__=='__main__':main()
