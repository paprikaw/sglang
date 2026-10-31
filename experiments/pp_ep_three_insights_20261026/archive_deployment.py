"""Validate the added deployment page and save a separate, immutable revision."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables_with_deployment'
OLD=ROOT/'deliverables'
STEM='PP_EP_deployment_and_three_insights_20261026'
ARCHIVE=ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/deployment_r2'


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not ARCHIVE.exists(),'Refusing to overwrite a saved revision'
    ppt=OUT/(STEM+'.pptx');pdf=OUT/(STEM+'.pdf')
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with zipfile.ZipFile(ppt)as package:
        assert package.testzip()is None
        names=[f'ppt/slides/slide{i}.xml'for i in range(1,5)]
        actual=[n for n in package.namelist()if n.startswith('ppt/slides/slide')and n.endswith('.xml')]
        assert sorted(actual)==names
        first=ET.fromstring(package.read(names[0]))
        text='\n'.join(node.text or''for node in first.findall('.//a:t',ns))
        for term in ['EP2','PP2','½ 专家','全部专家','Dispatch + Combine','同一微批','不同微批','阶段边界']:
            assert term in text,term
        geometries=[node.attrib['prst']for node in first.findall('.//a:prstGeom',ns)]
        assert geometries.count('upDownArrow')==4 and geometries.count('downArrow')==1
        assert len(first.findall('.//p:pic',ns))==0,'Deployment page must stay editable'
        for i,name in enumerate(names[1:],1):
            new=ET.fromstring(package.read(name))
            with zipfile.ZipFile(OLD/'PP_EP_three_insights_20261026.pptx')as old_package:
                old=ET.fromstring(old_package.read(f'ppt/slides/slide{i}.xml'))
            new_text=[n.text for n in new.findall('.//a:t',ns)]
            old_text=[n.text for n in old.findall('.//a:t',ns)]
            # Only the slide number changes on the three existing data pages.
            assert new_text[:2]==old_text[:2]
            assert new_text[4:]==old_text[4:]
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    assert 'Pages:           4'in info
    for name in ['latency.csv','expert_communication.csv']:
        assert (OUT/name).read_bytes()==(OLD/name).read_bytes(),name
        assert len(list(csv.DictReader((OUT/name).open())))==12
    for file in (OLD/'figures').glob('*.png'):
        assert sha(file)==sha(OUT/'figures'/file.name),file.name
    sources=json.loads((OUT/'sources.json').read_text())
    for item in sources:
        assert sha(Path(item['path']))==item['sha256']
    notes=json.loads((OUT/'slide_notes.json').read_text())
    assert len(notes)==4 and notes[0]['title'].startswith('EP2 分专家')
    bounds=json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0<=r['x'] and 0<=r['y'] and r['x']+r['width']<=40/3+.001 and r['y']+r['height']<=7.501 for r in bounds)
    validation=dict(passed=True,ppt_slides=4,pdf_pages=4,
        new_deployment_diagram_editable=True,ep_layer_exchange_arrows=4,pp_stage_transfer_arrows=1,
        existing_data_pages_preserved=True,latency_rows=12,component_rows=12,
        six_data_chart_pngs_byte_identical=True,source_hashes_verified=len(sources),
        visual_review='Full first-page preview and four-page contact sheet inspected',
        new_gpu_experiments=False,checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json',validation)
    for name in ['full_metrics.csv','stage_metrics.csv']:
        shutil.copy2(OLD/name,OUT/name)
    shutil.copy2(ROOT/'archive_deployment.py',OUT/'archive_deployment.py')
    shutil.copy2(OLD/'deck_engine.py',OUT/'deck_engine.py')
    shutil.copy2(OLD/'Noto_CJK_LICENSE.txt',OUT/'Noto_CJK_LICENSE.txt')
    shutil.copytree(OLD/'profile_data',OUT/'profile_data')
    manifest=json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_deployment_r2',
        title='EP2/PP2 deployment diagram followed by three measured insights',
        created_at=datetime.now(timezone.utc).isoformat(),revision_of=str(OLD),
        conceptual_diagram=dict(layers=4,gpus=2,
            ep='Each layer has half its experts on each GPU; per-layer dispatch/combine',
            pp='Two contiguous layer stages; boundary activations and two-microbatch pipeline'),
        artifacts=dict(pptx=ppt.name,pdf=pdf.name,full_metrics='full_metrics.csv',stage_metrics='stage_metrics.csv',
            raw_profiles='Referenced by profile_data/components_per_rank.csv'),
        key_results='Data and three insight pages unchanged; one editable deployment page prepended')
    dump(OUT/'manifest.json',manifest)
    (OUT/'summary.md').write_text('''# EP2 / PP2 部署对比与三组数据

共4页：第1页用4层、2GPU示意EP2每层分专家及逐层Dispatch/Combine；PP2按层划分阶段，并用微批A/B展示流水并行。第2–4页保留原GPU、模型、阶段三组insight，以及每组的端到端延迟和Expert/通信图。

图中箭头表示通信发生的位置；示意层数与时间槽不作为测量数据。数据表和六张数据图与三页版逐字节核验一致。部署图为原生可编辑PowerPoint形状和文字，详细来源及统计口径在备注和sources.json中。

没有新增GPU实验；原三页版及其归档保留。重建：使用build.py --with-deployment --out <output_directory>。PDF和PNG是同一布局的预览。
''')
    checks=[dict(path=str(p.relative_to(OUT)),sha256=sha(p))for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.suffix!='.zip' and p.name!='checksums.json']
    dump(OUT/'checksums.json',checks)
    bundle=OUT/'PP_EP_deployment_and_three_insights_data_20261026.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED)as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file()and p.suffix!='.zip':package.write(p,arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle)as package:assert package.testzip()is None
    shutil.copytree(OUT,ARCHIVE)
    for item in checks:assert sha(ARCHIVE/item['path'])==item['sha256']
    report=ROOT.parent/'experiment_summary/component_reports/Q08.md'
    with report.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- DEPLOYMENT DIAGRAM PPT R2 -->'
        assert marker not in text
        lines=text.splitlines(keepends=True)
        entry=f'\n{marker}\n**部署图 + 三组数据（4页）：** [PPT]({ppt}) · [PDF]({pdf})。首图对比EP2逐层专家交换、PP2阶段传输与微批流水；后续三组数据沿用精简版。\n\n'
        handle.seek(0);handle.write(lines[0]+entry+''.join(lines[1:]));handle.truncate()
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and 'reports/deployment_r2'not in text
        entry=f'- Deployment-diagram revision: {ARCHIVE}. Four slides: editable EP2/PP2 placement and communication diagram first, then the same three data insights. EP shards each layer\'s experts and exchanges tokens per layer; PP divides layers and passes stage activations, illustrated with a two-microbatch pipeline. Both12-row tables and six data-chart PNGs verified unchanged;4-pagePDF and PPT validated. No new GPU execution.\n\n'
        handle.seek(0);handle.write(text.replace(marker,entry+marker));handle.truncate()
    dump(ROOT/'deployment_archive_validation.json',dict(**validation,archive=str(ARCHIVE),files_verified=len(checks),bundle=str(bundle)))
    print(json.dumps(dict(**validation,archive=str(ARCHIVE)),ensure_ascii=False))


if __name__=='__main__':main()
