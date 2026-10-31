"""Archive the four-slide revision with a single intro and wider PP Expert blocks."""
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
OUT=ROOT/'deliverables_expert_width'
OLD=ROOT/'deliverables_vertical'
STEM='PP_EP_four_slides_20261026'
ARCHIVE=ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/expert_width_r6'


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not ARCHIVE.exists(),'Refusing to overwrite a saved revision'
    ppt=OUT/(STEM+'.pptx');pdf=OUT/(STEM+'.pdf')
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with zipfile.ZipFile(ppt)as package,zipfile.ZipFile(OLD/'PP_EP_attention_and_three_insights_20261026.pptx')as old:
        assert package.testzip()is None
        slides=[n for n in package.namelist()if n.startswith('ppt/slides/slide')and n.endswith('.xml')]
        assert len(slides)==4
        first=ET.fromstring(package.read('ppt/slides/slide1.xml'))
        text='\n'.join(n.text or''for n in first.findall('.//a:t',ns))
        assert '阶段 1 在上，阶段 2 在下'in text
        assert text.count('Expert · E（该层全部专家）')==2
        assert text.count('Expert · E/2')==4
        assert text.count('Attention · B4')==6
        assert not first.findall('.//p:pic',ns)
        presets=[n.attrib['prst']for n in first.findall('.//a:prstGeom',ns)]
        assert presets.count('downArrow')==17
        widths={'EAF1FF':[],'E5F5F0':[],'EDF0F5':[]}
        for shape in first.findall('.//p:sp',ns):
            fill=shape.find('./p:spPr/a:solidFill/a:srgbClr',ns)
            if fill is not None and fill.attrib['val']in widths:
                ext=shape.find('./p:spPr/a:xfrm/a:ext',ns)
                widths[fill.attrib['val']].append(int(ext.attrib['cx']))
        assert [len(widths[k])for k in widths]==[4,2,6]
        assert len(set(widths['EAF1FF']))==len(set(widths['E5F5F0']))==len(set(widths['EDF0F5']))==1
        assert abs(widths['E5F5F0'][0]-2*widths['EAF1FF'][0])<=2
        for i in range(2,5):
            new=ET.fromstring(package.read(f'ppt/slides/slide{i}.xml'))
            previous=ET.fromstring(old.read(f'ppt/slides/slide{i+1}.xml'))
            a=[n.text for n in new.findall('.//a:t',ns)]
            b=[n.text for n in previous.findall('.//a:t',ns)]
            assert a[:2]==b[:2]and a[4:]==b[4:]
            # This revision has the same data-page numbers as the previous four-page deck.
            name=f'previews/slide_{i:02d}.png'
            assert sha(OUT/name)==sha(ROOT/'deliverables_with_deployment'/name)
    assert 'Pages:           4'in subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    for name in ['latency.csv','expert_communication.csv']:
        assert (OUT/name).read_bytes()==(OLD/name).read_bytes()
        assert len(list(csv.DictReader((OUT/name).open())))==12
    for file in (OLD/'figures').glob('*.png'):assert sha(file)==sha(OUT/'figures'/file.name)
    for name in ['full_metrics.csv','stage_metrics.csv','deck_engine.py','Noto_CJK_LICENSE.txt','attention_placement_protocol.json']:
        shutil.copy2(OLD/name,OUT/name)
    sources=json.loads((OLD/'sources.json').read_text())
    for row in sources:assert sha(Path(row['path']))==row['sha256']
    dump(OUT/'sources.json',sources)
    assert len(json.loads((OUT/'slide_notes.json').read_text()))==4
    bounds=json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0<=r['x'] and 0<=r['y'] and r['x']+r['width']<=40/3+.001 and r['y']+r['height']<=7.501 for r in bounds)
    validation=dict(passed=True,ppt_slides=4,pdf_pages=4,
        original_second_diagram_now_first=True,duplicate_intro_removed=True,
        pp_expert_width_twice_ep_half_expert_width=True,pp_expert_label='E: all experts of this layer',
        attention_widths_unchanged_and_equal=True,all_17_progression_arrows_point_down=True,
        three_data_pages_verified_unchanged=True,latency_rows=12,component_rows=12,
        source_hashes_verified=len(sources),new_gpu_experiments=False,
        visual_review='First slide at full size and four-slide contact sheet inspected',
        checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json',validation)
    manifest=json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_expert_width_r6',
        title='One vertical deployment diagram with full-layer PP experts, followed by three insights',
        created_at=datetime.now(timezone.utc).isoformat(),revision_of=str(OLD),
        key_results='Old slide2 replaces old slide1; PP E expert blocks are twice the width of EP E/2 blocks; four slides total',
        artifacts=dict(pptx=ppt.name,pdf=pdf.name,full_metrics='full_metrics.csv',stage_metrics='stage_metrics.csv',
            protocol='attention_placement_protocol.json',raw_profiles='profile_data/components_per_rank.csv'))
    manifest['attention_diagram'].update(pp_expert_width_ratio_to_ep=2,
        width_semantics='Expert-set cardinality only; widths do not encode latency',slide=1)
    dump(OUT/'manifest.json',manifest)
    (OUT/'summary.md').write_text('''# 一张部署图与三组数据

共4页：统一上下结构的EP2/PP2部署示意 → GPU数据 → 模型数据 → 阶段数据。

原第二页替换原第一页，移除重复示意。PP2每个本地层的Expert长条表示该层全部E个专家，宽度为EP2每卡E/2专家块的2倍；Attention块维持同样宽度。Expert长度表示专家数量，流程图不编码耗时。三组数据表和六张数据图保留，三张数据页按四页版页码核验一致。

PPT文字和示意图原生可编辑；未启动新GPU实验。来源、配置和统计口径保留在备注、sources.json和profile_data中。重建：build.py --replace-intro --out <output_directory>。
''')
    shutil.copy2(ROOT/'archive_expert_width.py',OUT/'archive_expert_width.py')
    shutil.copytree(OLD/'profile_data',OUT/'profile_data')
    checks=[dict(path=str(p.relative_to(OUT)),sha256=sha(p))for p in sorted(OUT.rglob('*'))
        if p.is_file()and p.suffix!='.zip'and p.name!='checksums.json']
    dump(OUT/'checksums.json',checks)
    bundle=OUT/(STEM+'_data.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED)as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file()and p.suffix!='.zip':package.write(p,arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle)as package:assert package.testzip()is None
    shutil.copytree(OUT,ARCHIVE)
    for row in checks:assert sha(ARCHIVE/row['path'])==row['sha256']
    report=ROOT.parent/'experiment_summary/component_reports/Q08.md'
    with report.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- EXPERT WIDTH FOUR SLIDES R6 -->';assert marker not in text
        lines=text.splitlines(keepends=True)
        entry=f'\n{marker}\n**一张部署图 + 三组数据（4页）：** [PPT]({ppt}) · [PDF]({pdf})。统一上下图作为第一页；PP Expert加长，显示每个本地层完整E专家，与EP每卡E/2对比。\n\n'
        handle.seek(0);handle.write(lines[0]+entry+''.join(lines[1:]));handle.truncate()
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE)not in text
        entry=f'- Single-intro/Expert-width revision: {ARCHIVE}. Four slides: previous slide2 replaces slide1; then the three data insights. PP Expert blocks explicitly label all E experts of the local layer and are twice EP E/2 block width; six Attention blocks retain equal width. All17 arrows point down. Three data pages,12-row tables and six plots verified preserved. No new GPU execution.\n\n'
        handle.seek(0);handle.write(text.replace(marker,entry+marker));handle.truncate()
    dump(ROOT/'expert_width_archive_validation.json',dict(**validation,archive=str(ARCHIVE),files_verified=len(checks)))
    print(json.dumps(dict(**validation,archive=str(ARCHIVE)),ensure_ascii=False))


if __name__=='__main__':main()
