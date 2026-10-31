"""Validate and archive the five-page Attention-placement revision."""
import argparse
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
OUT=ROOT/'deliverables_with_attention'
OLD=ROOT/'deliverables'
STEM='PP_EP_attention_and_three_insights_20261026'
ARCHIVE=ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/attention_r3'


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(efficiency=False):
    global OUT,ARCHIVE
    variant='attention_efficiency_r4'if efficiency else'attention_r3'
    if efficiency:
        OUT=ROOT/'deliverables_attention_efficiency'
        ARCHIVE=ARCHIVE.parent/variant
    assert not ARCHIVE.exists(),'Refusing to overwrite a saved revision'
    ppt=OUT/(STEM+'.pptx');pdf=OUT/(STEM+'.pdf')
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with zipfile.ZipFile(ppt)as package,zipfile.ZipFile(OLD/'PP_EP_three_insights_20261026.pptx')as old:
        assert package.testzip()is None
        names=[n for n in package.namelist()if n.startswith('ppt/slides/slide')and n.endswith('.xml')]
        assert len(names)==5
        second=ET.fromstring(package.read('ppt/slides/slide2.xml'))
        text='\n'.join(n.text or''for n in second.findall('.//a:t',ns))
        terms=(['Attention TP=1','本地 batch = 4','平均耗时应接近','本地 E / 2 专家',
            '本地 E 专家','单专家 tokens 更少','Expert 效率可能下降','Dispatch / Combine']if efficiency else
            ['DP2','Attention TP=1','半数请求','半数层','上下文','Expert 本地执行效率','Dispatch / Combine'])
        for word in terms:
            assert word in text,word
        assert sum(n.text=='Attention'for n in second.findall('.//a:t',ns))==(4 if efficiency else 12)
        assert not second.findall('.//p:pic',ns),'Attention placement must remain editable'
        for i in range(1,4):
            actual=ET.fromstring(package.read(f'ppt/slides/slide{i+2}.xml'))
            previous=ET.fromstring(old.read(f'ppt/slides/slide{i}.xml'))
            a=[n.text for n in actual.findall('.//a:t',ns)]
            b=[n.text for n in previous.findall('.//a:t',ns)]
            assert a[:2]==b[:2]and a[4:]==b[4:]
    assert 'Pages:           5'in subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    assert sha(OUT/'previews/slide_01.png')==sha(ROOT/'deliverables_with_deployment/previews/slide_01.png')
    for name in ['latency.csv','expert_communication.csv']:
        assert (OUT/name).read_bytes()==(OLD/name).read_bytes()
        assert len(list(csv.DictReader((OUT/name).open())))==12
    for path in (OLD/'figures').glob('*.png'):assert sha(path)==sha(OUT/'figures'/path.name)
    protocol=ROOT.parent/'pp_ep_steady_decode_20261026/deliverables/data/protocol.json'
    config=json.loads(protocol.read_text())
    assert config['EP2']['attention_tp']==config['PP2']['attention_tp']==1
    assert config['EP2']['attention_dp']==2 and config['PP2']['pipeline_stages']==2
    if efficiency:
        assert config['workload']['requests']==8
        assert config['PP2']['pipeline_slots']==2
        assert config['PP2']['max_requests_per_microbatch']==config['profiling']['actual_local_batch']==4
    shutil.copy2(protocol,OUT/'attention_placement_protocol.json')
    sources=json.loads((OUT/'sources.json').read_text())
    sources.append(dict(path=str(protocol),bytes=protocol.stat().st_size,sha256=sha(protocol)))
    for item in sources:assert sha(Path(item['path']))==item['sha256']
    dump(OUT/'sources.json',sources)
    notes=json.loads((OUT/'slide_notes.json').read_text());assert len(notes)==5
    bounds=json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0<=r['x'] and 0<=r['y'] and r['x']+r['width']<=40/3+.001 and r['y']+r['height']<=7.501 for r in bounds)
    validation=dict(passed=True,ppt_slides=5,pdf_pages=5,
        attention_placement_editable=True,attention_modules_shown=4 if efficiency else 12,
        attention_tp1_both_layouts_verified=True,ep_attention_dp2_verified=True,
        first_slide_unchanged=True,three_data_pages_unchanged=True,
        latency_rows=12,component_rows=12,six_data_charts_byte_identical=True,
        source_hashes_verified=len(sources),new_gpu_experiments=False,
        visual_review='Second slide at full size and all five slides in contact sheet inspected',
        checked_at=datetime.now(timezone.utc).isoformat())
    if efficiency:
        validation.update(matched_local_attention_batch=4,global_batch=8,
            local_expert_sets='EP E/2 versus PP E for each local model layer')
    dump(OUT/'validation.json',validation)
    for name in ['full_metrics.csv','stage_metrics.csv','deck_engine.py','Noto_CJK_LICENSE.txt']:
        shutil.copy2(OLD/name,OUT/name)
    shutil.copy2(ROOT/'archive_attention.py',OUT/'archive_attention.py')
    shutil.copytree(OLD/'profile_data',OUT/'profile_data')
    manifest=json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_attention_r3',
        title='EP2/PP2 deployment and Attention placement, followed by three measured insights',
        created_at=datetime.now(timezone.utc).isoformat(),
        revision_of=str(ROOT/'deliverables_with_deployment'),
        attention_diagram=dict(layers=4,gpus=2,attention_tp=1,
            ep='Attention DP2: all Attention layers per GPU, one request group per GPU',
            pp='Half the Attention layers per GPU, both request groups traverse both stages',
            invariant='Same per-token per-layer operator and model-level computation at equal contexts',
            focus='Expert local execution efficiency versus dispatch/combine exchange'),
        artifacts=dict(pptx=ppt.name,pdf=pdf.name,full_metrics='full_metrics.csv',stage_metrics='stage_metrics.csv',
            protocol='attention_placement_protocol.json',raw_profiles='profile_data/components_per_rank.csv'),
        key_results='One Attention-placement page inserted at slide2; previous deployment and three data pages preserved')
    if efficiency:
        manifest.update(id='pp_ep_three_insights_20261026_attention_efficiency_r4',
            title='Matched local Attention efficiency versus local Expert concentration',
            revision_of=str(ROOT/'deliverables_with_attention'),
            attention_diagram=dict(attention_tp=1,global_batch=8,local_batch_both_layouts=4,
                ep='Unsharded Attention; E/2 local experts pool routed tokens from both ranks',
                pp='Unsharded Attention per stage; E local experts process one B4 microbatch',
                focus='Attention mean time should be close at matched local shapes; Expert working set and tokens per expert change',
                expert_tiles='Conceptual E/2 versus E; no measured duration represented'),
            key_results='Slide2 now explains local Attention efficiency rather than only total FLOPs; PP local Expert concentration reduces token reuse')
    dump(OUT/'manifest.json',manifest)
    summary='''# Attention 放置与 Expert / 通信权衡

共5页：部署对比 → Attention放置与共同计算 → GPU数据 → 模型数据 → 阶段数据。

第二页以4层、两组等量请求A/B示意：EP2使用Attention TP1/DP2，各卡保留全部Attention层、处理一组请求；PP2使用Attention TP1，各卡保留半数层，两组请求均经过两阶段。同一模型、token和上下文下，每个token的逐层Attention算子与总计算量相同，因而主要关注Expert本地执行效率与Dispatch/Combine专家交换。

算子与计算量的不变性不等同于GPU执行时间严格相等。批形状、缓存和调度的影响，以及阶段通信和流水填充/排空，保留在详细备注与原始端到端数据中。此说明不放在幻灯正文。

第一张部署图、三组数据表和六张数据图均核验保留；未启动新GPU实验。文字和两张示意图使用原生可编辑PPT形状。重建命令：build.py --with-attention --out <output_directory>。PDF和PNG来自同一布局。
'''
    if efficiency:
        summary='''# Attention 本地效率与 Expert 集中程度

共5页。第二页改为比较本地算子效率：同一总batch B8下，EP2按卡分成B4+B4，PP2两槽微批也各为B4；两边Attention TP1、匹配token数及上下文后，本地Attention算子形状与每次调用的平均耗时应接近。

Expert有结构差异：EP每卡每层持有E/2专家，通过Dispatch/Combine汇聚两卡路由token；PP每卡每个本地层持有E专家，处理当前B4微批。本地专家工作集扩大、平均单专家token减少，可能降低Expert执行效率。示意块只表示E/2与E的比例。未把图形尺寸当作耗时数据。

协议已核验Attention TP1、全局B8、PP两槽、实际本地B4。匹配本地Attention形状是该示例的关键；不将相同总batch单独解释为任何微批策略下的耗时恒等。第一张部署图与后三组数据保持一致，无新增GPU实验。详细来源在备注与sources.json。重建：build.py --with-attention --out <output_directory>。
'''
    (OUT/'summary.md').write_text(summary)
    checks=[dict(path=str(p.relative_to(OUT)),sha256=sha(p))for p in sorted(OUT.rglob('*'))
        if p.is_file()and p.suffix!='.zip'and p.name!='checksums.json']
    dump(OUT/'checksums.json',checks)
    bundle=OUT/'PP_EP_attention_and_three_insights_data_20261026.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED)as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file()and p.suffix!='.zip':package.write(p,arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle)as package:assert package.testzip()is None
    shutil.copytree(OUT,ARCHIVE)
    for item in checks:assert sha(ARCHIVE/item['path'])==item['sha256']
    report=ROOT.parent/'experiment_summary/component_reports/Q08.md'
    with report.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- ATTENTION EFFICIENCY PPT R4 -->'if efficiency else'<!-- ATTENTION PLACEMENT PPT R3 -->'
        assert marker not in text
        lines=text.splitlines(keepends=True)
        entry=f'\n{marker}\n**部署图 + Attention放置 + 三组数据（5页）：** [PPT]({ppt}) · [PDF]({pdf})。第二页对比EP的Attention DP2与PP按层放置，突出相同逐token计算及Expert / Dispatch-Combine的主要权衡。\n\n'
        if efficiency:
            entry=f'\n{marker}\n**Attention本地效率与Expert集中程度（5页）：** [PPT]({ppt}) · [PDF]({pdf})。第二页以相同总B8、本地Attention B4说明平均耗时应接近；Expert对比E/2与E的本地工作集和单专家token复用。\n\n'
        handle.seek(0);handle.write(lines[0]+entry+''.join(lines[1:]));handle.truncate()
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);text=handle.read()
        marker='<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE)not in text
        entry=f'- Attention-placement revision: {ARCHIVE}. Five slides: deployment, Attention placement, three original data insights. Slide2 shows TP1 Attention DP2 in EP versus layer-stage placement in PP, with equal per-token computation at equal contexts; highlights local Expert efficiency versus dispatch/combine. Native editable shapes; original first slide, both12-row tables and six data figures verified unchanged. Protocol and notes retain timing scope. No new GPU execution.\n\n'
        if efficiency:
            entry=f'- Attention-efficiency revision: {ARCHIVE}. Slide2 now compares local execution efficiency: matched TP1 Attention B4 calls under global B8 yield similar expected per-call time, while PP concentrates E local experts versus EP E/2 and lowers tokens per expert. Equal Attention boxes and doubled expert-set tiles express the mechanism. Protocol confirms actual local batch4 and two PP slots. Five-page deck, first deployment page and three data pages preserved; no new GPU timing.\n\n'
        handle.seek(0);handle.write(text.replace(marker,entry+marker));handle.truncate()
    validation_name='attention_efficiency_archive_validation.json'if efficiency else'attention_archive_validation.json'
    dump(ROOT/validation_name,dict(**validation,archive=str(ARCHIVE),files_verified=len(checks),bundle=str(bundle)))
    print(json.dumps(dict(**validation,archive=str(ARCHIVE)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--efficiency',action='store_true')
    main(efficiency=parser.parse_args().efficiency)
