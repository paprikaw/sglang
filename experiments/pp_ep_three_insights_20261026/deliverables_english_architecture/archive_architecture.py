"""Validate the appended architecture and archive the six-slide revision."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

from archive_english import dump, sha, NS

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'deliverables_english_architecture'
OLD = ROOT/'deliverables_english_domains_two_cases'
STEM = 'PP_EP_six_slides_EN_architecture_20261026'
ARCHIVE = ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/proposed_architecture_r11'


def main():
    assert not ARCHIVE.exists(), 'Refusing to overwrite a saved revision'
    assert json.loads((OLD/'validation.json').read_text())['passed']
    ppt, pdf = OUT/(STEM+'.pptx'), OUT/(STEM+'.pdf')
    old_ppt = OLD/json.loads((OLD/'manifest.json').read_text())['artifacts']['pptx']
    with zipfile.ZipFile(ppt) as package, zipfile.ZipFile(old_ppt) as prior:
        assert package.testzip() is None
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', n)]
        assert len(slides) == 6
        for n in range(1, 6):
            name = f'ppt/slides/slide{n}.xml'
            assert package.read(name) == prior.read(name), name
            assert sha(OUT/f'previews/slide_{n:02d}.png') == sha(OLD/f'previews/slide_{n:02d}.png')
        for name in slides + [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml', n)]:
            text = '\n'.join(n.text or '' for n in ET.fromstring(package.read(name)).findall('.//a:t', NS))
            assert not re.search(r'[\u3400-\u9fff]', text), name
        architecture = ET.fromstring(package.read('ppt/slides/slide6.xml'))
        text = '\n'.join(n.text or '' for n in architecture.findall('.//a:t', NS))
        for label in ['Lightweight Profiler', 'Performance Model', 'PP / EP Deployment',
                      'Live Workload', 'Autoscaler', 'PP scale-out', 'EP scale-out',
                      'Dispatch + Combine by domain', 'Measured latency & throughput']:
            assert label in text, label
        assert not architecture.findall('.//p:pic', NS)
        connectors = len(architecture.findall('.//p:cxnSp', NS))
        assert connectors == 10, connectors
    assert re.search(r'Pages:\s+6\b', subprocess.check_output(['pdfinfo', str(pdf)], text=True))
    assert sha(OUT/'figures/proposed_architecture.png') == sha(OUT/'previews/slide_06.png')
    assert re.search(r'Pages:\s+1\b', subprocess.check_output([
        'pdfinfo', str(OUT/'figures/proposed_architecture.pdf')], text=True))
    for name in ['latency.csv', 'expert_communication.csv', 'displayed_metrics.csv']:
        assert (OUT/name).read_bytes() == (OLD/name).read_bytes(), name
    for p in (OLD/'figures').glob('*.png'):
        assert sha(OUT/'figures'/p.name) == sha(p), p.name
    for name in ['plotted_throughput.csv', 'plotted_communication.csv']:
        assert sha(OUT/'domain_data'/name) == sha(OLD/'domain_data'/name)
    for name in ['profile_data', 'request_metrics', 'domain_data']:
        shutil.copytree(OLD/name, OUT/name, dirs_exist_ok=True)
    for name in ['Noto_CJK_LICENSE.txt', 'full_metrics.csv', 'stage_metrics.csv',
                 'request_metric_validation.json']:
        shutil.copy2(OLD/name, OUT/name)
    sources = {r['path']: r for r in json.loads((OLD/'sources.json').read_text())}
    sources.update({r['path']: r for r in json.loads((OUT/'sources.json').read_text())})
    for source in sources.values():
        assert sha(Path(source['path'])) == source['sha256'], source['path']
    dump(OUT/'sources.json', list(sources.values()))
    bounds = json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0 <= b['x'] and 0 <= b['y'] and b['x']+b['width'] <= 40/3+.001
               and b['y']+b['height'] <= 7.501 for b in bounds)
    assert len(json.loads((OUT/'slide_notes.json').read_text())) == 6
    design = json.loads((OUT/'architecture_design.json').read_text())
    assert not design['implemented'] and not design['empirical_speedup_claimed']
    validation = dict(passed=True, ppt_slides=6, pdf_pages=6, language='English',
        previous_five_slide_xml_preserved=True, previous_five_page_previews_preserved=True,
        all_previous_plotted_numeric_data_preserved=True, all_sources_verified=len(sources),
        architecture_native_shapes=True, architecture_connectors=connectors,
        proposed_architecture_has_no_fabricated_measurements=True, new_gpu_experiments=False,
        visual_review='Full-size architecture preview inspected; no text overflow or crossing flow arrows',
        checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json', validation)
    manifest = json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_proposed_architecture_r11',
                    title='Six English slides: EP/PP evidence and proposed deployment/autoscaling architecture',
                    created_at=validation['checked_at'], revision_of=str(OLD),
                    key_results='Adds a proposed Profiler -> Performance Model -> initial deployment workflow and model-guided PP/EP scale-out')
    manifest['artifacts'].update(pptx=ppt.name, pdf=pdf.name,
        architecture_png='figures/proposed_architecture.png',
        architecture_pdf='figures/proposed_architecture.pdf', architecture_specification='architecture_design.json')
    manifest['proposed_architecture'] = dict(slide=6, status='conceptual design',
        initial_placement='Profile Expert compute and communication, predict feasible PP x EP performance',
        runtime_scaling='Reuse the predictor to compare PP scale-out and EP scale-out under new load and topology',
        feedback='Measured serving latency and throughput update the model',
        implemented=False)
    dump(OUT/'manifest.json', manifest)
    (OUT/'summary.md').write_text('''# EP / PP deployment and proposed autoscaling architecture

Six English slides. The last slide proposes a lightweight Profiler, a Performance Model,
and an Autoscaler. The Profiler captures Expert execution and communication by payload
and domain. The model predicts latency/throughput for memory-feasible PP x EP layouts and
selects an initial configuration. The Autoscaler reuses the predictor to compare adding
pipeline stages with widening the expert group as load grows. Runtime measurements
calibrate the predictor. Model details and design scope are in the speaker notes.

The architecture uses editable native PowerPoint shapes and connectors. A standalone
PNG and one-page PDF are in figures/. The preceding five slide XML files, preview images,
plots, and numerical tables are unchanged from the verified two-domain-case revision.
No new GPU experiments were run. Rebuild: build_english.py --with-architecture.
''')
    for name in ['archive_architecture.py', 'archive_english.py']:
        shutil.copy2(ROOT/name, OUT/name)
    checks = [dict(path=str(p.relative_to(OUT)), sha256=sha(p)) for p in sorted(OUT.rglob('*'))
              if p.is_file() and p.suffix != '.zip' and p.name != 'checksums.json']
    dump(OUT/'checksums.json', checks)
    with zipfile.ZipFile(OUT/(STEM+'_data.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.zip':
                package.write(p, arcname=str(p.relative_to(OUT)))
    shutil.copytree(OUT, ARCHIVE)
    for r in checks:
        assert sha(ARCHIVE/r['path']) == r['sha256']
    report = ROOT.parent/'experiment_summary/component_reports/Q08.md'
    with report.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- PROPOSED ARCHITECTURE R11 -->'
        assert marker not in text
        lines = text.splitlines(keepends=True)
        entry = f'\n{marker}\n**English presentation with proposed architecture (6 slides):** [PPT]({ppt}) · [PDF]({pdf}). Adds the Profiler, initial PP/EP performance model, and model-guided PP/EP autoscaling. [Diagram]({OUT}/figures/proposed_architecture.png).\n\n'
        handle.seek(0)
        handle.write(lines[0]+entry+''.join(lines[1:]))
        handle.truncate()
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE) not in text
        entry = f'- Proposed architecture revision: {ARCHIVE}. Six English slides; final native diagram connects lightweight Expert/communication profiling, initial PP/EP performance prediction, and runtime PP-vs-EP scale-out decisions with feedback. Previous five slide XMLs/previews and all plotted numbers are preserved exactly. Conceptual design; no new GPU execution.\n\n'
        handle.seek(0)
        handle.write(text.replace(marker, entry+marker))
        handle.truncate()
    dump(ROOT/'architecture_archive_validation.json', dict(**validation, archive=str(ARCHIVE)))
    print(json.dumps(dict(**validation, archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
