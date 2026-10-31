"""Archive the targeted slide-2 change from TTFT to Decode output throughput."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

import archive_english as previous

ROOT, EXP = previous.ROOT, previous.EXP
OUT = ROOT/'deliverables_english_throughput'
OLD = ROOT/'deliverables_english'
STEM = 'PP_EP_four_slides_EN_throughput_20261026'
ARCHIVE = ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/english_throughput_r8'
dump, sha, rows = previous.dump, previous.sha, previous.rows


def main():
    assert not ARCHIVE.exists(), 'Refusing to overwrite a saved revision'
    assert json.loads((OLD/'validation.json').read_text())['passed']
    ppt, pdf = OUT/(STEM+'.pptx'), OUT/(STEM+'.pdf')
    with zipfile.ZipFile(ppt) as package:
        assert package.testzip() is None
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', n)]
        assert len(slides) == 4
        for name in slides + [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml', n)]:
            xml = ET.fromstring(package.read(name))
            text = '\n'.join(n.text or '' for n in xml.findall('.//a:t', previous.NS))
            assert not re.search(r'[\u3400-\u9fff]', text), name
    assert re.search(r'Pages:\s+4\b', subprocess.check_output(['pdfinfo', str(pdf)], text=True))
    for n in [1, 3, 4]:
        name = f'previews/slide_{n:02d}.png'
        assert sha(OUT/name) == sha(OLD/name), name
    for p in (OLD/'figures').glob('*.png'):
        if p.name != 'gpu_latency.png':
            assert sha(OUT/'figures'/p.name) == sha(p), p.name
    for name in ['latency.csv', 'expert_communication.csv']:
        assert (OUT/name).read_bytes() == (OLD/name).read_bytes(), name
    plotted = rows(OUT/'displayed_metrics.csv')
    old_plotted = rows(OLD/'displayed_metrics.csv')
    assert len(plotted) == len(old_plotted) == 24
    assert [r for r in plotted if r['slide'] != 'gpu'] == [r for r in old_plotted if r['slide'] != 'gpu']
    output_rows = [r for r in plotted if r['slide'] == 'gpu' and r['metric'] == 'throughput']
    assert len(output_rows) == 4
    assert not any(r['slide'] == 'gpu' and r['metric'] == 'ttft_s' for r in plotted)
    latency = rows(OUT/'latency.csv')
    for r in output_rows:
        matches = [x for x in latency if x['slide'] == 'gpu' and previous.identity(x) == previous.identity(r)]
        assert len(matches) == 1
        previous.close(r['display_scale'], 1)
        previous.close(r['displayed_value'], matches[0]['throughput'])
        assert matches[0]['throughput_unit'] == 'output tokens/s'
        assert r['display_label'] == 'Output throughput\n(tokens/s) ↑'
    sources = json.loads((OUT/'sources.json').read_text())
    for row in sources:
        assert sha(Path(row['path'])) == row['sha256']
    previous.OUT = OUT
    request_checks = previous.raw_metrics_validation(latency, sources)
    dump(OUT/'sources.json', sources)
    for name in ['Noto_CJK_LICENSE.txt', 'attention_placement_protocol.json']:
        shutil.copy2(OLD/name, OUT/name)
    shutil.copytree(OLD/'profile_data', OUT/'profile_data', dirs_exist_ok=True)
    shutil.copy2(OUT/'latency.csv', OUT/'full_metrics.csv')
    shutil.copy2(OUT/'expert_communication.csv', OUT/'stage_metrics.csv')
    shutil.copy2(ROOT/'archive_english.py', OUT/'archive_english.py')
    shutil.copy2(ROOT/'archive_english_throughput.py', OUT/'archive_english_throughput.py')
    manifest = json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_english_throughput_r8',
                    title='English EP/PP deployment insights: slide 2 compares Decode throughput',
                    created_at=datetime.now(timezone.utc).isoformat(), revision_of=str(OLD),
                    key_results='Slide 2 uses E2E latency and output throughput instead of TTFT; other three slides retained')
    manifest['artifacts'].update(pptx=ppt.name, pdf=pdf.name)
    manifest['workload']['throughput'] = ('Mean per-round output tokens / cohort makespan for Decode; '
                                          'fresh input tokens / cohort makespan for Prefill')
    dump(OUT/'manifest.json', manifest)
    validation = dict(passed=True, ppt_slides=4, pdf_pages=4, language='English',
                      slide_2_ttft_replaced_by_output_throughput=True,
                      output_throughput_unit='output tokens/s',
                      output_throughput_cells=4, raw_request_metric_cells_recomputed=len(request_checks),
                      other_three_slide_previews_unchanged=True,
                      original_latency_and_component_tables_unchanged=True,
                      source_hashes_verified=len(sources), new_gpu_experiments=False,
                      visual_review='Revised second slide inspected at full size',
                      checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json', validation)
    (OUT/'summary.md').write_text('''# EP / PP deployment: Decode throughput revision

The second slide now compares mean full-request E2E latency and Decode output throughput,
replacing its TTFT charts. Throughput is output tokens divided by cohort makespan, averaged
over three saved rounds, with 8 requests and 128 output tokens per request.

A100: EP2 72.1 versus PP2 97.3 output tokens/s. H100: EP2 636.6 versus PP2 509.8 output tokens/s.
The other three slide previews and both original numeric tables are preserved. All four
slides remain in English. Raw request records were checked again; no GPU experiment ran.

Rebuild from the workspace experiment root: build_english.py --gpu-throughput.
''')
    checks = [dict(path=str(p.relative_to(OUT)), sha256=sha(p)) for p in sorted(OUT.rglob('*'))
              if p.is_file() and p.suffix != '.zip' and p.name != 'checksums.json']
    dump(OUT/'checksums.json', checks)
    bundle = OUT/(STEM+'_data.zip')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.zip':
                package.write(p, arcname=str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle) as package:
        assert package.testzip() is None
    shutil.copytree(OUT, ARCHIVE)
    for row in checks:
        assert sha(ARCHIVE/row['path']) == row['sha256']
    report = EXP/'experiment_summary/component_reports/Q08.md'
    with report.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- ENGLISH THROUGHPUT FOUR SLIDES R8 -->'
        assert marker not in text
        lines = text.splitlines(keepends=True)
        entry = f'\n{marker}\n**English presentation — Decode throughput (4 slides):** [PPT]({ppt}) · [PDF]({pdf}). Slide 2 compares E2E latency and output tokens/s; its TTFT plots have been replaced.\n\n'
        handle.seek(0)
        handle.write(lines[0]+entry+''.join(lines[1:]))
        handle.truncate()
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE) not in text
        entry = f'- English/Decode-throughput revision: {ARCHIVE}. Only slide 2 changed: TTFT plots replaced with output throughput (A100 EP/PP 72.1/97.3 tokens/s; H100 636.6/509.8 tokens/s). E2E retained. Other three page previews, numeric tables and five figures preserved. Eight saved request-metric cells recomputed; no new GPU execution.\n\n'
        handle.seek(0)
        handle.write(text.replace(marker, entry+marker))
        handle.truncate()
    dump(ROOT/'english_throughput_archive_validation.json', dict(**validation, archive=str(ARCHIVE)))
    print(json.dumps(dict(**validation, archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
