"""Validate and archive the English deck with additional serving metrics."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
OUT = ROOT / 'deliverables_english'
OLD = ROOT / 'deliverables_expert_width'
STEM = 'PP_EP_four_slides_EN_20261026'
ARCHIVE = ROOT.parents[4] / 'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/english_metrics_r7'
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(path):
    return list(csv.DictReader(path.open()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert abs(float(a)-float(b)) < 1e-6, (a, b)


def identity(row):
    return tuple(row[key] for key in ['hardware', 'model', 'phase', 'arm'])


def raw_metrics_validation(latency, sources):
    expected = {identity(row): row for row in latency}
    h100 = {}
    for model, run in [
            ('122B', EXP/'ep2_pp2_qwen_h100_20261025/runs/h100_ep2_pp2_r7'),
            ('35B', EXP/'ep2_pp2_qwen35b_h100_20261025/runs/h100_qwen35b_ep2_pp2_r3')]:
        evidence_path = run/'analysis/validated_metrics.json'
        evidence = json.loads(evidence_path.read_text())
        assert evidence['passed']
        for row in evidence['sources']:
            p = Path(row['path'])
            if not p.is_absolute():
                p = ROOT.parents[1] / p
            if p.name.startswith('metrics.moderate.'):
                phase = p.name.split('.')[-2]
                h100[('H100', model, phase, p.parent.name)] = (p, row['sha256'])
    a100 = {}
    for row in rows(EXP/'experiment_summary/qwen_ep2_pp2_metrics.csv'):
        if row['workload'] == 'decode' and (
                row['configuration'] == 'EP2' or row['pipeline_slots'] == '2'):
            arm = 'ep2' if row['configuration'] == 'EP2' else 'pp2'
            a100[('A100', '122B', 'decode', arm)] = (Path(row['source_path']), row['source_sha256'])
    validation = []
    raw_dir = OUT/'request_metrics'
    raw_dir.mkdir(exist_ok=True)
    for key, expected_row in sorted(expected.items()):
        p, expected_hash = (h100 if key[0] == 'H100' else a100)[key]
        assert sha(p) == expected_hash, p
        data = json.loads(p.read_text())
        assert data['valid'] and not data['profiler_enabled'] and len(data['rounds']) == 3
        requests, rates = [], []
        for group in data['rounds']:
            cohort = group['requests']
            assert len(cohort) == 8
            makespan = max(r['finished'] for r in cohort) - min(r['started'] for r in cohort)
            close(makespan, group['makespan_s'])
            tokens = 4096 if key[2] == 'prefill' else 128
            rates.append(8*tokens/makespan)
            requests.extend(cohort)
        ttft = statistics.mean(r['ttft_s'] for r in requests)
        e2e = statistics.mean(r['e2e_s'] for r in requests)
        rate = statistics.mean(rates)
        close(ttft, expected_row['ttft_s'])
        close(e2e, expected_row['e2e_s'])
        close(rate, expected_row['throughput'])
        close(ttft, data['ttft_s']['mean'])
        if key[2] == 'decode':
            close(e2e, ttft + 127*float(expected_row['tpot_ms'])/1000)
            assert expected_row['throughput_unit'] == 'output tokens/s'
        else:
            close(ttft, e2e)
            assert expected_row['throughput_unit'] == 'input tokens/s'
        destination = raw_dir/('_'.join(key)+'.json')
        shutil.copy2(p, destination)
        sources.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p),
                            copied_as=str(destination.relative_to(OUT))))
        validation.append(dict(hardware=key[0], model=key[1], phase=key[2], arm=key[3],
                               requests=24, rounds=3, ttft_s=ttft, e2e_s=e2e,
                               throughput=rate, throughput_unit=expected_row['throughput_unit'],
                               source=str(p), source_sha256=sha(p)))
    assert len(validation) == 8
    dump(OUT/'request_metric_validation.json', dict(passed=True, cells=validation,
         throughput_definition='Mean per-round fresh-input tokens / cohort makespan for Prefill; output tokens / makespan for Decode'))
    return validation


def main():
    assert not ARCHIVE.exists(), 'Refusing to overwrite a saved revision'
    ppt, pdf = OUT/(STEM+'.pptx'), OUT/(STEM+'.pdf')
    with zipfile.ZipFile(ppt) as package:
        assert package.testzip() is None
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', n)]
        assert len(slides) == 4
        for name in slides + [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml', n)]:
            xml = ET.fromstring(package.read(name))
            text = '\n'.join(n.text or '' for n in xml.findall('.//a:t', NS))
            assert not re.search(r'[\u3400-\u9fff]', text), name
        first = ET.fromstring(package.read('ppt/slides/slide1.xml'))
        text = '\n'.join(n.text or '' for n in first.findall('.//a:t', NS))
        assert text.count('Attention · B4') == 6
        assert text.count('Expert · E/2') == 4
        assert text.count('Expert · E (all experts of this layer)') == 2
        assert not first.findall('.//p:pic', NS)
        assert sum(n.attrib['prst'] == 'downArrow' for n in first.findall('.//a:prstGeom', NS)) == 17
    assert re.search(r'Pages:\s+4\b', subprocess.check_output(['pdfinfo', str(pdf)], text=True))
    latency = rows(OUT/'latency.csv')
    old_latency = rows(OLD/'latency.csv')
    assert len(latency) == len(old_latency) == 12
    for new, old in zip(latency, old_latency):
        for key, value in old.items():
            assert new[key] == value, (key, new[key], value)
    components, old_components = rows(OUT/'expert_communication.csv'), rows(OLD/'expert_communication.csv')
    assert len(components) == len(old_components) == 12
    for new, old in zip(components, old_components):
        assert identity(new) == identity(old)
        for key in ['expert', 'exchange', 'pipeline']:
            close(new[key], old[key])
    displayed = rows(OUT/'displayed_metrics.csv')
    assert len(displayed) == 24
    for row in latency:
        matches = [r for r in displayed if identity(r) == identity(row) and r['slide'] == row['slide']]
        expected_supplement = 'ttft_s' if row['phase'] == 'decode' else 'throughput'
        assert {r['metric'] for r in matches} == {'e2e_s', expected_supplement}
        for r in matches:
            close(r['raw_value'], row[r['metric']])
            close(r['displayed_value'], float(r['raw_value'])*float(r['display_scale']))
    sources = json.loads((OUT/'sources.json').read_text())
    for row in sources:
        assert sha(Path(row['path'])) == row['sha256']
    request_checks = raw_metrics_validation(latency, sources)
    dump(OUT/'sources.json', sources)
    assert len(list((OUT/'figures').glob('*.png'))) == 6
    assert len(list((OUT/'figures').glob('*.pdf'))) == 6
    notes = json.loads((OUT/'slide_notes.json').read_text())
    assert len(notes) == 4 and not re.search(r'[\u3400-\u9fff]', json.dumps(notes, ensure_ascii=False))
    bounds = json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0 <= r['x'] and 0 <= r['y'] and r['x']+r['width'] <= 40/3+.001
               and r['y']+r['height'] <= 7.501 for r in bounds)
    for name in ['Noto_CJK_LICENSE.txt']:
        shutil.copy2(OLD/name, OUT/name)
    shutil.copytree(OLD/'profile_data', OUT/'profile_data', dirs_exist_ok=True)
    shutil.copy2(OUT/'latency.csv', OUT/'full_metrics.csv')
    shutil.copy2(OUT/'expert_communication.csv', OUT/'stage_metrics.csv')
    validation = dict(passed=True, ppt_slides=4, pdf_pages=4, language='English',
                      slide_and_speaker_note_text_contains_no_chinese=True,
                      latency_rows=12, component_rows=12, plotted_metric_rows=24,
                      decode_ttft_unique_cells=4, prefill_input_throughput_unique_cells=4,
                      raw_request_metrics_recomputed_cells=len(request_checks),
                      original_e2e_and_expert_communication_numbers_preserved=True,
                      sources_verified=len(sources), new_gpu_experiments=False,
                      visual_review='All four pages and three data pages inspected at full size',
                      checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json', validation)
    manifest = json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_english_metrics_r7',
                    title='English EP/PP deployment insights with Decode TTFT and Prefill throughput',
                    created_at=datetime.now(timezone.utc).isoformat(), revision_of=str(OLD),
                    language='English', key_results='Four English slides; Decode adds TTFT, Prefill adds input throughput; existing E2E and component values retained')
    manifest['workload'].update(ttft='Mean time to first streamed token within each workload; Decode uses cached prefix',
                                throughput='Prefill fresh input tokens / cohort makespan, mean across three rounds')
    manifest['artifacts'].update(pptx=ppt.name, pdf=pdf.name, displayed_metrics='displayed_metrics.csv',
                                 request_validation='request_metric_validation.json',
                                 raw_request_metrics='request_metrics/')
    dump(OUT/'manifest.json', manifest)
    (OUT/'summary.md').write_text('''# EP / PP deployment: English presentation

Four slides: vertical EP2/PP2 placement; GPU platform; model; Prefill versus Decode.
All slide text, charts, footers and speaker notes are in English.

Decode comparisons include mean TTFT in milliseconds. Prefill comparisons include input
throughput in k tokens/s. The mixed-phase slide shows Prefill input throughput and Decode
TTFT in their respective columns. The existing end-to-end latency and separated Expert,
dispatch/combine and PP communication values are retained.

Eight unique metric cells were recomputed from 24 requests each across three saved rounds.
Prefill throughput is fresh input tokens divided by each cohort's makespan, averaged across
rounds. Decode TTFT starts from the recorded cached prefix. Detailed protocols and provenance
are in the speaker notes, sources.json and request_metric_validation.json.

Rebuild from the workspace experiment root with build_english.py --out <output_directory>.
''')
    shutil.copy2(ROOT/'archive_english.py', OUT/'archive_english.py')
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
        marker = '<!-- ENGLISH METRICS FOUR SLIDES R7 -->'
        assert marker not in text
        lines = text.splitlines(keepends=True)
        entry = f'\n{marker}\n**English presentation (4 slides):** [PPT]({ppt}) · [PDF]({pdf}). Decode now includes TTFT; Prefill includes input throughput. All diagrams, chart labels and speaker notes are in English.\n\n'
        handle.seek(0)
        handle.write(lines[0]+entry+''.join(lines[1:]))
        handle.truncate()
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE) not in text
        entry = f'- English/serving-metrics revision: {ARCHIVE}. Four slides translated throughout, including diagram, charts, footers and speaker notes. Decode adds mean TTFT (A100 EP/PP 278.0/173.9 ms; H100 275.7/178.5 ms); Prefill adds input throughput (35B EP/PP 67.8/77.4k tokens/s; 122B 40.6/43.2k tokens/s). Eight raw request-metric cells recomputed across three saved rounds each. Original E2E and Expert/communication numbers preserved; no new GPU execution.\n\n'
        handle.seek(0)
        handle.write(text.replace(marker, entry+marker))
        handle.truncate()
    dump(ROOT/'english_archive_validation.json', dict(**validation, archive=str(ARCHIVE)))
    print(json.dumps(dict(**validation, archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
