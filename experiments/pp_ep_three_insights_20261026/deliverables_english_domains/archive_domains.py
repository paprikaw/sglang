"""Validate the appended communication-domain slide and save revision 9."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree as ET

from archive_english import dump, sha, rows, close, NS

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
OUT = ROOT/'deliverables_english_domains'
OLD = ROOT/'deliverables_english_throughput'
DOMAIN = EXP/'h100_ep_pp_comm_domain_comparison_20261026'
STEM = 'PP_EP_five_slides_EN_domains_20261026'
ARCHIVE = ROOT.parents[4]/'experiment_manager/experiments/pp_ep_three_insights_20261026/reports/communication_domains_r9'


def main():
    assert not ARCHIVE.exists(), 'Refusing to overwrite a saved revision'
    ppt, pdf = OUT/(STEM+'.pptx'), OUT/(STEM+'.pdf')
    with zipfile.ZipFile(ppt) as package:
        assert package.testzip() is None
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml', n)]
        assert len(slides) == 5
        for name in slides + [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml', n)]:
            xml = ET.fromstring(package.read(name))
            text = '\n'.join(n.text or '' for n in xml.findall('.//a:t', NS))
            assert not re.search(r'[\u3400-\u9fff]', text), name
        new = ET.fromstring(package.read('ppt/slides/slide5.xml'))
        text = '\n'.join(n.text or '' for n in new.findall('.//a:t', NS))
        for label in ['1 GPU · GPU-local', '2 GPUs · NVLink', '2 nodes · RDMA',
                      'Deployment throughput · EP / PP', 'Dispatch + Combine (ms)']:
            assert label in text
        assert len(new.findall('.//p:pic', NS)) == 2
    assert re.search(r'Pages:\s+5\b', subprocess.check_output(['pdfinfo', str(pdf)], text=True))
    for n in range(1, 5):
        name = f'previews/slide_{n:02d}.png'
        assert sha(OUT/name) == sha(OLD/name), name
    for name in ['latency.csv', 'expert_communication.csv', 'displayed_metrics.csv']:
        assert (OUT/name).read_bytes() == (OLD/name).read_bytes(), name
    for p in (OLD/'figures').glob('*.png'):
        assert sha(OUT/'figures'/p.name) == sha(p)
    comparisons = rows(OUT/'domain_data/e2e_comparisons.csv')
    formal = rows(OUT/'domain_data/full_metrics.csv')
    comm = rows(OUT/'domain_data/plotted_communication.csv')
    original_comm = rows(DOMAIN/'sources/communication_metrics.csv')
    assert len(comparisons) == 6 and len(formal) == 12 and len(comm) == 9
    for r in comparisons:
        selected = [m for m in formal if m['case'] == r['case'] and m['phase'] == r['phase']]
        assert len(selected) == 2
        ep = next(m for m in selected if m['layout'].startswith('EP'))
        pp = next(m for m in selected if m['layout'].startswith('PP'))
        close(ep['throughput'], r['ep_throughput'])
        close(pp['throughput'], r['pp_throughput'])
        close(float(ep['throughput'])/float(pp['throughput']), r['ep_over_pp'])
        close(100*(float(r['ep_over_pp'])-1), r['ep_vs_pp_pct'])
    for r in comm:
        assert r in original_comm and int(r['n']) == 3
        assert float(r['min_ms']) <= float(r['median_ms']) <= float(r['max_ms'])
    for r in rows(OUT/'domain_data/communication_comparisons.csv'):
        close(float(r['ep8_cross_median_ms'])/float(r['ep4_local_median_ms']), r['cross_over_local'])
    native = json.loads((DOMAIN/'sources/communication_manifest.json').read_text())
    model = json.loads((DOMAIN/'sources/eight_gpu_manifest.json').read_text())
    assert native['implementation']['native_sha256'] == model['deepep_custom']['sha256']
    assert not native['model']['loaded'] and not native['workload']['route_padding']
    source_checks = []
    for r in json.loads((DOMAIN/'source_manifest.json').read_text()):
        assert sha(Path(r['path'])) == r['sha256']
        assert sha(DOMAIN/r['snapshot']) == r['sha256']
        source_checks.append(r)
    dump(OUT/'domain_data/source_verification.json', dict(passed=True,
         original_source_hashes_verified=len(source_checks),
         same_comm_native=native['implementation']['native_sha256'],
         e2e_comparisons=6, communication_bars=9, min_max_whiskers=True,
         single_gpu_card='Concept only; no single-GPU timing is asserted',
         e2e_interpretation='Separate historical model/workload cohorts; ratios normalized within each pair',
         communication_scope='Standalone Dispatch+Combine, fixed global logical tokens; rank count and partition differ'))
    for name in ['profile_data', 'request_metrics']:
        shutil.copytree(OLD/name, OUT/name, dirs_exist_ok=True)
    for name in ['Noto_CJK_LICENSE.txt', 'full_metrics.csv', 'stage_metrics.csv', 'request_metric_validation.json']:
        shutil.copy2(OLD/name, OUT/name)
    sources = {r['path']: r for r in json.loads((OLD/'sources.json').read_text())}
    sources.update({r['path']: r for r in json.loads((OUT/'sources.json').read_text())})
    for i, r in enumerate(json.loads((OUT/'domain_data/attachments.json').read_text()), 1):
        p = Path(r['path'])
        target = OUT/'domain_data'/f'reference_figure_{i}.png'
        shutil.copy2(p, target)
        sources[str(p)] = dict(**r, bytes=p.stat().st_size,
                               copied_as=str(target.relative_to(OUT)))
    for r in sources.values():
        assert sha(Path(r['path'])) == r['sha256']
    dump(OUT/'sources.json', list(sources.values()))
    bounds = json.loads((OUT/'layout_bounds.json').read_text())
    assert all(0 <= r['x'] and 0 <= r['y'] and r['x']+r['width'] <= 40/3+.001
               and r['y']+r['height'] <= 7.501 for r in bounds)
    assert len(json.loads((OUT/'slide_notes.json').read_text())) == 5
    manifest = json.loads((OLD/'manifest.json').read_text())
    manifest.update(id='pp_ep_three_insights_20261026_communication_domains_r9',
                    title='Five English slides: EP/PP deployment and communication-domain boundaries',
                    created_at=datetime.now(timezone.utc).isoformat(), revision_of=str(OLD),
                    key_results='Append one communication-domain page with six throughput contrasts and nine standalone communication bars')
    manifest['artifacts'].update(pptx=ppt.name, pdf=pdf.name,
                                 communication_domain_data='domain_data/',
                                 domain_full_metrics='domain_data/full_metrics.csv')
    manifest['communication_domain_slide'] = dict(slide=5,
        conceptual_progression=['single GPU', 'NVLink-local GPUs', 'cross-node RDMA'],
        measurements='Original two user-supplied plots, redrawn from saved source tables',
        paired_layouts=['EP2 versus PP2', 'EP8 versus PP2xEP4'],
        e2e_models=['Qwen3.5-122B FP8', 'GLM-4.7-AWQ INT4'],
        comm_native_sha256=native['implementation']['native_sha256'],
        scope_notes='speaker_notes.md')
    dump(OUT/'manifest.json', manifest)
    validation = dict(passed=True, ppt_slides=5, pdf_pages=5, language='English',
        previous_four_page_previews_preserved=True, previous_three_data_tables_preserved=True,
        supplied_figure_datasets_integrated=True, throughput_comparisons=6,
        communication_bars=9, formal_domain_metric_rows=12,
        original_domain_sources_verified=len(source_checks), all_sources_verified=len(sources),
        single_gpu_schematic_has_no_fabricated_timings=True, new_gpu_experiments=False,
        visual_review='Fifth slide inspected at full size; small-message value labels staggered to avoid overlap',
        checked_at=datetime.now(timezone.utc).isoformat())
    dump(OUT/'validation.json', validation)
    (OUT/'summary.md').write_text('''# EP / PP deployment: communication-domain page

Five English slides. The fifth page adds the conceptual GPU-local -> NVLink -> cross-node
progression and redraws both supplied figure datasets. Six within-cohort throughput ratios
and nine standalone Dispatch+Combine medians are shown, retaining both local nodes and
the observed min/max whiskers. Twelve complete domain request-metric rows are included.

The paired deployments are EP2/PP2 on two GPUs and EP8/PP2xEP4 on eight GPUs. The Expert
execution versus repeated cross-domain exchange trade-off is the page's decision rule.
Models, cache starts, workloads and timing scopes are detailed in the English speaker notes.
The single-GPU card is conceptual. The original four page previews are preserved exactly.

Rebuild from the workspace experiment root: build_english.py --with-domains.
''')
    for name in ['archive_domains.py', 'archive_english.py']:
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
    report = EXP/'experiment_summary/component_reports/Q08.md'
    with report.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- COMMUNICATION DOMAINS FIVE SLIDES R9 -->'
        assert marker not in text
        lines = text.splitlines(keepends=True)
        entry = f'\n{marker}\n**English presentation with communication domains (5 slides):** [PPT]({ppt}) · [PDF]({pdf}). Added GPU-local → NVLink → cross-node progression, paired deployment throughput and standalone Dispatch+Combine data.\n\n'
        handle.seek(0)
        handle.write(lines[0]+entry+''.join(lines[1:]))
        handle.truncate()
    registry = Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    with registry.open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        marker = '<!-- end pp_ep_three_insights_20261026 -->'
        assert marker in text and str(ARCHIVE) not in text
        entry = f'- Communication-domain slide revision: {ARCHIVE}. Fifth English page integrates both user-supplied H100 datasets: six paired EP/PP throughput contrasts and nine standalone DeepEP Dispatch+Combine medians with min/max whiskers. Conceptual single-GPU/NVLink/cross-node progression. Two-GPU EP2/PP2 and eight-GPU EP8/PP2xEP4 scopes explicit; model/workload distinctions in notes. Original four page previews and numeric tables preserved. No new GPU execution.\n\n'
        handle.seek(0)
        handle.write(text.replace(marker, entry+marker))
        handle.truncate()
    dump(ROOT/'domains_archive_validation.json', dict(**validation, archive=str(ARCHIVE)))
    print(json.dumps(dict(**validation, archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
