"""Check the handoff against source observations and presentation structure."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables'


def rows(path):
    return list(csv.DictReader(path.open()))


def main():
    sources = json.loads((OUT/'data/source_hashes.json').read_text())
    for source in sources:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    metrics = rows(OUT/'data/native_metrics.csv')
    request_rows = rows(OUT/'data/requests.csv')
    round_rows = rows(OUT/'data/rounds.csv')
    assert len(metrics) == 4 and len(request_rows) == 96 and len(round_rows) == 12
    for row in metrics:
        hw, arm = row['hardware'], row['arm']
        selected = [r for r in request_rows if (r['hardware'],r['arm']) == (hw,arm)]
        rounds = [r for r in round_rows if (r['hardware'],r['arm']) == (hw,arm)]
        assert len(selected) == 24 and len(rounds) == 3
        assert math.isclose(float(row['output_tokens_per_s']),statistics.mean(float(r['output_tokens_per_s']) for r in rounds),rel_tol=1e-12)
        assert math.isclose(float(row['tpot_ms']),statistics.mean(float(r['tpot_ms']) for r in selected),rel_tol=1e-12)
        assert math.isclose(float(row['ttft_s']),statistics.mean(float(r['ttft_s']) for r in selected),rel_tol=1e-12)
        assert all((int(r['cached_tokens']),int(r['prompt_tokens']),int(r['output_tokens']))==(4096,4097,128)for r in selected)
    ranks = rows(OUT/'data/components_per_rank.csv')
    assert len(ranks) == 36 and all(int(r['selected_forwards'])==24 for r in ranks)
    assert sum(int(r['actual_layer_invocations']) for r in ranks)==25920
    assert len(rows(OUT/'data/components.csv'))==6
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(OUT/'PP_EP_deployment_decision_20261026.pptx') as z:
        assert z.testzip() is None
        slides=[n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
        assert len(slides)==12
        assert len([n for n in z.namelist() if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml')])==12
        for name in slides:
            xml=ET.fromstring(z.read(name))
            assert ''.join(xml.itertext()).strip()
            for run in xml.findall('.//a:r',ns):
                assert run.find('a:rPr/a:ea',ns).get('typeface')=='Noto Sans CJK SC'
    notes=json.loads((OUT/'slide_notes.json').read_text())
    assert len(notes)==12 and all(r['notes'] and r['sources'] for r in notes)
    bounds=json.loads((OUT/'layout_bounds.json').read_text())
    assert all(r['x']>=0 and r['y']>=0 and r['x']+r['width']<=40/3+.001 and r['y']+r['height']<=7.501 for r in bounds)
    info=subprocess.check_output(['pdfinfo',str(OUT/'PP_EP_deployment_decision_20261026.pdf')],text=True)
    assert 'Pages:           12' in info and '960 x 540 pts' in info
    # Check that the shareable artifact set contains no run authentication values.
    secrets=[]
    for name in ['a100_r2','h100_r2']:
        run=ROOT/'runs'/name
        secrets.append((run/'admin.key').read_bytes().strip())
        lease=json.loads((run/'dashboard_guard/gpu_registry_lease.json').read_text())
        secrets.append(lease['token'].encode())
        clean=json.loads((OUT/'evidence'/name/'guard_and_cleanup.json').read_text())
        assert clean['passed'] and clean['model_cache_passed'] and clean['cleanup_verified']
        assert clean['final_gpu_lease']['state']=='released'
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix != '.zip':
            content=path.read_bytes()
            assert all(secret not in content for secret in secrets if len(secret)>=16), path
            assert path.name not in ['admin.key','config.json','gpu_registry_lease.json']
    result=dict(passed=True,checked_at=datetime.now(timezone.utc).isoformat(),
        source_files_verified=len(sources),native_cells=4,raw_request_rows=96,formal_rounds=12,
        rank_profiles=36,selected_rank_forwards=864,actual_layer_calls=25920,
        ppt_slides=12,pdf_pages=12,notes_slides=12,shape_bounds_checked=len(bounds),
        no_run_authentication_values_in_deliverables=True,
        cleaned_and_released_runs=['a100_r2','h100_r2'],
        visual_review='all-slide contact sheet and full-size slides 5,6,11 inspected',
        rendering_limit='PIL layout previews and PDF checked; no PowerPoint/LibreOffice runtime installed')
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
