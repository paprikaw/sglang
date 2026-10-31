"""Archive small verified artifacts, not weights, KV, credentials or raw traces."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run):
    root = Path(__file__).resolve().parent
    analysis = run/'analysis'
    verified = json.loads((analysis/'validation.json').read_text())
    cfg = json.loads((run/'config.json').read_text())
    assert verified['passed'] and verified['trace_rows']==14
    assert json.loads((run/'worker_cleanup.json').read_text())['cleanup_verified']
    post = json.loads(Path(verified['guard_postcheck']).read_text())
    assert post['passed'] and post['scientific_success'] and post['cleanup_verified']
    result = subprocess.run([sys.executable,'-m','unittest','-v','test_protocol'],cwd=root,
        text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    (analysis/'tests.txt').write_text(result.stdout+result.stderr)
    ident='qwen_h100_ep2_pp2_prefill_components_20261025_r1'
    manager=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager')
    target=manager/'experiments'/ident
    target.mkdir(mode=0o700)
    copied=[]
    for p in sorted(analysis.iterdir()):
        if p.is_file():
            shutil.copyfile(p,target/p.name);copied.append(p.name)
    for name in ('topology.txt','devices.csv','completed.json','worker_cleanup.json'):
        shutil.copyfile(run/name,target/name);copied.append(name)
    with zipfile.ZipFile(target/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((run/'code').glob('*.py')):
            z.write(p,'frozen_run/'+p.name)
        for name in ('prepare_prefill_followup.py','prefill_followup.py','prefill_analysis.py',
                     'publish_prefill.py','archive_prefill.py','test_protocol.py','protocol.py'):
            z.write(root/name,'analysis_and_future_fix/'+name)
    copied.append('source.zip')
    with zipfile.ZipFile(target/'source.zip') as z:assert z.testzip() is None
    sources=[dict(path=str(run/'config.json'),sha256=sha(run/'config.json')),
             dict(path=verified['guard_postcheck'],sha256=sha(Path(verified['guard_postcheck'])))]
    sources += [dict(path=t['trace'],sha256=t['sha256']) for t in verified['traces']]
    manifest=dict(schema_version=1,id=ident,status='complete',phase='prefill_component_diagnostic',
        model='Qwen3.5-122B-A10B-FP8',model_revision=cfg['model_revision'],
        engine_commit=cfg['engine_commit'],node=cfg['node'],gpu='4x H100 80GB; EP2 GPU0/1, PP2 GPU2/3; NV6',
        source_run=str(run),reference_run=cfg['reference_run'],trace_ranks=14,cells=7,
        units='GPU kernel cumulative milliseconds per card per complete Prefill cohort; overlapping, not TPOT',
        sources=sources,tests=9,guard_postcheck_passed=True,owned_cleanup_verified=True,
        limitations=['one trace per cell; profiler instrumentation and allocator retries',
          'EP extra overlap Decode excluded and separately recorded',
          'PP NCCL residence includes waits/control, not pure payload time',
          'one empty-export snapshot hash reconciled after verified writer shutdown; original manifest unchanged'],
        cpu_only_future_fix='source.zip analysis_and_future_fix is not substituted into the completed frozen run',
        created_at=time.time(),artifacts=copied)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={str(p.relative_to(target)):sha(p) for p in target.iterdir() if p.is_file()}
    (target/'checksums.json').write_text(json.dumps(files,indent=2)+'\n')
    assert all(sha(target/name)==digest for name,digest in files.items())
    registry=Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md')
    section=f'''\n\n<!-- begin {ident} -->
### {ident}

Snapshot: `{target}`; linked under Q03 in the general experiment summary.

- Supplemental actual full-model H100 EP2/PP2 Prefill-only traces: same native FP8 W8A8/DeepGEMM/FA3/GDN path, same node178 NV6 pairs and capacities as accepted r7. Three common workloads (8x fresh4096; 8x cached26752+fresh512; 16x cached13376+fresh512), plus PP-only16x cached42176+fresh512. Output1; no HiCache, synthetic KV or MTP. Existing complete CPU weight Bank reused.
- All14 ranks verify exact fresh-token totals and full layer scope; GPU kernels attributed through CPU launch correlation, not CPU wall duration. Whole Attention, routed/shared Expert, Dispatch and Combine separated. Per-card Attention work is close; PP Expert kernel work is about1.5-1.9x EP. Decode reference retains narrower core/GEMM scope: DeepEP kernel duration is69-82% of routed GEMM, not negligible or an exclusive critical-path share.
- Seven single-cohort diagnostic cells, not new formal performance metrics. EP native overlap executes one extra Decode after output-one Prefill; excluded and separately recorded. EP8-request long-prefix cell has a source-IDLE/arrival wait in Dispatch; PP NCCL residence includes waits/control. Allocator retries and instrumentation limits retained. One empty-file export snapshot reconciled after verified writer exit using gzip/JSON, token/layer counts, mtimes and stable hashes; original evidence unchanged. Future export-completion guard fixed and nine CPU tests pass.
- Guard cache/scientific/cleanup postcheck passed, GPU lease released. Artifacts: Markdown tables, per-rank/aggregate CSVs, PNG/SVG, provenance and reconciliation JSON, test output, original-versus-future source ZIP, manifest and verified checksums. No weights/KV/private launch credentials/raw traces copied.

<!-- end {ident} -->
'''
    with registry.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        if f'<!-- begin {ident} -->' not in f.read():
            f.seek(0,2);f.write(section);f.flush()
    print(json.dumps(dict(archived=True,path=str(target),verified_files=len(files),tests=9)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    archive(p.parse_args().run.resolve())
