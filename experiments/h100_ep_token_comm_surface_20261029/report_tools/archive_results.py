"""Publish a verified CPU-only report snapshot; never copy private Guard tokens."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BASE=Path(__file__).resolve().parents[1]
WORKSPACE=BASE.parents[1]
MANAGER=Path('/data/gpfs/projects/punim2715/vllm_workbench/experiment_manager')
ID='h100_ep_token_comm_surface_20261029'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def update_index(path,body):
    start,end=f'<!-- begin {ID} -->',f'<!-- end {ID} -->'
    for _ in range(5):
        original=path.read_bytes()
        text=original.decode()
        if start in text:
            assert text.count(start)==text.count(end)==1
            a=text.index(start)
            b=text.index(end)+len(end)
            revised=text[:a]+body+text[b:]
        else:
            revised=text.rstrip()+'\n\n'+body+'\n'
        if revised.encode()==original:
            return
        fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
        with os.fdopen(fd,'wb') as stream:
            stream.write(revised.encode())
        os.chmod(name,path.stat().st_mode&0o777)
        if path.read_bytes()!=original:
            Path(name).unlink()
            continue
        os.replace(name,path)
        return
    raise RuntimeError('Concurrent registry change; snapshot preserved, index update deferred')


def archive(run):
    cfg=json.loads((run/'config.json').read_text())
    comparison=json.loads((run/'comparison.json').read_text())
    cleanup=json.loads((run/'task/cleanup.json').read_text())
    assert comparison['passed'] and comparison['aggregation']=='none'
    assert cleanup['passed'] and cleanup['gpu_idle'] and cleanup['model_processes_exited']
    spec=json.loads((run/'dashboard_spec.json').read_text())
    for item in spec['frozen_inputs']:
        assert sha(item['path'])==item['sha256'],item['path']
    # A postcheck must establish both scientific completion and owned cleanup.
    post=[]
    for path in sorted((run/'dashboard_guard/checks').glob('*-post.json')):
        obj=json.loads(path.read_text())
        post.append((path,obj))
    assert post, 'Guard postcheck required before publishing'
    assert any(obj.get('passed') is True for _,obj in post), 'Guard postcheck did not pass'
    target=MANAGER/'experiments'/ID
    assert not target.exists(), 'Preserve an existing snapshot; use explicit versioning'
    staged=BASE/'archive_staging'
    assert not staged.exists()
    staged.mkdir()
    subprocess.run(['/home/bxb1/conda/bin/python',str(BASE/'report_tools/build_report.py'),
        '--run',str(run),'--out',str(staged)],check=True)
    shutil.copy2(run/'task/cleanup.json',staged/'cleanup.json')
    shutil.copytree(run/'frozen/harness',staged/'frozen_harness',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,staged/'archive_results.py')
    shutil.copy2(BASE/'report_tools/build_report.py',staged/'build_report.py')
    shutil.copy2(BASE/'report_tools/audit_results.py',staged/'audit_results.py')
    assert json.loads((run/'independent_audit.json').read_text())['passed']
    shutil.copy2(run/'independent_audit.json',staged/'independent_audit.json')
    shutil.copy2(run/'environment_metadata.json',staged/'environment_metadata.json')
    assert json.loads((run/'geometry_sources.json').read_text())['passed']
    shutil.copy2(run/'geometry_sources.json',staged/'geometry_sources.json')
    rank_hashes=[]
    for phase in cfg['phases']:
        dest=staged/'rank_results'/phase['id']
        dest.mkdir(parents=True)
        for path in sorted((run/phase['id']).glob('result_rank*.json')):
            shutil.copy2(path,dest/path.name)
            rank_hashes.append(dict(path=str(path),sha256=sha(path)))
        for path in sorted((run/phase['id']).glob('*_topology.json')):
            shutil.copy2(path,dest/path.name)
    for node in cfg['nodes']:
        shutil.copy2(run/f'{node}_nic.json',staged/f'{node}_nic.json')
    for path,obj in post:
        # Postcheck contains source/result/cleanup checks, not authentication tokens.
        assert '"token"' not in json.dumps(obj)
        save(staged/'guard_postcheck'/path.name,obj)
    failed=[]
    for label in ('r1','r2','r3'):
        old=BASE/'runs'/label
        if not old.exists():
            continue
        row=dict(run=str(old),failure=json.loads((old/'controller_failure.json').read_text()),
            cleanup=json.loads((old/'task/cleanup.json').read_text()),accepted_measurements=0,
            preserved_point_files=len(list(old.glob('r*_*/point_*_rank0.json'))))
        if (old/'cpu_binding_diagnosis.json').exists():
            row['diagnosis']=json.loads((old/'cpu_binding_diagnosis.json').read_text())
        failed.append(row)
    save(staged/'failed_attempts.json',failed)
    sources=dict(source_run=str(run),spec_sha256=sha(run/'dashboard_spec.json'),
        frozen_inputs_sha256=sha(run/'frozen_inputs.json'),implementation=spec['sources'],
        native=cfg['native'],rank_results=rank_hashes,
        result_sha256=sha(run/'comparison.json'),cleanup_sha256=sha(run/'task/cleanup.json'),
        report_generator_sha256=sha(BASE/'report_tools/build_report.py'),
        default_presentation=dict(aggregation='none',selected_event_1based=1,rounds_separate=True),
        failed_history='failed_attempts.json',
        canonical_destination=str(target))
    save(staged/'sources.json',sources)
    (staged/'REPRODUCE.md').write_text(f'''# Reproduction

The accepted original run is `{run}`. All GPU measurements are frozen in that
run; no completed run or shared native library was edited. Read `REPORT.md`
for the exact original-token, chunk, topology, dtype and timing definitions.

CPU report: `/home/bxb1/conda/bin/python {target}/build_report.py --run {run} --out NEW_REPORT_DIR`.

GPU rerun: copy the frozen harness to a fresh directory, refresh only currently
RUNNING booked node/job identities, create a fresh run using `prepare.py`, and
run its `launch.json` command through the canonical Experiment Guard. Existing
CPU model/KV banks are not used or altered. No Slurm allocation is submitted;
workers run as normal steps of the specified existing Slurm jobs.

Every table cell is original aligned event 1/30, with no averaging or median.
All 30 iterations and all ranks are in CSV. N4096/8192/16384/32768 use measured
2/4/8/16 chunks of up to2048 global tokens identically across EP2/4/8.
EP8 spans two nodes; EP2/EP4 are node-local NVLink.
''')
    artifacts={str(path.relative_to(staged)):sha(path) for path in sorted(staged.rglob('*')) if path.is_file()}
    manifest=dict(id=ID,title='H100 EP2/4/8 × global original token communication surface, four geometries',
        created_at=datetime.now(timezone.utc).isoformat(),status='complete',source_run=str(run),
        implementation=spec['sources'],native=cfg['native'],testbed=dict(gpu='H100 80GB HBM3',nodes=cfg['nodes'],
            booked_jobs=cfg['booked_jobs'],ep=[2,4,8],placement='EP2/EP4 node-local NVLink; EP8 two-node NVLink + RDMA'),
        workload=dict(shapes=cfg['shapes'],global_original_tokens=cfg['token_sweep'],topk=8,dtype='bfloat16',
            global_chunk_limit=cfg['global_chunk_limit'],model_loaded=False,expert_compute=False,attention_compute=False),
        metrics=dict(formal_points=comparison['formal_points'],raw_rank_events=comparison['raw_rank_events'],
            selected_event_1based=1,aggregation='none',rounds_separate=True),
        guard_postcheck_passed=True,cleanup_verified=True,artifacts=artifacts,
        limits=['BF16 synthetic identity expert only','equal global2048-token chunking, large N is multi-call',
            'EP8 also changes communication domain','raw individual samples fluctuate','no PP/e2e/overlap measurements'])
    save(staged/'manifest.json',manifest)
    # Publish only after the human-readable report and all provenance are concrete.
    shutil.copytree(staged,target)
    entry=f'''<!-- begin {ID} -->
### H100：EP2/EP4/EP8 × 总 token 通信二维表（2026-10-29）

四种激活尺寸 GLM/HY3/35B/122B，统一 BF16 / top8，固定全 EP 组原始 token 总数8–32768。
EP2/EP4 单节点 NVLink，EP8 两节点 NVLink+RDMA；统一全组2048-token 分块，大档为完整2/4/8/16次交换实测。
624点、87360条 rank原始事件；每格固定事件1，分轮显示，Dispatch/Combine/合计绝对ms，无平均或relative；Guard校验与清理通过。
[报告]({target}/REPORT.md) · [交互表格]({target}/report.html) · [二维图]({target}/total_ms_graph_round1.png) · [原始CSV]({target}/absolute_event01.csv)
<!-- end {ID} -->'''
    for index in (Path('/home/bxb1/.codex/skills/llm-experiment-manager/references/experiment_registry.md'),
            WORKSPACE/'experiments/experiment_summary/README.md'):
        update_index(index,entry)
    return dict(snapshot=str(target),artifacts=len(artifacts),formal_points=624,raw_rank_events=87360)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(archive(parser.parse_args().run.resolve())))
