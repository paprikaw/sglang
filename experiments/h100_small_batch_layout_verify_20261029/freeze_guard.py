"""Retain the qualified Qwen admission contract and current shared lock ABI."""
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
WS=ROOT.parents[1]
BASE=WS/'experiments/qwen_a100_capacity_20261019_r1/guard_bundle'
CURRENT=Path('/home/bxb1/.codex/skills/llm-experiment-guard/scripts/gpu_registry.py')
TARGET=ROOT/'guard_bundle_posix_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    spec=importlib.util.spec_from_file_location('registry_under_test',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def contender(source,root,events):
    module=load(Path(source));r=module.SharedGpuRegistry(root)
    events.put('started')
    with r._locked():events.put('locked')


def main():
    if TARGET.exists():raise RuntimeError('Guard bundle is immutable; choose a new version')
    subprocess.run([sys.executable,'/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py'],check=True,stdout=subprocess.DEVNULL)
    shutil.copytree(BASE,TARGET,ignore=shutil.ignore_patterns('__pycache__'))
    (TARGET/'manifest.json').rename(TARGET/'BASE_MANIFEST.json')
    registry=TARGET/'skills/llm-experiment-guard/scripts/gpu_registry.py'
    shutil.copyfile(CURRENT,registry)
    differences=[]
    for p in BASE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json':
            q=TARGET/p.relative_to(BASE)
            if sha(p)!=sha(q):differences.append(str(p.relative_to(BASE)))
    assert differences==['skills/llm-experiment-guard/scripts/gpu_registry.py'],differences
    adapter=TARGET/'skills/llm-experiment-guard/scripts/model_cache_adapter.py'
    assert sha(adapter)==sha(BASE/'skills/llm-experiment-guard/scripts/model_cache_adapter.py')
    assert 'qwen3_5moe-canonical-v1' in adapter.read_text()
    assert sha(registry)==sha(CURRENT)
    # Both directions must honor the same writer lock. Test only a temporary
    # private registry; never claim nodes or edit the real shared registry here.
    ctx=multiprocessing.get_context('spawn');checks=[]
    for first,second in [(CURRENT,registry),(registry,CURRENT)]:
        with tempfile.TemporaryDirectory(prefix='guard-posix-compat-') as tmp:
            module=load(first);r=module.SharedGpuRegistry(tmp);events=ctx.Queue()
            with r._locked():
                child=ctx.Process(target=contender,args=(str(second),tmp,events));child.start()
                assert events.get(timeout=10)=='started'
                try:events.get(timeout=.25);raise AssertionError('Second writer entered an already locked registry')
                except queue.Empty:pass
            assert events.get(timeout=10)=='locked';child.join(timeout=10);assert child.exitcode==0
            checks.append(dict(first=str(first),second=str(second),mutual_exclusion=True))
    files=[dict(path=str(p),sha256=sha(p)) for p in sorted(TARGET.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json']
    manifest=dict(created_at=time.time(),base_bundle=str(BASE),base_manifest_sha256=sha(BASE/'manifest.json'),
        change='Replace only the GPU registry module with the current POSIX-lock implementation. '
        'Qwen support, source identity, capacity admission, authenticated cache usage and cleanup rules are unchanged.',
        actual_registry_source=dict(path=str(CURRENT),sha256=sha(CURRENT)),changed_files=differences,
        cache_adapter_sha256=sha(adapter),cpu_lock_tests=checks,frozen_inputs=files)
    (TARGET/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(bundle=str(TARGET),changed_files=differences,cpu_lock_tests_passed=len(checks),files=len(files))))

if __name__=='__main__':main()
