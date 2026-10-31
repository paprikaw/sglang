import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
OUT = ROOT / 'experiments/experiment_summary/sglang_version_tracking_20261031'
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, GIT_AUTHOR_NAME='Xu Bai', GIT_AUTHOR_EMAIL='bxb1@spartan-login2.hpc.unimelb.edu.au', GIT_COMMITTER_NAME='Xu Bai', GIT_COMMITTER_EMAIL='bxb1@spartan-login2.hpc.unimelb.edu.au')
NEW = []

def git(path, *args, env=None, data=None):
    return subprocess.check_output(['git', '-C', str(path), *args], input=data, env=env, stderr=subprocess.PIPE).decode().strip()

def snapshot(source, branch, parent, groups):
    source = Path(source)
    ref = 'refs/heads/' + branch
    if subprocess.run(['git', '-C', str(source), 'show-ref', '--verify', '--quiet', ref]).returncode == 0:
        raise RuntimeError('Ref already exists: ' + ref)
    base = git(source, 'rev-parse', parent)
    tip = base
    f = tempfile.NamedTemporaryFile(prefix='sglang_tracking_index_', delete=False)
    f.close()
    os.unlink(f.name)
    env = dict(ENV, GIT_INDEX_FILE=f.name)
    git(source, 'read-tree', tip, env=env)
    commits = []
    for message, names in groups:
        paths = sorted(set(names))
        if not paths:
            continue
        records = []
        for name in paths:
            p = source / name
            if not p.is_file() or p.is_symlink():
                raise RuntimeError('Expected regular source file: ' + str(p))
            content = p.read_bytes()
            if re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)', content):
                raise RuntimeError('Credential pattern in source: ' + str(p))
            records.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'blob': hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest(), 'bytes': len(content)})
        with tempfile.NamedTemporaryFile(prefix='sglang_tracking_paths_') as listing:
            listing.write(b'\0'.join(p.encode() for p in paths) + b'\0')
            listing.flush()
            git(source, 'add', '-f', '--pathspec-from-file=' + listing.name, '--pathspec-file-nul', env=env)
        tree = git(source, 'write-tree', env=env)
        if tree == git(source, 'rev-parse', tip + '^{tree}'):
            continue
        body = message + '\n\nArchive the current source without changing the original checkout, index,\nbranch, frozen experiment tags, or runtime. This preserves source history;\nit does not assert a new GPU qualification or performance result.\n\nOriginal worktree: ' + str(source) + '\nOriginal base: ' + base + '\n'
        commit = git(source, 'commit-tree', tree, '-p', tip, env=env, data=body.encode())
        blobs = {}
        for line in git(source, 'ls-tree', '-r', commit, '--', *paths).splitlines():
            meta, name = line.split('\t', 1)
            blobs[name] = meta.split()[2]
        for rec in records:
            if blobs.get(rec['path']) != rec['blob']:
                raise RuntimeError('Snapshot changed while reading: ' + rec['path'])
        tip = commit
        commits.append({'commit': commit, 'message': message, 'files': records})
        print(json.dumps({'commit': commit, 'branch': branch, 'files': len(paths), 'message': message}), flush=True)
    git(source, 'update-ref', ref, tip, '0' * 40)
    os.unlink(f.name)
    NEW.append({'source': str(source), 'branch': branch, 'base': base, 'head': tip, 'commits': commits})
    (OUT / 'new_commits.json').write_text(json.dumps(NEW, indent=2) + '\n')
    return tip

if __name__ == '__main__':
    root_pd = ['python/sglang/srt/disaggregation/common/conn.py', 'python/sglang/srt/disaggregation/mooncake/conn.py', 'python/sglang/srt/mem_cache/memory_pool.py', 'test/registered/disaggregation/test_disaggregation_pp.py', 'test/registered/unit/disaggregation/test_disaggregation_wire.py']
    root_verify = ['python/sglang/srt/environ.py', 'python/sglang/srt/layers/attention/triton_backend.py', 'python/sglang/kernels/ops/attention/verify_as_decode.py', 'test/registered/attention/test_verify_splitkv.py']
    snapshot(ROOT, 'codex/archive/sglang-development-20261031', 'HEAD', [('fix(pd): preserve native MHA PP1-to-PP2 transfer work', root_pd), ('feat(attention): archive opt-in SM80 causal verify decode path', root_verify)])
    repair = ROOT.parent / '.worktrees/e2e-repair'
    names = git(repair, 'diff', '--name-only', 'HEAD').splitlines()
    snapshot(repair, 'codex/archive/elastic-pp-fence-20261031', 'HEAD', [('fix(elastic-pp): snapshot live KV only at the drained scheduler fence', names)])
    tbo = ROOT / '.engine-worktrees/qwen35-ep2-tbo-20261027'
    names = git(tbo, 'diff', '--name-only', 'HEAD').splitlines() + ['python/sglang/srt/batch_overlap/qwen35_tbo.py']
    snapshot(tbo, 'codex/archive/qwen35-ep2-tbo-wip-20261031', 'HEAD', [('wip(qwen35): preserve experimental two-batch overlap implementation', names)])
    guard = ROOT / '.guard-worktrees/qwen-capacity-20261017'
    names = git(guard, 'diff', '--name-only', 'HEAD').splitlines()
    snapshot(guard, 'codex/archive/qwen-capacity-adapter-wip-20261031', 'HEAD', [('wip(cache): preserve Qwen capacity admission adapter changes', names)])
    selected = set(json.loads(Path('/tmp/sglang_filtered_sources_20261031.json').read_text()))
    # Frozen source copies are reproducibility inputs, unlike runtime logs and caches.
    topics = sorted({Path(p).parts[1] for p in selected})
    allowed = {'.py', '.sh', '.cu', '.cuh', '.cpp', '.h', '.patch', '.diff', '.md', '.yaml', '.yml', '.toml'}
    for topic in topics:
        for code in (ROOT / 'experiments' / topic).glob('runs/*/code'):
            for p in code.rglob('*'):
                if p.is_file() and not p.is_symlink() and p.suffix in allowed and '__pycache__' not in p.parts:
                    selected.add(str(p.relative_to(ROOT)))
    for p in ROOT.joinpath('tools').rglob('*'):
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:
            selected.add(str(p.relative_to(ROOT)))
    # Preserve the deliberately source-only dashboard worktree without its
    # unrelated staged deletion of the complete SGLang engine tree.
    runner = ROOT / '.runner-worktrees/experiment-dashboard-guard'
    dashboard_files = git(runner, 'ls-files', '--others', '--exclude-standard', 'tools').splitlines()
    snapshot(runner, 'codex/archive/experiment-dashboard-source-20261031', 'HEAD', [('chore(experiments): archive source-only experiment dashboard helpers', dashboard_files)])
    groups = {}
    for p in sorted(selected):
        parts = Path(p).parts
        if parts[0] == 'tools': key = 'tools/' + parts[1]
        elif parts[1] == 'experiment_summary': key = 'reports'
        elif 'agentx' in parts[1]: key = 'agentx'
        elif any(k in parts[1] for k in ['ep_pp', 'expert_surface', 'token_comm', 'single_layer', 'bandwidth', 'deepep']): key = 'component-and-communication'
        elif any(k in parts[1] for k in ['pp', 'prefill']): key = 'pipeline-and-prefill'
        else: key = 'benchmark-support'
        groups.setdefault(key, []).append(p)
    source_manifest = {'root': str(ROOT), 'selection': 'October 20-31 experiment source and frozen runs/*/code; compact summaries; local tools. Large logs, datasets, weights, KV bytes, generated trace payloads and upstream audit copies excluded.', 'file_count': len(selected), 'groups': {k: len(v) for k, v in groups.items()}}
    (OUT / 'source_selection.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    snapshot(ROOT, 'codex/sglang-optimization-records-20261031', 'fa52c3cd95d2c171c9eac9a8bfdc32296aa8c75f', [('chore(research): version ' + key + ' optimization sources and evidence', paths) for key, paths in sorted(groups.items())])
    print(json.dumps({'new_branches': len(NEW), 'new_commits': sum(len(x['commits']) for x in NEW), 'source_selection': source_manifest}), flush=True)
