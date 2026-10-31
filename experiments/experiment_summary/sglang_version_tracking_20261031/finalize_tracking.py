"""Commit and publish the archive index without changing an engine worktree."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path('/data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling')
OUT = Path(__file__).resolve().parent
BRANCH = 'codex/sglang-optimization-records-20261031'
TAG = 'archive/sglang-optimizations-20261031'
ENV = dict(os.environ, GIT_AUTHOR_NAME='Xu Bai', GIT_AUTHOR_EMAIL='bxb1@spartan-login2.hpc.unimelb.edu.au', GIT_COMMITTER_NAME='Xu Bai', GIT_COMMITTER_EMAIL='bxb1@spartan-login2.hpc.unimelb.edu.au')

def git(*args, env=None, data=None):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], input=data, env=env, stderr=subprocess.PIPE).decode().strip()

if __name__ == '__main__':
    old = git('rev-parse', BRANCH)
    assert old == '473e0c8fc1de85c4e4c691e04aaa6ce80d0c2e56', 'Archive already finalized or changed'
    index = tempfile.mktemp(prefix='sglang_tracking_final_index_')
    env = dict(ENV, GIT_INDEX_FILE=index)
    git('read-tree', old, env=env)
    paths = [str(p.relative_to(ROOT)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'final_receipt.json' and p.suffix != '.log']
    paths += ['experiments/experiment_summary/README.md']
    with tempfile.NamedTemporaryFile() as listing:
        listing.write(b'\0'.join(p.encode() for p in paths) + b'\0'); listing.flush()
        git('add', '-f', '--pathspec-from-file=' + listing.name, '--pathspec-file-nul', env=env)
    ignore = git('show', old + ':.gitignore') + '''

# Spartan experiment runtime data. Track source and compact reports explicitly.
/.engine-worktrees/
/.runner-worktrees/
/.guard-worktrees/
/.third_party/
/.dashboard-template-*/
/datasets/
/results/
/experiment-report-hub*/
/experiments/**/runs/
/experiments/**/logs/
/experiments/**/profiles/
/experiments/**/traces/
'''
    blob = git('hash-object', '-w', '--stdin', data=ignore.encode())
    git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',.gitignore', env=env)
    tree = git('write-tree', env=env)
    commit = git('commit-tree', tree, '-p', old, env=ENV, data=b'docs: index and verify SGLang optimization source archives\n\nPreserve per-change source hashes, original worktree identities, remote\nverification, CPU validation limits, and reproducible archival helpers.\nIgnore runtime data on this archival branch. No GPU runtime is changed.\n')
    git('update-ref', 'refs/heads/' + BRANCH, commit, old)
    git('tag', '-a', TAG, commit, '-m', 'Immutable SGLang optimization archive: 15 source commits, source and replay histories, provenance and verification. Archival status does not imply GPU qualification.', env=ENV)
    proc = subprocess.run(['git', '-C', str(ROOT), 'push', '--atomic', 'origin', 'refs/heads/' + BRANCH + ':refs/heads/' + BRANCH, 'refs/tags/' + TAG + ':refs/tags/' + TAG], capture_output=True, text=True, timeout=240)
    (OUT / 'final_push.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    verification = json.loads((OUT / 'remote_verification.json').read_text())
    for item in verification:
        repo = {'sglang': ROOT, 'replay': Path('/data/gpfs/projects/punim2715/vllm_workbench/sglang-cpu-kv-decode-replay'), 'guard': ROOT / '.guard-worktrees/qwen-capacity-20261017'}[item['label']]
        text = subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin'], text=True)
        remote = {s.split()[1]: s.split()[0] for s in text.splitlines()}
        for record in item['refs']:
            expected = commit if item['label'] == 'sglang' and record['ref'] == 'refs/heads/' + BRANCH else record['sha']
            assert remote.get(record['ref']) == expected, record
        before = (OUT / ('remote_before_' + {'sglang': '0', 'replay': '1', 'guard': '2'}[item['label']] + '.tsv')).read_text()
        before = {s.split()[1]: s.split()[0] for s in before.splitlines()}
        assert remote.get('refs/heads/main') == before.get('refs/heads/main')
        if item['label'] == 'sglang':
            assert remote.get('refs/tags/' + TAG) == git('rev-parse', 'refs/tags/' + TAG)
            assert remote.get('refs/tags/' + TAG + '^{}') == commit
    for branch in [BRANCH, 'codex/agentx-mha-pp-kv-layer-map-20261031']:
        git('branch', '--set-upstream-to=origin/' + branch, branch)
    receipt = {'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'archive_commit': commit, 'source_snapshot_commit': old, 'archive_branch': BRANCH, 'archive_tag': TAG, 'new_source_commits': 15, 'new_documentation_commits': 1, 'sglang_branches_verified': 83, 'sglang_tags_verified': 43, 'replay_branches_verified': 17, 'guard_branches_verified': 1, 'all_remote_refs_match': True, 'all_main_branches_unchanged': True}
    (OUT / 'final_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    os.unlink(index)
    print(json.dumps(receipt), flush=True)
