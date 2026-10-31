"""Download immutable official AgentX data; never execute downloaded code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / 'datasets/agentx_20261030'
REPOS = {
    'cc-traces-weka-062126': '23f152f6f0f9399a85901b89a6458def0ef16729',
    'cc-traces-weka-062126-256k': '8fecd2fc56694469f758f0afbbb6335ad3043740',
}

def download(repo, revision):
    dest = ROOT / repo
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    tree = json.loads((ROOT / (repo + '-tree.json')).read_text())
    for item in tree:
        if item['type'] != 'file':
            continue
        name = item['path']
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/semianalysisai/{repo}/resolve/{revision}/{name}?download=true'
        partial = target.with_suffix(target.suffix + '.partial')
        if not target.exists():
            for attempt in range(3):
                try:
                    h = hashlib.sha256()
                    count = 0
                    with urllib.request.urlopen(url, timeout=120) as src, partial.open('wb') as out:
                        while chunk := src.read(8 * 1024 * 1024):
                            h.update(chunk)
                            count += len(chunk)
                            out.write(chunk)
                    assert count == item['size'], (name, count, item['size'])
                    expected = item.get('lfs', {}).get('oid')
                    if expected:
                        assert h.hexdigest() == expected, (name, 'LFS SHA256 mismatch')
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        with target.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert target.stat().st_size == item['size']
        expected = item.get('lfs', {}).get('oid')
        if expected:
            assert digest == expected
        records.append(dict(path=str(target), bytes=target.stat().st_size, sha256=digest, url=url))
        print(json.dumps(dict(repo=repo, file=name, bytes=target.stat().st_size)), flush=True)
    manifest = dict(repo='semianalysisai/' + repo, revision=revision, license='Apache-2.0',
                    completed_unix=time.time(), files=records)
    (dest / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(lambda p: download(*p), REPOS.items()))
    (ROOT / 'manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')
