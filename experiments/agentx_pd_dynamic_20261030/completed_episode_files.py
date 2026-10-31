"""One completion gate for offline episode reports; no serving-side imports."""
import hashlib
import json
from pathlib import Path


def completed_formals(run):
    complete, pending = [], []
    for path in sorted(run.glob('results/*/*/*/formal.r*.json')):
        marker = path.with_name('completed.'+path.stem.split('.')[-1]+'.json')
        if not marker.exists():
            pending.append(str(path))
            continue
        assert json.loads(marker.read_text()).get('passed') is True, marker
        complete.append(path)
    return complete, pending


def gate_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
