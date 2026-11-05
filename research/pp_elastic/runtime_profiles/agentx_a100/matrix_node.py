"""Private node-local collector control and post-window artifact transfer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--level', type=int)
    p.add_argument('--export', type=Path)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = a.root / 'control'
    if a.level is not None:
        assert 0 <= a.level <= 2
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            os.pwrite(fd, bytes([a.level]), 0)
        finally:
            os.close(fd)
    files = []
    if a.export:
        a.export.mkdir(parents=True, exist_ok=True)
        for source in a.root.rglob('*.jsonl'):
            dest = a.export / source.relative_to(a.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.tmp')
            shutil.copyfile(source, tmp)
            tmp.replace(dest)
            files.append(dict(path=str(dest), bytes=dest.stat().st_size))
    print(json.dumps(dict(node=socket.gethostname(), monotonic_s=time.perf_counter(),
                         unix_ns=time.time_ns(), level=path.read_bytes()[0], files=files)))


if __name__ == '__main__':
    main()
