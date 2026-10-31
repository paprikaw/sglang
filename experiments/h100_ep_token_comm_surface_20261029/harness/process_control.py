"""Owned-process cleanup adapted from the validated communication launcher."""
import os
from pathlib import Path
import signal
import subprocess
import time

def process_record(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'parent': int(fields[1]), 'ticks': int(fields[19]), 'state': fields[0]}
    except (OSError, ValueError, IndexError):
        return None

def track_descendants(root_pid, owned):
    records = [record for path in Path('/proc').iterdir() if path.name.isdigit()
               if (record := process_record(int(path.name))) is not None]
    # Only currently matching identities may be parents. A stale owned PID
    # could have been reused by an unrelated process after it exited.
    parents = {record['pid'] for record in records
               if owned.get(record['pid']) == record['ticks']}
    changed = True
    while changed:
        changed = False
        for record in records:
            if record['pid'] not in parents and record['parent'] in parents:
                owned[record['pid']] = record['ticks']
                parents.add(record['pid'])
                changed = True

def alive_owned(owned):
    return [pid for pid, ticks in owned.items()
            if (record := process_record(pid)) and record['ticks'] == ticks and record['state'] != 'Z']

def stop_owned(child, owned):
    # Rank workers may start their own sessions: torchrun's process group alone
    # is insufficient. Start ticks prevent signalling an unrelated reused PID.
    track_descendants(child.pid, owned)
    for signum, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        for pid in alive_owned(owned):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while alive_owned(owned) and time.monotonic() < deadline:
            track_descendants(child.pid, owned)
            time.sleep(.05)
        if not alive_owned(owned):
            break
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return not alive_owned(owned)
