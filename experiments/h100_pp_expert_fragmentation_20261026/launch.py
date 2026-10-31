"""Supervise the reviewed Qwen Guard, with canonical backfill notifications."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--run', type=Path, required=True)
args = ap.parse_args()
run = args.run.absolute()
prep = json.loads((run / 'preparation.json').read_text())
notify = ['python3', '/home/bxb1/.codex/skills/llm-experiment-guard/scripts/backfill_trigger.py']
subprocess.run(notify, check=True)
print('Starting Guard-supervised H100 run:', run.name, flush=True)
try:
    with (run / 'supervisor.log').open('ab', buffering=0) as log:
        result = subprocess.run(prep['command'], env=dict(os.environ, **prep['required_environment']),
                                stdout=log, stderr=subprocess.STDOUT)
finally:
    subprocess.run(notify, check=False)
print('Guard exit:', result.returncode, flush=True)
raise SystemExit(result.returncode)
