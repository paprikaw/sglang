"""Matched E7 source, V4 branch timing and finite-episode deadline cohort."""
from pathlib import Path
import subprocess
import sys

from analyze_e6_controls import main

COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'
RUNS = ('eight_gpu_ep4_queue32_e7_v4_b7200_r1',
        'eight_gpu_pp2_pd_e7_v4_b7200_r1',
        'eight_gpu_ep2_split62_e7_v4_b7200_r1',
        'eight_gpu_col_single_e7_v4_b7200_r1')

if __name__ == '__main__':
    main(expected_commit=COMMIT, run_names=RUNS, cohort='E7_V4_B7200')
    root = Path(__file__).resolve().parent
    for name in RUNS:
        run = root/'runs'/name
        if any(run.glob('results/*/*/*/completed.r*.json')):
            subprocess.run([sys.executable, str(root/'analyze_forward_batches.py'),
                            '--run', str(run)], check=True)
            subprocess.run([sys.executable, str(root/'analyze_cache_owner_pressure.py'),
                            '--run', str(run)], check=True)
        if any(run.glob('results/pd_*/*/*/completed.r*.json')):
            subprocess.run([sys.executable, str(root/'analyze_pd_waits.py'),
                            '--run', str(run)], check=True)
