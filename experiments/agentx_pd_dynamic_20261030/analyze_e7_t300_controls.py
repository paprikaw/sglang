"""E7/native-deadline cohort, kept separate from the failed 120-second attempt."""
from pathlib import Path
import subprocess
import sys

from analyze_e6_controls import main

COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'
RUNS = ('eight_gpu_pd_ep4_hicache_retry_e7_t300_r2',
        'eight_gpu_ep4_native_queue32_e7_t300_r2',
        'eight_gpu_pp2_pd_controls_e7_t300_r1',
        'eight_gpu_ep2_split26_e7_t300_r1')

if __name__ == '__main__':
    main(expected_commit=COMMIT, run_names=RUNS, cohort='E7_T300')
    root = Path(__file__).resolve().parent
    for name in RUNS:
        run = root/'runs'/name
        if any(run.glob('results/pd_*/*/*/completed.r*.json')):
            subprocess.run([sys.executable, str(root/'analyze_pd_waits.py'),
                            '--run', str(run)], check=True)
