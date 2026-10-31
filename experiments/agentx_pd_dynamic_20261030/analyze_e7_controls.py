"""Independent E7 table; never pool previous-source performance samples."""
from analyze_e6_controls import main

COMMIT = '797f6488cf5fea2f88e38ccaeb1703443deec537'
RUNS = ('eight_gpu_pd_ep4_hicache_retry_e7_r1',
        'eight_gpu_ep4_native_queue32_e7_r1',
        'eight_gpu_pp2_pd_controls_e7_r1',
        'eight_gpu_ep2_split26_e7_r1')

if __name__ == '__main__':
    main(expected_commit=COMMIT, run_names=RUNS, cohort='E7')
