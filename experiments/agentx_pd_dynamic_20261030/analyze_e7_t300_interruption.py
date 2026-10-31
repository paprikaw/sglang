"""Keep the shared-lease interruption visible outside the active cohort."""
from analyze_e6_controls import main

if __name__ == '__main__':
    main(expected_commit='797f6488cf5fea2f88e38ccaeb1703443deec537',
         run_names=('eight_gpu_pd_ep4_hicache_retry_e7_t300_r1',),
         cohort='E7_T300_INTERRUPTED_R1')
