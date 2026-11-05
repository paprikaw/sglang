"""A100 BF16 native Triton with E10 cache and decode instrumentation."""
import os
def install():
    import a100_runtime_adapter, runtime_setup, pp_ready_send
    import torch
    a100_runtime_adapter.install()
    runtime_setup.install()
    pp_ready_send.install(torch)
    if os.environ.get('BENCH_PP2_LOOKAHEAD') == '1':
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    import forward_observer, cache_probe, phase_timing, matrix_observer
    forward_observer.install()
    cache_probe.install()
    phase_timing.install()
    matrix_observer.install()
