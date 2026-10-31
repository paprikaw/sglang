import os
def install():
    import h100_runtime_adapter, diagnostic_switch, fragmentation, ep_fragmentation, forward_observer
    h100_runtime_adapter.install()
    diagnostic_switch.install()
    fragmentation.install()
    ep_fragmentation.install()
    if os.environ.get("BENCH_PP2_LOOKAHEAD") == "1":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    forward_observer.install()

    import cache_probe
    cache_probe.install()
