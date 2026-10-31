"""Frozen, hardware-specific accepted adapter plus a diagnostic-only selector."""
import os


def install():
    if os.environ['EP_STEADY_HARDWARE'] == 'A100':
        import a100_runtime_adapter
        a100_runtime_adapter.install()
    else:
        import h100_runtime_adapter
        h100_runtime_adapter.install()
        import diagnostic_switch
        diagnostic_switch.install()

_previous_install = install
def install():
    _previous_install()
    import fragmentation
    fragmentation.install()
    import ep_fragmentation
    ep_fragmentation.install()

_shared_gemm_install = install
def install():
    _shared_gemm_install()
    if os.environ["EP2_PP2_ARM"] == "pp2":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
