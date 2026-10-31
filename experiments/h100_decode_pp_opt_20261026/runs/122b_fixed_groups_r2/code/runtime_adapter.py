"""Native model mathematics plus private PP scheduling."""
def install():
    import h100_runtime_adapter
    import pp_schedule_opt
    h100_runtime_adapter.install()
    pp_schedule_opt.install()
    import pp_transfer_audit, pp_fixed_cohort
    pp_transfer_audit.install()
    pp_fixed_cohort.install()
