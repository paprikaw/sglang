"""Native A100 Expert kernel shared by EP and PP; optional PP scheduling only."""
import os
def install():
    import a100_runtime_adapter
    a100_runtime_adapter.install()
    if os.environ["EP2_PP2_ARM"] == "pp2":
        import gap_schedule, pp_fixed_cohort, pp_transfer_audit
        gap_schedule.install()
        pp_fixed_cohort.install()
        pp_transfer_audit.install()
    else:
        import prefill_admission
        prefill_admission.install()
